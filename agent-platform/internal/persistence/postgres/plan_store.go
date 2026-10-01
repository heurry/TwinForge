package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func preparePlanRevision(callName string, previousRevision int, previousGoal, previousExplanation string, previousSteps []taskplan.Step, update taskplan.Update) (taskplan.Update, error) {
	if callName != "update_plan" {
		update.Steps = taskplan.ReconcileSteps(previousSteps, update.Steps)
		return update, nil
	}
	mode := strings.ToLower(strings.TrimSpace(update.ChangeMode))
	if previousRevision == 0 {
		if mode != "" && mode != "create" {
			return update, tool.NewContractErrorWithRepair("PLAN_CHANGE_MODE_INVALID", callName, "/change_mode", "create", mode, "a Workflow without a Plan can only create its initial Plan", "Set change_mode=create and omit base_revision.", nil, true)
		}
		update.ChangeMode = "create"
		update.Steps = taskplan.ReconcileSteps(nil, update.Steps)
		return update, nil
	}
	if update.BaseRevision == nil || *update.BaseRevision != previousRevision {
		actual := "missing"
		if update.BaseRevision != nil {
			actual = fmt.Sprintf("%d", *update.BaseRevision)
		}
		return update, tool.NewContractErrorWithRepair("PLAN_REVISION_CONFLICT", callName, "/base_revision", fmt.Sprintf("%d", previousRevision), actual, "full Plan mutation must compare-and-swap the current Workflow Plan revision", fmt.Sprintf("Reload the current Plan and retry once with base_revision=%d.", previousRevision), nil, true)
	}
	if mode != "extend" && mode != "replan" {
		return update, tool.NewContractErrorWithRepair("PLAN_CHANGE_MODE_REQUIRED", callName, "/change_mode", "extend or replan", mode, "an existing Workflow Plan cannot be replaced without an explicit mutation mode", "Use update_plan_step/revise_verification for local changes; use change_mode=extend or replan only for an explicit new-task extension or goal/dependency change.", nil, true)
	}
	if strings.TrimSpace(update.ReplanReason) == "" {
		return update, tool.NewContractErrorWithRepair("PLAN_REPLAN_REASON_REQUIRED", callName, "/replan_reason", "a concrete reason", "missing", "full Plan mutation requires an auditable reason", "Explain which user request, goal, dependency, or acceptance condition changed.", nil, true)
	}
	if mode == "extend" {
		existing := make(map[string]struct{}, len(previousSteps))
		for _, step := range previousSteps {
			existing[step.ID] = struct{}{}
		}
		for _, step := range update.Steps {
			if _, found := existing[step.ID]; found {
				return update, tool.NewContractErrorWithRepair("PLAN_EXTEND_NODE_CONFLICT", callName, "/steps", "only new node ids", step.ID, "extend cannot overwrite an existing node", "Keep the existing graph unchanged and provide only new nodes; use update_plan_step for an existing node.", nil, true)
			}
		}
		update.Goal = previousGoal
		if strings.TrimSpace(update.Explanation) == "" {
			update.Explanation = previousExplanation
		}
		update.Steps = taskplan.ReconcileSteps(previousSteps, append(append([]taskplan.Step(nil), previousSteps...), update.Steps...))
		return update, nil
	}
	retired := make(map[string]string, len(update.RetiredSteps))
	for _, item := range update.RetiredSteps {
		if strings.TrimSpace(item.ID) == "" || strings.TrimSpace(item.Reason) == "" {
			return update, tool.NewContractErrorWithRepair("PLAN_RETIREMENT_INVALID", callName, "/retired_steps", "id and non-empty reason", item.ID, "retired nodes need an explicit reason", "Provide {id, reason} for every intentionally removed unfinished node.", nil, true)
		}
		retired[item.ID] = item.Reason
	}
	proposed := make(map[string]struct{}, len(update.Steps))
	for _, step := range update.Steps {
		proposed[step.ID] = struct{}{}
	}
	for _, step := range previousSteps {
		if _, kept := proposed[step.ID]; kept || step.Status == taskplan.StatusCompleted || step.Status == taskplan.StatusSkipped {
			continue
		}
		if _, explicitlyRetired := retired[step.ID]; !explicitlyRetired {
			return update, tool.NewContractErrorWithRepair("PLAN_OPEN_NODE_OMITTED", callName, "/steps", "preserve the node or list it in retired_steps", step.ID, "replan omitted an unfinished node without an explicit retirement", "Add the unfinished node back, or add its id and reason to retired_steps.", nil, true)
		}
	}
	update.Steps = taskplan.ReconcileSteps(previousSteps, update.Steps)
	return update, nil
}

func (s *RunStore) UpsertTaskPlan(ctx context.Context, lease agent.Lease, tenantID string, call tool.Call, update taskplan.Update) (taskplan.Plan, error) {
	var err error
	update, err = taskplan.NormalizeUpdate(update)
	if err != nil {
		return taskplan.Plan{}, err
	}
	if err := update.Validate(); err != nil {
		return taskplan.Plan{}, err
	}
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return taskplan.Plan{}, err
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	var valid bool
	var planningPolicy string
	var workflowID string
	if err := tx.QueryRow(ctx, `SELECT true,workflow_id::text,COALESCE(NULLIF(binding_snapshot #>> '{spec,planning,policy}',''),'auto') FROM agent_platform.agent_runs WHERE id=$1::uuid AND tenant_id=$2 AND lease_owner=$3 AND lease_token=$4 AND status='running' FOR UPDATE`, lease.RunID, tenantID, lease.Owner, lease.Token).Scan(&valid, &workflowID, &planningPolicy); errors.Is(err, pgx.ErrNoRows) {
		return taskplan.Plan{}, agent.ErrLeaseLost
	} else if err != nil {
		return taskplan.Plan{}, err
	}
	var previous int
	var lastCallID string
	var previousGoal, previousExplanation string
	var previousRunID, previousLastModifiedRunID string
	var previousRaw []byte
	lookupErr := tx.QueryRow(ctx, `
		SELECT revision,last_call_id,goal,COALESCE(explanation,''),steps,
		       run_id::text,COALESCE(last_modified_run_id,run_id)::text
		FROM agent_platform.agent_task_plans
		WHERE tenant_id=$1 AND workflow_id=$2::uuid FOR UPDATE`, tenantID, workflowID).Scan(
		&previous, &lastCallID, &previousGoal, &previousExplanation, &previousRaw,
		&previousRunID, &previousLastModifiedRunID,
	)
	if lookupErr != nil && !errors.Is(lookupErr, pgx.ErrNoRows) {
		return taskplan.Plan{}, lookupErr
	}
	if lastCallID == call.ID {
		result, err := scanTaskPlan(tx.QueryRow(ctx, taskPlanSelect+` WHERE plan.tenant_id=$1 AND plan.workflow_id=$2::uuid`, tenantID, workflowID))
		if err != nil {
			return taskplan.Plan{}, err
		}
		result, err = hydrateTaskPlanNodeStatesTx(ctx, tx, result)
		if err != nil {
			return taskplan.Plan{}, err
		}
		if err := tx.Commit(ctx); err != nil {
			return taskplan.Plan{}, err
		}
		return result, nil
	}
	var previousSteps []taskplan.Step
	if len(previousRaw) != 0 {
		if err := json.Unmarshal(previousRaw, &previousSteps); err != nil {
			return taskplan.Plan{}, err
		}
		// The JSON graph is the model-authored definition, not the latest
		// execution state. Reconcile a mutation against the normalized node
		// projection so a stale model echo cannot erase receipts, attempts or
		// usage accumulated since the last Plan revision.
		previousPlan, err := hydrateTaskPlanNodeStatesTx(ctx, tx, taskplan.Plan{
			RunID:             previousRunID,
			LastModifiedRunID: previousLastModifiedRunID,
			WorkflowID:        workflowID,
			TenantID:          tenantID,
			Revision:          previous,
			Goal:              previousGoal,
			Explanation:       previousExplanation,
			Steps:             previousSteps,
		})
		if err != nil {
			return taskplan.Plan{}, err
		}
		previousSteps = previousPlan.Steps
	}
	if err := validateCriterionRevisionCAS(call.Name, previous, call.Arguments); err != nil {
		return taskplan.Plan{}, err
	}
	update, err = preparePlanRevision(call.Name, previous, previousGoal, previousExplanation, previousSteps, update)
	if err != nil {
		return taskplan.Plan{}, err
	}
	if err := update.Validate(); err != nil {
		return taskplan.Plan{}, err
	}
	steps, _ := json.Marshal(update.Steps)
	result, err := scanTaskPlan(tx.QueryRow(ctx, `INSERT INTO agent_platform.agent_task_plans(run_id,last_modified_run_id,workflow_id,tenant_id,revision,goal,explanation,steps,last_call_id) VALUES($1::uuid,$1::uuid,$2::uuid,$3,1,$4,$5,$6::jsonb,$7) ON CONFLICT(workflow_id) DO UPDATE SET last_modified_run_id=EXCLUDED.last_modified_run_id,revision=agent_platform.agent_task_plans.revision+1,goal=EXCLUDED.goal,explanation=EXCLUDED.explanation,steps=EXCLUDED.steps,last_call_id=EXCLUDED.last_call_id,updated_at=now() RETURNING plan_id::text,run_id::text,last_modified_run_id::text,workflow_id::text,tenant_id,revision,(SELECT COALESCE(NULLIF(goal,''),agent_platform.agent_task_plans.goal) FROM agent_platform.agent_workflows WHERE id=agent_platform.agent_task_plans.workflow_id),goal,COALESCE(explanation,''),steps,created_at,updated_at`, lease.RunID, workflowID, tenantID, update.Goal, update.Explanation, steps, call.ID))
	if err != nil {
		return taskplan.Plan{}, err
	}
	if err := s.persistPlanNodeStatesTx(ctx, tx, result); err != nil {
		return taskplan.Plan{}, err
	}
	eventType := event.PlanUpdated
	if previous == 0 {
		eventType = event.PlanCreated
		if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{RunID: lease.RunID, WorkflowID: workflowID, Type: event.ExecutionModeSelected, Turn: call.Turn, Step: call.Step, CallID: call.ID, Payload: mustJSON(map[string]any{"mode": "planned", "policy": planningPolicy, "source": "model", "reason": "durable_plan_created", "workflow_id": workflowID})}); err != nil {
			return taskplan.Plan{}, err
		}
	}
	if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{RunID: lease.RunID, WorkflowID: workflowID, Type: eventType, Turn: call.Turn, Step: call.Step, CallID: call.ID, Payload: mustJSON(map[string]any{"revision": result.Revision, "goal": result.Goal, "explanation": result.Explanation, "steps": result.Steps, "workflow_id": workflowID, "source_run_id": lease.RunID, "mutation_source": call.Name, "change_mode": update.ChangeMode, "base_revision": update.BaseRevision, "replan_reason": update.ReplanReason, "retired_steps": update.RetiredSteps})}); err != nil {
		return taskplan.Plan{}, err
	}
	if call.Name == "revise_verification" {
		var revision taskplan.CriterionRevision
		if err := json.Unmarshal(call.Arguments, &revision); err != nil {
			return taskplan.Plan{}, err
		}
		if _, err := s.appendEventTx(ctx, tx, tenantID, event.Input{RunID: lease.RunID, WorkflowID: workflowID, Type: event.VerificationSpecRevised, Turn: call.Turn, Step: call.Step, CallID: call.ID, Payload: mustJSON(map[string]any{
			"plan_revision": result.Revision, "plan_step_key": revision.StepID, "criterion_key": revision.CriterionID,
			"action": revision.Action, "reason": revision.Reason, "verification": revision.Verification, "base_revision": revision.BaseRevision, "workflow_id": workflowID,
		})}); err != nil {
			return taskplan.Plan{}, err
		}
	}
	if err := s.persistVerificationPlanTx(ctx, tx, result, call.Turn, call.Step, call.ID); err != nil {
		return taskplan.Plan{}, err
	}
	if _, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_workflows
		SET active_plan_id=$2::uuid, updated_at=now()
		WHERE id=$1::uuid AND tenant_id=$3::text`, workflowID, result.PlanID, tenantID); err != nil {
		return taskplan.Plan{}, fmt.Errorf("project active plan: %w", err)
	}
	if err := tx.Commit(ctx); err != nil {
		return taskplan.Plan{}, err
	}
	return result, nil
}

func validateCriterionRevisionCAS(callName string, previous int, arguments json.RawMessage) error {
	if callName != "revise_verification" {
		return nil
	}
	var revision taskplan.CriterionRevision
	if err := json.Unmarshal(arguments, &revision); err != nil {
		return err
	}
	if revision.BaseRevision == nil || *revision.BaseRevision == previous {
		return nil
	}
	return tool.NewContractErrorWithRepair(
		"PLAN_REVISION_CONFLICT", callName, "/base_revision", fmt.Sprintf("%d", previous), fmt.Sprintf("%d", *revision.BaseRevision),
		"Reviewer verification repair must compare-and-swap the Plan revision it inspected",
		fmt.Sprintf("Reload the current Plan and retry once with base_revision=%d.", previous), nil, true,
	)
}

// persistPlanNodeStatesTx projects the graph into explicit, queryable node
// fields. The model may propose a Plan, but scheduling and retry decisions use
// this projection plus committed receipts rather than parsing prose.
func (s *RunStore) persistPlanNodeStatesTx(ctx context.Context, tx pgx.Tx, plan taskplan.Plan) error {
	workflowID := strings.TrimSpace(plan.WorkflowID)
	if workflowID == "" && strings.TrimSpace(plan.RunID) != "" {
		if err := tx.QueryRow(ctx, `SELECT workflow_id::text FROM agent_platform.agent_task_plans WHERE run_id=$1::uuid`, plan.RunID).Scan(&workflowID); err != nil {
			return fmt.Errorf("resolve plan workflow: %w", err)
		}
	}
	if workflowID == "" {
		return errors.New("plan workflow_id is required for node state projection")
	}
	activeNodeIDs := make([]string, 0, len(plan.Steps))
	for _, step := range plan.Steps {
		activeNodeIDs = append(activeNodeIDs, step.ID)
		artifactIDs, err := json.Marshal(step.State.ArtifactIDs)
		if err != nil {
			return fmt.Errorf("marshal node artifacts: %w", err)
		}
		tests, err := json.Marshal(step.State.Tests)
		if err != nil {
			return fmt.Errorf("marshal node tests: %w", err)
		}
		nextNodes, err := json.Marshal(step.State.NextNodeIDs)
		if err != nil {
			return fmt.Errorf("marshal node transitions: %w", err)
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO agent_platform.agent_plan_node_states
				(workflow_id,node_id,plan_revision,node_revision,status,output,artifact_ids,tests,
				 input_tokens,output_tokens,total_tokens,cost_usd,duration_ms,attempts,
				 last_run_id,last_event_sequence,blocked_reason,retry_from_node_id,next_node_ids)
			VALUES ($1::uuid,$2,$3,GREATEST($3,1),$4,$5,$6::jsonb,$7::jsonb,$8,$9,$10,$11,$12,$13,
				$14::uuid,$15,$16,$17,$18::jsonb)
			ON CONFLICT (workflow_id,node_id) DO UPDATE SET
				plan_revision=EXCLUDED.plan_revision,
				node_revision=agent_platform.agent_plan_node_states.node_revision+1,
				status=EXCLUDED.status,output=EXCLUDED.output,
				artifact_ids=EXCLUDED.artifact_ids,tests=EXCLUDED.tests,
				input_tokens=EXCLUDED.input_tokens,output_tokens=EXCLUDED.output_tokens,
				total_tokens=EXCLUDED.total_tokens,cost_usd=EXCLUDED.cost_usd,
				duration_ms=EXCLUDED.duration_ms,attempts=EXCLUDED.attempts,
				last_run_id=EXCLUDED.last_run_id,last_event_sequence=EXCLUDED.last_event_sequence,
				blocked_reason=EXCLUDED.blocked_reason,retry_from_node_id=EXCLUDED.retry_from_node_id,
				next_node_ids=EXCLUDED.next_node_ids,updated_at=now()`,
			workflowID, step.ID, plan.Revision, step.State.Status, step.State.Output,
			artifactIDs, tests, step.State.Usage.InputTokens, step.State.Usage.OutputTokens,
			step.State.Usage.TotalTokens, step.State.Usage.CostUSD, step.State.Usage.DurationMS,
			step.State.Attempts, nullableUUID(step.State.LastRunID), step.State.LastEventSeq,
			step.State.BlockedReason, step.State.RetryFromNodeID, nextNodes); err != nil {
			return fmt.Errorf("persist plan node %s workflow=%s: %w", step.ID, workflowID, err)
		}
	}
	// agent_plan_node_states is the current scheduling projection. Historical
	// graph revisions remain in Plan events, while nodes omitted from a full
	// update_plan revision must leave the active projection atomically.
	if _, err := tx.Exec(ctx, `
		DELETE FROM agent_platform.agent_plan_node_states
		WHERE workflow_id=$1::uuid AND NOT (node_id = ANY($2::text[]))`, workflowID, activeNodeIDs); err != nil {
		return fmt.Errorf("retire omitted plan nodes for workflow=%s: %w", workflowID, err)
	}
	return nil
}

func nullableUUID(value string) *string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return &value
}

// GetTaskPlanForWorkflow resolves the durable Plan by its real owner. A new
// Run attempt must see the Plan before it has written any Plan revision.
func (s *RunStore) GetTaskPlanForWorkflow(ctx context.Context, tenantID, workflowID string) (taskplan.Plan, error) {
	return s.readTaskPlanSnapshot(ctx, func(tx pgx.Tx) taskPlanRow {
		return tx.QueryRow(ctx, taskPlanSelect+` WHERE plan.tenant_id=$1 AND plan.workflow_id=$2::uuid LIMIT 1`, tenantID, workflowID)
	})
}

// GetLatestTaskPlanForSession returns the most recent durable plan in a
// session. A continuation is a new Run, so run-scoped lookup alone would
// incorrectly expose only planning controls after a model timeout.
func (s *RunStore) GetLatestTaskPlanForSession(ctx context.Context, tenantID, sessionID string) (taskplan.Plan, error) {
	return s.readTaskPlanSnapshot(ctx, func(tx pgx.Tx) taskPlanRow {
		return tx.QueryRow(ctx, `
		SELECT plan.plan_id::text,plan.run_id::text,plan.last_modified_run_id::text,plan.workflow_id::text,plan.tenant_id,plan.revision,COALESCE(NULLIF(w.goal,''),plan.goal),plan.goal,COALESCE(plan.explanation,''),plan.steps,plan.created_at,plan.updated_at
		FROM agent_platform.agent_task_plans plan
		JOIN agent_platform.agent_workflows w ON w.id=plan.workflow_id
		WHERE plan.tenant_id=$1 AND w.tenant_id=$1 AND w.session_id=$2::uuid
		ORDER BY plan.updated_at DESC, plan.run_id DESC
		LIMIT 1`, tenantID, sessionID)
	})
}

// readTaskPlanSnapshot reads the model-authored graph and the platform-owned
// execution projection from one repeatable-read snapshot. Without this
// boundary, a concurrent Plan mutation could pair revision N's graph with
// revision N+1's node state and expose a state that never existed.
func (s *RunStore) readTaskPlanSnapshot(ctx context.Context, selectPlan func(pgx.Tx) taskPlanRow) (taskplan.Plan, error) {
	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return taskplan.Plan{}, fmt.Errorf("begin task plan snapshot: %w", err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	plan, err := scanTaskPlan(selectPlan(tx))
	if err != nil {
		return taskplan.Plan{}, err
	}
	plan, err = hydrateTaskPlanNodeStatesTx(ctx, tx, plan)
	if err != nil {
		return taskplan.Plan{}, err
	}
	if err := tx.Commit(ctx); err != nil {
		return taskplan.Plan{}, fmt.Errorf("commit task plan snapshot: %w", err)
	}
	return plan, nil
}

// hydrateTaskPlanNodeStatesTx overlays platform-owned execution and
// verification fields onto the model-authored Plan graph. Descriptions,
// dependencies and acceptance contracts remain sourced from agent_task_plans;
// status, receipts, attempts and usage come from normalized projections.
func hydrateTaskPlanNodeStatesTx(ctx context.Context, tx pgx.Tx, plan taskplan.Plan) (taskplan.Plan, error) {
	rows, err := tx.Query(ctx, `
		SELECT node_id,node_revision,status,COALESCE(output,''),artifact_ids,tests,
		       input_tokens,output_tokens,total_tokens,cost_usd,duration_ms,attempts,
		       COALESCE(last_run_id::text,''),last_event_sequence,
		       COALESCE(blocked_reason,''),COALESCE(retry_from_node_id,''),next_node_ids
		FROM agent_platform.agent_plan_node_states
		WHERE workflow_id=$1::uuid`, plan.WorkflowID)
	if err != nil {
		return taskplan.Plan{}, fmt.Errorf("load plan node states workflow=%s: %w", plan.WorkflowID, err)
	}
	defer rows.Close()
	states := make(map[string]taskplan.NodeState, len(plan.Steps))
	for rows.Next() {
		var nodeID string
		var state taskplan.NodeState
		var artifactsRaw, testsRaw, nextRaw []byte
		if err := rows.Scan(
			&nodeID, &state.Revision, &state.Status, &state.Output, &artifactsRaw, &testsRaw,
			&state.Usage.InputTokens, &state.Usage.OutputTokens, &state.Usage.TotalTokens,
			&state.Usage.CostUSD, &state.Usage.DurationMS, &state.Attempts,
			&state.LastRunID, &state.LastEventSeq, &state.BlockedReason,
			&state.RetryFromNodeID, &nextRaw,
		); err != nil {
			return taskplan.Plan{}, fmt.Errorf("scan plan node state workflow=%s: %w", plan.WorkflowID, err)
		}
		if err := json.Unmarshal(artifactsRaw, &state.ArtifactIDs); err != nil {
			return taskplan.Plan{}, fmt.Errorf("decode plan node %s artifacts: %w", nodeID, err)
		}
		if err := json.Unmarshal(testsRaw, &state.Tests); err != nil {
			return taskplan.Plan{}, fmt.Errorf("decode plan node %s tests: %w", nodeID, err)
		}
		if err := json.Unmarshal(nextRaw, &state.NextNodeIDs); err != nil {
			return taskplan.Plan{}, fmt.Errorf("decode plan node %s transitions: %w", nodeID, err)
		}
		states[nodeID] = state
	}
	if err := rows.Err(); err != nil {
		return taskplan.Plan{}, fmt.Errorf("iterate plan node states workflow=%s: %w", plan.WorkflowID, err)
	}
	for index := range plan.Steps {
		state, found := states[plan.Steps[index].ID]
		if !found {
			continue
		}
		plan.Steps[index].State = state
		plan.Steps[index].Status = state.Status
		plan.Steps[index].Result = state.Output
		taskplan.NormalizeNodeState(&plan.Steps[index])
	}
	verificationRows, err := tx.Query(ctx, `
		SELECT plan_step_key,criterion_key,status,
		       COALESCE(diagnostic_code,''),COALESCE(diagnostic_message,'')
		FROM agent_platform.agent_verification_intents
		WHERE tenant_id=$1::text AND run_id=$2::uuid AND plan_revision=$3`,
		plan.TenantID, plan.EffectiveRunID(), plan.Revision)
	if err != nil {
		return taskplan.Plan{}, fmt.Errorf("load verification intent projection workflow=%s: %w", plan.WorkflowID, err)
	}
	defer verificationRows.Close()
	type criterionProjection struct {
		status, reason, message string
	}
	criteria := make(map[string]criterionProjection)
	for verificationRows.Next() {
		var stepID, criterionID string
		var projection criterionProjection
		if err := verificationRows.Scan(&stepID, &criterionID, &projection.status, &projection.reason, &projection.message); err != nil {
			return taskplan.Plan{}, fmt.Errorf("scan verification intent projection workflow=%s: %w", plan.WorkflowID, err)
		}
		criteria[stepID+"\x00"+criterionID] = projection
	}
	if err := verificationRows.Err(); err != nil {
		return taskplan.Plan{}, fmt.Errorf("iterate verification intent projection workflow=%s: %w", plan.WorkflowID, err)
	}
	for stepIndex := range plan.Steps {
		for criterionIndex := range plan.Steps[stepIndex].AcceptanceCriteria {
			criterion := &plan.Steps[stepIndex].AcceptanceCriteria[criterionIndex]
			projection, found := criteria[plan.Steps[stepIndex].ID+"\x00"+criterion.ID]
			if !found {
				continue
			}
			criterion.Status = projection.status
			criterion.VerificationReason = projection.reason
			criterion.VerificationMessage = projection.message
			if projection.status != taskplan.CriterionPassed {
				criterion.Evidence = ""
				criterion.EvidenceCallIDs = nil
			}
		}
	}
	return taskplan.NormalizePlan(plan)
}

const taskPlanSelect = `SELECT plan.plan_id::text,plan.run_id::text,plan.last_modified_run_id::text,plan.workflow_id::text,plan.tenant_id,plan.revision,COALESCE(NULLIF(workflow.goal,''),plan.goal),plan.goal,COALESCE(plan.explanation,''),plan.steps,plan.created_at,plan.updated_at FROM agent_platform.agent_task_plans plan JOIN agent_platform.agent_workflows workflow ON workflow.id=plan.workflow_id`

type taskPlanRow interface{ Scan(...any) error }

func scanTaskPlan(row taskPlanRow) (taskplan.Plan, error) {
	var result taskplan.Plan
	var raw []byte
	err := row.Scan(&result.PlanID, &result.RunID, &result.LastModifiedRunID, &result.WorkflowID, &result.TenantID, &result.Revision, &result.OriginalGoal, &result.Goal, &result.Explanation, &raw, &result.CreatedAt, &result.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return taskplan.Plan{}, taskplan.ErrNotFound
	}
	if err != nil {
		return taskplan.Plan{}, err
	}
	if err := json.Unmarshal(raw, &result.Steps); err != nil {
		return taskplan.Plan{}, err
	}
	return taskplan.NormalizePlan(result)
}
