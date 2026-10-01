package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/jackc/pgx/v5"
)

// projectObservationTx materializes one append-only event into a compact,
// queryable Observation row. The Event Ledger remains authoritative; this
// projection is rebuilt safely from events when a schema or parent rule changes.
func (s *RunStore) projectObservationTx(ctx context.Context, tx pgx.Tx, tenantID string, committed event.Event) error {
	// Kept as a separate hook so appendEventTx can remain transaction-bound. The
	// concrete pgx transaction is accepted through the tiny Exec interface.
	payload := map[string]any{}
	_ = json.Unmarshal(committed.Payload, &payload)
	key, kind := observationKey(committed, payload)
	status := observationStatus(committed.Type)
	name := strings.TrimSpace(fmt.Sprint(payload["name"]))
	if name == "" || name == "<nil>" {
		name = strings.ReplaceAll(strings.ToLower(string(committed.Type)), "_", " ")
	}
	modelResolution := firstPayload(payload, "model_resolution_id")
	modelID := firstPayload(payload, "model_id", "model")
	toolVersionID := firstPayload(payload, "tool_version_id")
	promptVersion := firstPayload(payload, "prompt_version_id", "prompt_version")
	skillSetVersion := firstPayload(payload, "skillset_version_id", "skill_set_version_id")
	toolSetVersion := firstPayload(payload, "toolset_version_id", "tool_set_version_id")
	delegationID := firstPayload(payload, "delegation_id")
	childRunID := firstPayload(payload, "child_run_id")
	workflowID := strings.TrimSpace(committed.WorkflowID)
	if workflowID == "" {
		workflowID = committed.RunID
	}
	planNodeID := strings.TrimSpace(committed.PlanNodeID)
	if planNodeID == "" {
		planNodeID = firstPayload(payload, "plan_node_id", "plan_step_id", "plan_step_key")
	}
	decisionCycle := committed.DecisionCycle
	if decisionCycle == 0 {
		decisionCycle = committed.Turn
	}
	actionID := strings.TrimSpace(committed.ActionID)
	if actionID == "" {
		actionID = firstPayload(payload, "action_id")
	}
	if actionID == "" {
		actionID = committed.CallID
	}
	parentActionID := firstPayload(payload, "parent_action_id", "parent_action")
	inputTokens, outputTokens := usageTokens(payload)
	cost := payloadNumber(payload, "cost")
	if cost == 0 {
		cost = payloadNumber(payload, "total_cost")
	}
	latency := payloadNumber(payload, "latency_ms")
	var completed any
	if observationTerminal(committed.Type) {
		completed = committed.CreatedAt
	}
	_, err := tx.Exec(ctx, `
		INSERT INTO agent_platform.agent_observations
			(tenant_id,run_id,trace_id,observation_key,kind,name,status,sequence,last_sequence,
			turn_no,step_no,call_id,workflow_id,plan_node_id,decision_cycle,action_id,parent_action_id,delegation_id,child_run_id,agent_version_id,model_resolution_id,model_id,tool_version_id,
			 prompt_version_id,skillset_version_id,toolset_version_id,input_tokens,output_tokens,total_cost,
			 started_at,completed_at,duration_ms,detail_refs,metadata)
		VALUES($1,$2::uuid,$2,$3,$4,$5,$6,$7::bigint,$7::bigint,NULLIF($8::int,0),NULLIF($9::int,0),NULLIF($10::text,''),NULLIF($11::text,'')::uuid,NULLIF($12::text,''),NULLIF($13::int,0),NULLIF($14::text,''),NULLIF($15::text,''),NULLIF($16::text,''),NULLIF($17::text,''),
			NULLIF($18,'')::uuid,NULLIF($19,''),NULLIF($20,''),NULLIF($21,''),NULLIF($22,''),NULLIF($23,''),NULLIF($24,''),$25,$26,$27,$28,$29,$30,
			jsonb_build_object('events', jsonb_build_array($7::bigint)), '{}'::jsonb)
		ON CONFLICT(run_id,observation_key) DO UPDATE SET
			last_sequence=EXCLUDED.last_sequence,status=EXCLUDED.status,name=EXCLUDED.name,
			completed_at=COALESCE(EXCLUDED.completed_at,agent_platform.agent_observations.completed_at),
			duration_ms=CASE WHEN EXCLUDED.duration_ms>0 THEN EXCLUDED.duration_ms ELSE agent_platform.agent_observations.duration_ms END,
			input_tokens=GREATEST(agent_platform.agent_observations.input_tokens,EXCLUDED.input_tokens),
			output_tokens=GREATEST(agent_platform.agent_observations.output_tokens,EXCLUDED.output_tokens),
			total_cost=GREATEST(agent_platform.agent_observations.total_cost,EXCLUDED.total_cost),
			model_resolution_id=COALESCE(NULLIF(EXCLUDED.model_resolution_id,''),agent_platform.agent_observations.model_resolution_id),
			prompt_version_id=COALESCE(NULLIF(EXCLUDED.prompt_version_id,''),agent_platform.agent_observations.prompt_version_id),
			skillset_version_id=COALESCE(NULLIF(EXCLUDED.skillset_version_id,''),agent_platform.agent_observations.skillset_version_id),
			toolset_version_id=COALESCE(NULLIF(EXCLUDED.toolset_version_id,''),agent_platform.agent_observations.toolset_version_id),
			model_id=COALESCE(NULLIF(EXCLUDED.model_id,''),agent_platform.agent_observations.model_id),
			tool_version_id=COALESCE(NULLIF(EXCLUDED.tool_version_id,''),agent_platform.agent_observations.tool_version_id),
			workflow_id=COALESCE(EXCLUDED.workflow_id,agent_platform.agent_observations.workflow_id),
			plan_node_id=COALESCE(NULLIF(EXCLUDED.plan_node_id,''),agent_platform.agent_observations.plan_node_id),
			decision_cycle=COALESCE(EXCLUDED.decision_cycle,agent_platform.agent_observations.decision_cycle),
			action_id=COALESCE(NULLIF(EXCLUDED.action_id,''),agent_platform.agent_observations.action_id),
			parent_action_id=COALESCE(NULLIF(EXCLUDED.parent_action_id,''),agent_platform.agent_observations.parent_action_id),
			updated_at=now()`,
		tenantID, committed.RunID, key, kind, name, status, committed.Sequence, committed.Turn, committed.Step,
		committed.CallID, workflowID, planNodeID, decisionCycle, actionID, parentActionID, delegationID, childRunID,
		firstPayload(payload, "agent_version_id"), modelResolution, modelID, toolVersionID, promptVersion, skillSetVersion,
		toolSetVersion, inputTokens, outputTokens, cost, committed.CreatedAt, completed, latency)
	if err != nil {
		return fmt.Errorf("project observation %s: %w", committed.Type, err)
	}
	if err := s.linkObservationParent(ctx, tx, committed.RunID, key, kind, committed.Turn, committed.Step, payload); err != nil {
		return err
	}
	if err := s.projectPlanNodeStateTx(ctx, tx, committed, payload); err != nil {
		return err
	}
	return s.projectRuntimeScoreTx(ctx, tx, tenantID, committed, payload)
}

// projectPlanNodeStateTx folds committed observations into the normalized
// node projection. This is intentionally event-driven: token usage, latency
// and verification status come from platform receipts, never from model text.
func (s *RunStore) projectPlanNodeStateTx(ctx context.Context, tx pgx.Tx, committed event.Event, payload map[string]any) error {
	workflowID := strings.TrimSpace(committed.WorkflowID)
	if workflowID == "" {
		workflowID = committed.RunID
	}
	nodeID := strings.TrimSpace(committed.PlanNodeID)
	if nodeID == "" {
		nodeID = firstPayload(payload, "plan_node_id", "plan_step_id", "plan_step_key")
	}
	if nodeID == "" {
		return nil
	}
	inputTokens, outputTokens := usageTokens(payload)
	duration := int64(payloadNumber(payload, "latency_ms"))
	status := "in_progress"
	if committed.Type == event.VerificationCompleted {
		status = "completed"
	} else if committed.Type == event.VerificationFailed {
		status = "blocked"
	}
	blockedReason := ""
	if committed.Type == event.ToolFailed {
		blockedReason = firstPayload(payload, "error_code", "error", "reason")
	}
	testJSON, _ := json.Marshal(map[string]any{
		"name":           firstPayload(payload, "criterion_key", "criterion_id", "name"),
		"kind":           firstPayload(payload, "verification_kind", "kind"),
		"status":         map[bool]string{true: "passed", false: "failed"}[committed.Type == event.VerificationCompleted],
		"tool_call_id":   firstPayload(payload, "tool_call_id", "call_id"),
		"event_sequence": committed.Sequence,
	})
	if committed.Type != event.VerificationCompleted && committed.Type != event.VerificationFailed {
		testJSON = []byte("null")
	}
	_, err := tx.Exec(ctx, `
		INSERT INTO agent_platform.agent_plan_node_states
			(workflow_id,node_id,plan_revision,status,input_tokens,output_tokens,total_tokens,
			 duration_ms,attempts,last_run_id,last_event_sequence,blocked_reason,tests)
		SELECT $1::uuid,$2::text,existing.plan_revision,$3::text,$4::bigint,$5::bigint,$4::bigint+$5::bigint,$6::bigint,
			CASE WHEN $4::bigint+$5::bigint>0 THEN 1 ELSE 0 END,
			$7::uuid,$8,NULLIF($9::text,''),CASE WHEN $10::jsonb='null'::jsonb THEN '[]'::jsonb ELSE jsonb_build_array($10::jsonb) END
		FROM agent_platform.agent_plan_node_states AS existing
		WHERE existing.workflow_id=$1::uuid AND existing.node_id=$2::text
		ON CONFLICT (workflow_id,node_id) DO UPDATE SET
			status=CASE
				WHEN EXCLUDED.status IN ('completed','blocked') THEN EXCLUDED.status
				WHEN agent_platform.agent_plan_node_states.last_run_id IS DISTINCT FROM EXCLUDED.last_run_id
					AND agent_platform.agent_plan_node_states.status='blocked' THEN 'in_progress'
				ELSE agent_platform.agent_plan_node_states.status END,
			input_tokens=agent_platform.agent_plan_node_states.input_tokens+EXCLUDED.input_tokens,
			output_tokens=agent_platform.agent_plan_node_states.output_tokens+EXCLUDED.output_tokens,
			total_tokens=agent_platform.agent_plan_node_states.total_tokens+EXCLUDED.total_tokens,
			duration_ms=agent_platform.agent_plan_node_states.duration_ms+EXCLUDED.duration_ms,
			attempts=CASE
				WHEN agent_platform.agent_plan_node_states.last_run_id IS DISTINCT FROM EXCLUDED.last_run_id
					THEN GREATEST(agent_platform.agent_plan_node_states.attempts+1,EXCLUDED.attempts,1)
				ELSE GREATEST(agent_platform.agent_plan_node_states.attempts,EXCLUDED.attempts) END,
			last_event_sequence=CASE
				WHEN agent_platform.agent_plan_node_states.last_run_id IS DISTINCT FROM EXCLUDED.last_run_id
					THEN EXCLUDED.last_event_sequence
				ELSE GREATEST(agent_platform.agent_plan_node_states.last_event_sequence,EXCLUDED.last_event_sequence) END,
			blocked_reason=CASE
				WHEN NULLIF(EXCLUDED.blocked_reason,'') IS NOT NULL THEN EXCLUDED.blocked_reason
				WHEN agent_platform.agent_plan_node_states.last_run_id IS DISTINCT FROM EXCLUDED.last_run_id THEN NULL
				ELSE agent_platform.agent_plan_node_states.blocked_reason END,
			last_run_id=EXCLUDED.last_run_id,
			tests=CASE WHEN $10::jsonb='null'::jsonb THEN agent_platform.agent_plan_node_states.tests
				ELSE agent_platform.agent_plan_node_states.tests || CASE WHEN agent_platform.agent_plan_node_states.tests @> jsonb_build_array($10::jsonb) THEN '[]'::jsonb ELSE jsonb_build_array($10::jsonb) END END,
			updated_at=now()`,
		workflowID, nodeID, status, inputTokens, outputTokens, duration, committed.RunID, committed.Sequence, blockedReason, testJSON)
	if err != nil {
		return fmt.Errorf("project plan node state event=%s workflow=%s node=%s: %w", committed.Type, workflowID, nodeID, err)
	}
	return nil
}

func (s *RunStore) projectRuntimeScoreTx(ctx context.Context, tx pgx.Tx, tenantID string, committed event.Event, payload map[string]any) error {
	if committed.Type != event.VerificationCompleted && committed.Type != event.VerificationFailed && committed.Type != event.EvalCompleted {
		return nil
	}
	verdict := strings.ToLower(firstPayload(payload, "verdict", "status", "result"))
	value := 0.0
	if verdict == "passed" || verdict == "pass" || verdict == "success" || verdict == "completed" || committed.Type == event.EvalCompleted {
		value = 1
	}
	metadata, _ := json.Marshal(payload)
	_, err := tx.Exec(ctx, `INSERT INTO agent_platform.agent_scores
		(tenant_id,run_id,name,score_type,value,source,evaluator_version,metadata)
		VALUES($1,$2::uuid,'verification','numeric',$3,'runtime','observation-v1',$4::jsonb)`, tenantID, committed.RunID, value, metadata)
	if err != nil {
		return fmt.Errorf("project runtime score: %w", err)
	}
	return nil
}

func (s *RunStore) linkObservationParent(ctx context.Context, tx pgx.Tx, runID, key, kind string, turn, step int, payload map[string]any) error {
	parentKey := ""
	switch kind {
	case "turn":
		parentKey = "run"
	case "step":
		parentKey = fmt.Sprintf("turn:%d", turn)
	case "model", "context", "memory", "skill", "plan", "checkpoint", "verification":
		parentKey = fmt.Sprintf("step:%d:%d", turn, step)
	case "tool":
		parentKey = fmt.Sprintf("step:%d:%d", turn, step)
	case "agent":
		parentKey = fmt.Sprintf("step:%d:%d", turn, step)
	default:
		if step > 0 {
			parentKey = fmt.Sprintf("step:%d:%d", turn, step)
		} else if turn > 0 {
			parentKey = fmt.Sprintf("turn:%d", turn)
		}
	}
	if parentKey == "" {
		return nil
	}
	_, err := tx.Exec(ctx, `UPDATE agent_platform.agent_observations child SET parent_observation_id=parent.id,root_observation_id=COALESCE(parent.root_observation_id,parent.id),updated_at=now() FROM agent_platform.agent_observations parent WHERE child.run_id=$1::uuid AND child.observation_key=$2 AND parent.run_id=$1::uuid AND parent.observation_key=$3`, runID, key, parentKey)
	return err
}

func observationKey(current event.Event, payload map[string]any) (string, string) {
	switch current.Type {
	case event.WorkflowCreated, event.WorkflowRoutingDecided, event.RunAttemptCreated, event.WorkflowResumed, event.WorkflowSuspended, event.WorkflowCompleted,
		event.RunCreated, event.RunClaimed, event.RunResumed, event.RunSuspended, event.RunCancelRequested, event.RunCompleted, event.RunFailed, event.RunCancelled:
		return "run", "run"
	case event.TurnStarted, event.TurnCompleted:
		return fmt.Sprintf("turn:%d", current.Turn), "turn"
	case event.StepStarted, event.StepCompleted:
		return fmt.Sprintf("step:%d:%d", current.Turn, current.Step), "step"
	case event.ModelRequested, event.ModelCompleted, event.ModelFailed:
		return fmt.Sprintf("model:%d:%d", current.Turn, current.Step), "model"
	case event.ToolCalled, event.ToolApprovalRequested, event.ToolApprovalResolved, event.ToolCompleted, event.ToolFailed:
		return "tool:" + current.CallID, "tool"
	case event.DelegationRequested, event.DelegationCompleted, event.DelegationFailed:
		id := firstPayload(payload, "delegation_id", "child_run_id")
		if id == "" {
			id = current.CallID
		}
		return "agent:" + id, "agent"
	}
	return fmt.Sprintf("event:%d", current.Sequence), observationEventKind(current.Type)
}

func observationEventKind(kind event.Type) string {
	v := string(kind)
	switch {
	case strings.Contains(v, "SKILL"):
		return "skill"
	case strings.Contains(v, "MEMORY"):
		return "memory"
	case strings.Contains(v, "CONTEXT"):
		return "context"
	case strings.Contains(v, "PLAN"), strings.Contains(v, "USER_INPUT"), kind == event.ExecutionModeSelected:
		return "plan"
	case strings.Contains(v, "CHECKPOINT"):
		return "checkpoint"
	case strings.Contains(v, "VERIFICATION"):
		return "verification"
	default:
		return "lifecycle"
	}
}

func observationStatus(kind event.Type) string {
	switch kind {
	case event.ModelFailed, event.ToolFailed, event.RunFailed:
		return "failed"
	case event.RunCancelled:
		return "cancelled"
	case event.WorkflowCompleted:
		return "completed"
	case event.ToolApprovalRequested, event.UserInputRequested, event.RunSuspended, event.WorkflowSuspended:
		return "waiting"
	case event.ModelRequested, event.ToolCalled, event.StepStarted, event.TurnStarted, event.TurnCreated, event.RunCreated, event.RunAttemptCreated, event.RunClaimed, event.RunResumed, event.WorkflowCreated, event.WorkflowResumed:
		return "running"
	default:
		return "completed"
	}
}
func observationTerminal(kind event.Type) bool {
	switch kind {
	case event.ModelCompleted, event.ModelFailed, event.ToolCompleted, event.ToolFailed, event.StepCompleted, event.TurnCompleted, event.RunCompleted, event.RunFailed, event.RunCancelled, event.WorkflowCompleted, event.ToolApprovalRequested, event.ToolApprovalResolved, event.UserInputRequested, event.UserInputReceived:
		return true
	}
	return false
}
func firstPayload(payload map[string]any, keys ...string) string {
	for _, key := range keys {
		value := strings.TrimSpace(fmt.Sprint(payload[key]))
		if value != "" && value != "<nil>" {
			return value
		}
	}
	return ""
}
func payloadNumber(payload map[string]any, key string) float64 {
	value, _ := payload[key].(float64)
	return value
}
func usageTokens(payload map[string]any) (int64, int64) {
	usage, _ := payload["usage"].(map[string]any)
	if usage == nil {
		usage = payload
	}
	return int64(payloadNumber(usage, "input_tokens") + payloadNumber(usage, "prompt_tokens")), int64(payloadNumber(usage, "output_tokens") + payloadNumber(usage, "completion_tokens"))
}
