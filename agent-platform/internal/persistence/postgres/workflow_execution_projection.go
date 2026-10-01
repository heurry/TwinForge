package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
)

// projectWorkflowExecutionTx folds immutable events into the explicit
// DecisionCycle/ActionAttempt vocabulary. It is transaction-bound to the event
// append, so projections can never move ahead of the ledger fact they cite.
func projectWorkflowExecutionTx(ctx context.Context, tx pgx.Tx, tenantID string, committed event.Event) error {
	payload := map[string]any{}
	_ = json.Unmarshal(committed.Payload, &payload)
	cycle := committed.DecisionCycle
	if cycle == 0 {
		cycle = committed.Turn
	}
	planNodeID := strings.TrimSpace(committed.PlanNodeID)
	if planNodeID == "" {
		planNodeID = firstPayload(payload, "plan_node_id", "plan_step_id", "plan_step_key")
	}
	if cycle > 0 {
		if err := projectDecisionCycleTx(ctx, tx, tenantID, committed, cycle, planNodeID); err != nil {
			return err
		}
	}
	return projectActionAttemptTx(ctx, tx, tenantID, committed, payload, cycle, planNodeID)
}

func projectDecisionCycleTx(ctx context.Context, tx pgx.Tx, tenantID string, committed event.Event, cycle int, planNodeID string) error {
	status := "running"
	terminal := false
	switch committed.Type {
	case event.TurnCompleted:
		status, terminal = "completed", true
	case event.RunFailed:
		status, terminal = "failed", true
	case event.RunCancelled:
		status, terminal = "cancelled", true
	}
	modelIncrement := 0
	if committed.Type == event.ModelRequested {
		modelIncrement = 1
	}
	actionIncrement := 0
	if committed.Type == event.ToolCalled || committed.Type == event.DelegationRequested || committed.Type == event.UserInputRequested {
		actionIncrement = 1
	}
	_, err := tx.Exec(ctx, `
		INSERT INTO agent_platform.agent_decision_cycles(
			tenant_id,workflow_id,run_id,cycle_no,plan_node_id,
			workflow_generation,plan_revision,node_revision,workspace_revision,
			status,model_call_count,action_count,first_event_sequence,last_event_sequence,
			started_at,finished_at)
		SELECT $1::text,workflow.id,$2::uuid,$3::int,NULLIF($4::text,''),workflow.execution_generation,
		       COALESCE(plan.revision,0),COALESCE(node.node_revision,0),workflow.workspace_revision,
		       $5::text,$6::int,$7::int,$8::bigint,$8::bigint,$9::timestamptz,
		       CASE WHEN $10::boolean THEN $9::timestamptz ELSE NULL END
		FROM agent_platform.agent_workflows AS workflow
		LEFT JOIN agent_platform.agent_task_plans AS plan ON plan.workflow_id=workflow.id
		LEFT JOIN agent_platform.agent_plan_node_states AS node
		  ON node.workflow_id=workflow.id AND node.node_id=NULLIF($4,'')
		WHERE workflow.id=$11::uuid AND workflow.tenant_id=$1::text
		ON CONFLICT(run_id,cycle_no) DO UPDATE SET
			plan_node_id=COALESCE(agent_platform.agent_decision_cycles.plan_node_id,EXCLUDED.plan_node_id),
			status=CASE WHEN agent_platform.agent_decision_cycles.status IN ('completed','failed','cancelled')
				THEN agent_platform.agent_decision_cycles.status ELSE EXCLUDED.status END,
			model_call_count=agent_platform.agent_decision_cycles.model_call_count+EXCLUDED.model_call_count,
			action_count=agent_platform.agent_decision_cycles.action_count+EXCLUDED.action_count,
			last_event_sequence=GREATEST(agent_platform.agent_decision_cycles.last_event_sequence,EXCLUDED.last_event_sequence),
			finished_at=COALESCE(agent_platform.agent_decision_cycles.finished_at,EXCLUDED.finished_at),
			updated_at=now()`,
		tenantID, committed.RunID, cycle, planNodeID, status, modelIncrement, actionIncrement,
		committed.WorkflowSequence, committed.CreatedAt, terminal, committed.WorkflowID)
	if err != nil {
		return fmt.Errorf("project decision cycle run=%s cycle=%d: %w", committed.RunID, cycle, err)
	}
	return nil
}

func projectActionAttemptTx(ctx context.Context, tx pgx.Tx, tenantID string, committed event.Event, payload map[string]any, cycle int, planNodeID string) error {
	status, kind, terminal, ok := actionProjectionState(committed.Type)
	if !ok {
		return nil
	}
	if committed.Type == event.ToolApprovalResolved {
		decision := strings.ToLower(firstPayload(payload, "status", "decision"))
		if decision == "rejected" || decision == "denied" || decision == "expired" || decision == "cancelled" {
			status, terminal = "failed", true
		}
	}
	if strings.EqualFold(firstPayload(payload, "action_kind"), "mcp") {
		kind = "mcp"
	}
	actionID := strings.TrimSpace(committed.ActionID)
	if actionID == "" {
		actionID = firstPayload(payload, "action_id", "delegation_id")
	}
	if actionID == "" {
		actionID = strings.TrimSpace(committed.CallID)
	}
	if actionID == "" {
		return nil
	}
	name := firstPayload(payload, "name", "tool_name", "target_agent", "target_agent_version_id")
	if name == "" {
		name = strings.ToLower(string(committed.Type))
	}
	request := jsonField(committed.Payload, "arguments", "input")
	result := jsonField(committed.Payload, "result", "output")
	errorValue := jsonField(committed.Payload, "error")
	_, err := tx.Exec(ctx, `
		INSERT INTO agent_platform.agent_action_attempts(
			tenant_id,workflow_id,run_id,action_id,decision_cycle,plan_node_id,action_kind,name,
			status,workflow_generation,plan_revision,node_revision,workspace_revision,
			request,result,error,first_event_sequence,last_event_sequence,started_at,finished_at)
		SELECT $1::text,workflow.id,$2::uuid,$3::text,NULLIF($4::int,0),NULLIF($5::text,''),$6::text,$7::text,$8::text,
		       workflow.execution_generation,COALESCE(plan.revision,0),COALESCE(node.node_revision,0),workflow.workspace_revision,
		       $9::jsonb,NULLIF($10::jsonb,'null'::jsonb),NULLIF($11::jsonb,'null'::jsonb),
		       $12::bigint,$12::bigint,$13::timestamptz,
		       CASE WHEN $14::boolean THEN $13::timestamptz ELSE NULL END
		FROM agent_platform.agent_workflows AS workflow
		LEFT JOIN agent_platform.agent_task_plans AS plan ON plan.workflow_id=workflow.id
		LEFT JOIN agent_platform.agent_plan_node_states AS node
		  ON node.workflow_id=workflow.id AND node.node_id=NULLIF($5,'')
		WHERE workflow.id=$15::uuid AND workflow.tenant_id=$1::text
		ON CONFLICT(run_id,action_id) DO UPDATE SET
			decision_cycle=COALESCE(agent_platform.agent_action_attempts.decision_cycle,EXCLUDED.decision_cycle),
			plan_node_id=COALESCE(agent_platform.agent_action_attempts.plan_node_id,EXCLUDED.plan_node_id),
			status=CASE WHEN agent_platform.agent_action_attempts.status IN ('completed','failed','cancelled')
				THEN agent_platform.agent_action_attempts.status ELSE EXCLUDED.status END,
			request=CASE WHEN agent_platform.agent_action_attempts.request='{}'::jsonb THEN EXCLUDED.request ELSE agent_platform.agent_action_attempts.request END,
			result=COALESCE(EXCLUDED.result,agent_platform.agent_action_attempts.result),
			error=COALESCE(EXCLUDED.error,agent_platform.agent_action_attempts.error),
			last_event_sequence=GREATEST(agent_platform.agent_action_attempts.last_event_sequence,EXCLUDED.last_event_sequence),
			finished_at=COALESCE(agent_platform.agent_action_attempts.finished_at,EXCLUDED.finished_at),
			updated_at=now()`,
		tenantID, committed.RunID, actionID, cycle, planNodeID, kind, name, status,
		request, result, errorValue, committed.WorkflowSequence, committed.CreatedAt, terminal, committed.WorkflowID)
	if err != nil {
		return fmt.Errorf("project action attempt run=%s action=%s: %w", committed.RunID, actionID, err)
	}
	return nil
}

func actionProjectionState(kind event.Type) (status, actionKind string, terminal, ok bool) {
	switch kind {
	case event.ToolCalled:
		return "started", "tool", false, true
	case event.ToolApprovalRequested:
		return "waiting_approval", "tool", false, true
	case event.ToolApprovalResolved:
		return "started", "tool", false, true
	case event.ToolCompleted:
		return "completed", "tool", true, true
	case event.ToolFailed:
		return "failed", "tool", true, true
	case event.DelegationRequested:
		return "waiting_agent", "agent", false, true
	case event.DelegationCompleted:
		return "completed", "agent", true, true
	case event.DelegationFailed:
		return "failed", "agent", true, true
	case event.UserInputRequested:
		return "waiting_user", "control", false, true
	case event.UserInputReceived:
		return "completed", "control", true, true
	default:
		return "", "", false, false
	}
}

func jsonField(raw json.RawMessage, keys ...string) json.RawMessage {
	root := map[string]json.RawMessage{}
	if json.Unmarshal(raw, &root) != nil {
		return json.RawMessage(`{}`)
	}
	for _, key := range keys {
		if value, exists := root[key]; exists && len(value) != 0 {
			return value
		}
	}
	return json.RawMessage(`{}`)
}
