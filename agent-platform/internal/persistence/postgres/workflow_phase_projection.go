package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/workflow"
)

// projectWorkflowPhaseEventTx advances the outer deterministic state machine
// from a committed runtime fact. Only the active root Run may change the
// Workflow phase; delegated child events remain visible without taking task
// ownership from their parent.
func projectWorkflowPhaseEventTx(ctx context.Context, tx pgx.Tx, tenantID string, committed event.Event) error {
	var currentRaw, activeRunID string
	err := tx.QueryRow(ctx, `
		SELECT phase,COALESCE(active_run_id::text,'')
		FROM agent_platform.agent_workflows
		WHERE id=$1::uuid AND tenant_id=$2::text
		FOR UPDATE`, committed.WorkflowID, tenantID).Scan(&currentRaw, &activeRunID)
	if errors.Is(err, pgx.ErrNoRows) {
		return errors.New("workflow phase projection target does not exist")
	}
	if err != nil {
		return fmt.Errorf("load workflow phase: %w", err)
	}
	if activeRunID != committed.RunID {
		return nil
	}
	current := workflow.Status(currentRaw)
	payload := map[string]any{}
	_ = json.Unmarshal(committed.Payload, &payload)
	next, ok := workflowPhaseForEvent(committed.Type, payload, current)
	if !ok || next == current || current.Terminal() {
		return nil
	}
	if err := workflow.ValidateTransition(current, next); err != nil {
		return fmt.Errorf("project workflow phase from %s: %w", committed.Type, err)
	}
	tag, err := tx.Exec(ctx, `
		UPDATE agent_platform.agent_workflows
		SET phase=$4::text,updated_at=now()
		WHERE id=$1::uuid AND tenant_id=$2::text AND active_run_id=$3::uuid AND phase=$5::text`,
		committed.WorkflowID, tenantID, committed.RunID, string(next), string(current))
	if err != nil {
		return fmt.Errorf("persist workflow phase %s: %w", next, err)
	}
	if tag.RowsAffected() != 1 {
		return errors.New("workflow phase changed concurrently")
	}
	return nil
}

func workflowPhaseForEvent(kind event.Type, payload map[string]any, current workflow.Status) (workflow.Status, bool) {
	switch kind {
	case event.ReviewDecisionRecorded:
		if strings.EqualFold(firstPayload(payload, "verdict"), "pass") && firstPayload(payload, "parse_error") == "" {
			return workflow.StatusRunning, true
		}
		return workflow.StatusReplanning, true
	case event.ProgressReviewCreated:
		switch strings.ToLower(firstPayload(payload, "action")) {
		case "replan":
			return workflow.StatusReplanning, true
		case "review":
			return workflow.StatusReviewing, true
		case "ask_user":
			// The strategy has narrowed the next decision to ask_user, but the
			// Workflow is not actually waiting until USER_INPUT_REQUESTED commits.
			// Keeping the current phase also avoids an impossible intermediate
			// transition when recovery escalates from replanning.
			return "", false
		default:
			return workflow.StatusReflecting, true
		}
	case event.PlanCompletionBlocked:
		if current == workflow.StatusReplanning || current == workflow.StatusReviewing {
			return "", false
		}
		return workflow.StatusVerifying, true
	case event.FinalOutputRejected, event.VerificationLoopDetected:
		if current == workflow.StatusReplanning || current == workflow.StatusReviewing {
			return "", false
		}
		return workflow.StatusReviewing, true
	case event.PlanCreated, event.PlanUpdated:
		return workflow.StatusRunning, true
	case event.ModelRequested:
		if current == workflow.StatusReflecting || current == workflow.StatusVerifying || current == workflow.StatusReviewing {
			return workflow.StatusRunning, true
		}
	case event.ToolApprovalRequested:
		return workflow.StatusWaitingApproval, true
	case event.UserInputRequested:
		return workflow.StatusWaitingUser, true
	case event.DelegationRequested:
		return workflow.StatusWaitingAgent, true
	case event.ToolApprovalResolved, event.UserInputReceived, event.DelegationCompleted, event.DelegationFailed:
		return workflow.StatusReady, true
	}
	return "", false
}
