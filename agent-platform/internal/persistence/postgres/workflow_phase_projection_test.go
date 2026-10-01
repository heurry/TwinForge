package postgres

import (
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/workflow"
)

func TestWorkflowPhaseForProgressReviewAction(t *testing.T) {
	tests := []struct {
		action string
		want   workflow.Status
	}{
		{action: "retry", want: workflow.StatusReflecting},
		{action: "replan", want: workflow.StatusReplanning},
		{action: "review", want: workflow.StatusReviewing},
	}
	for _, test := range tests {
		got, ok := workflowPhaseForEvent(event.ProgressReviewCreated, map[string]any{"action": test.action}, workflow.StatusRunning)
		if !ok || got != test.want {
			t.Fatalf("action=%s phase=%s ok=%v want=%s", test.action, got, ok, test.want)
		}
	}
}

func TestWorkflowPhaseWaitsForActualUserInputRequest(t *testing.T) {
	if _, ok := workflowPhaseForEvent(event.ProgressReviewCreated, map[string]any{"action": "ask_user"}, workflow.StatusReplanning); ok {
		t.Fatal("strategy selection must not enter waiting_user before USER_INPUT_REQUESTED commits")
	}
	if got, ok := workflowPhaseForEvent(event.UserInputRequested, nil, workflow.StatusReplanning); !ok || got != workflow.StatusWaitingUser {
		t.Fatalf("user input request phase=%s ok=%v", got, ok)
	}
}

func TestWorkflowPhaseKeepsReplanningUntilPlanUpdate(t *testing.T) {
	if _, ok := workflowPhaseForEvent(event.ModelRequested, nil, workflow.StatusReplanning); ok {
		t.Fatal("model request must not leave replanning before a Plan revision commits")
	}
	if got, ok := workflowPhaseForEvent(event.PlanUpdated, nil, workflow.StatusReplanning); !ok || got != workflow.StatusRunning {
		t.Fatalf("plan update phase=%s ok=%v", got, ok)
	}
}

func TestWorkflowPhaseForReviewerDecision(t *testing.T) {
	tests := []struct {
		name    string
		payload map[string]any
		want    workflow.Status
	}{
		{name: "pass", payload: map[string]any{"verdict": "pass"}, want: workflow.StatusRunning},
		{name: "changes", payload: map[string]any{"verdict": "changes_required"}, want: workflow.StatusReplanning},
		{name: "blocked", payload: map[string]any{"verdict": "blocked"}, want: workflow.StatusReplanning},
		{name: "invalid pass", payload: map[string]any{"verdict": "pass", "parse_error": "invalid output"}, want: workflow.StatusReplanning},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, ok := workflowPhaseForEvent(event.ReviewDecisionRecorded, test.payload, workflow.StatusReady)
			if !ok || got != test.want {
				t.Fatalf("phase=%s ok=%v want=%s", got, ok, test.want)
			}
			if err := workflow.ValidateTransition(workflow.StatusReady, got); err != nil {
				t.Fatalf("projected transition is invalid: %v", err)
			}
		})
	}
}
