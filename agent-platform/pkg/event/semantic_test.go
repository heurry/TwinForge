package event

import (
	"encoding/json"
	"testing"
)

func TestNormalizeSemanticMapsLegacyCoordinates(t *testing.T) {
	input := NormalizeSemantic(Input{
		RunID: "run-1", Turn: 2, Step: 3, CallID: "call-1",
		Payload: json.RawMessage(`{"plan_step_id":"build"}`),
	})
	if input.WorkflowID != "run-1" || input.DecisionCycle != 2 || input.ActionID != "call-1" || input.PlanNodeID != "build" {
		t.Fatalf("unexpected semantic aliases: %+v", input)
	}
	workflowID, planNodeID, cycle, actionID := SemanticFromPayload(input.Payload)
	if workflowID != "run-1" || planNodeID != "build" || cycle != 2 || actionID != "call-1" {
		t.Fatalf("unexpected payload envelope: %q %q %d %q", workflowID, planNodeID, cycle, actionID)
	}
}

func TestNormalizeSemanticLeavesNonObjectPayloadUntouched(t *testing.T) {
	payload := json.RawMessage(`[]`)
	input := NormalizeSemantic(Input{RunID: "run-1", Payload: payload})
	if string(input.Payload) != string(payload) {
		t.Fatalf("payload changed: %s", input.Payload)
	}
}

func TestNormalizeSemanticKeepsTurnIdentityAcrossRunAttempts(t *testing.T) {
	input := NormalizeSemantic(Input{
		RunID: "attempt-2", WorkflowID: "workflow-1", Turn: 2,
		Payload: json.RawMessage(`{"kind":"model"}`),
	})
	if input.TurnID != "workflow-1:2" {
		t.Fatalf("turn id = %q, want workflow-1:2", input.TurnID)
	}
	workflowID, _, _, _ := SemanticFromPayload(input.Payload)
	if workflowID != "workflow-1" {
		t.Fatalf("workflow id in semantic envelope = %q", workflowID)
	}
}
