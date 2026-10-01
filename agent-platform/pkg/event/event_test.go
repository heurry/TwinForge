package event

import (
	"context"
	"encoding/json"
	"testing"
)

func TestMemoryStoreAcceptsCompleteLifecycle(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	inputs := []Input{
		{RunID: "run-1", Type: RunCreated},
		{RunID: "run-1", Type: TurnStarted, Turn: 1},
		{RunID: "run-1", Type: StepStarted, Turn: 1, Step: 1},
		{RunID: "run-1", Type: ModelRequested, Turn: 1, Step: 1},
		{RunID: "run-1", Type: ModelCompleted, Turn: 1, Step: 1},
		{RunID: "run-1", Type: ToolCalled, Turn: 1, Step: 1, CallID: "call-1"},
		{RunID: "run-1", Type: ToolCompleted, Turn: 1, Step: 1, CallID: "call-1"},
		{RunID: "run-1", Type: StepCompleted, Turn: 1, Step: 1},
		{RunID: "run-1", Type: TurnCompleted, Turn: 1},
		{RunID: "run-1", Type: RunCompleted},
	}
	for i, input := range inputs {
		event, err := store.Append(ctx, input)
		if err != nil {
			t.Fatalf("append %s: %v", input.Type, err)
		}
		if event.Sequence != int64(i+1) {
			t.Fatalf("sequence = %d, want %d", event.Sequence, i+1)
		}
	}
}

func TestMemoryStoreAcceptsMemoryLifecycleEventsOutsideStep(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	for _, eventType := range []Type{RunCreated, MemoryExtractionRequested, MemoryExtractionStarted, MemoryCandidateProposed, MemoryCreated, MemoryUpdated, MemoryDeleted, MemoryMerged, MemorySuperseded, MemoryReviewRequired, MemoryExtractionCompleted, MemoryVerified, MemoryContradicted, MemoryTeamPromotionRequested, MemoryTeamPromoted, MemorySourceRevisionObserved, RunCompleted} {
		if _, err := store.Append(ctx, Input{RunID: "run-memory", Type: eventType}); err != nil {
			t.Fatalf("append %s: %v", eventType, err)
		}
	}
}

func TestMemoryStoreAcceptsReviewDecisionAfterToolResult(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	ctx := context.Background()
	inputs := []Input{
		{RunID: "run-review", Type: RunCreated},
		{RunID: "run-review", Type: TurnStarted, Turn: 1},
		{RunID: "run-review", Type: StepStarted, Turn: 1, Step: 1},
		{RunID: "run-review", Type: ToolCalled, Turn: 1, Step: 1, CallID: "review-call"},
		{RunID: "run-review", Type: ToolCompleted, Turn: 1, Step: 1, CallID: "review-call"},
		{RunID: "run-review", Type: ReviewDecisionRecorded, Turn: 1, Step: 1, CallID: "review-call", Payload: json.RawMessage(`{"decision_id":"review-decision:review-call","verdict":"pass"}`)},
	}
	for _, input := range inputs {
		if _, err := store.Append(ctx, input); err != nil {
			t.Fatalf("append %s: %v", input.Type, err)
		}
	}
}

func TestMemoryStoreRejectsUnpairedToolResult(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	for _, input := range []Input{
		{RunID: "run-1", Type: RunCreated},
		{RunID: "run-1", Type: TurnStarted, Turn: 1},
		{RunID: "run-1", Type: StepStarted, Turn: 1, Step: 1},
	} {
		if _, err := store.Append(ctx, input); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Append(ctx, Input{
		RunID: "run-1", Type: ToolCompleted, Turn: 1, Step: 1, CallID: "missing",
	}); err == nil {
		t.Fatal("unpaired tool result must be rejected")
	}
}

func TestMemoryStoreRejectsRunCompletionWithOpenStep(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	store := NewMemoryStore()
	for _, input := range []Input{
		{RunID: "run-1", Type: RunCreated},
		{RunID: "run-1", Type: TurnStarted, Turn: 1},
		{RunID: "run-1", Type: StepStarted, Turn: 1, Step: 1},
	} {
		if _, err := store.Append(ctx, input); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := store.Append(ctx, Input{RunID: "run-1", Type: RunCompleted}); err == nil {
		t.Fatal("run with open step must not complete")
	}
}

func TestMemoryStoreAssignsWorkflowSequenceAcrossRuns(t *testing.T) {
	t.Parallel()

	store := NewMemoryStore()
	first, err := store.Append(context.Background(), Input{RunID: "run-1", WorkflowID: "workflow-1", Type: RunCreated})
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.Append(context.Background(), Input{RunID: "run-2", WorkflowID: "workflow-1", Type: RunCreated})
	if err != nil {
		t.Fatal(err)
	}
	if first.Sequence != 1 || second.Sequence != 1 || first.WorkflowSequence != 1 || second.WorkflowSequence != 2 {
		t.Fatalf("sequences first=%+v second=%+v", first, second)
	}
}
