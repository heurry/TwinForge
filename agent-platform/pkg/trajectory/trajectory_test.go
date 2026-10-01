package trajectory

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
)

func TestProjectMergesModelAndToolPairsWithStableOrder(t *testing.T) {
	base := time.Date(2026, 9, 7, 1, 0, 0, 0, time.UTC)
	events := []event.Event{
		{Input: event.Input{RunID: "run", Type: event.StepStarted, Turn: 1, Step: 1}, Sequence: 1, CreatedAt: base},
		{Input: event.Input{RunID: "run", Type: event.ModelRequested, Turn: 1, Step: 1}, Sequence: 2, CreatedAt: base.Add(time.Millisecond)},
		{Input: event.Input{RunID: "run", Type: event.ModelCompleted, Turn: 1, Step: 1, Payload: json.RawMessage(`{"model_id":"m","latency_ms":7}`)}, Sequence: 3, CreatedAt: base.Add(8 * time.Millisecond)},
		{Input: event.Input{RunID: "run", Type: event.ToolCalled, Turn: 1, Step: 1, CallID: "c1", Payload: json.RawMessage(`{"name":"read_file"}`)}, Sequence: 4, CreatedAt: base.Add(9 * time.Millisecond)},
		{Input: event.Input{RunID: "run", Type: event.ToolCompleted, Turn: 1, Step: 1, CallID: "c1", Payload: json.RawMessage(`{"name":"read_file","latency_ms":4}`)}, Sequence: 5, CreatedAt: base.Add(13 * time.Millisecond)},
	}
	records := Project(events)
	if len(records) != 3 || records[1].Kind != "model" || records[2].Kind != "tool" {
		t.Fatalf("records = %+v", records)
	}
	if records[1].Status != StatusCompleted || records[1].DurationMS != 7 || len(records[1].EventSequences) != 2 {
		t.Fatalf("model record = %+v", records[1])
	}
	if records[2].ParentID != records[0].ID || records[2].CallID != "c1" {
		t.Fatalf("tool hierarchy = %+v", records[2])
	}
	if again := Project(events); again[2].ID != records[2].ID {
		t.Fatal("record id is not stable")
	}
}

func TestPaginateUsesStableSequenceCursor(t *testing.T) {
	records := []Record{{Sequence: 2}, {Sequence: 4}, {Sequence: 7}}
	page := Paginate(records, 2, 1)
	if len(page.Records) != 1 || page.Records[0].Sequence != 4 || !page.HasMore || page.NextCursor != 4 || page.Total != 3 {
		t.Fatalf("page = %+v", page)
	}
}

func TestPaginateReturnsEmptyArrayForEmptyProjection(t *testing.T) {
	page := Paginate(nil, 0, 100)
	encoded, err := json.Marshal(page)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) == "" || !containsJSONToken(encoded, `"records":[]`) {
		t.Fatalf("empty trajectory records must encode as an array: %s", encoded)
	}
}

func containsJSONToken(value []byte, token string) bool {
	return string(value) == token || strings.Contains(string(value), token)
}

func TestProjectMarksUserQuestionAsWaitingPlanRecord(t *testing.T) {
	records := Project([]event.Event{{
		Input:    event.Input{RunID: "run", Type: event.UserInputRequested, Turn: 1, Step: 2, Payload: json.RawMessage(`{"question":"continue?"}`)},
		Sequence: 1, CreatedAt: time.Now(),
	}})
	if len(records) != 1 || records[0].Kind != "plan" || records[0].Status != StatusWaiting || records[0].Summary != "continue?" {
		t.Fatalf("record = %+v", records)
	}
}

func TestProjectClassifiesExecutionModeAsPlanDecision(t *testing.T) {
	records := Project([]event.Event{{
		Input: event.Input{RunID: "run", Type: event.ExecutionModeSelected, Turn: 1, Step: 1,
			Payload: json.RawMessage(`{"mode":"planned","policy":"auto"}`)},
		Sequence: 1, CreatedAt: time.Now(),
	}})
	if len(records) != 1 || records[0].Kind != "plan" || records[0].Summary != "planned · auto" {
		t.Fatalf("record = %+v", records)
	}
}
