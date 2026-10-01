package protocol

import (
	"encoding/json"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

func TestDecodeStructuredActionIntent(t *testing.T) {
	decision, err := Decode(model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "a1", Name: "read_file", Arguments: json.RawMessage(`{"path":"x"}`)}}})
	if err != nil || decision.Kind != KindActionIntent || len(decision.ToolCalls) != 1 {
		t.Fatalf("decision=%+v err=%v", decision, err)
	}
}

func TestDecodeRejectsLegacyMarkup(t *testing.T) {
	_, err := Decode(model.TextMessage(model.RoleAssistant, `<toolcall><function=run_command>`))
	if err != ErrLegacyToolMarkup {
		t.Fatalf("err=%v, want ErrLegacyToolMarkup", err)
	}
}

func TestDecodeRejectsNonObjectArguments(t *testing.T) {
	_, err := Decode(model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "a1", Name: "x", Arguments: json.RawMessage(`[]`)}}})
	if err == nil {
		t.Fatal("expected object schema error")
	}
}
