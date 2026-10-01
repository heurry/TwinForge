package harness

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

func TestCheckpointProjectionRemovesDuplicationAndCompactsHistoricalFailure(t *testing.T) {
	assistant := model.Message{ID: "assistant-failed", Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "failed", Name: "update_plan", Arguments: json.RawMessage(`{"goal":"ship","steps":"` + strings.Repeat("x", 4000) + `"}`)}}}
	failed := model.TextMessage(model.RoleTool, `{"error_code":"TOOL_SCHEMA_INVALID","error":"`+strings.Repeat("bad ", 1000)+`","correction":"add one assertion"}`)
	failed.Name = "update_plan"
	failed.ToolCallID = "failed"
	failed.Metadata = map[string]string{"tool_failed": "true"}
	checkpoint := Checkpoint{Messages: []model.Message{model.TextMessage(model.RoleUser, "task"), assistant, failed}}

	projected := ProjectCheckpointForStorage(checkpoint)
	originalJSON, _ := json.Marshal(checkpoint)
	projectedJSON, _ := json.Marshal(projected)
	if len(projectedJSON)*2 >= len(originalJSON) {
		t.Fatalf("checkpoint projection did not materially shrink state: original=%d projected=%d", len(originalJSON), len(projectedJSON))
	}
	if projected.Messages[0].Content != "" || len(projected.Messages[0].Parts) == 0 {
		t.Fatalf("text message was not canonicalized: %+v", projected.Messages[0])
	}
	if len(projected.Messages) != 2 {
		t.Fatalf("historical failure should become one observation: %+v", projected.Messages)
	}
	history := projected.Messages[1]
	if history.Role != model.RoleUser || len(history.ToolCalls) != 0 || history.Metadata["runtime.tool_history"] != "true" {
		t.Fatalf("failure remained executable instead of becoming Tool History: %+v", history)
	}
	if !strings.Contains(history.TextContent(), `"executable":false`) || !strings.Contains(history.TextContent(), `"status":"failed"`) {
		t.Fatalf("bounded failure observation is incomplete: %s", history.TextContent())
	}
	if strings.Contains(history.TextContent(), strings.Repeat("bad ", 400)) {
		t.Fatalf("failure result was not bounded: %d", len(history.TextContent()))
	}
	restored := RestoreCheckpointFromStorage(projected)
	if restored.Messages[0].Content != "task" || restored.Messages[1].Content == "" {
		t.Fatalf("checkpoint was not rehydrated: %+v", restored.Messages)
	}
}

func TestCheckpointFailureProjectionPreservesDiagnosticTail(t *testing.T) {
	assistant := model.Message{ID: "assistant-run", Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "run-1", Name: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["test.py"]}`)}}}
	stderr := "Traceback start\n" + strings.Repeat("frame detail\n", 120) + "RuntimeError: final-root-cause"
	content, _ := json.Marshal(map[string]any{
		"error_code": "TOOL_EXECUTION_FAILED", "failure_kind": "process_exit", "exit_code": 1,
		"diagnostic": "RuntimeError: final-root-cause", "stderr": stderr,
		"correction": "repair code before retry",
	})
	failed := model.TextMessage(model.RoleTool, string(content))
	failed.Name = "run_command"
	failed.ToolCallID = "run-1"
	failed.Metadata = map[string]string{"tool_failed": "true"}

	projected := ProjectCheckpointForStorage(Checkpoint{Messages: []model.Message{assistant, failed}})
	if len(projected.Messages) != 1 {
		t.Fatalf("projected messages=%d: %+v", len(projected.Messages), projected.Messages)
	}
	text := projected.Messages[0].TextContent()
	for _, want := range []string{`"failure_kind":"process_exit"`, `"exit_code":1`, `RuntimeError: final-root-cause`, `"stderr_tail"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("failure projection lost %q: %s", want, text)
		}
	}
	if len(text) > 4000 {
		t.Fatalf("failure projection is unbounded: %d", len(text))
	}
}

func TestCheckpointProjectionPreservesPendingToolArguments(t *testing.T) {
	raw := json.RawMessage(`{"path":"out.py","content":"` + strings.Repeat("x", 4000) + `"}`)
	message := model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "pending", Name: "write_file", Arguments: raw}}}
	projected := ProjectCheckpointForStorage(Checkpoint{Messages: []model.Message{message}, PendingToolCalls: []model.ToolCall{{ID: "pending", Name: "write_file", Arguments: raw}}})
	if string(projected.Messages[0].ToolCalls[0].Arguments) != string(raw) {
		t.Fatalf("pending arguments changed: %s", projected.Messages[0].ToolCalls[0].Arguments)
	}
}

func TestCheckpointProjectionDropsMessagesCoveredByCollapseState(t *testing.T) {
	state, err := json.Marshal(map[string]any{
		"generation": 2, "summary": "bounded history", "covered_message_ids": []string{"old-user", "old-assistant"},
	})
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := Checkpoint{ContextState: state, Messages: []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "old-user", Role: model.RoleUser, Content: strings.Repeat("old ", 1000)},
		{ID: "old-assistant", Role: model.RoleAssistant, Content: strings.Repeat("answer ", 1000)},
		{ID: "recent", Role: model.RoleUser, Content: "continue"},
	}}
	projected := ProjectCheckpointForStorage(checkpoint)
	if len(projected.Messages) != 2 || projected.Messages[0].ID != "system" || projected.Messages[1].ID != "recent" {
		t.Fatalf("covered checkpoint history was retained: %+v", projected.Messages)
	}
}

func TestCheckpointProjectionTurnsVerboseWriteIntoNonExecutablePreview(t *testing.T) {
	raw := json.RawMessage(`{"path":"pkg/cli.py","content":"print('start')\n` + strings.Repeat("x", 4000) + `"}`)
	assistant := model.Message{ID: "assistant-write", Role: model.RoleAssistant, Content: "write cli", ToolCalls: []model.ToolCall{{ID: "write-1", Name: "write_file", Arguments: raw}}}
	result := model.TextMessage(model.RoleTool, `{"path":"pkg/cli.py","bytes":4015,"file_sha256":"abc"}`)
	result.Name = "write_file"
	result.ToolCallID = "write-1"

	projected := ProjectCheckpointForStorage(Checkpoint{Messages: []model.Message{assistant, result}})
	if len(projected.Messages) != 1 {
		t.Fatalf("messages=%d, want one Tool History observation: %+v", len(projected.Messages), projected.Messages)
	}
	history := projected.Messages[0]
	if history.Role != model.RoleUser || len(history.ToolCalls) != 0 {
		t.Fatalf("verbose write remained executable: %+v", history)
	}
	text := history.TextContent()
	for _, want := range []string{`<tool-history executable="false"`, `"argument_fields":["content","path"]`, `"content_preview"`, `print('start')`, `"do_not_copy_as_tool_call":true`} {
		if !strings.Contains(text, want) {
			t.Fatalf("Tool History missing %q: %s", want, text)
		}
	}
	if len(text) >= len(raw) {
		t.Fatalf("Tool History did not bound source content: history=%d raw=%d", len(text), len(raw))
	}
}

func TestCheckpointProjectionMigratesLegacyInternalArgumentProjection(t *testing.T) {
	assistant := model.Message{ID: "legacy", Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{
		ID: "write-legacy", Name: "write_file", Arguments: json.RawMessage(`{"path":"pkg/cli.py","event_backed":true,"arguments_sha256":"sha256:old"}`),
	}}}
	result := model.TextMessage(model.RoleTool, `{"path":"pkg/cli.py","bytes":512}`)
	result.Name = "write_file"
	result.ToolCallID = "write-legacy"

	projected := ProjectCheckpointForStorage(Checkpoint{Messages: []model.Message{assistant, result}})
	if len(projected.Messages) != 1 || len(projected.Messages[0].ToolCalls) != 0 {
		t.Fatalf("legacy projection remained a model-callable Tool Call: %+v", projected.Messages)
	}
	text := projected.Messages[0].TextContent()
	if strings.Contains(text, `"argument_fields":["arguments_sha256"`) || strings.Contains(text, `"event_backed":true`) {
		t.Fatalf("platform-owned fields leaked into argument preview: %s", text)
	}
}

func TestCheckpointRestoreMigratesLegacyInternalArgumentProjectionBeforeModelUse(t *testing.T) {
	assistant := model.Message{ID: "legacy", Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{
		ID: "write-legacy", Name: "write_file", Arguments: json.RawMessage(`{"path":"pkg/cli.py","event_backed":true,"arguments_sha256":"sha256:old"}`),
	}}}
	result := model.TextMessage(model.RoleTool, `{"path":"pkg/cli.py","bytes":512}`)
	result.Name = "write_file"
	result.ToolCallID = "write-legacy"

	restored := RestoreCheckpointFromStorage(Checkpoint{Messages: []model.Message{assistant, result}})
	if len(restored.Messages) != 1 || len(restored.Messages[0].ToolCalls) != 0 {
		t.Fatalf("legacy Checkpoint reached the model as a callable projection: %+v", restored.Messages)
	}
	if !strings.Contains(restored.Messages[0].TextContent(), `<tool-history executable="false"`) {
		t.Fatalf("legacy Checkpoint was not converted to Tool History: %s", restored.Messages[0].TextContent())
	}
}

func TestCheckpointProjectionKeepsOnlyFiveRecentCompletedToolInteractions(t *testing.T) {
	messages := make([]model.Message, 0, 14)
	for index := 1; index <= 7; index++ {
		callID := "call-" + string(rune('0'+index))
		assistant := model.Message{ID: "assistant-" + callID, Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: callID, Name: "read_file", Arguments: json.RawMessage(`{"path":"file-` + callID + `.txt"}`)}}}
		result := model.TextMessage(model.RoleTool, `{"path":"file-`+callID+`.txt","content":"ok"}`)
		result.Name = "read_file"
		result.ToolCallID = callID
		messages = append(messages, assistant, result)
	}

	projected := ProjectCheckpointForStorage(Checkpoint{Messages: messages})
	if len(projected.Messages) != 10 {
		t.Fatalf("messages=%d, want five assistant/tool pairs: %+v", len(projected.Messages), projected.Messages)
	}
	joined, _ := json.Marshal(projected.Messages)
	if strings.Contains(string(joined), "call-1") || strings.Contains(string(joined), "call-2") {
		t.Fatalf("older completed Tool interactions were retained: %s", joined)
	}
	for index := 3; index <= 7; index++ {
		callID := "call-" + string(rune('0'+index))
		if !strings.Contains(string(joined), callID) {
			t.Fatalf("recent interaction %s missing: %s", callID, joined)
		}
	}
}
