package context

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

func TestCompactMessagesPreservesSystemAndRecentHistory(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "system contract")}
	for index := 0; index < 12; index++ {
		messages = append(messages, model.TextMessage(model.RoleUser, strings.Repeat("older context ", 20)))
		messages = append(messages, model.TextMessage(model.RoleAssistant, strings.Repeat("observation ", 20)))
	}
	messages = append(messages, model.TextMessage(model.RoleUser, "current objective"))
	compacted, report, err := CompactMessages(messages, 500)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Compacted || report.RemovedMessages == 0 || report.AfterTokens > 500 {
		t.Fatalf("report = %+v", report)
	}
	if compacted[0].Role != model.RoleSystem || compacted[len(compacted)-1].TextContent() != "current objective" {
		t.Fatalf("mandatory messages not preserved: %+v", compacted)
	}
	if !strings.Contains(compacted[1].TextContent(), "context_summary") {
		t.Fatalf("summary missing: %+v", compacted[1])
	}
}

func TestCompactMessagesFallsBackToTaskAnchorForOversizedLatestToolExchange(t *testing.T) {
	t.Parallel()
	messages := []model.Message{
		model.TextMessage(model.RoleSystem, "system contract"),
		model.TextMessage(model.RoleUser, "example question"),
		model.TextMessage(model.RoleAssistant, "example answer"),
		model.TextMessage(model.RoleUser, "the original task must survive"),
		{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call-1", Name: "update_plan", Arguments: json.RawMessage(`{"steps":"` + strings.Repeat("oversized ", 300) + `"}`)}}},
		model.TextMessage(model.RoleTool, strings.Repeat("large tool error ", 300)),
	}
	compacted, report, err := CompactMessages(messages, 160)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Compacted || len(compacted) < 2 || compacted[1].TextContent() != "the original task must survive" {
		t.Fatalf("task anchor not preserved: report=%+v messages=%+v", report, compacted)
	}
	anchorCount := 0
	for _, message := range compacted {
		if message.TextContent() == "the original task must survive" {
			anchorCount++
		}
	}
	if anchorCount != 1 {
		t.Fatalf("task anchor must appear exactly once after fallback compaction, got %d", anchorCount)
	}
	if report.AfterTokens > 160 {
		t.Fatalf("compacted result exceeds budget: %+v", report)
	}
}

func TestCompactMessagesAlwaysPreservesExactOriginalTask(t *testing.T) {
	t.Parallel()
	task := `{"question":"read docs/Harness面试复习文档-问题版.md at start_line=381 and require VALIDATION: passed"}`
	taskMessage := model.TextMessage(model.RoleUser, task)
	taskMessage.Metadata = map[string]string{TaskAnchorMetadataKey: "true"}
	messages := []model.Message{
		model.TextMessage(model.RoleSystem, strings.Repeat("system contract ", 12)),
		model.TextMessage(model.RoleUser, "example question"),
		model.TextMessage(model.RoleAssistant, "example answer"),
		taskMessage,
	}
	for index := 0; index < 8; index++ {
		messages = append(messages,
			model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call", Name: "list_files", Arguments: json.RawMessage(`{"path":"."}`)}}},
			model.TextMessage(model.RoleTool, strings.Repeat("large directory result ", 30)),
		)
	}
	compacted, report, err := CompactMessages(messages, 360)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, message := range compacted {
		if message.Role == model.RoleUser && message.TextContent() == task {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("exact task was lost: report=%+v messages=%+v", report, compacted)
	}
	if report.AfterTokens > 360 {
		t.Fatalf("compacted result exceeds budget: %+v", report)
	}
	compacted = append(compacted,
		model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "next", Name: "read_file", Arguments: json.RawMessage(`{"path":"docs/file.md","start_line":21}`)}}},
		model.TextMessage(model.RoleTool, strings.Repeat("next observation ", 80)),
	)
	compactedAgain, _, err := CompactMessages(compacted, 360)
	if err != nil {
		t.Fatal(err)
	}
	found = false
	for _, message := range compactedAgain {
		if message.Metadata[TaskAnchorMetadataKey] == "true" && message.TextContent() == task {
			found = true
		}
	}
	if !found {
		t.Fatalf("exact task was lost after a second compaction: %+v", compactedAgain)
	}
}
