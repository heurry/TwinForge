package context

import (
	stdcontext "context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

func TestBudgetBuilderKeepsSystemCurrentAndNewestHistory(t *testing.T) {
	t.Parallel()
	builder, err := NewBudgetBuilder(BudgetConfig{MaxInputTokens: 70, ReserveOutputTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	result, err := builder.Build(stdcontext.Background(), Request{Messages: []model.Message{
		{Role: model.RoleSystem, Content: "system"},
		{Role: model.RoleUser, Content: strings.Repeat("old ", 120)},
		{Role: model.RoleAssistant, Content: "old answer"},
		{Role: model.RoleUser, Content: "new question"},
		{Role: model.RoleAssistant, Content: "new answer"},
		{Role: model.RoleUser, Content: "current"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 4 || result.Messages[0].Role != model.RoleSystem ||
		result.Messages[1].Content != "new question" || result.Messages[2].Content != "new answer" ||
		result.Messages[3].Content != "current" {
		t.Fatalf("selected messages = %+v", result.Messages)
	}
	if result.Manifest.ContextHash == "" || result.Manifest.InputTokens <= 0 {
		t.Fatalf("manifest = %+v", result.Manifest)
	}
}

func TestBudgetBuilderNeverDropsMandatorySkillMessages(t *testing.T) {
	t.Parallel()
	builder, err := NewBudgetBuilder(BudgetConfig{MaxInputTokens: 55, ReserveOutputTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	result, err := builder.Build(stdcontext.Background(), Request{MandatoryPrefix: 3, Messages: []model.Message{
		{Role: model.RoleSystem, Content: "base policy"},
		{Role: model.RoleSystem, Content: "mandatory skill safety rule"},
		{Role: model.RoleUser, Content: "mandatory skill example"},
		{Role: model.RoleUser, Content: strings.Repeat("old ", 80)},
		{Role: model.RoleAssistant, Content: "old answer"},
		{Role: model.RoleUser, Content: "current"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 4 || result.Messages[1].Content != "mandatory skill safety rule" || result.Manifest.MandatoryMessageCount != 3 {
		t.Fatalf("mandatory context was not preserved: messages=%+v manifest=%+v", result.Messages, result.Manifest)
	}
}

func TestModelMessageBudgetReservesToolSchemasAndOutput(t *testing.T) {
	t.Parallel()
	tools := []model.ToolSchema{{
		Name: "read_file", Description: strings.Repeat("read workspace files safely ", 20),
		Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
	}}
	budget, toolTokens, err := ModelMessageBudget(4096, 256, tools)
	if err != nil {
		t.Fatal(err)
	}
	if toolTokens <= 24 || budget >= 4096-256 || budget+toolTokens+256 > 4096 {
		t.Fatalf("budget=%d tool_tokens=%d", budget, toolTokens)
	}
}

func TestEstimateTokensUsesCanonicalParts(t *testing.T) {
	t.Parallel()
	message := model.TextMessage(model.RoleUser, strings.Repeat("token ", 40))
	message.Content = ""
	if EstimateTokens(message) < 40 {
		t.Fatalf("parts were not included: %d", EstimateTokens(message))
	}
}

func TestEstimateTokensIncludesToolCallArguments(t *testing.T) {
	t.Parallel()
	plain := model.Message{Role: model.RoleAssistant}
	withCall := plain
	withCall.ToolCalls = []model.ToolCall{{Name: "update_plan", Arguments: json.RawMessage(`{"steps":"` + strings.Repeat("long plan ", 80) + `"}`)}}
	if EstimateTokens(withCall) <= EstimateTokens(plain)+100 {
		t.Fatalf("tool arguments were not budgeted: plain=%d tool_call=%d", EstimateTokens(plain), EstimateTokens(withCall))
	}
}

func TestAllocateSectionBudgetsNeverExceedsMessageBudget(t *testing.T) {
	allocation := AllocateSectionBudgets(7000, 4096, 1024, 1024, 3072)
	if allocation.ReservedTokens > 7000 {
		t.Fatalf("section allocation exceeded budget: %+v", allocation)
	}
	if allocation.RecentTurnTokens != 4096 || allocation.MemoryTokens != 1024 || allocation.KnowledgeTokens != 1024 || allocation.ToolResultTokens != 856 {
		t.Fatalf("unexpected priority allocation: %+v", allocation)
	}
}

func TestAllocateSectionBudgetsReservesSummaryAndStaticInstructions(t *testing.T) {
	allocation := AllocateSectionBudgetsWithStatic(7000, 4096, 1024, 1024, 3072, 1536, 3072)
	if allocation.ReservedTokens > 7000 || allocation.SummaryTokens != 1536 || allocation.StaticInstructionTokens != 3072 {
		t.Fatalf("invalid reserved allocation: %+v", allocation)
	}
	if allocation.RecentTurnTokens != 2392 || allocation.MemoryTokens != 0 || allocation.KnowledgeTokens != 0 || allocation.ToolResultTokens != 0 {
		t.Fatalf("mutable sections were not reduced after reserves: %+v", allocation)
	}
}
