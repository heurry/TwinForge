package context

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

func TestProtectedFailureProjectionDropsLargeToolArguments(t *testing.T) {
	arguments := json.RawMessage(`{"path":"src/app.py","content":"` + strings.Repeat("source ", 2000) + `"}`)
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "task", Role: model.RoleUser, Content: "repair"},
		{ID: "assistant", Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call-1", Name: "write_file", Arguments: arguments}}},
		{ID: "tool", Role: model.RoleTool, Name: "write_file", Content: `{"error":"provider detail","content":{"error_code":"TOOL_SCHEMA_INVALID","correction":"fix path and retry"}}`, Metadata: map[string]string{"tool_failed": "true"}},
	}
	projected := projectProtectedFailureMessages(messages)
	if len(projected[2].ToolCalls) != 1 || len(projected[2].ToolCalls[0].Arguments) > maxFailureArgumentsChars {
		t.Fatalf("assistant failure arguments were not bounded: %s", projected[2].ToolCalls[0].Arguments)
	}
	if strings.Contains(string(projected[2].ToolCalls[0].Arguments), "source source") || strings.Contains(string(projected[2].ToolCalls[0].Arguments), "arguments_omitted") || strings.Contains(string(projected[2].ToolCalls[0].Arguments), "tool_name") {
		t.Fatal("source content leaked into protected failure projection")
	}
	if !strings.Contains(projected[3].TextContent(), "TOOL_SCHEMA_INVALID") || !strings.Contains(projected[3].TextContent(), "fix path and retry") {
		t.Fatalf("structured correction was lost: %s", projected[3].TextContent())
	}
	if len(messages[2].ToolCalls[0].Arguments) <= len(projected[2].ToolCalls[0].Arguments) {
		t.Fatal("durable failure arguments were mutated")
	}
}

func TestToolHistoryObservationUsesToolBudgetAndPreservesFailureSignal(t *testing.T) {
	message := model.TextMessage(model.RoleUser, strings.Repeat("history ", 80))
	message.Metadata = map[string]string{
		ContextSectionMetadataKey:     ContextSectionRecent,
		RuntimeControlMetadataKey:     "true",
		ToolHistoryMetadataKey:        "true",
		ToolHistoryFailureMetadataKey: "true",
	}
	groups := collapseGroups([]model.Message{message})
	if len(groups) != 1 || groups[0].toolTokens == 0 {
		t.Fatalf("Tool History did not consume Tool Result budget: %+v", groups)
	}
	if !groups[0].failure {
		t.Fatalf("failed Tool History lost its recovery signal: %+v", groups[0])
	}
}

func TestRuntimeControlMessageIsCompressibleWhileHumanTaskRemainsProtected(t *testing.T) {
	task := model.TextMessage(model.RoleUser, "keep this exact human task")
	task.Metadata = map[string]string{TaskAnchorMetadataKey: "true"}
	runtime := model.TextMessage(model.RoleUser, strings.Repeat("runtime recovery guidance ", 200))
	runtime.Metadata = map[string]string{
		ContextSectionMetadataKey: ContextSectionRuntime,
		RuntimeControlMetadataKey: "true",
	}
	projected, _, _, err := ProjectMessagesWithOptions(context.Background(), []model.Message{
		model.TextMessage(model.RoleSystem, "system contract"), task, runtime,
	}, 180, CollapseState{}, ProjectionOptions{SummaryTokens: 40, TargetRatio: 0.7})
	if err != nil {
		t.Fatal(err)
	}
	joined := ""
	for _, message := range projected {
		joined += message.TextContent()
	}
	if !strings.Contains(joined, "keep this exact human task") {
		t.Fatalf("human task was not protected: %+v", projected)
	}
	if strings.Count(joined, "runtime recovery guidance") > 2 {
		t.Fatalf("runtime control message was retained verbatim: %+v", projected)
	}
}

func TestProjectMessagesDoesNotMutateDurableHistory(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "task", Role: model.RoleUser, Content: "keep this exact task"},
	}
	for index := 0; index < 10; index++ {
		messages = append(messages,
			model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("old observation ", 40)},
			model.Message{ID: "tool-" + string(rune('a'+index)), Role: model.RoleTool, ToolCallID: "call", Content: strings.Repeat("tool result ", 40)},
		)
	}
	originalCount := len(messages)
	projected, state, report, err := ProjectMessages(messages, 420, CollapseState{})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Compacted || state.Generation != 1 || len(projected) >= originalCount {
		t.Fatalf("projection report=%+v state=%+v projected=%d original=%d", report, state, len(projected), originalCount)
	}
	if len(messages) != originalCount || messages[2].TextContent() == projected[2].TextContent() {
		t.Fatalf("durable history was mutated or projection was not applied")
	}
	foundTask := false
	for _, message := range projected {
		if message.ID == "task" && message.TextContent() == "keep this exact task" {
			foundTask = true
		}
	}
	if !foundTask {
		t.Fatalf("exact user task was not preserved: %+v", projected)
	}
}

func TestProjectMessagesUsesProactiveTriggerAndSemanticSummary(t *testing.T) {
	messages := []model.Message{{ID: "system", Role: model.RoleSystem, Content: "contract"}, {ID: "task", Role: model.RoleUser, Content: "keep exact task"}}
	for index := 0; index < 6; index++ {
		messages = append(messages,
			model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("fact ", 30)},
			model.Message{ID: "tool-" + string(rune('a'+index)), Role: model.RoleTool, Content: strings.Repeat("evidence ", 30)},
		)
	}
	projected, state, report, err := ProjectMessagesWithOptions(context.Background(), messages, 900, CollapseState{}, ProjectionOptions{
		TriggerRatio:     0.75,
		RecentTurnTokens: 360,
		ToolResultTokens: 180,
		Summarize:        func(context.Context, SummaryInput) (string, error) { return "- API gateway: Kong, not nginx.", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Compacted || report.SummaryMode != "semantic" || state.Generation != 1 || len(projected) >= len(messages) {
		t.Fatalf("unexpected proactive semantic projection: report=%+v state=%+v projected=%d", report, state, len(projected))
	}
	if !strings.Contains(state.Summary, "Kong") {
		t.Fatalf("semantic fact was not retained: %s", state.Summary)
	}
}

func TestProjectMessagesRunsCollapseBarrierBeforeRemovingMessages(t *testing.T) {
	messages := []model.Message{{ID: "system", Role: model.RoleSystem, Content: "contract"}, {ID: "task", Role: model.RoleUser, Content: "task"}}
	for index := 0; index < 6; index++ {
		messages = append(messages,
			model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("fact ", 30)},
			model.Message{ID: "tool-" + string(rune('a'+index)), Role: model.RoleTool, Content: strings.Repeat("evidence ", 30)},
		)
	}
	called := 0
	_, _, report, err := ProjectMessagesWithOptions(context.Background(), messages, 900, CollapseState{}, ProjectionOptions{
		TriggerRatio: 0.75,
		BeforeCollapse: func(_ context.Context, input CollapseBarrierInput) error {
			called++
			if input.SourceHash == "" || len(input.Messages) == 0 {
				t.Fatalf("invalid barrier input: %+v", input)
			}
			return nil
		},
	})
	if err != nil || !report.Compacted || called != 1 {
		t.Fatalf("barrier called=%d report=%+v err=%v", called, report, err)
	}
}

func TestProjectMessagesCarriesPreviousCollapseSummary(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "task", Role: model.RoleUser, Content: "task"},
	}
	for index := 0; index < 8; index++ {
		messages = append(messages,
			model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("fact ", 70)},
			model.Message{ID: "tool-" + string(rune('a'+index)), Role: model.RoleTool, Content: strings.Repeat("evidence ", 70)},
		)
	}
	_, firstState, firstReport, err := ProjectMessages(messages, 360, CollapseState{})
	if err != nil || !firstReport.Compacted {
		t.Fatalf("first projection report=%+v err=%v", firstReport, err)
	}
	for index := 8; index < 12; index++ {
		messages = append(messages,
			model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("new fact ", 70)},
			model.Message{ID: "tool-" + string(rune('a'+index)), Role: model.RoleTool, Content: strings.Repeat("new evidence ", 70)},
		)
	}
	_, secondState, secondReport, err := ProjectMessages(messages, 360, firstState)
	if err != nil {
		t.Fatal(err)
	}
	if !secondReport.Compacted || secondState.Generation <= firstState.Generation || !strings.Contains(secondState.Summary, "prior collapse") {
		t.Fatalf("previous summary was not carried forward: first=%+v second=%+v report=%+v", firstState, secondState, secondReport)
	}
}

func TestProjectMessagesReusesCoveredMessagesIncrementally(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "task", Role: model.RoleUser, Content: "keep exact task"},
		{ID: "old-assistant", Role: model.RoleAssistant, Content: strings.Repeat("old observation ", 40)},
		{ID: "old-tool", Role: model.RoleTool, Content: strings.Repeat("old result ", 40)},
		{ID: "new-assistant", Role: model.RoleAssistant, Content: "new observation"},
	}
	state := CollapseState{
		Generation:        3,
		Summary:           "<context_collapse generation=\"3\">old facts</context_collapse>",
		CoveredMessageIDs: []string{"old-assistant", "old-tool"},
	}
	projected, next, report, err := ProjectMessagesWithOptions(context.Background(), messages, 220, state, ProjectionOptions{TriggerRatio: 0.8})
	if err != nil {
		t.Fatal(err)
	}
	if report.Compacted || next.Generation != state.Generation {
		t.Fatalf("covered range was compacted again: report=%+v state=%+v", report, next)
	}
	joined := ""
	for _, message := range projected {
		joined += message.TextContent() + "\n"
	}
	if !strings.Contains(joined, "old facts") || !strings.Contains(joined, "new observation") {
		t.Fatalf("reused projection lost summary or new tail: %s", joined)
	}
	if strings.Contains(joined, "old observation") || strings.Contains(joined, "old result") {
		t.Fatalf("covered messages were reintroduced: %s", joined)
	}
}

func TestProjectMessagesProtectsRecentToolFailures(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "task", Role: model.RoleUser, Content: "repair task"},
	}
	for index := 0; index < 5; index++ {
		messages = append(messages,
			model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("old work ", 40)},
			model.Message{ID: "tool-" + string(rune('a'+index)), Role: model.RoleTool, Content: `{"error_code":"TOOL_SCHEMA_INVALID","correction":"use the offered schema"}`, Metadata: map[string]string{"tool_failed": "true"}},
		)
	}
	projected, _, report, err := ProjectMessagesWithOptions(context.Background(), messages, 700, CollapseState{}, ProjectionOptions{ToolResultTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Compacted {
		t.Fatalf("expected compaction: %+v", report)
	}
	joined := ""
	for _, message := range projected {
		joined += message.TextContent() + "\n"
	}
	if !strings.Contains(joined, "TOOL_SCHEMA_INVALID") || !strings.Contains(joined, "use the offered schema") {
		t.Fatalf("recent tool failure was discarded: %s", joined)
	}
}

func TestStripRebuildableRuntimeBlocksKeepsStaticSystemContract(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "static contract\n<RUNTIME_DURABLE_PLAN>old plan</RUNTIME_DURABLE_PLAN>\n<RUNTIME_TOOL_PROJECTION>{\"tools\":[]}</RUNTIME_TOOL_PROJECTION>\n<RUNTIME_EXECUTION_LEDGER>STATE={}</RUNTIME_EXECUTION_LEDGER>"},
		{ID: "task", Role: model.RoleUser, Content: "keep exact task and the literal marker <RUNTIME_DURABLE_PLAN>"},
	}
	projected := StripRebuildableRuntimeBlocks(messages)
	if len(projected) != 2 || projected[0].TextContent() != "static contract" {
		t.Fatalf("runtime blocks were not removed without preserving static text: %+v", projected)
	}
	if projected[0].ID != "system" || projected[1].TextContent() != "keep exact task and the literal marker <RUNTIME_DURABLE_PLAN>" {
		t.Fatalf("message identity or task changed: %+v", projected)
	}
}

func TestStripRebuildableRuntimeBlocksDropsEmptyRuntimeSystemMessage(t *testing.T) {
	messages := []model.Message{{ID: "system", Role: model.RoleSystem, Content: "<RUNTIME_TOOL_PROJECTION>{}</RUNTIME_TOOL_PROJECTION>"}}
	if projected := StripRebuildableRuntimeBlocks(messages); len(projected) != 0 {
		t.Fatalf("empty runtime-only system message survived: %+v", projected)
	}
}

func TestProjectMessagesReplacesPreviousSummarySlot(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "old-summary", Role: model.RoleUser, Content: "<context_collapse generation=\"1\">stale</context_collapse>"},
		{ID: "task", Role: model.RoleUser, Content: "task"},
	}
	for index := 0; index < 8; index++ {
		messages = append(messages,
			model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("fact ", 60)},
			model.Message{ID: "tool-" + string(rune('a'+index)), Role: model.RoleTool, Content: strings.Repeat("evidence ", 60)},
		)
	}
	projected, _, report, err := ProjectMessagesWithOptions(context.Background(), messages, 480, CollapseState{}, ProjectionOptions{TargetRatio: 0.65})
	if err != nil || !report.Compacted {
		t.Fatalf("projection report=%+v err=%v", report, err)
	}
	count := 0
	for _, message := range projected {
		if isCollapseSummaryMessage(message) {
			count++
			if message.Metadata[ContextSectionMetadataKey] != ContextSectionSummary {
				t.Fatalf("summary slot missing section metadata: %+v", message.Metadata)
			}
		}
		if message.ID == "old-summary" {
			t.Fatal("stale summary survived projection")
		}
	}
	if count != 1 {
		t.Fatalf("expected exactly one rolling summary, got %d: %+v", count, projected)
	}
}

func TestProjectMessagesCooldownSkipsSmallProactiveGrowth(t *testing.T) {
	messages := []model.Message{{ID: "system", Role: model.RoleSystem, Content: "contract"}, {ID: "task", Role: model.RoleUser, Content: "task"}}
	for index := 0; index < 6; index++ {
		messages = append(messages, model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("fact ", 40)})
	}
	before := 0
	for _, message := range messages {
		before += EstimateTokens(message)
	}
	state := CollapseState{Generation: 1, Summary: "prior", LastCompactionMessageCount: len(messages) - 1, LastCompactionBeforeTokens: before - 50}
	projected, next, report, err := ProjectMessagesWithOptions(context.Background(), messages, 520, state, ProjectionOptions{TriggerRatio: 0.75, MinMessagesBetweenCollapses: 4, MinTokensBetweenCollapses: 2048})
	if err != nil {
		t.Fatal(err)
	}
	if report.Compacted || next.Generation != state.Generation || len(projected) != len(messages) {
		t.Fatalf("small proactive growth was not held back: report=%+v state=%+v projected=%d", report, next, len(projected))
	}
}

func TestProjectMessagesCooldownDoesNotBlockHardBudget(t *testing.T) {
	messages := []model.Message{{ID: "system", Role: model.RoleSystem, Content: "contract"}, {ID: "task", Role: model.RoleUser, Content: "task"}}
	for index := 0; index < 10; index++ {
		messages = append(messages, model.Message{ID: "assistant-" + string(rune('a'+index)), Role: model.RoleAssistant, Content: strings.Repeat("fact ", 80)})
	}
	state := CollapseState{Generation: 1, Summary: "prior", LastCompactionMessageCount: len(messages) - 1, LastCompactionBeforeTokens: 1}
	_, next, report, err := ProjectMessagesWithOptions(context.Background(), messages, 300, state, ProjectionOptions{TriggerRatio: 0.85, TargetRatio: 0.65, MinMessagesBetweenCollapses: 100, MinTokensBetweenCollapses: 100000})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Compacted || next.Generation <= state.Generation {
		t.Fatalf("hard budget was incorrectly held by cooldown: report=%+v state=%+v", report, next)
	}
}

func TestCollapseStateRebaseForContinuationKeepsMemoryLedger(t *testing.T) {
	state := CollapseState{Generation: 4, Summary: "old run", CoveredMessageIDs: []string{"m-1"}, CoveredMessageCount: 1, SourceHash: "hash", FailureStreak: 2, LastCompactionMessageCount: 10, LastCompactionBeforeTokens: 900, SurfacedMemories: []SurfacedMemory{{ID: "memory-1"}}, LastMemoryExtractionSequence: 7}
	state.RebaseForContinuation()
	if state.Generation != 4 || state.Summary != "old run" || state.SourceHash != "hash" || len(state.CoveredMessageIDs) != 1 || state.LastCompactionMessageCount != 0 || state.LastCompactionBeforeTokens != 0 || state.FailureStreak != 0 {
		t.Fatalf("continuation projection boundary was not preserved: %+v", state)
	}
	if len(state.SurfacedMemories) != 1 || state.LastMemoryExtractionSequence != 7 {
		t.Fatalf("memory ledger was unexpectedly discarded: %+v", state)
	}
}

func TestContinuationProjectionReusesCoveredMessages(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "task", Role: model.RoleUser, Content: "original task", Metadata: map[string]string{TaskAnchorMetadataKey: "true"}},
		{ID: "old-assistant", Role: model.RoleAssistant, Content: "old observation"},
		{ID: "new-user", Role: model.RoleUser, Content: "continue"},
	}
	state := CollapseState{Generation: 1, Summary: "<context_collapse generation=\"1\">old observation</context_collapse>", CoveredMessageIDs: []string{"old-assistant"}, CoveredMessageCount: 1, SourceHash: "hash"}
	state.RebaseForContinuation()
	projected, _, _, err := ProjectMessagesWithOptions(context.Background(), messages, 500, state, ProjectionOptions{TriggerRatio: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	for _, message := range projected {
		if message.ID == "old-assistant" {
			t.Fatalf("covered message was reintroduced after continuation: %+v", projected)
		}
	}
	if !strings.Contains(strings.Join(messageTexts(projected), "\n"), "old observation") {
		t.Fatalf("previous summary was not reused: %+v", projected)
	}
}

func TestMemoryMergeCannotRollbackCollapseProjection(t *testing.T) {
	state := CollapseState{
		Generation: 6, Summary: "latest summary", CoveredMessageIDs: []string{"old-1", "old-2"},
		LastMemoryExtractionSequence: 10,
	}
	state.MergeMemoryState(MemoryState{
		SurfacedMemories:             []SurfacedMemory{{ID: "repair-rule", RevisionKey: "v2"}},
		RecentToolEvidence:           []string{"write_file:TOOL_SCHEMA_INVALID"},
		LastMemoryExtractionSequence: 8,
	})
	if state.Generation != 6 || state.Summary != "latest summary" || len(state.CoveredMessageIDs) != 2 {
		t.Fatalf("memory merge rolled back collapse-owned fields: %+v", state)
	}
	if len(state.SurfacedMemories) != 1 || len(state.RecentToolEvidence) != 1 {
		t.Fatalf("memory-owned fields were not merged: %+v", state)
	}
	if state.LastMemoryExtractionSequence != 10 {
		t.Fatalf("memory extraction sequence regressed: %+v", state)
	}
}

func TestMemoryExtractionCursorIsRunScoped(t *testing.T) {
	memory := MemoryState{MemoryExtractionRunID: "run-old", LastMemoryExtractionSequence: 537}
	memory.RebaseMemoryExtractionCursor("run-new", false)
	if memory.MemoryExtractionRunID != "run-new" || memory.LastMemoryExtractionSequence != 0 {
		t.Fatalf("cross-run memory cursor was not reset: %+v", memory)
	}
	memory.LastMemoryExtractionSequence = 12
	memory.RebaseMemoryExtractionCursor("run-new", true)
	if memory.LastMemoryExtractionSequence != 12 {
		t.Fatalf("same-run takeover reset its extraction cursor: %+v", memory)
	}
}

func TestMemoryMergeReplacesCursorAcrossRuns(t *testing.T) {
	state := CollapseState{MemoryExtractionRunID: "run-old", LastMemoryExtractionSequence: 537}
	state.MergeMemoryState(MemoryState{MemoryExtractionRunID: "run-new", LastMemoryExtractionSequence: 12})
	if state.MemoryExtractionRunID != "run-new" || state.LastMemoryExtractionSequence != 12 {
		t.Fatalf("cross-run cursor used numeric max instead of owner replacement: %+v", state)
	}
}

func TestProjectionRestoresSummaryAfterCoveredMessagesWerePruned(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "task", Role: model.RoleUser, Content: "original task", Metadata: map[string]string{TaskAnchorMetadataKey: "true"}},
		{ID: "new-user", Role: model.RoleUser, Content: "continue"},
	}
	state := CollapseState{Generation: 2, Summary: "<context_collapse>durable fact</context_collapse>", CoveredMessageIDs: []string{"already-pruned"}}
	projected, next, report, err := ProjectMessagesWithOptions(context.Background(), messages, 500, state, ProjectionOptions{TriggerRatio: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	if next.Generation != 2 || !report.Projected || !strings.Contains(strings.Join(messageTexts(projected), "\n"), "durable fact") {
		t.Fatalf("summary was not restored from durable state: report=%+v projected=%+v", report, projected)
	}
}

func TestNewCollapseAdvancesAfterOldCoveredMessagesWerePruned(t *testing.T) {
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: "contract"},
		{ID: "task", Role: model.RoleUser, Content: "original task", Metadata: map[string]string{TaskAnchorMetadataKey: "true"}},
	}
	for index := 0; index < 12; index++ {
		messages = append(messages, model.Message{ID: fmt.Sprintf("new-%d", index), Role: model.RoleAssistant, Content: strings.Repeat("new evidence ", 40)})
	}
	state := CollapseState{Generation: 8, Summary: "<context_collapse>prior facts</context_collapse>", CoveredMessageIDs: []string{"old-1", "old-2"}, SourceHash: hashIDs([]string{"old-1", "old-2"})}
	_, next, report, err := ProjectMessagesWithOptions(context.Background(), messages, 300, state, ProjectionOptions{TriggerRatio: 0.8, TargetRatio: 0.6, SummaryTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Compacted || next.Generation != state.Generation+1 || next.SourceHash == state.SourceHash {
		t.Fatalf("new collapse did not advance after physical pruning: report=%+v state=%+v", report, next)
	}
	for _, id := range next.CoveredMessageIDs {
		if id == "old-1" || id == "old-2" {
			t.Fatalf("pruned coverage ID leaked into new boundary: %+v", next.CoveredMessageIDs)
		}
	}
}

func TestDeterministicCollapsePreservesCanonicalWorkspacePath(t *testing.T) {
	messages := []model.Message{{ID: "call", Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{Name: "read_file", Arguments: json.RawMessage(`{"path":"tmp/agent-long-task-demo/issue-tracker/issue_tracker/db.py"}`)}}}}
	summary, mode := buildCollapseSummaryWithOptions(context.Background(), messages, "", 256, 1, ProjectionOptions{})
	if mode != "deterministic" || !strings.Contains(summary, "tmp/agent-long-task-demo/issue-tracker/issue_tracker/db.py") {
		t.Fatalf("mode=%s summary=%s", mode, summary)
	}
}

func TestRollingCollapseCarriesForwardCanonicalWorkspacePath(t *testing.T) {
	previous := `<context_collapse generation="1"><canonical_workspace_paths>
tmp/agent-long-task-demo/issue-tracker/issue_tracker/db.py
</canonical_workspace_paths>
- prior fact
</context_collapse>`
	summary, _ := buildCollapseSummaryWithOptions(context.Background(), []model.Message{{ID: "new", Role: model.RoleUser, Content: "continue"}}, previous, 256, 2, ProjectionOptions{})
	if !strings.Contains(summary, "tmp/agent-long-task-demo/issue-tracker/issue_tracker/db.py") {
		t.Fatalf("rolling summary lost canonical path: %s", summary)
	}
}

func messageTexts(messages []model.Message) []string {
	texts := make([]string, 0, len(messages))
	for _, message := range messages {
		texts = append(texts, message.TextContent())
	}
	return texts
}
