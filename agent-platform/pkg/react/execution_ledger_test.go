package react

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	contextpkg "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/context"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestExecutionLedgerTracksReadRangesAcrossReplacement(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract"), model.TextMessage(model.RoleUser, "task")}
	for _, start := range []int{1, 21, 41} {
		arguments, _ := json.Marshal(map[string]any{"path": "docs/test.md", "start_line": start, "line_count": 20})
		messages = updateExecutionLedger(messages, model.ToolCall{Name: "read_file", Arguments: arguments}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	}
	ledger := loadExecutionLedger(messages)
	if ledger.ToolCalls != 3 || ledger.Succeeded != 3 || ledger.Failed != 0 {
		t.Fatalf("unexpected totals: %+v", ledger)
	}
	if ledger.ByTool["read_file"].Succeeded != 3 || ledger.ByTool["write_file"].Succeeded != 0 {
		t.Fatalf("unexpected per-tool totals: %+v", ledger.ByTool)
	}
	want := []string{"1+20", "21+20", "41+20"}
	if got := ledger.ReadFiles["docs/test.md"].Succeeded; !reflect.DeepEqual(got, want) {
		t.Fatalf("read ranges = %#v, want %#v", got, want)
	}
	guidance := 0
	for _, message := range messages {
		if message.Metadata[executionGuidanceMetadata] == "true" {
			guidance++
		}
	}
	if guidance != 1 {
		t.Fatalf("ledger guidance accumulated: count=%d messages=%+v", guidance, messages)
	}
}

func TestExecutionLedgerRecordsFailedPrerequisite(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	call := model.ToolCall{Name: "edit_file", Arguments: json.RawMessage(`{"path":"tmp/report.md"}`)}
	messages = updateExecutionLedger(messages, call, tool.Result{Content: json.RawMessage(`{"error":"missing"}`), IsError: true, Error: "missing"})
	ledger := loadExecutionLedger(messages)
	if ledger.Failed != 1 || !reflect.DeepEqual(ledger.Edits, []string{"tmp/report.md:failed"}) || len(ledger.Errors) != 1 {
		t.Fatalf("failed action not recorded: %+v", ledger)
	}
}

func TestExecutionLedgerPreservesExactSuccessfulEvidenceReceipts(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "call-write-1", Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py","content":"print('ok')"}`)}, tool.Result{Content: json.RawMessage(`{"path":"game.py","sha256":"abc"}`)})
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "call-plan-1", Name: "update_plan_step", Arguments: json.RawMessage(`{"step_id":"1","status":"in_progress"}`)}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	ledger := loadExecutionLedger(messages)
	if len(ledger.SuccessfulEvidence) != 1 || ledger.SuccessfulEvidence[0].CallID != "call-write-1" || ledger.SuccessfulEvidence[0].Tool != "write_file" {
		t.Fatalf("successful evidence receipt missing or polluted by control tools: %+v", ledger.SuccessfulEvidence)
	}
	if !strings.Contains(messages[0].TextContent(), "never invent a receipt") || !strings.Contains(messages[0].TextContent(), "call-write-1") {
		t.Fatalf("model-visible evidence guidance missing: %s", messages[0].TextContent())
	}
}

func TestExecutionLedgerTracksLatestFileManifestSeparatelyFromArtifacts(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	result := tool.Result{
		Content: json.RawMessage(`{"path":"src/game.py","bytes":42,"line_count":3,"file_sha256":"file-hash-2","content_sha256":"file-hash-2","syntax_status":"unverified"}`),
		Meta: map[string]string{
			"artifact_1":        "artifact-2",
			"artifact_1_kind":   "workspace_file",
			"artifact_1_sha256": "artifact-hash-2",
		},
	}
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "call-2", Name: "append_file", Arguments: json.RawMessage(`{"path":"src/game.py","content":"next"}`)}, result)
	ledger := loadExecutionLedger(messages)
	file, ok := ledger.Files["src/game.py"]
	if !ok {
		t.Fatalf("latest file manifest missing: %+v", ledger)
	}
	if file.Revision != 1 || file.FileSHA256 != "file-hash-2" || file.ArtifactID != "artifact-2" || file.ArtifactSHA256 != "artifact-hash-2" || file.LastOperation != "append_file" || file.LineCount != 3 {
		t.Fatalf("unexpected file manifest: %+v", file)
	}
	joined := ""
	for _, message := range messages {
		joined += message.TextContent()
	}
	if !strings.Contains(joined, "file_sha256=file-hash-2") || !strings.Contains(joined, "syntax_status=unverified") {
		t.Fatalf("file state was not surfaced in runtime guidance: %s", joined)
	}
}

func TestExecutionLedgerBindsPyCompileFailureToExactFile(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "write-1", Name: "write_file", Arguments: json.RawMessage(`{"path":"src/game.py","content":"broken"}`)}, tool.Result{Content: json.RawMessage(`{"path":"src/game.py","bytes":6,"line_count":1,"file_sha256":"file-hash","syntax_status":"unverified"}`)})
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "check-1", Name: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["-m","py_compile","src/game.py"]}`)}, tool.Result{Content: json.RawMessage(`{"exit_code":1,"stderr":"src/game.py:1:1: SyntaxError: invalid syntax"}`), IsError: true, Error: "command exited with code 1"})
	file := loadExecutionLedger(messages).Files["src/game.py"]
	if file.SyntaxStatus != "failed" || !strings.Contains(file.VerificationError, "SyntaxError") {
		t.Fatalf("py_compile failure was not bound to file: %+v", file)
	}
	if !strings.Contains(messages[len(messages)-1].TextContent(), "Repair the reported range") {
		t.Fatalf("repair guidance missing: %+v", messages)
	}
}

func TestExecutionLedgerModelProjectionIsBoundedWithoutMutatingDurableState(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract"), model.TextMessage(model.RoleUser, "build the project")}
	for index := 0; index < 12; index++ {
		callID := fmt.Sprintf("call-write-%02d", index)
		arguments := json.RawMessage(fmt.Sprintf(`{"path":"part-%02d.py","content":"%s"}`, index, strings.Repeat("large source payload ", 20)))
		messages = updateExecutionLedger(messages, model.ToolCall{ID: callID, Name: "write_file", Arguments: arguments}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	}
	durableBefore := executionLedgerJSON(messages)
	projected := projectExecutionLedgerForModel(messages)
	durableAfter := executionLedgerJSON(messages)

	if !reflect.DeepEqual(durableBefore, durableAfter) {
		t.Fatal("model projection mutated the durable execution ledger")
	}
	if estimateMessages(projected) >= estimateMessages(messages) {
		t.Fatalf("projection was not smaller: full=%d projected=%d", estimateMessages(messages), estimateMessages(projected))
	}
	ledger := loadExecutionLedger(projected)
	if ledger.ToolCalls != 12 || ledger.Succeeded != 12 || len(ledger.SuccessfulEvidence) != 8 {
		t.Fatalf("projected counters or bounded evidence are invalid: %+v", ledger)
	}
	latest := ledger.SuccessfulEvidence[len(ledger.SuccessfulEvidence)-1]
	if latest.CallID != "call-write-11" || !strings.Contains(latest.Arguments, `"path":"part-11.py"`) || strings.Contains(latest.Arguments, "large source payload") {
		t.Fatalf("latest evidence receipt was not compacted safely: %+v", latest)
	}
}

func TestExecutionLedgerModelProjectionHasHardCharacterCeiling(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract"), model.TextMessage(model.RoleUser, "build")}
	for index := 0; index < 80; index++ {
		path := fmt.Sprintf("src/module-%02d.py", index)
		arguments := json.RawMessage(fmt.Sprintf(`{"path":%q,"start_line":1,"line_count":200}`, path))
		result := tool.Result{Content: json.RawMessage(`{"path":"src/module.py","bytes":8000,"line_count":200,"file_sha256":"hash","syntax_status":"unverified"}`)}
		messages = updateExecutionLedger(messages, model.ToolCall{ID: fmt.Sprintf("call-%03d", index), Name: "read_file", Arguments: arguments}, result)
	}
	projected := projectExecutionLedgerForModel(messages)
	if got := len([]rune(projected[0].TextContent())); got > modelLedgerMaxChars+600 {
		t.Fatalf("ledger projection exceeds hard ceiling with wrapper: chars=%d ceiling=%d", got, modelLedgerMaxChars+600)
	}
	if strings.Contains(projected[0].TextContent(), `"line_count":200`) {
		t.Fatalf("projection leaked unbounded read arguments: %s", projected[0].TextContent())
	}
}

func TestExecutionLedgerTracksConsecutiveFailuresForToolBackoff(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	failed := tool.Result{Content: json.RawMessage(`{"error":"exists"}`), IsError: true, Error: "exists"}
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "call-1", Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "call-2", Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py","content":"next"}`)}, failed)
	ledger := loadExecutionLedger(messages)
	if ledger.FailureStreakTool != "write_file" || ledger.FailureStreakCount != 2 {
		t.Fatalf("failure streak = %s/%d", ledger.FailureStreakTool, ledger.FailureStreakCount)
	}
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "call-3", Name: "append_file", Arguments: json.RawMessage(`{"path":"game.py","content":"next"}`)}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	ledger = loadExecutionLedger(messages)
	if ledger.FailureStreakTool != "" || ledger.FailureStreakCount != 0 {
		t.Fatalf("successful recovery did not reset failure streak: %+v", ledger)
	}
}

func TestExecutionLedgerCountsListingsUntilWorkspaceMutation(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	for index := 0; index < 3; index++ {
		messages = updateExecutionLedger(messages, model.ToolCall{Name: "list_files", Arguments: json.RawMessage(`{"path":"."}`)}, tool.Result{Content: json.RawMessage(`{"entries":[]}`)})
	}
	if got := loadExecutionLedger(messages).ListsSinceMutation; got != 3 {
		t.Fatalf("list observations = %d, want 3", got)
	}
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "append_file", Arguments: json.RawMessage(`{"path":"game.py","content":"next"}`)}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	if got := loadExecutionLedger(messages).ListsSinceMutation; got != 0 {
		t.Fatalf("list observations after mutation = %d, want 0", got)
	}
}

func TestProgressReviewSurfacesCrossToolStagnationOnce(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	failed := tool.Result{Content: json.RawMessage(`{"error":"invalid"}`), IsError: true, Error: "invalid"}
	// Alternate tools so the exact same-tool failure-streak guard cannot hide
	// the broader no-progress condition.
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "read_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "update_plan", Arguments: json.RawMessage(`{"goal":"build"}`)}, failed)
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["-m","py_compile","game.py"]}`)}, failed)

	updated, review := ensureProgressReview(messages, false)
	if review == nil || review.Generation != 1 || review.Phase != "repair" || review.Action != "retry" || review.FailedToolCalls != 3 || review.WorkspaceMutations != 0 {
		t.Fatalf("unexpected progress review: %+v", review)
	}
	if !strings.Contains(updated[len(updated)-1].TextContent(), "STRATEGY CHECKPOINT #1") {
		t.Fatalf("strategy checkpoint was not injected: %+v", updated)
	}
	if _, duplicate := ensureProgressReview(updated, false); duplicate != nil {
		t.Fatalf("same observations emitted a duplicate progress review: %+v", duplicate)
	}
}

func TestProgressReviewIsPreservedByLedgerProjection(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	for index := 0; index < 8; index++ {
		messages = updateExecutionLedger(messages, model.ToolCall{Name: "update_plan_step", Arguments: json.RawMessage(`{"step_id":"build","status":"in_progress"}`)}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	}
	updated, review := ensureProgressReview(messages, false)
	if review == nil || !strings.Contains(review.Reason, "without a new workspace mutation") {
		t.Fatalf("no-progress review = %+v", review)
	}
	projected := projectExecutionLedgerForModel(updated)
	got := loadExecutionLedger(projected).ProgressReview
	if got == nil || got.Generation != review.Generation || got.Reason != review.Reason {
		t.Fatalf("projection lost progress review: got=%+v want=%+v", got, review)
	}
	if loadExecutionLedger(projected).ProgressReviewBaseline != nil {
		t.Fatal("durable progress-review baseline leaked into the model projection")
	}
}

func TestProgressReviewDoesNotRepeatWithoutNewToolObservation(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	failed := tool.Result{Content: json.RawMessage(`{"error":"missing content"}`), IsError: true, Error: "missing content"}
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	updated, first := ensureProgressReview(messages, false)
	if first == nil || first.Generation != 1 {
		t.Fatalf("first strategy checkpoint = %+v", first)
	}
	if _, second := ensureProgressReview(updated, false); second != nil {
		t.Fatalf("strategy checkpoint repeated without a new tool result: %+v", second)
	}
}

func TestProgressReviewAllowsBoundedRecoveryWindow(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	failed := tool.Result{Content: json.RawMessage(`{"error":"invalid"}`), IsError: true, Error: "invalid"}
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	messages, first := ensureProgressReview(messages, false)
	if first == nil {
		t.Fatal("initial repair review was not created")
	}
	for index := 0; index < 2; index++ {
		messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
		var repeated *progressReview
		messages, repeated = ensureProgressReview(messages, false)
		if repeated != nil {
			t.Fatalf("review repeated before recovery window ended: %+v", repeated)
		}
	}
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	messages, next := ensureProgressReview(messages, false)
	if next == nil || next.Generation != first.Generation+1 || next.Action != "replan" {
		t.Fatalf("review was not renewed after bounded recovery window: %+v", next)
	}
	for index := 0; index < 3; index++ {
		messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	}
	_, third := ensureProgressReview(messages, false)
	if third == nil || third.Generation != 3 || third.Action != "review" {
		t.Fatalf("repeated no-progress recovery did not escalate to review: %+v", third)
	}
}

func TestProgressReviewBaselineSurvivesSuccessfulRecovery(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	failed := tool.Result{Content: json.RawMessage(`{"error":"invalid"}`), IsError: true, Error: "invalid"}
	for _, name := range []string{"read_file", "update_plan", "run_command"} {
		messages = updateExecutionLedger(messages, model.ToolCall{Name: name, Arguments: json.RawMessage(`{}`)}, failed)
	}
	messages, first := ensureProgressReview(messages, false)
	if first == nil {
		t.Fatal("initial progress review was not created")
	}
	// A successful implementation action clears the active instruction but the
	// checkpoint baseline must remain, otherwise the three historical failures
	// immediately create another generation.
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py","content":"ok"}`)}, tool.Result{Content: json.RawMessage(`{"path":"game.py"}`)})
	ledger := loadExecutionLedger(messages)
	if ledger.ProgressReview != nil || ledger.ProgressReviewBaseline == nil || ledger.ProgressReviewBaseline.Generation != first.Generation {
		t.Fatalf("review lifecycle is inconsistent: %+v", ledger)
	}
	if _, repeated := ensureProgressReview(messages, false); repeated != nil {
		t.Fatalf("historical failures were counted again after successful recovery: %+v", repeated)
	}
}

func TestSingleFileFocusTriggersReviewAndHardBudget(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "system")}
	for index := 0; index < 8; index++ {
		messages = updateExecutionLedger(messages, model.ToolCall{ID: fmt.Sprintf("read-%d", index), Name: "read_file", Arguments: json.RawMessage(`{"path":"db.py","start_line":1,"line_count":80}`)}, tool.Result{Content: json.RawMessage(`{"content":"x"}`)})
	}
	messages, review := ensureProgressReview(messages, false)
	if review == nil || review.Phase != "decompose" || review.Action != "replan" || !strings.Contains(review.Reason, "db.py") {
		t.Fatalf("file-focus review = %+v", review)
	}
	for index := 8; index < 14; index++ {
		messages = updateExecutionLedger(messages, model.ToolCall{ID: fmt.Sprintf("edit-%d", index), Name: "edit_file", Arguments: json.RawMessage(`{"path":"db.py"}`)}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	}
	err := validateFileFocusBudget(messages, model.ToolCall{Name: "read_file", Arguments: json.RawMessage(`{"path":"db.py"}`)})
	contractErr, ok := tool.AsContractError(err)
	if !ok || contractErr.Code != "FILE_FOCUS_BUDGET_EXCEEDED" {
		t.Fatalf("focus budget error = %#v", err)
	}
	if err := validateFileFocusBudget(messages, model.ToolCall{Name: "read_file", Arguments: json.RawMessage(`{"path":"models.py"}`)}); err != nil {
		t.Fatalf("different module should be allowed: %v", err)
	}
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "verify", Name: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["-m","py_compile","db.py"]}`)}, tool.Result{Content: json.RawMessage(`{"exit_code":0}`)})
	if err := validateFileFocusBudget(messages, model.ToolCall{Name: "edit_file", Arguments: json.RawMessage(`{"path":"db.py"}`)}); err != nil {
		t.Fatalf("successful verification should reset focus: %v", err)
	}
}

func TestBoundedLiveTranscriptRestoresFullExecutionLedger(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract"), model.TextMessage(model.RoleUser, "task")}
	for index := 0; index < 5; index++ {
		messages = updateExecutionLedger(messages, model.ToolCall{ID: fmt.Sprintf("call-%d", index), Name: "read_file", Arguments: json.RawMessage(fmt.Sprintf(`{"path":"file-%d"}`, index))}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	}
	full := executionLedgerJSON(messages)
	bounded := contextpkg.StripRebuildableRuntimeBlocks(messages)
	restored := restoreExecutionLedger(bounded, full)
	ledger := loadExecutionLedger(restored)
	if ledger.ToolCalls != 5 || len(ledger.SuccessfulEvidence) != 5 {
		t.Fatalf("live compaction lost execution receipts: %+v", ledger)
	}
}

func TestProgressReviewProjectsDeterministicToolSurface(t *testing.T) {
	all := []model.ToolSchema{
		{Name: "read_file"}, {Name: "list_files"}, {Name: "write_file"},
		{Name: "run_command"}, {Name: "update_plan"}, {Name: "update_plan_step"},
		{Name: "delegate_agent"}, {Name: "ask_user"},
	}
	base := []model.Message{model.TextMessage(model.RoleSystem, "system")}
	replan := replaceExecutionLedger(base, executionLedger{ProgressReview: &progressReview{Action: "replan"}})
	got := schemaNames(applyProgressReviewToolProjection(replan, withoutToolSchema(all, "update_plan"), all))
	if strings.Join(got, ",") != "update_plan" {
		t.Fatalf("replan tools = %v", got)
	}
	if choice := toolChoiceForMessages(replan, applyProgressReviewToolProjection(replan, all, all)); choice != "required" {
		t.Fatalf("replan tool choice = %q", choice)
	}
	review := replaceExecutionLedger(base, executionLedger{ProgressReview: &progressReview{Action: "review"}})
	got = schemaNames(applyProgressReviewToolProjection(review, withoutToolSchemas(all, "delegate_agent", "update_plan"), all))
	if strings.Join(got, ",") != "delegate_agent" {
		t.Fatalf("review tools = %v", got)
	}
	if choice := toolChoiceForMessages(review, applyProgressReviewToolProjection(review, all, all)); choice != "required" {
		t.Fatalf("review tool choice = %q", choice)
	}
	withoutReviewer := withoutToolSchema(all, "delegate_agent")
	got = schemaNames(applyProgressReviewToolProjection(review, withoutReviewer, withoutReviewer))
	if strings.Join(got, ",") != "update_plan" {
		t.Fatalf("review fallback tools = %v", got)
	}
	userDecision := replaceExecutionLedger(base, executionLedger{ProgressReview: &progressReview{Action: "ask_user"}})
	got = schemaNames(applyProgressReviewToolProjection(userDecision, withoutToolSchema(all, "ask_user"), all))
	if strings.Join(got, ",") != "ask_user" || toolChoiceForMessages(userDecision, applyProgressReviewToolProjection(userDecision, all, all)) != "required" {
		t.Fatalf("ask_user projection = %v", got)
	}
}

func TestProgressReviewExhaustionRequiresUserDecision(t *testing.T) {
	if action := progressReviewAction("repair", 4); action != "review" {
		t.Fatalf("generation 4 action = %s", action)
	}
	if action := progressReviewAction("repair", 5); action != "ask_user" {
		t.Fatalf("generation 5 action = %s", action)
	}
}

func TestUserDecisionResetsAutonomousRecoveryGeneration(t *testing.T) {
	review := &progressReview{Generation: 5, Phase: "clarify", Action: "ask_user", ToolCalls: 20, FailedToolCalls: 8, WorkspaceMutations: 2}
	messages := replaceExecutionLedger([]model.Message{model.TextMessage(model.RoleSystem, "system")}, executionLedger{
		ToolCalls: 20, Failed: 8, ProgressReview: review, ProgressReviewBaseline: review,
	})
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "ask_user"}, tool.Result{Content: json.RawMessage(`{"answer":"use staging"}`)})
	ledger := loadExecutionLedger(messages)
	if ledger.ProgressReview != nil || ledger.ProgressReviewBaseline == nil || ledger.ProgressReviewBaseline.Generation != 0 || ledger.ProgressReviewBaseline.ToolCalls != 21 {
		t.Fatalf("user decision did not reset bounded recovery: %+v", ledger)
	}
}

func TestSuccessfulReplanReleasesDeterministicToolProjection(t *testing.T) {
	all := []model.ToolSchema{{Name: "read_file"}, {Name: "write_file"}, {Name: "update_plan"}}
	messages := []model.Message{model.TextMessage(model.RoleSystem, "system")}
	messages = replaceExecutionLedger(messages, executionLedger{
		ProgressReview:         &progressReview{Generation: 2, Action: "replan"},
		ProgressReviewBaseline: &progressReview{Generation: 2, Action: "replan"},
	})
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "update_plan", Arguments: json.RawMessage(`{"goal":"finish","steps":[]}`)}, tool.Result{Content: json.RawMessage(`{"status":"updated"}`)})
	ledger := loadExecutionLedger(messages)
	if ledger.ProgressReview != nil {
		t.Fatalf("successful replan left active review = %+v", ledger.ProgressReview)
	}
	if ledger.ProgressReviewBaseline == nil || ledger.ProgressReviewBaseline.Generation != 2 {
		t.Fatalf("successful replan lost review baseline = %+v", ledger.ProgressReviewBaseline)
	}
	got := schemaNames(applyProgressReviewToolProjection(messages, all, all))
	if strings.Join(got, ",") != "read_file,update_plan,write_file" {
		t.Fatalf("post-replan tools remain projected = %v", got)
	}
}

func TestFailedAutomaticReviewerFallsBackToReplan(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "system")}
	review := &progressReview{Generation: 3, Phase: "review", Action: "review", Reason: "stalled", ToolCalls: 12}
	messages = replaceExecutionLedger(messages, executionLedger{ToolCalls: 12, ProgressReview: review, ProgressReviewBaseline: review})
	updated, fallback := fallbackReviewerToReplan(messages, "reviewer failed")
	if fallback == nil || fallback.Action != "replan" || fallback.Phase != "decompose" || fallback.Generation != 4 || fallback.Reason != "reviewer failed" {
		t.Fatalf("fallback = %+v", fallback)
	}
	ledger := loadExecutionLedger(updated)
	if ledger.ProgressReview == nil || ledger.ProgressReview.Action != "replan" || ledger.ProgressReviewBaseline == nil || ledger.ProgressReviewBaseline.Generation != 4 {
		t.Fatalf("fallback ledger = %+v", ledger)
	}
}

func TestStructuredReviewerChangesRequirePlanCASReplan(t *testing.T) {
	call := model.ToolCall{ID: automaticReviewerCallPrefix + "g3", Name: "delegate_agent", Arguments: json.RawMessage(`{"target_agent_version_id":"reviewer","mode":"sync","input":{"base_plan_revision":9}}`)}
	messages := []model.Message{model.TextMessage(model.RoleSystem, "system")}
	review := &progressReview{Generation: 3, Phase: "review", Action: "review", Reason: "stalled", ToolCalls: 12}
	messages = replaceExecutionLedger(messages, executionLedger{ToolCalls: 12, ProgressReview: review, ProgressReviewBaseline: review})
	messages = append(messages, automaticReviewerMessage("run", 1, 1, call))
	result := tool.Result{Content: json.RawMessage(`{"child_run_id":"child","status":"completed","output":{"verdict":"changes_required","summary":"split the persistence work","findings":[{"severity":"high","summary":"mixed responsibilities","evidence":"db.py combines schema and transport"}],"recommended_plan_changes":[{"operation":"modify_step","step_id":"implement-db","description":"separate persistence boundaries","reason":"separate persistence boundaries"},{"operation":"modify_step","step_id":"wire-store","description":"use the new boundary","reason":"use the new boundary"},{"operation":"retire_step","step_id":"legacy-db","reason":"obsolete"},{"operation":"modify_step","step_id":"verify-db","description":"verify the corrected boundary","reason":"verify the corrected boundary"}]}}`)}
	messages = updateExecutionLedger(messages, call, result)
	ledger := loadExecutionLedger(messages)
	if ledger.ReviewerDecision == nil || ledger.ReviewerDecision.Verdict != "changes_required" || ledger.ReviewerDecision.BasePlanRevision != 9 {
		t.Fatalf("Reviewer decision = %+v", ledger.ReviewerDecision)
	}
	if len(ledger.ReviewerDecision.RecommendedPlanChanges) != 4 {
		t.Fatalf("Reviewer changes were truncated before atomic Plan compilation: %+v", ledger.ReviewerDecision.RecommendedPlanChanges)
	}
	if ledger.ProgressReview == nil || ledger.ProgressReview.Action != "replan" || !strings.Contains(ledger.ProgressReview.Reason, "Plan revision 9") {
		t.Fatalf("Reviewer did not force CAS replan: %+v", ledger.ProgressReview)
	}
	validUpdate := model.ToolCall{Name: "update_plan", Arguments: json.RawMessage(`{"base_revision":9}`)}
	if err := validateReviewerPlanCAS(messages, validUpdate); err != nil {
		t.Fatalf("matching Reviewer Plan revision was rejected: %v", err)
	}
	staleUpdate := model.ToolCall{Name: "update_plan", Arguments: json.RawMessage(`{"base_revision":10}`)}
	if err := validateReviewerPlanCAS(messages, staleUpdate); err == nil || !strings.Contains(err.Error(), "revision") {
		t.Fatalf("mismatched Reviewer Plan revision was accepted: %v", err)
	}
	conflict := tool.Result{IsError: true, Error: "Plan revision conflict", Content: json.RawMessage(`{"error_code":"PLAN_REVISION_CONFLICT"}`)}
	messages = updateExecutionLedger(messages, validUpdate, conflict)
	ledger = loadExecutionLedger(messages)
	if ledger.ReviewerDecision != nil || ledger.ProgressReview == nil || !strings.Contains(ledger.ProgressReview.Reason, "stale") {
		t.Fatalf("database CAS conflict did not retire stale Reviewer advice: %+v", ledger)
	}
}

func TestBlockedReviewerRequiresFocusedUserDecision(t *testing.T) {
	call := model.ToolCall{ID: automaticReviewerCallPrefix + "blocked", Name: "delegate_agent", Arguments: json.RawMessage(`{"input":{"base_plan_revision":4}}`)}
	review := &progressReview{Generation: 3, Phase: "review", Action: "review", Reason: "stalled"}
	messages := replaceExecutionLedger([]model.Message{model.TextMessage(model.RoleSystem, "system")}, executionLedger{ProgressReview: review, ProgressReviewBaseline: review})
	messages = append(messages, automaticReviewerMessage("run", 1, 1, call))
	result := tool.Result{Content: json.RawMessage(`{"child_run_id":"child","output":{"verdict":"blocked","summary":"required deployment target is unknown","findings":[{"severity":"high","summary":"missing target","evidence":"workspace has no deployment configuration"}],"recommended_plan_changes":[]}}`)}
	messages = updateExecutionLedger(messages, call, result)
	ledger := loadExecutionLedger(messages)
	if ledger.ProgressReview == nil || ledger.ProgressReview.Action != "ask_user" || ledger.ProgressReview.Phase != "clarify" || ledger.ProgressReview.Confidence != "low" {
		t.Fatalf("blocked Reviewer escalation = %+v", ledger.ProgressReview)
	}
	if followUp := automaticReviewerFollowUp(messages, []model.ToolCall{call}); followUp == nil || followUp.Action != "ask_user" {
		t.Fatalf("blocked Reviewer follow-up = %+v", followUp)
	}
}

func TestReviewerVerificationCASConflictRetiresStaleAdvice(t *testing.T) {
	decision := &reviewerDecision{CallID: "review", BasePlanRevision: 6, Verdict: "changes_required"}
	review := &progressReview{Generation: 3, Action: "replan", Reason: "repair verification"}
	messages := replaceExecutionLedger([]model.Message{model.TextMessage(model.RoleSystem, "system")}, executionLedger{ReviewerDecision: decision, ProgressReview: review, ProgressReviewBaseline: review})
	call := model.ToolCall{Name: "revise_verification", Arguments: json.RawMessage(`{"base_revision":6}`)}
	conflict := tool.Result{IsError: true, Error: "Plan revision conflict", Content: json.RawMessage(`{"error_code":"PLAN_REVISION_CONFLICT"}`)}
	messages = updateExecutionLedger(messages, call, conflict)
	ledger := loadExecutionLedger(messages)
	if ledger.ReviewerDecision != nil || ledger.ProgressReview == nil || !strings.Contains(ledger.ProgressReview.Reason, "stale") {
		t.Fatalf("verification CAS conflict did not retire stale advice: %+v", ledger)
	}
}

func TestCachedReadObservationIsInvalidatedByMutation(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	read := model.ToolCall{Name: "read_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}
	messages = updateExecutionLedger(messages, read, tool.Result{Content: json.RawMessage(`{"path":"game.py","content":"old"}`)})
	if _, ok := cachedReadObservation(messages, read); !ok {
		t.Fatal("successful read was not cached in the durable execution ledger")
	}
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py","content":"next"}`)}, tool.Result{Content: json.RawMessage(`{"path":"game.py","file_sha256":"new"}`)})
	if _, ok := cachedReadObservation(messages, read); ok {
		t.Fatal("workspace mutation did not invalidate the cached observation")
	}
}

func TestExecutionLedgerDurablyGatesDeterministicCommandRetryUntilMutation(t *testing.T) {
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract")}
	call := model.ToolCall{Name: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["test.py"]}`)}
	failure := tool.Result{IsError: true, Error: "command exited with code 1", Content: json.RawMessage(`{"failure_kind":"process_exit","exit_code":1,"diagnostic":"AssertionError: failed"}`)}
	messages = updateExecutionLedger(messages, call, failure)
	if err := validateDurableCommandRecovery(messages, call); err == nil || !strings.Contains(err.Error(), "unchanged workspace") {
		t.Fatalf("durable command retry was not blocked: %v", err)
	}
	// A read is observation only and must not unlock a deterministic retry.
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "read_file", Arguments: json.RawMessage(`{"path":"test.py"}`)}, tool.Result{Content: json.RawMessage(`{"path":"test.py"}`)})
	if err := validateDurableCommandRecovery(messages, call); err == nil {
		t.Fatal("read-only observation unlocked deterministic retry")
	}
	// A successful file mutation advances the durable workspace revision.
	messages = updateExecutionLedger(messages, model.ToolCall{Name: "edit_file", Arguments: json.RawMessage(`{"path":"test.py","old_text":"a","new_text":"b"}`)}, tool.Result{Content: json.RawMessage(`{"path":"test.py","file_sha256":"next","bytes":1,"line_count":1}`)})
	if err := validateDurableCommandRecovery(messages, call); err != nil {
		t.Fatalf("workspace mutation did not unlock corrected retry: %v", err)
	}
	encoded := executionLedgerJSON(messages)
	restored := restoreExecutionLedger([]model.Message{model.TextMessage(model.RoleSystem, "contract")}, encoded)
	if ledger := loadExecutionLedger(restored); ledger.WorkspaceRevision != 1 || len(ledger.CommandFailures) != 1 {
		t.Fatalf("durable retry state was not checkpoint-safe: %+v", ledger)
	}
}
