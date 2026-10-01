package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	contextpkg "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/context"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/react"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestMemoryCandidateFilteringUsesSurfacedRevisionAndRecentToolEvidence(t *testing.T) {
	now := time.Now().UTC()
	project := agent.Memory{ID: "project-1", SourceLayer: agent.MemoryLayerAuto, SemanticType: agent.MemoryTypeProject, ContentHash: "hash-1", UpdatedAt: now, Description: "API gateway uses Kong, not nginx"}
	reference := agent.Memory{ID: "reference-1", SourceLayer: agent.MemoryLayerAuto, SemanticType: agent.MemoryTypeReference, ContentHash: "hash-2", UpdatedAt: now, Title: "Testing guide", Description: "Testing guide is in docs/testing.md"}
	state := contextpkg.MemoryState{SurfacedMemories: []contextpkg.SurfacedMemory{{ID: project.ID, RevisionKey: project.ContentHash, Turn: 3}}}
	recent := []model.Message{{Role: model.RoleTool, Content: "read docs/testing.md and inspect the testing guide"}}
	selected, suppressed := filterMemoryCandidates([]agent.Memory{project, reference}, state, recent, 4, 3)
	if len(selected) != 0 {
		t.Fatalf("selected = %+v, want both candidates suppressed by their independent rules", selected)
	}
	if len(suppressed) != 2 || !strings.Contains(strings.Join(suppressed, ","), "already_surfaced") || !strings.Contains(strings.Join(suppressed, ","), "recent_tool_evidence") {
		t.Fatalf("suppressed = %v", suppressed)
	}
	state = contextpkg.MemoryState{}
	markMemoryContextState(&state, []agent.Memory{project}, recent, 4)
	if len(state.SurfacedMemories) != 1 || state.SurfacedMemories[0].RevisionKey != project.ContentHash || len(state.RecentToolEvidence) != 1 {
		t.Fatalf("state = %+v", state)
	}
}

func TestFailureMemoryLookupCanResurfaceRelevantFeedback(t *testing.T) {
	now := time.Now().UTC()
	feedback := agent.Memory{ID: "feedback-1", SemanticType: agent.MemoryTypeFeedback, ContentHash: "hash-1", UpdatedAt: now, Description: "When update_plan fails validation, follow the correction exactly."}
	state := contextpkg.MemoryState{SurfacedMemories: []contextpkg.SurfacedMemory{{ID: feedback.ID, RevisionKey: feedback.ContentHash, Turn: 3}}}

	selected, suppressed := filterMemoryCandidatesForFailure([]agent.Memory{feedback}, state, nil, 4, 3)
	if len(selected) != 1 || selected[0].ID != feedback.ID {
		t.Fatalf("selected=%v, suppressed=%v; failure recovery must be allowed to resurface the repair rule", selected, suppressed)
	}
	if len(suppressed) != 0 {
		t.Fatalf("suppressed=%v", suppressed)
	}
}

func TestToolSpecificMemoryCandidatesExcludeUnrelatedSchemaRules(t *testing.T) {
	memories := []agent.Memory{
		{ID: "write", Title: "write_file content maxLength", Body: "write_file chunks"},
		{ID: "plan", Title: "Plan verification", Body: "update_plan assertions must be non-empty"},
		{ID: "read", Title: "read_file path", Body: "read_file requires an existing parent"},
	}
	selected := toolSpecificMemoryCandidates(memories, "update_plan")
	if len(selected) != 1 || selected[0].ID != "plan" {
		t.Fatalf("tool-specific memories = %+v", selected)
	}
}

func TestToolFailureRecoveryContextUsesExactCurrentContract(t *testing.T) {
	write := buildToolFailureRecoveryContext(react.ToolFailure{
		ToolName: "write_file", ErrorCode: "TOOL_SCHEMA_INVALID",
		Error: "content maxLength is 8192", Arguments: json.RawMessage(`{"path":"pkg/core.py","content":"` + strings.Repeat("x", 7000) + `"}`),
	})
	if write.Metadata[react.ToolFailureRecoveryMetadata] != "true" ||
		write.Metadata[react.ToolFailureRecoveryToolMetadata] != "write_file" ||
		write.Metadata[react.FileChunkRecoveryMetadata] != "true" ||
		write.Metadata[react.FileChunkRecoveryPathMetadata] != "pkg/core.py" ||
		!strings.Contains(write.TextContent(), "&lt;=6000") ||
		!strings.Contains(write.TextContent(), "append") {
		t.Fatalf("write recovery reminder = %+v", write)
	}
	run := buildToolFailureRecoveryContext(react.ToolFailure{
		ToolName: "run_command", ErrorCode: "TOOL_SCHEMA_INVALID",
		Error: "missing properties: 'args'",
	})
	if !strings.Contains(run.TextContent(), "command=python3") || !strings.Contains(run.TextContent(), "args") {
		t.Fatalf("run_command recovery reminder = %+v", run)
	}
	revision := buildToolFailureRecoveryContext(react.ToolFailure{
		ToolName: "revise_verification", ErrorCode: "TOOL_SCHEMA_INVALID",
		Error: "additionalProperties 'tool_hints' not allowed",
	})
	if !strings.Contains(revision.TextContent(), "never include tool_hints") {
		t.Fatalf("revision recovery reminder = %+v", revision)
	}
}

func TestRunCommandProcessExitRecoveryUsesDiagnosticNotSchemaAdvice(t *testing.T) {
	exitCode := 1
	failure := react.ToolFailure{
		ToolName: "run_command", ErrorCode: "TOOL_EXECUTION_FAILED", FailureKind: "process_exit",
		Error: "command exited with code 1", Diagnostic: "RuntimeError: final-root-cause", ExitCode: &exitCode, HasStderr: true,
	}
	message := buildToolFailureRecoveryContext(failure)
	text := message.TextContent()
	if !strings.Contains(text, "passed Schema and Sandbox policy") || !strings.Contains(text, "RuntimeError: final-root-cause") || !strings.Contains(text, "workspace mutation") {
		t.Fatalf("process-exit recovery=%s", text)
	}
	if strings.Contains(text, "requires command and args") || strings.Contains(text, "python3 -c is not allowed") {
		t.Fatalf("process-exit recovery fell back to parameter advice: %s", text)
	}
}

func TestFailureApplicableMemoryCandidatesRejectsContradictedPremise(t *testing.T) {
	memories := []agent.Memory{
		{ID: "missing", Title: "run_command failure with missing stderr", Description: "Use a wrapper when no stderr is returned."},
		{ID: "repeat", Title: "run_command deterministic retry", Description: "Repair the workspace before retrying."},
	}
	selected := failureApplicableMemoryCandidates(memories, react.ToolFailure{ToolName: "run_command", FailureKind: "process_exit", HasStderr: true})
	if len(selected) != 1 || selected[0].ID != "repeat" {
		t.Fatalf("applicable memories=%+v", selected)
	}
}

func TestStaticMemoryContextSeparatesConfiguredAndHistoricalLayers(t *testing.T) {
	messages := buildStaticMemoryContext([]agent.StaticMemoryDocument{
		{ID: "project", SourceLayer: agent.MemoryLayerProject, Path: "CLAUDE.md", ContentHash: "hash-project", Content: "Use Kong."},
		{ID: "auto", SourceLayer: agent.MemoryLayerAuto, Path: ".agent/memory/auto/gateway.md", ContentHash: "hash-auto", Content: "Historical gateway note."},
	}, 256)
	if len(messages) != 2 || messages[0].Role != model.RoleSystem || messages[1].Role != model.RoleUser {
		t.Fatalf("messages = %#v", messages)
	}
	if !strings.Contains(messages[0].Content, "configured-instruction") || !strings.Contains(messages[0].Content, "Kong") {
		t.Fatalf("configured static memory = %s", messages[0].Content)
	}
	if !strings.Contains(messages[1].Content, "historical-untrusted-data") || !strings.Contains(messages[1].Content, "Historical gateway note") {
		t.Fatalf("historical static memory = %s", messages[1].Content)
	}
}

func TestMemoryRetrievalAuditReasonParsing(t *testing.T) {
	ids, reasons := parseSuppressedMemoryReasons([]string{"memory-1:already_surfaced", "memory-2:recent_tool_evidence"})
	if !reflect.DeepEqual(ids, []string{"memory-1", "memory-2"}) || reasons["memory-2"] != "recent_tool_evidence" {
		t.Fatalf("ids=%v reasons=%v", ids, reasons)
	}
}

type manifestRouterProvider struct{}

func (manifestRouterProvider) Complete(_ context.Context, request model.Request) (model.Response, error) {
	if len(request.Messages) != 2 || request.Metadata["purpose"] != "memory_manifest_router" {
		return model.Response{}, fmt.Errorf("unexpected router request: %+v", request)
	}
	if !strings.Contains(request.Messages[1].TextContent(), "Kong") {
		return model.Response{}, fmt.Errorf("router request omitted memory excerpt: %s", request.Messages[1].TextContent())
	}
	return model.Response{ModelID: "router-test", Message: model.TextMessage(model.RoleAssistant, `{"memory_ids":["m-2","unknown","m-2"]}`)}, nil
}

func TestRouteMemoryManifestUsesAllowlistAndCapsResults(t *testing.T) {
	candidates := []agent.Memory{
		{ID: "m-1", Title: "first", Description: "first candidate", Body: "API gateway uses Kong, not nginx.", RecallScore: 0.8},
		{ID: "m-2", Title: "second", Description: "second candidate", RecallScore: 0.7},
		{ID: "m-3", Title: "third", Description: "third candidate", RecallScore: 0.6},
	}
	routed, modelID := routeMemoryManifest(context.Background(), manifestRouterProvider{}, "run-1", "gateway", candidates, 2)
	if modelID != "router-test" || len(routed) != 1 || routed[0].ID != "m-2" {
		t.Fatalf("routed=%+v model=%q", routed, modelID)
	}
}

func TestTruncateMemoryExcerptKeepsBoundedPrefix(t *testing.T) {
	value := "line-1\nline-2\nline-3\nline-4"
	got := truncateMemoryExcerpt(value, 2, 100)
	if got != "line-1\nline-2" {
		t.Fatalf("excerpt=%q", got)
	}
	got = truncateMemoryExcerpt("0123456789", 0, 5)
	if got != "01234…" {
		t.Fatalf("rune-bounded excerpt=%q", got)
	}
}

func TestMemoryStaleWarningUsesSemanticTypeDefaults(t *testing.T) {
	if memoryStaleWarningDays(agent.MemoryTypeProject) != 2 ||
		memoryStaleWarningDays(agent.MemoryTypeReference) != 7 ||
		memoryStaleWarningDays(agent.MemoryTypeFeedback) != 30 ||
		memoryStaleWarningDays(agent.MemoryTypeUser) != 90 {
		t.Fatalf("unexpected stale-warning defaults")
	}
}

func TestReActProcessorUsesPinnedSnapshot(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-1", Type: event.RunCreated})
	resolver := &fakeExecutionResolver{events: events}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	spec := validProcessorSpec()
	snapshot, _ := json.Marshal(bindingSnapshot{
		AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: spec,
	})
	output, err := processor.Process(context.Background(), agent.Run{
		ID: "run-1", AgentVersionID: "version-1", Input: json.RawMessage(`{"question":"hello"}`),
		BindingSnapshot: snapshot,
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(output) != `{"answer":"ok"}` {
		t.Fatalf("output = %s", output)
	}
	if resolver.promptRef != spec.PromptRef || !reflect.DeepEqual(resolver.modelBinding, spec.Model) || resolver.toolSetRef != spec.ToolSetRef {
		t.Fatalf("resolver did not receive pinned refs: %+v", resolver)
	}
	committed := events.Events("run-1")
	if len(committed) < 3 || committed[1].Type != event.TurnCreated || committed[2].Type != event.TurnStarted || committed[len(committed)-1].Type != event.TurnCompleted {
		t.Fatalf("events = %+v", committed)
	}
}

func TestReActProcessorAutoPolicyAnswersDirectlyAndRecordsMode(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-auto", Type: event.RunCreated})
	resolver := &noPlanResolver{fakeExecutionResolver: fakeExecutionResolver{events: events}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	spec := validProcessorSpec()
	spec.Planning.Policy = agent.PlanningPolicyAuto
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-auto", Version: 1, SpecHash: "sha256", Spec: spec})
	if _, err := processor.Process(context.Background(), agent.Run{ID: "run-auto", AgentVersionID: "version-auto", Input: json.RawMessage(`{"question":"hello"}`), BindingSnapshot: snapshot}); err != nil {
		t.Fatal(err)
	}
	var selected bool
	for _, committed := range events.Events("run-auto") {
		selected = selected || committed.Type == event.ExecutionModeSelected && strings.Contains(string(committed.Payload), `"mode":"conversational"`)
	}
	if !selected {
		t.Fatalf("auto direct answer did not record execution mode: %+v", events.Events("run-auto"))
	}
}

func TestReActProcessorRequiredPolicyRejectsFinalAnswerWithoutPlan(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-required", Type: event.RunCreated})
	provider := &capturingFinalProvider{}
	resolver := &noPlanResolver{fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: provider}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	spec := validProcessorSpec()
	spec.Planning.Policy = agent.PlanningPolicyRequired
	spec.Harness.MaxSteps = 2
	spec.Runtime.MaxModelCalls = 2
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-required", Version: 1, SpecHash: "sha256", Spec: spec})
	_, err = processor.Process(context.Background(), agent.Run{ID: "run-required", AgentVersionID: "version-required", Input: json.RawMessage(`{"question":"do work"}`), BindingSnapshot: snapshot})
	if err == nil || !strings.Contains(err.Error(), "verification recovery loop detected") {
		t.Fatalf("required policy should not complete without a Plan, got %v", err)
	}
	if len(provider.requests) != 2 {
		t.Fatalf("model calls = %d, want 2", len(provider.requests))
	}
	var blocked, loopDetected int
	for _, committed := range events.Events("run-required") {
		if committed.Type == event.PlanCompletionBlocked {
			blocked++
		}
		if committed.Type == event.VerificationLoopDetected {
			loopDetected++
		}
	}
	if blocked != 2 || loopDetected != 1 {
		t.Fatalf("completion blocks = %d, loop detections = %d, events = %+v", blocked, loopDetected, events.Events("run-required"))
	}
}

func TestExecutionProtocolIsConditional(t *testing.T) {
	t.Parallel()
	if auto := executionProtocol(agent.PlanningPolicyAuto, "", nil); !strings.Contains(auto, "answer directly") || strings.Contains(auto, "requires durable planning") {
		t.Fatalf("auto protocol is not adaptive: %s", auto)
	}
	if required := executionProtocol(agent.PlanningPolicyRequired, "", nil); !strings.Contains(required, "requires durable planning") {
		t.Fatalf("required protocol is not strict: %s", required)
	}
	if disabled := executionProtocol(agent.PlanningPolicyDisabled, "", nil); !strings.Contains(disabled, "conversational-only") || !strings.Contains(disabled, "Do not create a Plan") {
		t.Fatalf("disabled protocol exposes execution: %s", disabled)
	}
}

func TestExecutionProtocolIncludesModelSafeEnvironmentFacts(t *testing.T) {
	got := executionProtocol(agent.PlanningPolicyAuto, "python-game:v1", []string{"pygame==2.6.1"})
	if !strings.Contains(got, "python-game:v1") || !strings.Contains(got, "pygame==2.6.1") || !strings.Contains(got, "run_command") {
		t.Fatalf("runtime environment was not exposed to the model: %s", got)
	}
}

func TestExecutionProtocolMatchesFileToolChunkContract(t *testing.T) {
	got := executionProtocol(agent.PlanningPolicyAuto, "", nil)
	if !strings.Contains(got, "hard content limit is 8192 characters") || !strings.Contains(got, "at most 6000 characters") || !strings.Contains(got, "append_file continues") {
		t.Fatalf("file chunk contract missing from execution protocol: %s", got)
	}
	if strings.Contains(got, "For source files use write_file once") {
		t.Fatal("execution protocol still tells the model to make one oversized write_file call")
	}
}

func TestRequiresDurableExecution(t *testing.T) {
	if !requiresDurableExecution(json.RawMessage(`{"question":"做一个简易版本贪吃蛇的游戏，用python写"}`)) {
		t.Fatal("artifact creation request should require durable execution")
	}
	if requiresDurableExecution(json.RawMessage(`{"question":"你好"}`)) || requiresDurableExecution(json.RawMessage(`{"question":"你这次执行遇到了什么问题"}`)) {
		t.Fatal("conversational requests should not require a Plan")
	}
}

func TestNormalizeUpdatePlanArgumentsRecoversLeadingStepsArray(t *testing.T) {
	raw := json.RawMessage(`{"goal":"build","steps":"[{\"id\":\"build\",\"description\":\"write game\",\"status\":\"in_progress\"}],\"explanation\":\"start\"}"}`)
	normalized, err := normalizeUpdatePlanArguments(raw)
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Steps []taskplan.Step `json:"steps"`
	}
	if err := json.Unmarshal(normalized, &payload); err != nil || len(payload.Steps) != 1 || payload.Steps[0].ID != "build" {
		t.Fatalf("unexpected normalized Plan: %s err=%v", normalized, err)
	}
}

func TestReActProcessorReturnsCompletedCheckpointWithoutProviders(t *testing.T) {
	t.Parallel()
	resolver := &completedCheckpointResolver{}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := json.Marshal(bindingSnapshot{
		AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: validProcessorSpec(),
	})
	output, err := processor.Process(context.Background(), agent.Run{
		ID: "run-checkpoint", AgentVersionID: "version-1", BindingSnapshot: snapshot,
	})
	if err != nil || string(output) != `{"answer":"checkpoint"}` {
		t.Fatalf("output = %s, error = %v", output, err)
	}
}

func TestReActProcessorStartsNewTurnForCompletedCheckpointFromPriorRun(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-new", Type: event.RunCreated})
	provider := &capturingFinalProvider{}
	resolver := &completedPriorRunResolver{fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: provider}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: validProcessorSpec()})
	output, err := processor.Process(context.Background(), agent.Run{
		ID: "run-new", WorkflowID: "workflow-1", AgentVersionID: "version-1",
		Input: json.RawMessage(`{"question":"追加要求"}`), BindingSnapshot: snapshot,
	})
	if err != nil || string(output) != `{"answer":"resumed"}` {
		t.Fatalf("output = %s, error = %v", output, err)
	}
	if len(provider.requests) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(provider.requests))
	}
	if provider.requests[0].Messages[len(provider.requests[0].Messages)-1].TextContent() != `{"question":"追加要求"}` {
		t.Fatalf("new Turn input was not appended: %+v", provider.requests[0].Messages)
	}
}

func TestReActProcessorCarriesInputWhenResumingPriorRunCheckpoint(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-new", Type: event.RunCreated})
	provider := &capturingFinalProvider{}
	resolver := &partialPriorRunResolver{fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: provider}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: validProcessorSpec()})
	if _, err := processor.Process(context.Background(), agent.Run{
		ID: "run-new", WorkflowID: "workflow-1", AgentVersionID: "version-1",
		Input: json.RawMessage(`{"question":"继续修复"}`), BindingSnapshot: snapshot,
	}); err != nil {
		t.Fatal(err)
	}
	if len(provider.requests) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(provider.requests))
	}
	last := provider.requests[0].Messages[len(provider.requests[0].Messages)-1]
	if last.Role != model.RoleUser || last.TextContent() != `{"question":"继续修复"}` {
		t.Fatalf("continuation input was not visible to model: %+v", provider.requests[0].Messages)
	}
}

func TestReActProcessorAutomaticRetryDoesNotDuplicateUserInput(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-retry", Type: event.RunCreated})
	provider := &capturingFinalProvider{}
	resolver := &partialPriorRunResolver{fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: provider}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: validProcessorSpec()})
	if _, err := processor.Process(context.Background(), agent.Run{
		ID: "run-retry", WorkflowID: "workflow-1", AgentVersionID: "version-1", TriggerType: "automatic_retry",
		Input: json.RawMessage(`{"question":"original task"}`), BindingSnapshot: snapshot,
	}); err != nil {
		t.Fatal(err)
	}
	if len(provider.requests) != 1 {
		t.Fatalf("provider calls = %d, want 1", len(provider.requests))
	}
	for _, message := range provider.requests[0].Messages {
		if message.TextContent() == `{"question":"original task"}` {
			t.Fatalf("automatic retry duplicated original input as a new continuation: %+v", provider.requests[0].Messages)
		}
	}
}

func TestReActProcessorDoesNotCarryAttemptLocalStateAcrossRuns(t *testing.T) {
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-new", Type: event.RunCreated})
	provider := &capturingFinalProvider{}
	resolver := &statefulPriorRunResolver{
		fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: provider},
		checkpoint: harness.Checkpoint{
			RunID: "run-new", SourceRunID: "run-old", Turn: 2, NextStep: 7,
			Messages:         []model.Message{model.TextMessage(model.RoleSystem, "system"), model.TextMessage(model.RoleUser, "old task")},
			Usage:            model.Usage{InputTokens: 100, OutputTokens: 20, TotalTokens: 120},
			PendingToolCalls: []model.ToolCall{{ID: "old-call", Name: "write_file", Arguments: json.RawMessage(`{"path":"stale.py","content":"stale"}`)}},
			ActiveToolCallID: "old-call",
		},
	}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: validProcessorSpec()})
	if _, err := processor.Process(context.Background(), agent.Run{
		ID: "run-new", WorkflowID: "workflow-1", AgentVersionID: "version-1",
		Input: json.RawMessage(`{"question":"continue safely"}`), BindingSnapshot: snapshot,
	}); err != nil {
		t.Fatal(err)
	}
	if len(resolver.saved) == 0 {
		t.Fatal("new Run did not persist a checkpoint")
	}
	terminal := resolver.saved[len(resolver.saved)-1]
	if terminal.Usage.TotalTokens != 0 || len(terminal.PendingToolCalls) != 0 || terminal.ActiveToolCallID != "" {
		t.Fatalf("prior Run attempt-local state leaked into continuation: %+v", terminal)
	}
}

func TestTurnCompleteMemoryQueueFailureDoesNotFailCompletedRun(t *testing.T) {
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-memory-degraded", Type: event.RunCreated})
	resolver := &failingMemoryWriterResolver{fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: &capturingFinalProvider{}}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	spec := validProcessorSpec()
	spec.Memory = agent.MemoryPolicy{Enabled: true, AutoExtract: true, ReadScopes: []string{agent.MemoryScopeSession}, MaxRecall: 5}
	spec.Context.MemoryTokens = 128
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: spec})
	output, err := processor.Process(context.Background(), agent.Run{
		ID: "run-memory-degraded", AgentVersionID: "version-1", Input: json.RawMessage(`{"question":"answer"}`), BindingSnapshot: snapshot,
	})
	if err != nil || string(output) != `{"answer":"resumed"}` {
		t.Fatalf("output=%s err=%v", output, err)
	}
	foundFailure := false
	for _, recorded := range events.Events("run-memory-degraded") {
		foundFailure = foundFailure || recorded.Type == event.MemoryExtractionFailed
	}
	if !foundFailure {
		t.Fatal("degraded memory extraction was not recorded")
	}
}

func TestMemoryExtractionDefaultsToOneTurnCompleteJob(t *testing.T) {
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-memory-once", Type: event.RunCreated})
	resolver := &recordingMemoryWriterResolver{fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: &capturingFinalProvider{}}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	spec := validProcessorSpec()
	spec.Memory = agent.MemoryPolicy{Enabled: true, AutoExtract: true, ReadScopes: []string{agent.MemoryScopeSession}, MaxRecall: 5}
	spec.Context.MemoryTokens = 128
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: spec})
	if _, err := processor.Process(context.Background(), agent.Run{
		ID: "run-memory-once", AgentVersionID: "version-1", Input: json.RawMessage(`{"question":"answer"}`), BindingSnapshot: snapshot,
	}); err != nil {
		t.Fatal(err)
	}
	if len(resolver.triggers) != 1 || resolver.triggers[0] != "turn_complete" {
		t.Fatalf("memory extraction triggers=%v, want one turn_complete job", resolver.triggers)
	}
}

func TestReActProcessorInjectsBudgetedScopedMemoryAndRecordsManifest(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-memory", Type: event.RunCreated})
	provider := &capturingFinalProvider{}
	resolver := &fakeExecutionResolver{events: events, modelProvider: provider, memories: []agent.Memory{{
		ID: "memory-1", Scope: agent.MemoryScopeSession, Kind: "preference", Content: "用户偏好稳定的 AIBrix 路由", RecallScore: 0.75,
	}}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	spec := validProcessorSpec()
	spec.Memory = agent.MemoryPolicy{Enabled: true, ReadScopes: []string{agent.MemoryScopeSession}, MaxRecall: 5, MinimumScore: 0.2}
	spec.Context.MemoryTokens = 256
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-memory", Version: 1, SpecHash: "sha256", Spec: spec})
	_, err = processor.Process(context.Background(), agent.Run{ID: "run-memory", AgentVersionID: "version-memory", Input: json.RawMessage(`{"question":"用户偏好什么路由"}`), BindingSnapshot: snapshot})
	if err != nil {
		t.Fatal(err)
	}
	memoryVisible := false
	if len(provider.requests) == 1 {
		for _, message := range provider.requests[0].Messages {
			memoryVisible = memoryVisible || strings.Contains(message.Content, "AIBrix")
		}
	}
	if !memoryVisible {
		t.Fatalf("model request did not contain recalled memory: %+v", provider.requests)
	}
	var retrieved, manifested bool
	for _, committed := range events.Events("run-memory") {
		if committed.Type == event.MemoryRetrieved {
			retrieved = strings.Contains(string(committed.Payload), "memory-1")
		}
		if committed.Type == event.ContextBuilt {
			manifested = strings.Contains(string(committed.Payload), "memory-1")
		}
	}
	if !retrieved || !manifested {
		t.Fatalf("memory events missing: retrieved=%v manifested=%v", retrieved, manifested)
	}
}

func TestReActProcessorCompilesIdentityBeforePromptAndRecordsDigest(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-identity", Type: event.RunCreated})
	provider := &capturingFinalProvider{}
	resolver := &fakeExecutionResolver{events: events, modelProvider: provider}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	spec := validProcessorSpec()
	spec.Identity = agent.Identity{DisplayName: "Reviewer", Role: "Release reviewer", Goal: "Decide from evidence", Boundaries: []string{"Do not change traffic"}}
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-identity", Version: 1, SpecHash: "sha256", Spec: spec})
	_, err = processor.Process(context.Background(), agent.Run{ID: "run-identity", AgentVersionID: "version-identity", Input: json.RawMessage(`{"question":"review"}`), BindingSnapshot: snapshot})
	if err != nil {
		t.Fatal(err)
	}
	if len(provider.requests) != 1 || len(provider.requests[0].Messages) < 2 || !strings.Contains(provider.requests[0].Messages[0].Content, "Role: Release reviewer") || !strings.Contains(provider.requests[0].Messages[0].Content, "You are a test agent") {
		t.Fatalf("identity/prompt ordering was not preserved: %+v", provider.requests)
	}
	var recorded bool
	for _, committed := range events.Events("run-identity") {
		if committed.Type == event.IdentityCompiled && strings.Contains(string(committed.Payload), "sha256:") && strings.Contains(string(committed.Payload), "Release reviewer") {
			recorded = true
		}
	}
	if !recorded {
		t.Fatal("compiled identity event with digest was not recorded")
	}
}

func TestReActProcessorResumesAtCheckpointNextStep(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	seed := []event.Input{
		{RunID: "run-resume", Type: event.RunCreated},
		{RunID: "run-resume", Type: event.TurnStarted, Turn: 1},
		{RunID: "run-resume", Type: event.StepStarted, Turn: 1, Step: 1},
		{RunID: "run-resume", Type: event.StepCompleted, Turn: 1, Step: 1},
		{RunID: "run-resume", Type: event.CheckpointCreated, Turn: 1},
	}
	for _, input := range seed {
		if _, err := events.Append(context.Background(), input); err != nil {
			t.Fatal(err)
		}
	}
	provider := &capturingFinalProvider{}
	resolver := &partialCheckpointResolver{
		fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: provider},
	}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := json.Marshal(bindingSnapshot{
		AgentVersionID: "version-1", Version: 1, SpecHash: "sha256", Spec: validProcessorSpec(),
	})
	output, err := processor.Process(context.Background(), agent.Run{
		ID: "run-resume", AgentVersionID: "version-1", BindingSnapshot: snapshot,
	})
	if err != nil || string(output) != `{"answer":"resumed"}` {
		t.Fatalf("output = %s, error = %v", output, err)
	}
	if len(provider.requests) != 1 || provider.requests[0].Messages[len(provider.requests[0].Messages)-1].Role != model.RoleTool {
		t.Fatalf("resumed model requests = %+v", provider.requests)
	}
	committed := events.Events("run-resume")
	turnStarted := 0
	for _, item := range committed {
		if item.Type == event.TurnStarted {
			turnStarted++
		}
	}
	if turnStarted != 1 {
		t.Fatalf("resume emitted another TurnStarted: %+v", committed)
	}
}

func TestReActProcessorRollsStepLimitIntoNextTurn(t *testing.T) {
	t.Parallel()
	events := event.NewMemoryStore()
	_, _ = events.Append(context.Background(), event.Input{RunID: "run-rollover", Type: event.RunCreated})
	provider := &rolloverProvider{}
	resolver := &rolloverResolver{fakeExecutionResolver: fakeExecutionResolver{events: events, modelProvider: provider}}
	processor, err := NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	spec := validProcessorSpec()
	spec.Harness.MaxTurns = 2
	spec.Harness.MaxSteps = 2
	spec.Runtime.MaxModelCalls = 4
	snapshot, _ := json.Marshal(bindingSnapshot{AgentVersionID: "version-rollover", Version: 1, SpecHash: "sha256", Spec: spec})
	output, err := processor.Process(context.Background(), agent.Run{ID: "run-rollover", AgentVersionID: "version-rollover", Input: json.RawMessage(`{"question":"finish the work"}`), BindingSnapshot: snapshot})
	if err != nil {
		t.Fatal(err)
	}
	if string(output) != `{"answer":"rolled over"}` || provider.calls != 3 {
		t.Fatalf("output=%s calls=%d", output, provider.calls)
	}
	turns := 0
	for _, committed := range events.Events("run-rollover") {
		if committed.Type == event.TurnStarted {
			turns++
		}
	}
	if turns != 2 {
		t.Fatalf("turn starts=%d, want 2", turns)
	}
}

type fakeExecutionResolver struct {
	events        event.Sink
	promptRef     agent.VersionRef
	modelBinding  agent.ModelBinding
	toolSetRef    agent.VersionRef
	modelProvider model.Provider
	memories      []agent.Memory
}

func (r *fakeExecutionResolver) ResolvePrompt(_ context.Context, _ string, ref agent.VersionRef) (string, error) {
	r.promptRef = ref
	return "You are a test agent.", nil
}

func (r *fakeExecutionResolver) ResolveModel(_ context.Context, _ agent.Run, binding agent.ModelBinding) (model.Provider, agent.ModelResolution, error) {
	r.modelBinding = binding
	resolution := agent.ModelResolution{
		SelectionPolicy: binding.EffectiveSelectionPolicy(), Provider: binding.Provider,
		ServiceRef: binding.ServiceRef, ModelID: binding.ModelID,
	}
	if r.modelProvider != nil {
		return r.modelProvider, resolution, nil
	}
	return finalProvider{}, resolution, nil
}

func (r *fakeExecutionResolver) ResolveTools(_ context.Context, _ agent.Run, ref agent.VersionRef) (tool.Executor, error) {
	r.toolSetRef = ref
	return tool.NewRegistry(), nil
}

func (r *fakeExecutionResolver) EventSink(agent.Run) (event.Sink, error) { return r.events, nil }

func (r *fakeExecutionResolver) LoadCheckpoint(context.Context, agent.Run) (harness.Checkpoint, bool, error) {
	return harness.Checkpoint{}, false, nil
}

func (r *fakeExecutionResolver) CheckpointSink(agent.Run) (harness.CheckpointSink, error) {
	return nil, nil
}

func (r *fakeExecutionResolver) SessionHistory(context.Context, agent.Run, int) ([]model.Message, error) {
	return nil, nil
}

func (r *fakeExecutionResolver) RecallMemories(context.Context, agent.Run, agent.MemoryPolicy, string) ([]agent.Memory, error) {
	return r.memories, nil
}

type finalProvider struct{}

func (finalProvider) Complete(context.Context, model.Request) (model.Response, error) {
	return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: `{"answer":"ok"}`}}, nil
}

type completedCheckpointResolver struct{ fakeExecutionResolver }

type completedPriorRunResolver struct{ fakeExecutionResolver }

func (r *completedPriorRunResolver) LoadCheckpoint(context.Context, agent.Run) (harness.Checkpoint, bool, error) {
	return harness.Checkpoint{
		RunID: "run-new", SourceRunID: "run-old", Turn: 1, NextStep: 3, Completed: true,
		Messages: []model.Message{{Role: model.RoleUser, Content: `{"question":"old"}`}, {Role: model.RoleAssistant, Content: `{"answer":"old"}`}},
		Answer:   model.Message{Role: model.RoleAssistant, Content: `{"answer":"old"}`},
	}, true, nil
}

type partialPriorRunResolver struct{ fakeExecutionResolver }

func (r *partialPriorRunResolver) LoadCheckpoint(_ context.Context, run agent.Run) (harness.Checkpoint, bool, error) {
	return harness.Checkpoint{
		RunID: run.ID, SourceRunID: "run-old", Turn: 1, NextStep: 2,
		Messages: []model.Message{{Role: model.RoleSystem, Content: "system"}, {Role: model.RoleAssistant, Content: "working"}},
	}, true, nil
}

type statefulPriorRunResolver struct {
	fakeExecutionResolver
	checkpoint harness.Checkpoint
	saved      []harness.Checkpoint
}

type failingMemoryWriterResolver struct{ fakeExecutionResolver }

type recordingMemoryWriterResolver struct {
	fakeExecutionResolver
	triggers []string
}

func (*failingMemoryWriterResolver) EnqueueMemoryWriteJob(context.Context, agent.Run, string, string, string, int64, int64) error {
	return errors.New("memory queue unavailable")
}

func (r *recordingMemoryWriterResolver) EnqueueMemoryWriteJob(_ context.Context, _ agent.Run, _, trigger, _ string, _, _ int64) error {
	r.triggers = append(r.triggers, trigger)
	return nil
}

func (r *statefulPriorRunResolver) LoadCheckpoint(context.Context, agent.Run) (harness.Checkpoint, bool, error) {
	return r.checkpoint, true, nil
}

func (r *statefulPriorRunResolver) CheckpointSink(agent.Run) (harness.CheckpointSink, error) {
	return r, nil
}

func (r *statefulPriorRunResolver) Save(_ context.Context, checkpoint harness.Checkpoint) error {
	r.saved = append(r.saved, checkpoint)
	r.checkpoint = checkpoint
	return nil
}

type noPlanResolver struct{ fakeExecutionResolver }

func (*noPlanResolver) GetTaskPlanForWorkflow(context.Context, string, string) (taskplan.Plan, error) {
	return taskplan.Plan{}, taskplan.ErrNotFound
}

func (r *completedCheckpointResolver) LoadCheckpoint(context.Context, agent.Run) (harness.Checkpoint, bool, error) {
	return harness.Checkpoint{
		RunID: "run-checkpoint", Turn: 1, NextStep: 3, Completed: true,
		Answer: model.Message{Role: model.RoleAssistant, Content: `{"answer":"checkpoint"}`},
	}, true, nil
}

type partialCheckpointResolver struct{ fakeExecutionResolver }

func (r *partialCheckpointResolver) LoadCheckpoint(context.Context, agent.Run) (harness.Checkpoint, bool, error) {
	return harness.Checkpoint{
		RunID: "run-resume", Turn: 1, NextStep: 2,
		Messages: []model.Message{
			{Role: model.RoleSystem, Content: "system"},
			{Role: model.RoleUser, Content: `{"question":"lookup"}`},
			{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call-1", Name: "lookup", Arguments: json.RawMessage(`{}`)}}},
			{Role: model.RoleTool, ToolCallID: "call-1", Name: "lookup", Content: `{"value":7}`},
		},
		Usage: model.Usage{TotalTokens: 10},
	}, true, nil
}

type capturingFinalProvider struct{ requests []model.Request }

func (p *capturingFinalProvider) Complete(_ context.Context, request model.Request) (model.Response, error) {
	p.requests = append(p.requests, request)
	return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: `{"answer":"resumed"}`}}, nil
}

type rolloverProvider struct{ calls int }

func (p *rolloverProvider) Complete(_ context.Context, _ model.Request) (model.Response, error) {
	p.calls++
	if p.calls <= 2 {
		return model.Response{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: fmt.Sprintf("call-%d", p.calls), Name: "noop", Arguments: json.RawMessage(`{}`)}}}}, nil
	}
	return model.Response{Message: model.Message{Role: model.RoleAssistant, Content: `{"answer":"rolled over"}`}}, nil
}

type rolloverResolver struct {
	fakeExecutionResolver
	checkpoint    harness.Checkpoint
	hasCheckpoint bool
}

func (r *rolloverResolver) ResolveTools(_ context.Context, _ agent.Run, ref agent.VersionRef) (tool.Executor, error) {
	r.toolSetRef = ref
	registry := tool.NewRegistry()
	err := registry.Register(tool.Definition{Name: "noop", Version: "1", Description: "noop", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	})
	return registry, err
}

func (r *rolloverResolver) LoadCheckpoint(context.Context, agent.Run) (harness.Checkpoint, bool, error) {
	return r.checkpoint, r.hasCheckpoint, nil
}

func (r *rolloverResolver) CheckpointSink(agent.Run) (harness.CheckpointSink, error) { return r, nil }

func (r *rolloverResolver) Save(_ context.Context, checkpoint harness.Checkpoint) error {
	r.checkpoint = checkpoint
	r.hasCheckpoint = true
	return nil
}

func validProcessorSpec() agent.Spec {
	return agent.Spec{
		Name: "generic-agent", Harness: agent.HarnessSpec{Name: "react-v1", MaxTurns: 1, MaxSteps: 4},
		Model:       agent.ModelBinding{Provider: "openai-compatible", ServiceRef: "model-service", ModelID: "test-model"},
		PromptRef:   agent.VersionRef{ID: "prompt", Version: "1"},
		ToolSetRef:  agent.VersionRef{ID: "tools", Version: "1"},
		InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: json.RawMessage(`{"type":"object"}`),
		Context: agent.ContextPolicy{MaxInputTokens: 2048, ReserveOutputTokens: 256},
		Runtime: agent.RuntimePolicy{MaxModelCalls: 4, MaxToolCalls: 4},
	}
}

func TestValidateFinalOutputRejectsInternalEnvelope(t *testing.T) {
	t.Parallel()
	schema := json.RawMessage(`{"type":"object","required":["content"],"properties":{"content":{"type":"string"}}}`)
	for _, content := range []string{
		"<context_summary>compacted state</context_summary>",
		`{"content":"<runtime_durable_plan>internal</runtime_durable_plan>"}`,
		"<think>private reasoning</think>visible",
	} {
		if err := validateFinalOutput(schema, model.TextMessage(model.RoleAssistant, content)); err == nil {
			t.Fatalf("expected internal output %q to be rejected", content)
		}
	}
}

func TestValidateFinalOutputAcceptsUserFacingContent(t *testing.T) {
	t.Parallel()
	schema := json.RawMessage(`{"type":"object","required":["content"],"properties":{"content":{"type":"string"}}}`)
	if err := validateFinalOutput(schema, model.TextMessage(model.RoleAssistant, "任务已完成，文件已验证。")); err != nil {
		t.Fatal(err)
	}
}
