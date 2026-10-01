package react

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/approval"
	contextpkg "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/context"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/delegation"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/harness"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	reviewcontract "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/review"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

type scriptedProvider struct {
	responses []model.Response
	requests  []model.Request
}

type protocolErrorProvider struct {
	calls    int
	requests []model.Request
}

type deadlineThenSuccessProvider struct {
	calls    int
	requests []model.Request
}

func (p *deadlineThenSuccessProvider) Complete(_ context.Context, request model.Request) (model.Response, error) {
	p.calls++
	p.requests = append(p.requests, request)
	if p.calls == 1 {
		return model.Response{}, context.DeadlineExceeded
	}
	return model.Response{Message: model.TextMessage(model.RoleAssistant, "recovered")}, nil
}

func (p *protocolErrorProvider) Complete(_ context.Context, request model.Request) (model.Response, error) {
	p.calls++
	p.requests = append(p.requests, request)
	return model.Response{}, errors.New("model_protocol_error: legacy tool markup received; expected structured tool_calls")
}

type checkpointRecorder struct{ last harness.Checkpoint }

func (r *checkpointRecorder) Save(_ context.Context, checkpoint harness.Checkpoint) error {
	r.last = checkpoint
	return nil
}

func (p *scriptedProvider) Complete(_ context.Context, request model.Request) (model.Response, error) {
	p.requests = append(p.requests, request)
	if len(p.responses) == 0 {
		return model.Response{}, errors.New("script exhausted")
	}
	response := p.responses[0]
	p.responses = p.responses[1:]
	return response, nil
}

func TestRunnerSchedulesReviewerWithoutModelGeneratedArguments(t *testing.T) {
	provider := &scriptedProvider{}
	registry := tool.NewRegistry()
	target := delegation.Target{
		AgentID: "11111111-1111-1111-1111-111111111111", AgentKey: "reviewer-agent", AgentName: "Reviewer Agent",
		AgentVersionID: "22222222-2222-2222-2222-222222222222", Modes: []string{"sync"}, Available: true,
	}
	parameters := jsonPayload(map[string]any{
		"type": "object", "required": []string{"target_agent_version_id", "mode", "input"},
		"properties": map[string]any{
			"target_agent_version_id": map[string]any{"type": "string"},
			"mode":                    map[string]any{"type": "string", "enum": []string{"sync"}},
			"input":                   map[string]any{"type": "object"},
		},
		"additionalProperties": false, "x-allowed-targets": []delegation.Target{target},
	})
	executions := 0
	if err := registry.Register(tool.Definition{Name: "delegate_agent", Version: "2", Risk: tool.RiskInternal, ExecutionMode: tool.ExecutionSerial, InputSchema: parameters}, func(_ context.Context, call tool.Call) (tool.Result, error) {
		executions++
		var request delegation.Request
		if err := json.Unmarshal(call.Arguments, &request); err != nil {
			t.Fatal(err)
		}
		if request.TargetAgentVersionID != target.AgentVersionID || request.Mode != "sync" {
			t.Fatalf("runtime reviewer request = %+v", request)
		}
		var input map[string]any
		if err := json.Unmarshal(request.Input, &input); err != nil || input["reason"] != "stalled" || input["base_plan_revision"] != float64(7) {
			t.Fatalf("runtime reviewer input = %s decoded=%+v error=%v", request.Input, input, err)
		}
		if executions == 1 {
			return tool.Result{}, delegation.ErrPending
		}
		return tool.Result{Content: json.RawMessage(`{"delegation_id":"delegation-1","child_run_id":"child-1","status":"completed","output":{"verdict":"pass","summary":"review passed","findings":[],"recommended_plan_changes":[]}}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	events := event.NewMemoryStore()
	if _, err := events.Append(context.Background(), event.Input{RunID: "runtime-review-run", Type: event.RunCreated}); err != nil {
		t.Fatal(err)
	}
	checkpoints := &checkpointRecorder{}
	runner, err := New(provider, registry, events, Config{
		MaxSteps: 1, Checkpoints: checkpoints, ExternalRunLifecycle: true, WorkflowID: "runtime-review-run",
		PlanContext: func(context.Context) (string, error) {
			return `Runtime durable plan (source of truth): {"revision":7,"goal":"finish"}` + "\nWork on the active node.", nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	messages := []model.Message{model.TextMessage(model.RoleSystem, "system"), model.TextMessage(model.RoleUser, "continue")}
	review := &progressReview{Generation: 3, Phase: "review", Action: "review", Reason: "stalled", ToolCalls: 12, FailedToolCalls: 4}
	messages = replaceExecutionLedger(messages, executionLedger{ToolCalls: 12, Failed: 4, ProgressReview: review, ProgressReviewBaseline: review})
	_, runErr := runner.Run(context.Background(), harness.Request{RunID: "runtime-review-run", Messages: messages})
	if !errors.Is(runErr, delegation.ErrPending) || len(checkpoints.last.PendingToolCalls) != 1 || !containsAutomaticReviewerCall(checkpoints.last.Messages, checkpoints.last.PendingToolCalls) {
		t.Fatalf("Reviewer pause error=%v checkpoint=%+v", runErr, checkpoints.last)
	}
	paused := checkpoints.last
	_, runErr = runner.Run(context.Background(), harness.Request{
		RunID: paused.RunID, Messages: paused.Messages, Turn: paused.Turn, StartStep: paused.NextStep,
		Resume: true, Usage: paused.Usage, PendingToolCalls: paused.PendingToolCalls,
		ActiveToolCallID: paused.ActiveToolCallID, ExecutionLedger: paused.ExecutionLedger,
	})
	if !errors.Is(runErr, ErrStepLimit) {
		t.Fatalf("run error = %v, want step limit after scheduler action", runErr)
	}
	if executions != 2 {
		t.Fatalf("runtime Reviewer executions=%d, want pending plus resumed completion", executions)
	}
	if len(provider.requests) != 0 {
		t.Fatalf("Reviewer scheduler unexpectedly called model %d times", len(provider.requests))
	}
	if len(checkpoints.last.PendingToolCalls) != 0 || checkpoints.last.ActiveToolCallID != "" {
		t.Fatalf("Reviewer scheduler left pending checkpoint = %+v", checkpoints.last)
	}
	if len(checkpoints.last.Messages) < 2 || checkpoints.last.Messages[len(checkpoints.last.Messages)-1].Role != model.RoleTool {
		t.Fatalf("resumed Reviewer result is not paired with its synthetic assistant call: %+v", checkpoints.last.Messages)
	}
	ledger := loadExecutionLedger(checkpoints.last.Messages)
	if ledger.ReviewerDecision == nil || ledger.ReviewerDecision.Verdict != "pass" || ledger.ReviewerDecision.BasePlanRevision != 7 || ledger.ProgressReview != nil {
		t.Fatalf("structured Reviewer decision was not applied: %+v", ledger)
	}
	foundSchedulerAction := false
	foundDecision := false
	for _, committed := range events.Events("runtime-review-run") {
		var payload map[string]any
		if json.Unmarshal(committed.Payload, &payload) != nil {
			continue
		}
		switch committed.Type {
		case event.ToolCalled:
			if payload["decision_source"] == "runtime_scheduler" && payload["scheduler_action"] == "review" {
				foundSchedulerAction = true
			}
		case event.ReviewDecisionRecorded:
			if payload["decision_id"] == "review-decision:"+committed.CallID && payload["verdict"] == "pass" && payload["base_plan_revision"] == float64(7) {
				foundDecision = true
			}
		}
	}
	if !foundSchedulerAction {
		t.Fatal("runtime Reviewer action was not identified in the event ledger")
	}
	if !foundDecision {
		t.Fatal("structured Reviewer decision was not recorded as a semantic event")
	}
}

func TestAutomaticReviewerAuditSourceCannotBeClaimedByModelCallID(t *testing.T) {
	call := model.ToolCall{ID: automaticReviewerCallPrefix + "spoofed", Name: "delegate_agent", Arguments: json.RawMessage(`{"mode":"sync","input":{}}`)}
	modelAuthored := []model.Message{{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{call}}}
	if containsAutomaticReviewerCall(modelAuthored, []model.ToolCall{call}) {
		t.Fatal("model-authored call_id prefix was trusted as a runtime scheduler action")
	}
	runtimeAuthored := []model.Message{automaticReviewerMessage("run", 1, 1, call)}
	if !containsAutomaticReviewerCall(runtimeAuthored, []model.ToolCall{call}) {
		t.Fatal("runtime-authored Reviewer call was not recognized")
	}
}

func TestRunnerAppliesSafeReviewerPlanPatchWithoutParentModel(t *testing.T) {
	provider := &scriptedProvider{}
	registry := tool.NewRegistry()
	target := delegation.Target{
		AgentID: "11111111-1111-1111-1111-111111111111", AgentKey: "reviewer-agent", AgentName: "Reviewer Agent",
		AgentVersionID: "22222222-2222-2222-2222-222222222222", Modes: []string{"sync"}, Available: true,
	}
	delegateSchema := jsonPayload(map[string]any{
		"type": "object", "required": []string{"target_agent_version_id", "mode", "input"},
		"properties": map[string]any{
			"target_agent_version_id": map[string]any{"type": "string"},
			"mode":                    map[string]any{"type": "string", "enum": []string{"sync"}},
			"input":                   map[string]any{"type": "object"},
		},
		"additionalProperties": false, "x-allowed-targets": []delegation.Target{target},
	})
	if err := registry.Register(tool.Definition{Name: "delegate_agent", Version: "2", Risk: tool.RiskInternal, ExecutionMode: tool.ExecutionSerial, InputSchema: delegateSchema}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"child_run_id":"child-review","output":{"verdict":"changes_required","summary":"correct the build node","findings":[{"severity":"medium","summary":"interface mismatch","evidence":"build.go calls the old interface"}],"recommended_plan_changes":[{"operation":"modify_step","step_id":"build","description":"build against the current interface","reason":"interface changed","depends_on":[]}]}}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	planSchema := json.RawMessage(`{"type":"object","required":["goal","steps","change_mode","base_revision"],"properties":{"goal":{"type":"string"},"steps":{"type":"array"},"change_mode":{"const":"replan"},"base_revision":{"type":"integer"},"replan_reason":{"type":"string"},"explanation":{"type":"string"},"retired_steps":{"type":"array"}},"additionalProperties":false}`)
	var applied taskplan.Update
	if err := registry.Register(tool.Definition{Name: "update_plan", Version: "12", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: planSchema}, func(_ context.Context, call tool.Call) (tool.Result, error) {
		if err := json.Unmarshal(call.Arguments, &applied); err != nil {
			t.Fatal(err)
		}
		return tool.Result{Content: json.RawMessage(`{"revision":8,"status":"updated"}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	plan := taskplan.Plan{Revision: 7, Goal: "ship", Steps: []taskplan.Step{{
		ID: "build", Description: "build old interface", Status: taskplan.StatusInProgress,
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{ID: "exists", Description: "build.go exists", Status: taskplan.CriterionPending, Verification: taskplan.VerificationSpec{Kind: "file_exists", Target: "build.go"}}},
	}}}
	events := event.NewMemoryStore()
	if _, err := events.Append(context.Background(), event.Input{RunID: "runtime-plan-patch", Type: event.RunCreated}); err != nil {
		t.Fatal(err)
	}
	checkpoints := &checkpointRecorder{}
	runner, err := New(provider, registry, events, Config{
		MaxSteps: 1, Checkpoints: checkpoints, ExternalRunLifecycle: true, WorkflowID: "runtime-plan-patch",
		PlanContext: func(context.Context) (string, error) {
			return `Runtime durable plan (source of truth): {"revision":7,"goal":"ship"}`, nil
		},
		PlanSnapshot: func(context.Context) (taskplan.Plan, error) { return plan, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	messages := []model.Message{model.TextMessage(model.RoleSystem, "system"), model.TextMessage(model.RoleUser, "continue")}
	review := &progressReview{Generation: 3, Phase: "review", Action: "review", Reason: "stalled", ToolCalls: 12, FailedToolCalls: 4}
	messages = replaceExecutionLedger(messages, executionLedger{ToolCalls: 12, Failed: 4, ProgressReview: review, ProgressReviewBaseline: review})
	_, runErr := runner.Run(context.Background(), harness.Request{RunID: "runtime-plan-patch", Messages: messages})
	if !errors.Is(runErr, ErrStepLimit) {
		t.Fatalf("run error = %v", runErr)
	}
	if len(provider.requests) != 0 {
		t.Fatalf("parent model was called %d times", len(provider.requests))
	}
	if applied.BaseRevision == nil || *applied.BaseRevision != 7 || applied.ChangeMode != "replan" || len(applied.Steps) != 1 || applied.Steps[0].Description != "build against the current interface" {
		t.Fatalf("applied Plan patch = %+v", applied)
	}
	ledger := loadExecutionLedger(checkpoints.last.Messages)
	if ledger.ReviewerDecision != nil || ledger.ProgressReview != nil {
		t.Fatalf("successful scheduler patch did not retire review state: %+v", ledger)
	}
	actions := map[string]bool{}
	for _, committed := range events.Events("runtime-plan-patch") {
		if committed.Type != event.ToolCalled {
			continue
		}
		var payload map[string]any
		if json.Unmarshal(committed.Payload, &payload) == nil && payload["decision_source"] == "runtime_scheduler" {
			actions[fmt.Sprint(payload["scheduler_action"])] = true
		}
	}
	if !actions["review"] || !actions["review_plan_patch"] {
		t.Fatalf("scheduler actions = %+v", actions)
	}
}

func TestRunnerAppliesReviewerVerificationPatchThroughNarrowTool(t *testing.T) {
	registry := tool.NewRegistry()
	schema := json.RawMessage(`{"type":"object","required":["step_id","criterion_id","action","reason","base_revision","verification"],"properties":{"step_id":{"type":"string"},"criterion_id":{"type":"string"},"action":{"const":"replace"},"reason":{"type":"string"},"base_revision":{"type":"integer"},"verification":{"type":"object"}},"additionalProperties":false}`)
	var applied taskplan.CriterionRevision
	if err := registry.Register(tool.Definition{Name: "revise_verification", Version: "2", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: schema}, func(_ context.Context, call tool.Call) (tool.Result, error) {
		if err := json.Unmarshal(call.Arguments, &applied); err != nil {
			t.Fatal(err)
		}
		return tool.Result{Content: json.RawMessage(`{"revision":6,"status":"updated"}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	events := event.NewMemoryStore()
	for _, input := range []event.Input{
		{RunID: "runtime-verification-patch", Type: event.RunCreated},
		{RunID: "runtime-verification-patch", Type: event.TurnStarted, Turn: 1},
		{RunID: "runtime-verification-patch", Type: event.StepStarted, Turn: 1, Step: 1},
	} {
		if _, err := events.Append(context.Background(), input); err != nil {
			t.Fatal(err)
		}
	}
	plan := taskplan.Plan{Revision: 5, Goal: "ship", Steps: []taskplan.Step{{
		ID: "build", Description: "build", Status: taskplan.StatusInProgress,
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{ID: "exists", Description: "old target exists", Status: taskplan.CriterionPending, Verification: taskplan.VerificationSpec{Kind: "file_exists", Target: "old.go"}}},
	}}}
	runner, err := New(&scriptedProvider{}, registry, events, Config{MaxSteps: 1, WorkflowID: "runtime-verification-patch", PlanSnapshot: func(context.Context) (taskplan.Plan, error) { return plan, nil }})
	if err != nil {
		t.Fatal(err)
	}
	decision := reviewerDecision{
		CallID: automaticReviewerCallPrefix + "verify", BasePlanRevision: 5, Verdict: "changes_required", Summary: "repair target",
		Findings: []reviewcontract.Finding{{Severity: "medium", Summary: "wrong target", Evidence: "file moved"}},
		RecommendedPlanChanges: []reviewcontract.PlanChange{{
			Operation: "revise_verification", StepID: "build", CriterionID: "exists", Reason: "file moved",
			Verification: taskplan.VerificationSpec{Kind: "file_exists", Target: "new.go"},
		}},
	}
	review := &progressReview{Generation: 3, Action: "replan"}
	messages := replaceExecutionLedger([]model.Message{model.TextMessage(model.RoleSystem, "system")}, executionLedger{ProgressReview: review, ProgressReviewBaseline: review, ReviewerDecision: &decision})
	attempted, failed, err := runner.tryApplyReviewerPlanPatch(context.Background(), "runtime-verification-patch", "workspace", 1, 1, &messages, model.Usage{}, decision, []model.ToolSchema{{Name: "revise_verification", Parameters: schema}})
	if err != nil || failed || !attempted {
		t.Fatalf("attempted=%v failed=%v error=%v", attempted, failed, err)
	}
	if applied.BaseRevision == nil || *applied.BaseRevision != 5 || applied.StepID != "build" || applied.CriterionID != "exists" || applied.Verification.Target != "new.go" {
		t.Fatalf("applied verification patch = %+v", applied)
	}
	ledger := loadExecutionLedger(messages)
	if ledger.ReviewerDecision != nil || ledger.ProgressReview != nil {
		t.Fatalf("verification patch did not retire Reviewer state: %+v", ledger)
	}
	found := false
	for _, committed := range events.Events("runtime-verification-patch") {
		if committed.Type == event.ToolCalled && strings.Contains(string(committed.Payload), `"scheduler_action":"review_verification_patch"`) {
			found = true
		}
	}
	if !found {
		t.Fatal("verification patch scheduler action was not audited")
	}
}

func TestRunnerCheckpointsDecisionBoundaryBeforeModelFailure(t *testing.T) {
	provider := &protocolErrorProvider{}
	checkpoints := &checkpointRecorder{}
	events := event.NewMemoryStore()
	if _, err := events.Append(context.Background(), event.Input{RunID: "run-model-timeout", Type: event.RunCreated}); err != nil {
		t.Fatal(err)
	}
	runner, err := New(provider, tool.NewRegistry(), events, Config{MaxSteps: 1, Checkpoints: checkpoints, ExternalRunLifecycle: true})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runner.Run(context.Background(), harness.Request{
		RunID: "run-model-timeout", Messages: []model.Message{model.TextMessage(model.RoleUser, "continue the unfinished task")},
	})
	if err == nil {
		t.Fatal("model failure was expected")
	}
	if checkpoints.last.RunID != "run-model-timeout" || checkpoints.last.Turn != 1 || checkpoints.last.NextStep != 1 {
		t.Fatalf("decision-boundary checkpoint position = %+v", checkpoints.last)
	}
	if len(checkpoints.last.Messages) == 0 || checkpoints.last.Messages[len(checkpoints.last.Messages)-1].TextContent() != "continue the unfinished task" {
		t.Fatalf("run input was not durable before model request: %+v", checkpoints.last.Messages)
	}
}

func TestRunnerRetriesProviderDeadlineWithDegradedOutputBudget(t *testing.T) {
	provider := &deadlineThenSuccessProvider{}
	store := event.NewMemoryStore()
	runner, err := New(provider, tool.NewRegistry(), store, Config{MaxSteps: 1, MaxTokens: 2048, ModelTimeoutRetries: 1})
	if err != nil {
		t.Fatal(err)
	}
	result, err := runner.Run(context.Background(), harness.Request{RunID: "run-deadline-retry", Messages: []model.Message{model.TextMessage(model.RoleUser, "continue")}})
	if err != nil || result.Answer.TextContent() != "recovered" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if provider.calls != 2 || len(provider.requests) != 2 || provider.requests[1].MaxTokens != 1536 {
		t.Fatalf("deadline retry requests=%+v", provider.requests)
	}
	var failed, requested int
	for _, committed := range store.Events("run-deadline-retry") {
		switch committed.Type {
		case event.ModelFailed:
			failed++
		case event.ModelRequested:
			requested++
		}
	}
	if failed != 1 || requested != 2 {
		t.Fatalf("deadline trajectory failed=%d requested=%d", failed, requested)
	}
}

func TestRunnerExecutesToolThenReturnsFinalAnswer(t *testing.T) {
	t.Parallel()

	provider := &scriptedProvider{responses: []model.Response{
		{
			Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{
				ID: "call-1", Name: "orders.get", Arguments: json.RawMessage(`{"order_id":"42"}`),
			}}},
			Provider: "mock", ModelID: "mock-model", Usage: model.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12},
		},
		{
			Message:  model.Message{Role: model.RoleAssistant, Content: "订单 42 已发货"},
			Provider: "mock", ModelID: "mock-model", Usage: model.Usage{InputTokens: 15, OutputTokens: 5, TotalTokens: 20},
		},
	}}
	tools := tool.NewRegistry()
	if err := tools.Register(tool.Definition{
		Name: "orders.get", Version: "1", Description: "read order",
		InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial,
	}, func(_ context.Context, call tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"status":"shipped"}`), Meta: map[string]string{"plan_node_id": "lookup"}}, nil
	}); err != nil {
		t.Fatal(err)
	}
	store := event.NewMemoryStore()
	planProgressCalls := 0
	runner, err := New(provider, tools, store, Config{MaxSteps: 4, MaxTokens: 256, PlanProgress: func(_ context.Context, call tool.Call) error {
		planProgressCalls++
		if call.ID != "call-1" || call.Name != "orders.get" || call.PlanNodeID != "lookup" || call.PlanStepID != "lookup" {
			t.Fatalf("Plan progress call = %+v", call)
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}

	result, err := runner.Run(context.Background(), harness.Request{
		RunID: "run-1", Messages: []model.Message{{Role: model.RoleUser, Content: "查询订单 42"}},
	})
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if result.Answer.Content != "订单 42 已发货" || result.Steps != 2 {
		t.Fatalf("unexpected result: %+v", result)
	}
	if result.Usage.TotalTokens != 32 {
		t.Fatalf("total tokens = %d, want 32", result.Usage.TotalTokens)
	}
	if len(provider.requests) != 2 {
		t.Fatalf("model calls = %d, want 2", len(provider.requests))
	}
	if planProgressCalls != 1 {
		t.Fatalf("Plan progress calls = %d, want 1", planProgressCalls)
	}
	secondMessages := provider.requests[1].Messages
	if got := secondMessages[len(secondMessages)-1]; got.Role != model.RoleTool || got.ToolCallID != "call-1" {
		t.Fatalf("tool result was not returned to the model: %+v", got)
	}
	events := store.Events("run-1")
	if events[len(events)-1].Type != event.RunCompleted {
		t.Fatalf("last event = %s, want %s", events[len(events)-1].Type, event.RunCompleted)
	}
}

func TestModelRequestEventsDeduplicateUnchangedToolSchemas(t *testing.T) {
	provider := &scriptedProvider{responses: []model.Response{
		{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call-1", Name: "read_file", Arguments: json.RawMessage(`{}`)}}}},
		{Message: model.TextMessage(model.RoleAssistant, "done")},
	}}
	tools := tool.NewRegistry()
	if err := tools.Register(tool.Definition{Name: "read_file", Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	store := event.NewMemoryStore()
	runner, err := New(provider, tools, store, Config{MaxSteps: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-schema-dedup", Messages: []model.Message{model.TextMessage(model.RoleUser, "inspect")}}); err != nil {
		t.Fatal(err)
	}
	requests := store.Events("run-schema-dedup")
	seen := 0
	for _, recorded := range requests {
		if recorded.Type != event.ModelRequested {
			continue
		}
		seen++
		var payload map[string]any
		if err := json.Unmarshal(recorded.Payload, &payload); err != nil {
			t.Fatal(err)
		}
		if payload["tool_schema_digest"] == nil {
			t.Fatalf("model request omitted tool schema digest: %s", recorded.Payload)
		}
		if seen == 1 && payload["tools"] == nil {
			t.Fatalf("first schema revision was not recorded: %s", recorded.Payload)
		}
		if seen == 2 && (payload["tools"] != nil || payload["tool_schema_ref"] != "previous_model_request") {
			t.Fatalf("unchanged schema was duplicated: %s", recorded.Payload)
		}
	}
	if seen != 2 {
		t.Fatalf("MODEL_REQUESTED events=%d, want 2", seen)
	}
}

func TestProjectRuntimeAwareMessagesRebuildsRuntimeBlocksAfterCollapse(t *testing.T) {
	system := "static contract\n<RUNTIME_DURABLE_PLAN>old plan</RUNTIME_DURABLE_PLAN>\n<RUNTIME_TOOL_PROJECTION>old tools</RUNTIME_TOOL_PROJECTION>\n<RUNTIME_EXECUTION_LEDGER>old ledger</RUNTIME_EXECUTION_LEDGER>"
	messages := []model.Message{
		{ID: "system", Role: model.RoleSystem, Content: system},
		{ID: "task", Role: model.RoleUser, Content: "keep this exact task"},
	}
	for index := 0; index < 8; index++ {
		messages = append(messages,
			model.Message{ID: fmt.Sprintf("assistant-%d", index), Role: model.RoleAssistant, Content: strings.Repeat("old execution ", 30)},
			model.Message{ID: fmt.Sprintf("tool-%d", index), Role: model.RoleTool, Content: strings.Repeat("old result ", 30)},
		)
	}
	ledger := json.RawMessage(`{"tool_calls":8,"succeeded":8}`)
	projected, _, report, err := projectRuntimeAwareMessages(
		context.Background(),
		messages,
		520,
		contextpkg.CollapseState{},
		contextpkg.ProjectionOptions{TriggerRatio: 0.8, TargetRatio: 0.4},
		ledger,
		"current plan",
		map[string]any{"tools": []string{"read_file"}},
		true,
	)
	if err != nil {
		t.Fatalf("runtime-aware projection failed: %v", err)
	}
	if !report.Compacted || report.AfterTokens > 520 {
		t.Fatalf("unexpected projection report: %+v", report)
	}
	joined := ""
	for _, message := range projected {
		joined += message.TextContent() + "\n"
	}
	for _, required := range []string{"static contract", "keep this exact task", "current plan", "RUNTIME_TOOL_PROJECTION", "RUNTIME_EXECUTION_LEDGER"} {
		if !strings.Contains(joined, required) {
			t.Fatalf("rebuilt projection lost %q: %s", required, joined)
		}
	}
	if strings.Contains(joined, "old plan") || strings.Contains(joined, "old tools") || strings.Contains(joined, "old ledger") {
		t.Fatalf("stale runtime projection survived: %s", joined)
	}
}

func TestRunnerRecordsProtocolRetryAndStepFailure(t *testing.T) {
	provider := &protocolErrorProvider{}
	store := event.NewMemoryStore()
	runner, err := New(provider, tool.NewRegistry(), store, Config{MaxSteps: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{
		RunID:    "run-protocol-failure",
		Messages: []model.Message{model.TextMessage(model.RoleUser, "execute")},
	}); err == nil || !strings.Contains(err.Error(), "model_protocol_error") {
		t.Fatalf("expected protocol failure, got %v", err)
	}
	if provider.calls != 2 {
		t.Fatalf("protocol recovery calls = %d, want 2", provider.calls)
	}
	if len(provider.requests) != 2 || provider.requests[0].ToolChoice != "none" || provider.requests[1].ToolChoice != "none" {
		t.Fatalf("no-tool finalizer must explicitly select tool_choice=none: %+v", provider.requests)
	}
	if len(provider.requests) != 2 || len(provider.requests[1].Messages) == 0 || !strings.Contains(provider.requests[1].Messages[len(provider.requests[1].Messages)-1].TextContent(), "offers no tools") {
		t.Fatalf("protocol recovery did not explain the empty tool projection: %+v", provider.requests)
	}
	var failedModels, requested int
	var stepFailed bool
	for _, committed := range store.Events("run-protocol-failure") {
		switch committed.Type {
		case event.ModelFailed:
			failedModels++
		case event.ModelRequested:
			requested++
		case event.StepFailed:
			stepFailed = true
		case event.StepCompleted:
			if strings.Contains(string(committed.Payload), `"status":"failed"`) {
				t.Fatal("model failure was recorded as STEP_COMPLETED")
			}
		}
	}
	if failedModels != 2 || requested != 2 || !stepFailed {
		t.Fatalf("protocol trajectory incomplete: model_failed=%d model_requested=%d step_failed=%v", failedModels, requested, stepFailed)
	}
}

func TestRunnerMergesPromptAndSkillSystemMessages(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{
		Message: model.TextMessage(model.RoleAssistant, "ok"),
	}}}
	runner, err := New(provider, tool.NewRegistry(), event.NewMemoryStore(), Config{MaxSteps: 1})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runner.Run(context.Background(), harness.Request{RunID: "run-system", Messages: []model.Message{
		model.TextMessage(model.RoleSystem, "base prompt"),
		model.TextMessage(model.RoleSystem, "skill instructions"),
		model.TextMessage(model.RoleUser, "question"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	got := provider.requests[0].Messages
	if len(got) != 2 || got[0].Role != model.RoleSystem || got[0].TextContent() != "base prompt\n\nskill instructions" {
		t.Fatalf("normalized messages = %+v", got)
	}
}

func TestModelToolProjectionSeparatesVisibilityAuthorizationApprovalAndWaiting(t *testing.T) {
	payload := modelToolProjectionPayload(
		[]model.ToolSchema{
			{Name: "read_file", ExecutionAllowed: true, RuntimeState: "ready"},
			{Name: "run_command", Risk: "HIGH_RISK", ExecutionAllowed: true, RuntimeState: "waiting_approval"},
			{Name: "delegate_agent", ExecutionAllowed: false, RuntimeState: "waiting_external"},
		},
		[]model.ToolSchema{{Name: "read_file", ExecutionAllowed: true}},
		"auto", true, true, true,
	)
	layers, ok := payload["tools"].([]map[string]any)
	if !ok || len(layers) != 3 {
		t.Fatalf("unexpected compact projection: %#v", payload)
	}
	if layers[0]["schema_visible"] != true || layers[0]["execution_allowed"] != true {
		t.Fatalf("ready tool projection = %#v", layers[0])
	}
	if layers[1]["schema_visible"] != false || layers[1]["approval_required"] != true {
		t.Fatalf("approval tool projection = %#v", layers[1])
	}
	if layers[2]["capability_available"] != true || layers[2]["execution_allowed"] != false || layers[2]["runtime_state"] != "projected_out" {
		t.Fatalf("waiting tool projection = %#v", layers[2])
	}
}

func TestRunnerShrinksOutputBudgetWhenMandatoryContextNeedsSpace(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "ok")}}}
	runner, err := New(provider, tool.NewRegistry(), event.NewMemoryStore(), Config{
		MaxSteps: 1, MaxTokens: 400, ContextWindowTokens: 1000,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runner.Run(context.Background(), harness.Request{RunID: "run-adaptive-output", Messages: []model.Message{
		model.TextMessage(model.RoleSystem, strings.Repeat("contract ", 35)),
		model.TextMessage(model.RoleUser, strings.Repeat("任", 400)),
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got := provider.requests[0].MaxTokens; got >= 400 || got < 128 {
		t.Fatalf("adaptive max tokens = %d, want [128,400)", got)
	}
}

func TestRunnerProjectsLargeDurableLedgerBeforeContextCompaction(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}}}
	checkpoints := &checkpointRecorder{}
	events := event.NewMemoryStore()
	messages := []model.Message{model.TextMessage(model.RoleSystem, "mandatory execution contract")}
	task := model.TextMessage(model.RoleUser, "build the requested project")
	task.Metadata = map[string]string{contextpkg.TaskAnchorMetadataKey: "true"}
	messages = append(messages, task)
	for index := 0; index < 12; index++ {
		arguments := json.RawMessage(fmt.Sprintf(`{"path":"module-%02d.py","content":"%s"}`, index, strings.Repeat("source payload ", 24)))
		messages = updateExecutionLedger(messages, model.ToolCall{ID: fmt.Sprintf("evidence-%02d", index), Name: "write_file", Arguments: arguments}, tool.Result{Content: json.RawMessage(`{"ok":true}`)})
	}
	fullTokens := estimateMessages(messages)
	runner, err := New(provider, tool.NewRegistry(), events, Config{
		MaxSteps: 1, MaxTokens: 128, ContextWindowTokens: 1000, Checkpoints: checkpoints,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := runner.Run(context.Background(), harness.Request{RunID: "run-ledger-projection", Messages: messages})
	if err != nil || result.Answer.TextContent() != "done" {
		t.Fatalf("run failed with projected ledger: result=%+v err=%v", result, err)
	}
	if len(provider.requests) != 1 || estimateMessages(provider.requests[0].Messages) >= fullTokens {
		t.Fatalf("provider did not receive a compact projection: full=%d request=%d", fullTokens, estimateMessages(provider.requests[0].Messages))
	}
	var durable executionLedger
	if err := json.Unmarshal(checkpoints.last.ExecutionLedger, &durable); err != nil {
		t.Fatal(err)
	}
	if len(durable.SuccessfulEvidence) != 12 || durable.SuccessfulEvidence[0].CallID != "evidence-00" {
		t.Fatalf("checkpoint lost full evidence ledger: %+v", durable.SuccessfulEvidence)
	}
	for _, recorded := range events.Events("run-ledger-projection") {
		if recorded.Type == event.ContextCompacted {
			t.Fatalf("bounded ledger projection was incorrectly reported as context compaction: %+v", recorded)
		}
	}
}

func TestRuntimeProjectionDropsOptionalToolCapabilityBlockBeforeExactTask(t *testing.T) {
	task := model.TextMessage(model.RoleUser, "keep the exact requested outcome")
	task.Metadata = map[string]string{contextpkg.TaskAnchorMetadataKey: "true"}
	base := []model.Message{model.TextMessage(model.RoleSystem, "system contract"), task}
	payload := map[string]any{"tools": strings.Repeat("optional capability state ", 500)}
	withRuntime := injectRuntimeToolProjection(base, payload)
	projected, _, _, err := projectRuntimeAwareMessages(context.Background(), withRuntime, 220, contextpkg.CollapseState{}, contextpkg.ProjectionOptions{SummaryTokens: 32, TargetRatio: 0.7}, json.RawMessage(`{}`), "", payload, true)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(projected[0].TextContent(), runtimeToolProjectionStart) {
		t.Fatalf("optional tool projection was not dropped: %s", projected[0].TextContent())
	}
	if !strings.Contains(projected[1].TextContent(), "exact requested outcome") {
		t.Fatalf("task anchor was lost: %+v", projected)
	}
}

func TestRunnerKeepsPlanToolVisibleWhileWorkAdvances(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{
		{Message: model.TextMessage(model.RoleAssistant, "step complete")},
		{Message: model.TextMessage(model.RoleAssistant, "done")},
	}}
	tools := tool.NewRegistry()
	for _, name := range []string{"update_plan", "update_plan_step", "revise_verification", "ask_user", "read_file"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	guardCalls := 0
	runner, err := New(provider, tools, event.NewMemoryStore(), Config{
		MaxSteps: 2,
		CompletionGuard: func(context.Context) (CompletionBlock, error) {
			guardCalls++
			if guardCalls == 1 {
				return CompletionBlock{Reason: "open_plan_work", Instruction: "update the durable plan now"}, nil
			}
			return CompletionBlock{}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-plan-window", Messages: []model.Message{model.TextMessage(model.RoleUser, "task")}}); err != nil {
		t.Fatal(err)
	}
	if !hasToolSchema(provider.requests[0].Tools, "update_plan") {
		t.Fatal("update_plan should remain visible during normal plan execution")
	}
	if !hasToolSchema(provider.requests[1].Tools, "update_plan") || !hasToolSchema(provider.requests[1].Tools, "revise_verification") {
		t.Fatal("completion guard must preserve recovery tools")
	}
}

func TestRunnerFusesRepeatedCompletionBlock(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}, {Message: model.TextMessage(model.RoleAssistant, "still done")}}}
	store := event.NewMemoryStore()
	runner, err := New(provider, tool.NewRegistry(), store, Config{MaxSteps: 3, CompletionGuard: func(context.Context) (CompletionBlock, error) {
		return CompletionBlock{Reason: "invalid_evidence", StepID: "verify", CriterionID: "syntax", VerificationKind: "python_syntax", Target: "game.py", RequiredAction: "repair contract", Instruction: "repair the same contract"}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runner.Run(context.Background(), harness.Request{RunID: "run-loop-fuse", Messages: []model.Message{model.TextMessage(model.RoleUser, "task")}})
	if err == nil || !strings.Contains(err.Error(), "verification recovery loop detected") {
		t.Fatalf("loop error = %v", err)
	}
	found := false
	for _, committed := range store.Events("run-loop-fuse") {
		if committed.Type == event.VerificationLoopDetected {
			found = true
		}
	}
	if !found {
		t.Fatal("missing verification loop event")
	}
}

func TestRunnerUsesCompactStepUpdatesDuringActivePlan(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}}}
	tools := tool.NewRegistry()
	for _, name := range []string{"update_plan", "update_plan_step"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	runner, err := New(provider, tools, event.NewMemoryStore(), Config{MaxSteps: 1, PlanState: func(context.Context) (bool, bool, error) { return true, true, nil }})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-compact-plan-tool", Messages: []model.Message{model.TextMessage(model.RoleUser, "task")}}); err != nil {
		t.Fatal(err)
	}
	if hasToolSchema(provider.requests[0].Tools, "update_plan") || !hasToolSchema(provider.requests[0].Tools, "update_plan_step") {
		t.Fatalf("active Plan did not select the compact update API: %+v", provider.requests[0].Tools)
	}
}

func TestRunnerKeepsToolsAfterDurablePlanIsClosed(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}}}
	tools := tool.NewRegistry()
	for _, name := range []string{"update_plan", "update_plan_step", "ask_user", "read_file", "write_file", "run_command"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	runner, err := New(provider, tools, event.NewMemoryStore(), Config{
		MaxSteps:  1,
		PlanState: func(context.Context) (bool, bool, error) { return true, false, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-closed-plan", Messages: []model.Message{model.TextMessage(model.RoleUser, "task")}}); err != nil {
		t.Fatal(err)
	}
	if len(provider.requests[0].Tools) != 5 || hasToolSchema(provider.requests[0].Tools, "update_plan") {
		t.Fatalf("closed continuation Plan must preserve action tools but hide full update_plan: %+v", provider.requests[0].Tools)
	}
}

func TestRunnerExposesFullPlanMutationOnlyForExplicitReplan(t *testing.T) {
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}}}
	tools := tool.NewRegistry()
	for _, name := range []string{"update_plan", "update_plan_step", "revise_verification", "read_file"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	runner, err := New(provider, tools, event.NewMemoryStore(), Config{MaxSteps: 1, PlanMutationMode: "replan", PlanState: func(context.Context) (bool, bool, error) { return true, true, nil }})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-explicit-replan", Messages: []model.Message{model.TextMessage(model.RoleUser, "replace dependencies")}}); err != nil {
		t.Fatal(err)
	}
	if !hasToolSchema(provider.requests[0].Tools, "update_plan") || !hasToolSchema(provider.requests[0].Tools, "update_plan_step") || !hasToolSchema(provider.requests[0].Tools, "revise_verification") {
		t.Fatalf("explicit replan did not expose full Plan mutation surface: %+v", provider.requests[0].Tools)
	}
}

func TestRunnerReopensToolSurfaceForExplicitPlanContinuation(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "continued")}}}
	tools := tool.NewRegistry()
	for _, name := range []string{"update_plan_step", "read_file", "run_command"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	store := event.NewMemoryStore()
	runner, err := New(provider, tools, store, Config{
		MaxSteps:                      1,
		PlanState:                     func(context.Context) (bool, bool, error) { return true, false, nil },
		PlanContinuationRequiresTools: func(context.Context) (bool, error) { return true, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-plan-continuation", Messages: []model.Message{model.TextMessage(model.RoleUser, "repair the verification")}}); err != nil {
		t.Fatal(err)
	}
	if !hasToolSchema(provider.requests[0].Tools, "run_command") || !hasToolSchema(provider.requests[0].Tools, "read_file") {
		t.Fatalf("continuation lost substantive tools: %+v", provider.requests[0].Tools)
	}
	found := false
	for _, committed := range store.Events("run-plan-continuation") {
		if committed.Type == event.ToolSchemaProjected && strings.Contains(string(committed.Payload), `"continuation_requires_tools":true`) {
			found = true
		}
	}
	if !found {
		t.Fatal("missing structured tool projection event")
	}
}

func TestRunnerKeepsRecoveryToolsBeforePlanExists(t *testing.T) {
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}}}
	tools := tool.NewRegistry()
	for _, name := range []string{"update_plan", "update_plan_step", "ask_user", "read_file", "write_file"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	runner, err := New(provider, tools, event.NewMemoryStore(), Config{MaxSteps: 1, PlanState: func(context.Context) (bool, bool, error) { return false, false, nil }})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-planning-only", Messages: []model.Message{model.TextMessage(model.RoleUser, "task")}}); err != nil {
		t.Fatal(err)
	}
	if len(provider.requests[0].Tools) != 5 || !hasToolSchema(provider.requests[0].Tools, "write_file") || !hasToolSchema(provider.requests[0].Tools, "read_file") {
		t.Fatalf("pre-plan recovery tools were hidden: %+v", provider.requests[0].Tools)
	}
}

func TestRunnerDisabledPlanningOffersOnlyAskUser(t *testing.T) {
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}}}
	tools := tool.NewRegistry()
	for _, name := range []string{"update_plan", "ask_user", "read_file", "write_file"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	store := event.NewMemoryStore()
	runner, err := New(provider, tools, store, Config{MaxSteps: 1, PlanningPolicy: "disabled"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-conversation-only", Messages: []model.Message{model.TextMessage(model.RoleUser, "hello")}}); err != nil {
		t.Fatal(err)
	}
	if len(provider.requests[0].Tools) != 1 || provider.requests[0].Tools[0].Name != "ask_user" {
		t.Fatalf("disabled policy tools = %+v", provider.requests[0].Tools)
	}
	var selected bool
	for _, committed := range store.Events("run-conversation-only") {
		if committed.Type == event.ExecutionModeSelected && strings.Contains(string(committed.Payload), `"mode":"conversational"`) {
			selected = true
		}
	}
	if !selected {
		t.Fatal("conversational execution mode was not recorded")
	}
}

func TestRunnerRequiredPlanningNeedsDurableState(t *testing.T) {
	_, err := New(&scriptedProvider{}, tool.NewRegistry(), event.NewMemoryStore(), Config{MaxSteps: 1, PlanningPolicy: "required"})
	if err == nil || !strings.Contains(err.Error(), "PlanState") {
		t.Fatalf("expected required planning configuration error, got %v", err)
	}
}

func TestRunnerKeepsToolsOutsideActiveTodoHints(t *testing.T) {
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}}}
	tools := tool.NewRegistry()
	for _, name := range []string{"update_plan", "update_plan_step", "ask_user", "read_file", "write_file"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	runner, err := New(provider, tools, event.NewMemoryStore(), Config{
		MaxSteps:           1,
		PlanState:          func(context.Context) (bool, bool, error) { return true, true, nil },
		PlanNeedsUserInput: func(context.Context) (bool, error) { return false, nil },
		PlanTools:          func(context.Context) ([]string, error) { return []string{"write_file"}, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-active-tools", Messages: []model.Message{model.TextMessage(model.RoleUser, "task")}}); err != nil {
		t.Fatal(err)
	}
	if !hasToolSchema(provider.requests[0].Tools, "read_file") || hasToolSchema(provider.requests[0].Tools, "update_plan") || hasToolSchema(provider.requests[0].Tools, "ask_user") || !hasToolSchema(provider.requests[0].Tools, "write_file") || !hasToolSchema(provider.requests[0].Tools, "update_plan_step") {
		t.Fatalf("active Todo projection selected the wrong execution surface: %+v", provider.requests[0].Tools)
	}
}

func TestRunnerKeepsConsecutivelyFailingToolVisible(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "recover")}}}
	tools := tool.NewRegistry()
	for _, name := range []string{"write_file", "append_file", "list_files", "update_plan_step", "ask_user"} {
		name := name
		if err := tools.Register(tool.Definition{Name: name, Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	messages := []model.Message{model.TextMessage(model.RoleSystem, "contract"), model.TextMessage(model.RoleUser, "task")}
	failed := tool.Result{Content: json.RawMessage(`{"error":"exists"}`), IsError: true, Error: "exists"}
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "call-1", Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, failed)
	messages = updateExecutionLedger(messages, model.ToolCall{ID: "call-2", Name: "write_file", Arguments: json.RawMessage(`{"path":"game.py","content":"next"}`)}, failed)
	runner, err := New(provider, tools, event.NewMemoryStore(), Config{
		MaxSteps:           1,
		PlanState:          func(context.Context) (bool, bool, error) { return true, true, nil },
		PlanNeedsUserInput: func(context.Context) (bool, error) { return false, nil },
		PlanTools:          func(context.Context) ([]string, error) { return []string{"write_file", "append_file"}, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "run-tool-backoff", Messages: messages}); err != nil {
		t.Fatal(err)
	}
	if !hasToolSchema(provider.requests[0].Tools, "write_file") || !hasToolSchema(provider.requests[0].Tools, "append_file") {
		t.Fatalf("consecutive failure must not hide recovery tools: %+v", provider.requests[0].Tools)
	}
	if hasToolSchema(provider.requests[0].Tools, "list_files") || hasToolSchema(provider.requests[0].Tools, "ask_user") {
		t.Fatalf("repair phase retained non-progress controls: %+v", provider.requests[0].Tools)
	}
}

func TestRuntimeControlToolsNeverBackOff(t *testing.T) {
	for _, name := range []string{"update_plan", "update_plan_step", "ask_user"} {
		if !isRuntimeControlTool(name) {
			t.Fatalf("control tool %q was not protected", name)
		}
	}
	if isRuntimeControlTool("write_file") {
		t.Fatal("substantive write tool must remain eligible for failure backoff")
	}
}

func hasToolSchema(schemas []model.ToolSchema, name string) bool {
	for _, schema := range schemas {
		if schema.Name == name {
			return true
		}
	}
	return false
}

func TestRunnerRefreshesPlanContextAndBlocksEarlyFinalAnswer(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{
		{Message: model.TextMessage(model.RoleAssistant, "premature")},
		{Message: model.TextMessage(model.RoleAssistant, "done")},
	}}
	planRevision := 0
	guardChecks := 0
	store := event.NewMemoryStore()
	runner, err := New(provider, tool.NewRegistry(), store, Config{
		MaxSteps: 3,
		PlanContext: func(context.Context) (string, error) {
			planRevision++
			return fmt.Sprintf("plan revision %d", planRevision), nil
		},
		CompletionGuard: func(context.Context) (CompletionBlock, error) {
			guardChecks++
			if guardChecks == 1 {
				return CompletionBlock{
					Reason: "invalid_evidence", Instruction: "continue open plan",
					CriterionID: "syntax", VerificationKind: "python_syntax", Target: "snake.py",
					RequiredTools: []string{"run_command"}, DeclaredTools: []string{"read_file"},
				}, nil
			}
			return CompletionBlock{}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := runner.Run(context.Background(), harness.Request{RunID: "run-plan-guard", Messages: []model.Message{
		model.TextMessage(model.RoleSystem, "base"), model.TextMessage(model.RoleUser, "work"),
	}})
	if err != nil || result.Answer.TextContent() != "done" || result.Steps != 2 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if len(provider.requests) != 2 {
		t.Fatalf("requests = %d", len(provider.requests))
	}
	first := provider.requests[0].Messages[0].TextContent()
	second := provider.requests[1].Messages[0].TextContent()
	if !strings.Contains(first, "plan revision 1") || strings.Contains(second, "plan revision 1") || !strings.Contains(second, "plan revision 2") {
		t.Fatalf("plan contexts were not replaced: first=%q second=%q", first, second)
	}
	if got := provider.requests[1].Messages[len(provider.requests[1].Messages)-1].TextContent(); got != "continue open plan" {
		t.Fatalf("continuation = %q", got)
	}
	foundBlocked := false
	for _, committed := range store.Events("run-plan-guard") {
		if committed.Type == event.PlanCompletionBlocked {
			foundBlocked = true
			var payload CompletionBlock
			if err := json.Unmarshal(committed.Payload, &payload); err != nil {
				t.Fatal(err)
			}
			if payload.CriterionID != "syntax" || payload.VerificationKind != "python_syntax" || len(payload.RequiredTools) != 1 || payload.RequiredTools[0] != "run_command" {
				t.Fatalf("completion block payload = %+v", payload)
			}
		}
	}
	if !foundBlocked {
		t.Fatal("missing plan completion blocked event")
	}
}

func TestRunnerModelViewDoesNotExposeLegacyCheckpointProjectionAsToolCall(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "done")}}}
	runner, err := New(provider, tool.NewRegistry(), event.NewMemoryStore(), Config{MaxSteps: 1})
	if err != nil {
		t.Fatal(err)
	}
	assistant := model.Message{ID: "legacy-assistant", Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{
		ID: "legacy-write", Name: "write_file", Arguments: json.RawMessage(`{"path":"pkg/cli.py","event_backed":true,"arguments_sha256":"sha256:old"}`),
	}}}
	result := model.TextMessage(model.RoleTool, `{"path":"pkg/cli.py","bytes":512}`)
	result.Name = "write_file"
	result.ToolCallID = "legacy-write"

	_, err = runner.Run(context.Background(), harness.Request{RunID: "run-safe-tool-history", Messages: []model.Message{
		model.TextMessage(model.RoleSystem, "base"),
		model.TextMessage(model.RoleUser, "continue"),
		assistant,
		result,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(provider.requests) != 1 {
		t.Fatalf("requests=%d, want 1", len(provider.requests))
	}
	joined, _ := json.Marshal(provider.requests[0].Messages)
	if strings.Contains(string(joined), `"tool_calls":[{"id":"legacy-write"`) || strings.Contains(string(joined), `"event_backed":true`) {
		t.Fatalf("legacy internal projection reached the model as a Tool Call: %s", joined)
	}
	if !strings.Contains(string(joined), `tool-history`) || !strings.Contains(string(joined), `do_not_copy_as_tool_call`) {
		t.Fatalf("bounded Tool History observation missing: %s", joined)
	}
}

func TestRunnerRejectsFinalAnswerBeforeCompletion(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{{Message: model.TextMessage(model.RoleAssistant, "invalid")}}}
	store := event.NewMemoryStore()
	runner, err := New(provider, tool.NewRegistry(), store, Config{
		MaxSteps:       1,
		ValidateAnswer: func(model.Message) error { return errors.New("output schema mismatch") },
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runner.Run(context.Background(), harness.Request{RunID: "run-invalid-answer", Messages: []model.Message{model.TextMessage(model.RoleUser, "question")}})
	if err == nil {
		t.Fatal("invalid answer must fail the run")
	}
	for _, committed := range store.Events("run-invalid-answer") {
		if committed.Type == event.RunCompleted || committed.Type == event.CheckpointCreated {
			t.Fatalf("invalid answer produced terminal success event %s", committed.Type)
		}
	}
}

func TestRunnerContinuesAfterFinalAnswerGuardRejectsInternalOutput(t *testing.T) {
	t.Parallel()
	provider := &scriptedProvider{responses: []model.Response{
		{Message: model.TextMessage(model.RoleAssistant, "<context_summary>internal state</context_summary>")},
		{Message: model.TextMessage(model.RoleAssistant, "verified result")},
	}}
	store := event.NewMemoryStore()
	runner, err := New(provider, tool.NewRegistry(), store, Config{
		MaxSteps: 2,
		FinalAnswerGuard: func(answer model.Message) (string, error) {
			if strings.HasPrefix(answer.TextContent(), "<context_summary>") {
				return "return a user-facing verified result", nil
			}
			return "", nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := runner.Run(context.Background(), harness.Request{RunID: "run-final-guard", Messages: []model.Message{model.TextMessage(model.RoleUser, "question")}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer.TextContent() != "verified result" || result.Steps != 2 {
		t.Fatalf("result=%+v", result)
	}
	foundRejected := false
	for _, committed := range store.Events("run-final-guard") {
		if committed.Type == event.FinalOutputRejected {
			foundRejected = true
		}
		if committed.Type == event.RunCompleted && committed.Step == 1 {
			t.Fatal("rejected output completed the run")
		}
	}
	if !foundRejected {
		t.Fatal("missing final output rejected event")
	}
}

func TestRunnerRecordsToolFailureAndLetsModelRecover(t *testing.T) {
	t.Parallel()

	provider := &scriptedProvider{responses: []model.Response{
		{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call-1", Name: "missing", Arguments: json.RawMessage(`{}`)}}}},
		{Message: model.Message{Role: model.RoleAssistant, Content: "工具不可用"}},
	}}
	runner, err := New(provider, tool.NewRegistry(), event.NewMemoryStore(), Config{MaxSteps: 2})
	if err != nil {
		t.Fatal(err)
	}
	result, err := runner.Run(context.Background(), harness.Request{
		RunID: "run-2", Messages: []model.Message{{Role: model.RoleUser, Content: "run"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer.Content != "工具不可用" {
		t.Fatalf("answer = %q", result.Answer.Content)
	}
}

func TestRunnerStopsAtStepLimit(t *testing.T) {
	t.Parallel()

	provider := &scriptedProvider{responses: []model.Response{{
		Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call-1", Name: "ping", Arguments: json.RawMessage(`{}`)}}},
	}}}
	tools := tool.NewRegistry()
	if err := tools.Register(tool.Definition{
		Name: "ping", Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`),
		Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial,
	}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	store := event.NewMemoryStore()
	runner, err := New(provider, tools, store, Config{MaxSteps: 1})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runner.Run(context.Background(), harness.Request{
		RunID: "run-3", Messages: []model.Message{{Role: model.RoleUser, Content: "run"}},
	})
	if !errors.Is(err, ErrStepLimit) {
		t.Fatalf("error = %v, want ErrStepLimit", err)
	}
	events := store.Events("run-3")
	if events[len(events)-1].Type != event.RunFailed {
		t.Fatalf("last event = %s, want %s", events[len(events)-1].Type, event.RunFailed)
	}
}

func TestRunnerCanTemporarilyDisableStepLimit(t *testing.T) {
	t.Parallel()

	provider := &scriptedProvider{responses: []model.Response{
		{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call-1", Name: "ping", Arguments: json.RawMessage(`{}`)}}}},
		{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "call-2", Name: "ping", Arguments: json.RawMessage(`{}`)}}}},
		{Message: model.TextMessage(model.RoleAssistant, "done")},
	}}
	tools := tool.NewRegistry()
	if err := tools.Register(tool.Definition{
		Name: "ping", Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`),
		Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial,
	}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	runner, err := New(provider, tools, event.NewMemoryStore(), Config{MaxSteps: 1, DisableStepLimit: true})
	if err != nil {
		t.Fatal(err)
	}
	result, err := runner.Run(context.Background(), harness.Request{
		RunID: "run-unlimited", Messages: []model.Message{{Role: model.RoleUser, Content: "run"}},
	})
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if result.Answer.TextContent() != "done" || len(provider.requests) != 3 {
		t.Fatalf("result=%+v model_calls=%d, want final answer after 3 calls", result, len(provider.requests))
	}
}

func TestRunnerMarksToolFailureAsFailedStepBeforeRecovery(t *testing.T) {
	t.Parallel()

	provider := &scriptedProvider{responses: []model.Response{
		{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "check-1", Name: "check", Arguments: json.RawMessage(`{}`)}}}},
		{Message: model.TextMessage(model.RoleAssistant, "repaired")},
	}}
	tools := tool.NewRegistry()
	if err := tools.Register(tool.Definition{
		Name: "check", Version: "1", InputSchema: json.RawMessage(`{"type":"object"}`),
		Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial,
	}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{IsError: true, Content: json.RawMessage(`{"exit_code":1,"stderr":"broken"}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	store := event.NewMemoryStore()
	runner, err := New(provider, tools, store, Config{MaxSteps: 2})
	if err != nil {
		t.Fatal(err)
	}
	result, err := runner.Run(context.Background(), harness.Request{
		RunID: "run-tool-failure-step", Messages: []model.Message{model.TextMessage(model.RoleUser, "repair")},
	})
	if err != nil || result.Answer.Content != "repaired" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	var failed, completed bool
	for _, committed := range store.Events("run-tool-failure-step") {
		if committed.Type == event.StepFailed && strings.Contains(string(committed.Payload), `"reason":"tool_failure"`) {
			failed = true
		}
		if committed.Type == event.StepCompleted && strings.Contains(string(committed.Payload), `"status":"completed"`) && committed.Step == 1 {
			completed = true
		}
	}
	if !failed {
		t.Fatal("tool failure did not close the decision step as STEP_FAILED")
	}
	if completed {
		t.Fatal("tool failure was falsely recorded as STEP_COMPLETED")
	}
}

func TestRunnerResumesPendingApprovedToolWithoutRepeatingModelCall(t *testing.T) {
	provider := &scriptedProvider{responses: []model.Response{
		{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{ID: "write-1", Name: "write", Arguments: json.RawMessage(`{"value":"x"}`)}}}},
		{Message: model.TextMessage(model.RoleAssistant, "done")},
	}}
	executions := 0
	tools := tool.NewRegistry()
	if err := tools.Register(tool.Definition{Name: "write", Version: "1", Description: "write", InputSchema: json.RawMessage(`{"type":"object"}`), Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial}, func(context.Context, tool.Call) (tool.Result, error) {
		executions++
		if executions == 1 {
			return tool.Result{}, approval.ErrRequired
		}
		return tool.Result{Content: json.RawMessage(`{"written":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	checkpoints := &checkpointRecorder{}
	events := event.NewMemoryStore()
	if _, err := events.Append(context.Background(), event.Input{RunID: "run-approval", Type: event.RunCreated}); err != nil {
		t.Fatal(err)
	}
	runner, err := New(provider, tools, events, Config{MaxSteps: 3, Checkpoints: checkpoints, ExternalRunLifecycle: true})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runner.Run(context.Background(), harness.Request{RunID: "run-approval", Messages: []model.Message{model.TextMessage(model.RoleUser, "write")}})
	if !errors.Is(err, approval.ErrRequired) || len(checkpoints.last.PendingToolCalls) != 1 || checkpoints.last.ActiveToolCallID != "write-1" {
		t.Fatalf("pause err=%v checkpoint=%+v", err, checkpoints.last)
	}
	paused := checkpoints.last
	result, err := runner.Run(context.Background(), harness.Request{RunID: paused.RunID, Messages: paused.Messages, Turn: paused.Turn, StartStep: paused.NextStep, Resume: true, Usage: paused.Usage, PendingToolCalls: paused.PendingToolCalls, ActiveToolCallID: paused.ActiveToolCallID, ExecutionLedger: paused.ExecutionLedger})
	if err != nil || result.Answer.Content != "done" {
		t.Fatalf("resume result=%+v err=%v", result, err)
	}
	if len(provider.requests) != 2 || executions != 2 {
		t.Fatalf("model requests=%d executions=%d", len(provider.requests), executions)
	}
	if len(checkpoints.last.ExecutionLedger) == 0 || !strings.Contains(string(checkpoints.last.ExecutionLedger), `"write"`) {
		t.Fatalf("successful resumed action was not stored in durable execution ledger: %s", checkpoints.last.ExecutionLedger)
	}
}

func TestStructuredToolFailureExposesExactFileContract(t *testing.T) {
	result := structuredToolFailure("write_file", tool.NewContractErrorWithRepair(
		"TOOL_SCHEMA_INVALID", "write_file", "/content", "maxLength: 8192", "9184 characters",
		"content exceeds maximum", "Correct the content field", nil, true,
	), nil, nil, "", false)
	var payload map[string]any
	if err := json.Unmarshal(result.Content, &payload); err != nil {
		t.Fatal(err)
	}
	correction, _ := payload["correction"].(string)
	if !strings.Contains(correction, "8192") || !strings.Contains(correction, "6000") || !strings.Contains(correction, "append_file") || !strings.Contains(correction, "Preserve requirements") {
		t.Fatalf("correction=%q; expected exact chunking guidance", correction)
	}
	contract, ok := payload["tool_contract"].(map[string]any)
	if !ok || contract["content_max_length"] != float64(8192) || contract["recovery_chunk_max_length"] != float64(6000) {
		t.Fatalf("tool_contract=%#v; expected content_max_length=8192", payload["tool_contract"])
	}
}

func TestActiveFileChunkRecoveryRejectsBoundaryRetry(t *testing.T) {
	recovery := model.TextMessage(model.RoleUser, "split the rejected file")
	recovery.Metadata = map[string]string{
		ToolFailureRecoveryMetadata:     "true",
		ToolFailureRecoveryToolMetadata: "write_file",
		FileChunkRecoveryMetadata:       "true",
		FileChunkRecoveryPathMetadata:   "pkg/core.py",
	}
	tooLarge := model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"pkg/core.py","content":"` + strings.Repeat("x", FileChunkRecoverySafeChars+1) + `"}`)}
	err := validateActiveFileChunkRecovery([]model.Message{recovery}, tooLarge)
	if err == nil || !strings.Contains(err.Error(), "too close") {
		t.Fatalf("boundary retry error=%v", err)
	}
	wrongPath := model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"pkg/other.py","content":"small"}`)}
	if err := validateActiveFileChunkRecovery([]model.Message{recovery}, wrongPath); err == nil || !strings.Contains(err.Error(), "changed") {
		t.Fatalf("wrong-path recovery error=%v", err)
	}
	valid := model.ToolCall{Name: "write_file", Arguments: json.RawMessage(`{"path":"pkg/core.py","content":"` + strings.Repeat("x", FileChunkRecoverySafeChars) + `"}`)}
	if err := validateActiveFileChunkRecovery([]model.Message{recovery}, valid); err != nil {
		t.Fatalf("safe recovery rejected: %v", err)
	}
}

func TestParseToolFailurePreservesProcessDiagnostic(t *testing.T) {
	result := tool.Result{IsError: true, Error: "command exited with code 1", Content: json.RawMessage(`{"error_code":"TOOL_EXECUTION_FAILED","failure_kind":"process_exit","exit_code":1,"stderr":"frame\nValueError: root","correction":"repair code"}`)}
	failure := parseToolFailure("run_command", json.RawMessage(`{"command":"python3","args":["test.py"]}`), result)
	if failure.FailureKind != "process_exit" || failure.ExitCode == nil || *failure.ExitCode != 1 || !failure.HasStderr || failure.Diagnostic != "ValueError: root" || !strings.Contains(failure.StderrTail, "ValueError: root") {
		t.Fatalf("parsed failure=%+v", failure)
	}
}

func TestStructuredToolFailureExplainsDuplicateRead(t *testing.T) {
	result := structuredToolFailure("read_file", tool.NewContractError(
		"DUPLICATE_READ_RANGE", "read_file", "path", "different range", "same range",
		"duplicate read_file range", true,
	), nil, nil, "", false)
	var payload map[string]any
	if err := json.Unmarshal(result.Content, &payload); err != nil {
		t.Fatal(err)
	}
	correction, _ := payload["correction"].(string)
	if !strings.Contains(correction, "Do not repeat") || !strings.Contains(correction, "different") {
		t.Fatalf("correction=%q; expected duplicate-read recovery", correction)
	}
}

func TestReplaceToolFailureRecoveryMessagesKeepsUnresolvedContractsForOtherTools(t *testing.T) {
	old := model.TextMessage(model.RoleUser, "old recovery")
	old.Metadata = map[string]string{ToolFailureRecoveryMetadata: "true", ToolFailureRecoveryToolMetadata: "write_file"}
	current := model.TextMessage(model.RoleUser, "current recovery")
	current.Metadata = map[string]string{ToolFailureRecoveryMetadata: "true", ToolFailureRecoveryToolMetadata: "read_file"}
	ordinary := model.TextMessage(model.RoleUser, "task")
	updated := replaceToolFailureRecoveryMessages([]model.Message{ordinary, old}, []model.Message{current})
	if len(updated) != 3 || updated[0].TextContent() != "task" || updated[1].TextContent() != "old recovery" || updated[2].TextContent() != "current recovery" {
		t.Fatalf("recovery replacement = %+v", updated)
	}
}

func TestToolSuccessClearsOnlyMatchingRecoveryContract(t *testing.T) {
	write := model.TextMessage(model.RoleUser, "write recovery")
	write.Metadata = map[string]string{ToolFailureRecoveryMetadata: "true", ToolFailureRecoveryToolMetadata: "write_file"}
	read := model.TextMessage(model.RoleUser, "read recovery")
	read.Metadata = map[string]string{ToolFailureRecoveryMetadata: "true", ToolFailureRecoveryToolMetadata: "read_file"}
	updated := clearToolFailureRecoveryMessagesForTool([]model.Message{write, read}, "write_file")
	if len(updated) != 1 || updated[0].TextContent() != "read recovery" {
		t.Fatalf("recovery cleanup = %+v", updated)
	}
}

func TestFailureMemoryReplacesPriorLookupAndDeduplicatesInitialMemory(t *testing.T) {
	initial := model.TextMessage(model.RoleUser, "initial memory")
	initial.Metadata = map[string]string{contextpkg.ContextSectionMetadataKey: contextpkg.ContextSectionMemory, contextpkg.MemoryIDsMetadataKey: "memory-a"}
	old := model.TextMessage(model.RoleUser, "old failure lookup")
	old.Metadata = map[string]string{ToolFailureMemoryMetadata: "true", ToolFailureRecoveryToolMetadata: "read_file", contextpkg.MemoryIDsMetadataKey: "memory-b"}
	duplicate := model.TextMessage(model.RoleUser, "duplicate failure lookup")
	duplicate.Metadata = map[string]string{ToolFailureMemoryMetadata: "true", ToolFailureRecoveryToolMetadata: "write_file", contextpkg.MemoryIDsMetadataKey: "memory-a"}
	updated := replaceToolFailureRecoveryMessages([]model.Message{initial, old}, []model.Message{duplicate})
	if len(updated) != 1 || updated[0].TextContent() != "initial memory" {
		t.Fatalf("failure memory replacement = %+v", updated)
	}
}
