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

type delegationNormalizationProvider struct{ calls int }

func (p *delegationNormalizationProvider) Complete(_ context.Context, _ model.Request) (model.Response, error) {
	p.calls++
	if p.calls == 1 {
		return model.Response{Message: model.Message{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{
			ID: "review-1", Name: "delegate_agent", Arguments: json.RawMessage(`{"target_agent":"reviewer-agent","mode":"sync","input":"{\"task\":\"review db.py\"}"}`),
		}}}}, nil
	}
	return model.Response{Message: model.TextMessage(model.RoleAssistant, "review complete")}, nil
}

func TestNormalizeUpdatePlanArgumentsAcceptsTaskAlias(t *testing.T) {
	normalized, err := normalizeUpdatePlanArguments(json.RawMessage(`{"goal":"g","steps":[{"id":"1","task":"inspect","status":"in_progress"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatal(err)
	}
	step := payload["steps"].([]any)[0].(map[string]any)
	if step["description"] != "inspect" {
		t.Fatalf("description alias not normalized: %s", normalized)
	}
	if _, exists := step["task"]; exists {
		t.Fatalf("task alias should be removed: %s", normalized)
	}
}

func TestNormalizeUpdatePlanArgumentsAcceptsCriteriaAlias(t *testing.T) {
	normalized, err := normalizeRuntimeToolArguments("update_plan", json.RawMessage(`{"goal":"g","steps":[{"id":"1","description":"build","status":"pending","accept_criteria":[{"id":"a","description":"exists","status":"pending"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(normalized), `"accept_criteria"`) || !strings.Contains(string(normalized), `"acceptance_criteria"`) {
		t.Fatalf("criteria alias not normalized: %s", normalized)
	}
}

func TestNormalizeRuntimeToolArgumentsConfinesWorkspaceAlias(t *testing.T) {
	normalized, err := normalizeRuntimeToolArguments("write_file", json.RawMessage(`{"path":"/workspace/src/game.py","content":"print('ok')"}`))
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["path"] != "src/game.py" {
		t.Fatalf("workspace alias not normalized: %s", normalized)
	}

	untouched, err := normalizeRuntimeToolArguments("read_file", json.RawMessage(`{"path":"/etc/passwd"}`))
	if err != nil {
		t.Fatal(err)
	}
	if string(untouched) != `{"path":"/etc/passwd"}` {
		t.Fatalf("unapproved absolute path must stay visible to confinement checks: %s", untouched)
	}
}

func TestNormalizeRuntimeToolArgumentsDecodesDelegationInputObject(t *testing.T) {
	normalized, err := normalizeRuntimeToolArguments("delegate_agent", json.RawMessage(`{"target_agent":"reviewer-agent","mode":"sync","input":"{\"task\":\"review db.py\",\"paths\":[\"db.py\"]}"}`))
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatal(err)
	}
	input, ok := payload["input"].(map[string]any)
	if !ok || input["task"] != "review db.py" {
		t.Fatalf("delegation input was not decoded: %s", normalized)
	}
}

func TestNormalizeRuntimeToolArgumentsKeepsPlainDelegationStringForSchemaError(t *testing.T) {
	raw := json.RawMessage(`{"target_agent":"reviewer-agent","mode":"sync","input":"review db.py"}`)
	normalized, err := normalizeRuntimeToolArguments("delegate_agent", raw)
	if err != nil || string(normalized) != string(raw) {
		t.Fatalf("plain string must remain visible to strict schema validation: normalized=%s err=%v", normalized, err)
	}
}

func TestRunnerNormalizesStringEncodedDelegationBeforeStrictSchema(t *testing.T) {
	provider := &delegationNormalizationProvider{}
	registry := tool.NewRegistry()
	called := false
	if err := registry.Register(tool.Definition{
		Name: "delegate_agent", Version: "2", Risk: tool.RiskInternal, ExecutionMode: tool.ExecutionSerial,
		InputSchema: json.RawMessage(`{"type":"object","required":["target_agent","mode","input"],"properties":{"target_agent":{"type":"string"},"mode":{"enum":["sync"]},"input":{"type":"object"}},"additionalProperties":false}`),
	}, func(_ context.Context, call tool.Call) (tool.Result, error) {
		called = true
		var payload struct {
			Input map[string]any `json:"input"`
		}
		if err := json.Unmarshal(call.Arguments, &payload); err != nil || payload.Input["task"] != "review db.py" {
			t.Fatalf("normalized delegate payload = %s err=%v", call.Arguments, err)
		}
		return tool.Result{Content: json.RawMessage(`{"status":"completed","output":{"findings":[]}}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	runner, err := react.New(provider, registry, event.NewMemoryStore(), react.Config{MaxSteps: 2, NormalizeToolArguments: normalizeRuntimeToolArguments})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), harness.Request{RunID: "delegate-normalization", Messages: []model.Message{model.TextMessage(model.RoleUser, "review")}}); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("delegate_agent provider was not executed")
	}
}

func TestNormalizeUpdatePlanArgumentsUnwrapsSameNamePayload(t *testing.T) {
	normalized, err := normalizeUpdatePlanArguments(json.RawMessage(`{"update_plan":"{\"goal\":\"g\",\"revision\":2,\"steps\":[{\"id\":\"1\",\"description\":\"build\",\"status\":\"in_progress\",\"acceptance_criteria\":[{\"id\":\"a\",\"description\":\"exists\",\"status\":\"pending\"}]}]}"}`))
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatal(err)
	}
	if payload["goal"] != "g" || payload["steps"] == nil {
		t.Fatalf("wrapped payload not normalized: %s", normalized)
	}
	if _, exists := payload["revision"]; exists {
		t.Fatalf("runtime-only revision should be removed: %s", normalized)
	}
}

func TestNormalizeUpdatePlanArgumentsDecodesStringEncodedSteps(t *testing.T) {
	normalized, err := normalizeUpdatePlanArguments(json.RawMessage(`{"goal":"g","steps":"[{\"id\":\"1\",\"description\":\"build\",\"status\":\"in_progress\",\"acceptance_criteria\":[{\"id\":\"a\",\"description\":\"exists\",\"status\":\"pending\",\"verification\":{\"kind\":\"file_exists\",\"target\":\"game.py\"}}]}]"}`))
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatal(err)
	}
	if _, ok := payload["steps"].([]any); !ok {
		t.Fatalf("string-encoded steps were not normalized: %s", normalized)
	}
}

func TestNormalizeUpdatePlanArgumentsStripsPlatformProjection(t *testing.T) {
	raw := json.RawMessage(`{"goal":"g","graph_state":{"status":"in_progress"},"steps":[{"id":"1","description":"build","status":"in_progress","state":{"attempts":3,"tests":[{"name":"python_syntax","status":"passed"}]},"acceptance_criteria":[{"id":"a","description":"exists","status":"pending","origin":"agent_inferred","enforcement":"advisory","evidence":"fake"}]}]}`)
	normalized, err := normalizeUpdatePlanArguments(raw)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatal(err)
	}
	if _, ok := payload["graph_state"]; ok {
		t.Fatalf("graph_state leaked into model input: %s", normalized)
	}
	step := payload["steps"].([]any)[0].(map[string]any)
	if _, ok := step["state"]; ok {
		t.Fatalf("state leaked into model input: %s", normalized)
	}
	criterion := step["acceptance_criteria"].([]any)[0].(map[string]any)
	for _, key := range []string{"origin", "enforcement", "evidence"} {
		if _, ok := criterion[key]; ok {
			t.Fatalf("platform criterion field %q leaked: %s", key, normalized)
		}
	}
}

func TestNormalizeUpdatePlanArgumentsRecoversCompleteStepsFromTruncatedString(t *testing.T) {
	// Simulates an OpenAI-compatible model stopping after a complete first
	// step but before closing the outer JSON array/object.
	truncated := json.RawMessage(`{"goal":"g","steps":"[{\"id\":\"1\",\"description\":\"build\"},{\"id\":\"2\",\"description\":\""}`)
	normalized, err := normalizeUpdatePlanArguments(truncated)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(normalized, &payload); err != nil {
		t.Fatal(err)
	}
	steps, ok := payload["steps"].([]any)
	if !ok || len(steps) != 1 {
		t.Fatalf("expected one recovered complete step, got %s", normalized)
	}
}

func TestExecutablePlanArgumentsDoNotLeakPlatformOwnedFields(t *testing.T) {
	arguments, err := executablePlanArguments(taskplan.Update{Goal: "game", Steps: []taskplan.Step{{
		ID: "write", Description: "write game", Status: taskplan.StatusInProgress,
		State: taskplan.NodeState{Status: taskplan.StatusInProgress, Attempts: 3, Output: "platform projection", NextNodeIDs: []string{"verify"}},
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
			ID: "exists", Description: "file exists", Status: taskplan.CriterionInvalid,
			Enforcement: taskplan.EnforcementAdvisory, Origin: taskplan.OriginAgentInferred,
			Verification:       taskplan.VerificationSpec{Kind: "file_exists", Target: "game.py"},
			VerificationReason: taskplan.VerificationReasonSpecInvalid, VerificationMessage: "old diagnostic",
			Evidence: "old evidence", EvidenceCallIDs: []string{"call-1"},
		}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"state", "enforcement", "origin", "verification_reason", "verification_message", "evidence", "evidence_call_ids"} {
		if strings.Contains(string(arguments), `"`+forbidden+`"`) {
			t.Fatalf("platform-owned field %q leaked across public Tool boundary: %s", forbidden, arguments)
		}
	}
	if !strings.Contains(string(arguments), `"status":"pending"`) || !strings.Contains(string(arguments), `"kind":"file_exists"`) {
		t.Fatalf("executable plan lost model-owned intent: %s", arguments)
	}
}

func TestBoundedToolExecutorSuppressesIdenticalLoop(t *testing.T) {
	registry := tool.NewRegistry()
	calls := 0
	if err := registry.Register(tool.Definition{
		Name: "list_files", Version: "1", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial,
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}, func(context.Context, tool.Call) (tool.Result, error) {
		calls++
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &boundedToolExecutor{delegate: registry, timeout: time.Second, limit: 10}
	call := tool.Call{Name: "list_files", Arguments: json.RawMessage(`{"path":"."}`)}
	for attempt := 1; attempt <= 2; attempt++ {
		result, err := executor.Execute(context.Background(), call)
		if err != nil {
			t.Fatal(err)
		}
		if attempt < 2 && result.IsError {
			t.Fatalf("attempt %d unexpectedly suppressed", attempt)
		}
		if attempt == 2 && (!result.IsError || !strings.Contains(result.Error, "repeated tool call suppressed")) {
			t.Fatalf("second identical call was not suppressed: %+v", result)
		}
	}
	if calls != 1 {
		t.Fatalf("underlying tool calls = %d, want 1", calls)
	}
}

func TestBoundedToolExecutorRequiresMutationBeforeDeterministicCommandRetry(t *testing.T) {
	registry := tool.NewRegistry()
	runCalls := 0
	if err := registry.Register(tool.Definition{Name: "run_command", Version: "1", Risk: tool.RiskHigh, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		runCalls++
		if runCalls == 1 {
			return tool.Result{Content: json.RawMessage(`{"failure_kind":"process_exit","exit_code":1,"diagnostic":"test failed"}`), IsError: true, Error: "command exited with code 1"}, nil
		}
		return tool.Result{Content: json.RawMessage(`{"exit_code":0}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"read_file", "edit_file"} {
		name := name
		if err := registry.Register(tool.Definition{Name: name, Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	executor := &boundedToolExecutor{delegate: registry, limit: 10}
	run := tool.Call{Name: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["test.py"]}`)}
	if result, err := executor.Execute(context.Background(), run); err != nil || !result.IsError {
		t.Fatalf("initial deterministic failure: result=%+v err=%v", result, err)
	}
	if _, err := executor.Execute(context.Background(), tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"test.py"}`)}); err != nil {
		t.Fatal(err)
	}
	if result, err := executor.Execute(context.Background(), run); err != nil || !result.IsError || !strings.Contains(string(result.Content), `"failure_kind":"repeated_without_change"`) {
		t.Fatalf("unchanged retry was not blocked: result=%+v err=%v", result, err)
	}
	if _, err := executor.Execute(context.Background(), tool.Call{Name: "edit_file", Arguments: json.RawMessage(`{"path":"test.py"}`)}); err != nil {
		t.Fatal(err)
	}
	if result, err := executor.Execute(context.Background(), run); err != nil || result.IsError {
		t.Fatalf("retry after mutation failed: result=%+v err=%v", result, err)
	}
	if runCalls != 2 {
		t.Fatalf("underlying run calls=%d, want 2", runCalls)
	}
}

func TestBoundedToolExecutorReturnsCachedSuccessfulReadRange(t *testing.T) {
	registry := tool.NewRegistry()
	calls := 0
	if err := registry.Register(tool.Definition{Name: "read_file", Version: "1", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		calls++
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &boundedToolExecutor{delegate: registry, limit: 10}
	first := tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"docs/test.md","start_line":1,"line_count":20}`)}
	if result, err := executor.Execute(context.Background(), first); err != nil || result.IsError {
		t.Fatalf("first read failed: result=%+v err=%v", result, err)
	}
	if result, err := executor.Execute(context.Background(), tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"docs/test.md","line_count":20}`)}); err != nil || result.IsError || result.Meta["cache_hit"] != "true" {
		t.Fatalf("equivalent duplicate read did not return cache: result=%+v err=%v", result, err)
	}
	if calls != 1 {
		t.Fatalf("underlying read calls=%d, want 1", calls)
	}
}

func TestBoundedToolExecutorAllowsReadAgainAfterFileMutation(t *testing.T) {
	registry := tool.NewRegistry()
	for _, name := range []string{"read_file", "append_file"} {
		name := name
		if err := registry.Register(tool.Definition{Name: name, Version: "1", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
			return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	executor := &boundedToolExecutor{delegate: registry, limit: 10}
	read := tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}
	if result, err := executor.Execute(context.Background(), read); err != nil || result.IsError {
		t.Fatalf("first read failed: result=%+v err=%v", result, err)
	}
	if result, err := executor.Execute(context.Background(), tool.Call{Name: "append_file", Arguments: json.RawMessage(`{"file_path":"game.py","text":"next"}`)}); err != nil || result.IsError {
		t.Fatalf("append failed: result=%+v err=%v", result, err)
	}
	if result, err := executor.Execute(context.Background(), read); err != nil || result.IsError {
		t.Fatalf("post-mutation read should be allowed: result=%+v err=%v", result, err)
	}
}

func TestPreserveCanonicalWorkspacePathsDoesNotShorten(t *testing.T) {
	input := contextpkg.SummaryInput{Generation: 2, Budget: 512, Messages: []model.Message{{Role: model.RoleAssistant, ToolCalls: []model.ToolCall{{Name: "read_file", Arguments: json.RawMessage(`{"path":"tmp/agent-long-task-demo/issue-tracker/issue_tracker/db.py"}`)}}}}}
	summary := preserveCanonicalWorkspacePaths("- edited issue_tracker/db.py", input)
	if !strings.Contains(summary, "tmp/agent-long-task-demo/issue-tracker/issue_tracker/db.py") {
		t.Fatalf("canonical path missing: %s", summary)
	}
}

func TestExactFailureMemoryCandidatesRequireToolAndTaxonomy(t *testing.T) {
	memories := []agent.Memory{
		{ID: "exact", Title: "write_file TOOL_SCHEMA_INVALID schema_invalid"},
		{ID: "wrong-kind", Title: "write_file PATH_NOT_FOUND"},
		{ID: "wrong-tool", Title: "read_file TOOL_SCHEMA_INVALID"},
	}
	selected := exactFailureMemoryCandidates(memories, react.ToolFailure{ToolName: "write_file", ErrorCode: "TOOL_SCHEMA_INVALID", FailureKind: "schema_invalid"})
	if len(selected) != 1 || selected[0].ID != "exact" {
		t.Fatalf("selected=%+v", selected)
	}
}

func TestValidatePlanToolOrderBlocksLaterStepPath(t *testing.T) {
	plan := taskplan.Plan{Steps: []taskplan.Step{
		{ID: "read-source", Status: taskplan.StatusInProgress, Description: "read docs/source.md"},
		{ID: "write-report", Status: taskplan.StatusPending, Description: "write tmp/report.md"},
	}}
	err := validatePlanToolOrder(plan, tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"tmp/report.md"}`)}, nil)
	if err == nil || !strings.Contains(err.Error(), "plan order violation") {
		t.Fatalf("later step path was not blocked: %v", err)
	}
	if err := validatePlanToolOrder(plan, tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"docs/source.md"}`)}, nil); err != nil {
		t.Fatalf("active step path was blocked: %v", err)
	}
}

func TestValidatePlanToolOrderTreatsTodoHintsAsGuidance(t *testing.T) {
	plan := taskplan.Plan{Steps: []taskplan.Step{{
		ID: "build", Status: taskplan.StatusInProgress, Description: "build project", ToolHints: []string{"write_file", "append_file"},
	}}}
	if err := validatePlanToolOrder(plan, tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}, nil); err != nil {
		t.Fatalf("tool outside active Todo hints should remain available for repair: %v", err)
	}
	if err := validatePlanToolOrder(plan, tool.Call{Name: "append_file", Arguments: json.RawMessage(`{"path":"game.py","content":"next"}`)}, nil); err != nil {
		t.Fatalf("allowed active Todo tool was rejected: %v", err)
	}
}

func TestIsNoopPlanUpdateIgnoresDescriptionRewording(t *testing.T) {
	current := taskplan.Plan{Steps: []taskplan.Step{
		{ID: "1", Status: taskplan.StatusInProgress, Description: "read source"},
		{ID: "2", Status: taskplan.StatusPending, Description: "write report", DependsOn: []string{"1"}},
	}}
	arguments := json.RawMessage(`{"goal":"translated goal","steps":[{"id":"1","description":"读取源文件","status":"in_progress"},{"id":"2","description":"写报告","status":"pending","depends_on":["1"]}]}`)
	if !isNoopPlanUpdate(current, arguments) {
		t.Fatal("description-only rewrite should be treated as a no-op")
	}
	changed := json.RawMessage(`{"goal":"g","steps":[{"id":"1","description":"read","status":"completed","result":"done"},{"id":"2","description":"write","status":"in_progress","depends_on":["1"]}]}`)
	if isNoopPlanUpdate(current, changed) {
		t.Fatal("status/result progress must not be treated as a no-op")
	}
}

func TestIsNoopPlanUpdateDetectsAcceptanceEvidence(t *testing.T) {
	current := taskplan.Plan{Steps: []taskplan.Step{{
		ID: "1", Status: taskplan.StatusInProgress, Description: "build",
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{ID: "syntax", Description: "syntax", Status: taskplan.CriterionPending}},
	}}}
	changed := json.RawMessage(`{"goal":"g","steps":[{"id":"1","description":"build","status":"completed","acceptance_criteria":[{"id":"syntax","description":"syntax","status":"passed","evidence":"py_compile exit 0"}]}]}`)
	if isNoopPlanUpdate(current, changed) {
		t.Fatal("acceptance evidence update must not be treated as a no-op")
	}
}

func TestPlanRequiredToolExecutorAllowsObservationButBlocksMutationWithoutPlan(t *testing.T) {
	registry := tool.NewRegistry()
	if err := registry.Register(tool.Definition{Name: "list_files", Version: "1", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := registry.Register(tool.Definition{Name: "write_file", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{}, taskplan.ErrNotFound
	}}
	if _, err := executor.Execute(context.Background(), tool.Call{Name: "list_files", Arguments: json.RawMessage(`{}`)}); err != nil {
		t.Fatalf("list_files should be allowed before a Plan: %v", err)
	}
	if _, err := executor.Execute(context.Background(), tool.Call{Name: "write_file", Arguments: json.RawMessage(`{}`)}); err == nil || !strings.Contains(err.Error(), "update_plan") {
		t.Fatalf("write_file should require a Plan, error = %v", err)
	}
}

func TestPlanRequiredToolExecutorKeepsInternalPolicyOutsidePublicSchema(t *testing.T) {
	registry := tool.NewRegistry()
	if err := registry.Register(tool.Definition{
		Name: "write_file", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial,
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	const publicPlanSchema = `{
		"type":"object",
		"properties":{"steps":{"type":"array","items":{
			"type":"object",
			"properties":{"acceptance_criteria":{"type":"array","items":{
				"type":"object",
				"properties":{"id":{},"description":{},"status":{},"verification":{}},
				"additionalProperties":false
			}}}
		}}}
	}`
	var captured json.RawMessage
	if err := registry.Register(tool.Definition{
		Name: "update_plan", Version: "1", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial,
		InputSchema: json.RawMessage(publicPlanSchema),
	}, func(_ context.Context, call tool.Call) (tool.Result, error) {
		captured = append(json.RawMessage(nil), call.Arguments...)
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{}, taskplan.ErrNotFound
	}}
	call := tool.Call{Name: "update_plan", Arguments: json.RawMessage(`{"goal":"game","graph_state":{"status":"in_progress"},"steps":[{"id":"write","description":"write game","status":"in_progress","state":{"attempts":2,"output":"old projection"},"tool_hints":["write_file"],"acceptance_criteria":[{"id":"exists","description":"file exists","status":"pending","origin":"agent_inferred","enforcement":"advisory","verification":{"kind":"file_exists","target":"game.py"}}]}]}`)}
	result, err := executor.Execute(context.Background(), call)
	if err != nil || result.IsError {
		t.Fatalf("valid public Plan was rejected after internal normalization: result=%+v err=%v", result, err)
	}
	if strings.Contains(string(captured), `"origin"`) || strings.Contains(string(captured), `"enforcement"`) {
		t.Fatalf("platform policy leaked into Registry input: %s", captured)
	}
}

func TestPlanRequiredToolExecutorAllowsToolAfterPlan(t *testing.T) {
	registry := tool.NewRegistry()
	if err := registry.Register(tool.Definition{Name: "read_file", Version: "1", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{Steps: []taskplan.Step{{ID: "read", Description: "read", Status: taskplan.StatusInProgress}}}, nil
	}}
	result, err := executor.Execute(context.Background(), tool.Call{Name: "read_file", Arguments: json.RawMessage(`{}`)})
	if err != nil || result.IsError {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestRenderPlanContextIncludesActiveStepAndBoundsLargeText(t *testing.T) {
	contextValue := renderPlanContext(taskplan.Plan{PlanID: "plan-1", Revision: 2, OriginalGoal: "original user goal", Goal: strings.Repeat("goal", 300), Steps: []taskplan.Step{{
		ID: "inspect", Description: strings.Repeat("read", 300), Status: taskplan.StatusInProgress,
	}}})
	if !strings.Contains(contextValue, `"plan_id":"plan-1"`) || !strings.Contains(contextValue, `"revision":2`) || !strings.Contains(contextValue, `"original_goal":"original user goal"`) || !strings.Contains(contextValue, `"active_nodes"`) || !strings.Contains(contextValue, `"status":"in_progress"`) {
		t.Fatalf("context = %q", contextValue)
	}
	if len([]rune(contextValue)) > 1400 {
		t.Fatalf("rendered plan context is unexpectedly large: %d", len([]rune(contextValue)))
	}
}

func TestRenderPlanContextKeepsLateActiveStepWithinBoundedWindow(t *testing.T) {
	steps := make([]taskplan.Step, 64)
	for index := range steps {
		status := taskplan.StatusCompleted
		if index == 40 {
			status = taskplan.StatusInProgress
		} else if index > 40 {
			status = taskplan.StatusPending
		}
		steps[index] = taskplan.Step{
			ID: fmt.Sprintf("step-%02d", index), Status: status,
			Description: strings.Repeat("long task description ", 20),
			ToolHints:   []string{"read_file", "write_file", "run_command"},
			AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
				ID: "criterion", Description: strings.Repeat("observable result ", 20), Status: taskplan.CriterionPending,
				Evidence: strings.Repeat("large evidence ", 20), EvidenceCallIDs: []string{"one", "two", "three", "four", "five"},
			}},
		}
	}
	contextValue := renderPlanContext(taskplan.Plan{Revision: 9, Goal: strings.Repeat("goal ", 100), Steps: steps})
	if !strings.Contains(contextValue, `"id":"step-40"`) || !strings.Contains(contextValue, `"status":"in_progress"`) {
		t.Fatalf("late active step missing from context: %s", contextValue)
	}
	if len([]rune(contextValue)) > 3800 {
		t.Fatalf("rendered large plan context is unexpectedly large: %d", len([]rune(contextValue)))
	}
}

func TestRenderPlanContextOmitsUnboundedVerificationPayloads(t *testing.T) {
	plan := taskplan.Plan{Revision: 12, Goal: strings.Repeat("goal ", 200), Steps: []taskplan.Step{{
		ID: "build", Description: "build the project", Status: taskplan.StatusInProgress,
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
			ID: "verify", Description: "run the complete verification", Status: taskplan.CriterionPending,
			Verification: taskplan.VerificationSpec{
				Kind: "command_exit_zero", Target: "run.py",
				Arguments:  json.RawMessage(`{"command":"python3","args":["-c","` + strings.Repeat("large-argument ", 500) + `"]}`),
				Assertions: []taskplan.VerificationAssertion{{Path: "exit_code", Operator: "equals", Value: json.RawMessage(`0`)}},
			},
		}},
	}}}
	contextValue := renderPlanContext(plan)
	if len([]rune(contextValue)) > 4300 {
		t.Fatalf("plan projection exceeds bounded runtime prompt: %d", len([]rune(contextValue)))
	}
	if strings.Contains(contextValue, "large-argument") || strings.Contains(contextValue, `"arguments"`) {
		t.Fatalf("raw verification arguments leaked into model projection: %s", contextValue)
	}
	if !strings.Contains(contextValue, `"kind":"command_exit_zero"`) || !strings.Contains(contextValue, `"target":"run.py"`) {
		t.Fatalf("verification identity was lost: %s", contextValue)
	}
}

func TestPlanRequiredToolExecutorPropagatesPlanReadFailure(t *testing.T) {
	executor := &planRequiredToolExecutor{delegate: tool.NewRegistry(), load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{}, errors.New("database unavailable")
	}}
	_, err := executor.Execute(context.Background(), tool.Call{Name: "read_file"})
	if err == nil || !strings.Contains(err.Error(), "database unavailable") {
		t.Fatalf("error = %v", err)
	}
}

func TestPlanRequiredToolExecutorRejectsUnavailableStepToolHintRevision(t *testing.T) {
	registry := tool.NewRegistry()
	if err := registry.Register(tool.Definition{Name: "update_plan_step", Version: "3", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{}, taskplan.ErrNotFound
	}}
	_, err := executor.Execute(context.Background(), tool.Call{Name: "update_plan_step", Arguments: json.RawMessage(`{"step_id":"build","status":"in_progress","tool_hints":["missing_tool"]}`)})
	if err == nil || !strings.Contains(err.Error(), "unavailable tool_hint") {
		t.Fatalf("error = %v", err)
	}
}

func TestPlanRequiredToolExecutorBindsSubstantiveCallToActiveStep(t *testing.T) {
	registry := tool.NewRegistry()
	var captured tool.Call
	if err := registry.Register(tool.Definition{Name: "read_file", Version: "1", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(_ context.Context, call tool.Call) (tool.Result, error) {
		captured = call
		return tool.Result{Content: json.RawMessage(`{"content":"ok"}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{Steps: []taskplan.Step{{
			ID: "inspect", Description: "inspect the file", Status: taskplan.StatusInProgress, ToolHints: []string{"read_file"},
		}}}, nil
	}}
	if _, err := executor.Execute(context.Background(), tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"game.py"}`)}); err != nil {
		t.Fatal(err)
	}
	if captured.PlanStepID != "inspect" {
		t.Fatalf("plan step binding = %q, want inspect", captured.PlanStepID)
	}
}

func TestPlanRequiredToolExecutorBlocksMutationWhenActiveStepCannotAdvance(t *testing.T) {
	registry := tool.NewRegistry()
	called := false
	if err := registry.Register(tool.Definition{Name: "write_file", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		called = true
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{Steps: []taskplan.Step{{
			ID: "legacy", Description: "legacy step", Status: taskplan.StatusInProgress,
			AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
				ID: "old", Description: "old provider", Status: taskplan.CriterionUnsupported,
				Verification: taskplan.VerificationSpec{Kind: "tool_result_observation"},
			}},
		}}}, nil
	}}
	_, err := executor.Execute(context.Background(), tool.Call{Name: "write_file", Arguments: json.RawMessage(`{"path":"main.py","content":"pass"}`)})
	contractErr, ok := tool.AsContractError(err)
	if !ok || contractErr.Code != "PLAN_PROGRESS_CONTRACT_INVALID" || !strings.Contains(contractErr.Correction, "revise_verification") {
		t.Fatalf("error = %#v, want repairable Plan progress contract failure", err)
	}
	if called {
		t.Fatal("mutation reached provider despite an unadvanceable Plan step")
	}
}

func TestPlanRequiredToolExecutorAllowsMutationWithExecutableVerification(t *testing.T) {
	registry := tool.NewRegistry()
	var captured tool.Call
	if err := registry.Register(tool.Definition{Name: "write_file", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(_ context.Context, call tool.Call) (tool.Result, error) {
		captured = call
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{Steps: []taskplan.Step{{
			ID: "write", Description: "write file", Status: taskplan.StatusInProgress,
			AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
				ID: "exists", Description: "file exists", Status: taskplan.CriterionPending,
				Verification: taskplan.VerificationSpec{Kind: "file_exists", Target: "main.py"},
			}},
		}}}, nil
	}}
	if _, err := executor.Execute(context.Background(), tool.Call{Name: "write_file", Arguments: json.RawMessage(`{"path":"main.py","content":"pass"}`)}); err != nil {
		t.Fatal(err)
	}
	if captured.PlanStepID != "write" || captured.PlanNodeID != "write" {
		t.Fatalf("call was not bound to active Plan node: %+v", captured)
	}
}

func TestPlanRequiredToolExecutorBlocksSubstantiveToolWithoutExecutableNode(t *testing.T) {
	registry := tool.NewRegistry()
	called := false
	if err := registry.Register(tool.Definition{Name: "write_file", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		called = true
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{Steps: []taskplan.Step{{
			ID: "implement", Description: "implement the feature", Status: taskplan.StatusCompleted,
			AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
				ID: "review", Description: "review the result", Status: taskplan.CriterionPending,
				Enforcement: taskplan.EnforcementAdvisory,
			}},
		}}}, nil
	}}
	_, err := executor.Execute(context.Background(), tool.Call{Name: "write_file", Arguments: json.RawMessage(`{"path":"main.py","content":"pass"}`)})
	contractErr, ok := tool.AsContractError(err)
	if !ok || contractErr.Code != "PLAN_NO_ACTIVE_NODE" || !strings.Contains(contractErr.Correction, "update_plan_step") {
		t.Fatalf("error = %#v, want recoverable missing active-node failure", err)
	}
	if called {
		t.Fatal("substantive tool reached provider without an executable Plan node")
	}
}

func TestPlanRequiredToolExecutorAllowsObservationWithoutExecutableNode(t *testing.T) {
	registry := tool.NewRegistry()
	called := false
	if err := registry.Register(tool.Definition{Name: "read_file", Version: "1", Risk: tool.RiskRead, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		called = true
		return tool.Result{Content: json.RawMessage(`{"content":"ok"}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{Steps: []taskplan.Step{{ID: "done", Status: taskplan.StatusCompleted}}}, nil
	}}
	if _, err := executor.Execute(context.Background(), tool.Call{Name: "read_file", Arguments: json.RawMessage(`{"path":"main.py"}`)}); err != nil {
		t.Fatalf("read-only observation should remain available: %v", err)
	}
	if !called {
		t.Fatal("read-only observation did not reach provider")
	}
}

func TestPlanRequiredToolExecutorRequiresExplicitStaleNodeRecovery(t *testing.T) {
	registry := tool.NewRegistry()
	called := false
	if err := registry.Register(tool.Definition{Name: "write_file", Version: "1", Risk: tool.RiskLowWrite, ExecutionMode: tool.ExecutionSerial, InputSchema: json.RawMessage(`{"type":"object"}`)}, func(context.Context, tool.Call) (tool.Result, error) {
		called = true
		return tool.Result{Content: json.RawMessage(`{"ok":true}`)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	executor := &planRequiredToolExecutor{delegate: registry, load: func(context.Context) (taskplan.Plan, error) {
		return taskplan.Plan{Steps: []taskplan.Step{{
			ID: "verify", Description: "verify", Status: taskplan.StatusStale,
			AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
				ID: "syntax", Description: "syntax", Status: taskplan.CriterionStale,
				Verification: taskplan.VerificationSpec{Kind: "python_syntax", Target: "main.py"},
			}},
		}}}, nil
	}}
	_, err := executor.Execute(context.Background(), tool.Call{Name: "write_file", Arguments: json.RawMessage(`{"path":"main.py","content":"pass"}`)})
	contractErr, ok := tool.AsContractError(err)
	if !ok || contractErr.Code != "PLAN_NODE_STALE" || !strings.Contains(contractErr.Correction, "update_plan_step") {
		t.Fatalf("error = %#v, want explicit stale-node recovery", err)
	}
	if called {
		t.Fatal("stale node mutation reached provider before recovery")
	}
}

func TestPlanRequiredToolExecutorDoesNotBypassUnsupportedRequiredCriterion(t *testing.T) {
	step := taskplan.Step{AcceptanceCriteria: []taskplan.AcceptanceCriterion{
		{ID: "advisory", Description: "file exists", Status: taskplan.CriterionPending, Verification: taskplan.VerificationSpec{Kind: "file_exists", Target: "main.py"}},
		{ID: "release", Description: "release policy", Status: taskplan.CriterionUnsupported, Enforcement: taskplan.EnforcementReleaseGate, Verification: taskplan.VerificationSpec{Kind: "retired_provider"}},
	}}
	if planStepHasExecutableVerification(step) {
		t.Fatal("an executable advisory criterion must not bypass an unsupported release gate")
	}
}

func TestValidateExecutableVerificationContractsRejectsUnexecutablePlanAtCreation(t *testing.T) {
	update, err := taskplan.NormalizeUpdate(taskplan.Update{Goal: "build", Steps: []taskplan.Step{{
		ID: "implement", Description: "implement", Status: taskplan.StatusInProgress,
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
			ID: "lint", Description: "lint succeeds", Status: taskplan.CriterionPending,
			Verification: taskplan.VerificationSpec{Kind: "lint"},
		}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	err = validateExecutableVerificationContracts(update, nil)
	contractErr, ok := tool.AsContractError(err)
	if !ok || contractErr.Code != "PLAN_VERIFICATION_PROVIDER_REQUIRED" || !strings.Contains(contractErr.Expected, "python_syntax") || !strings.Contains(contractErr.Correction, "Do not skip the only criterion") {
		t.Fatalf("error = %#v, want executable-provider creation guard", err)
	}
}

func TestExpandPlanToolHintsAddsChunkContinuation(t *testing.T) {
	definitions := []tool.Definition{{Name: "write_file"}, {Name: "append_file"}, {Name: "edit_file"}, {Name: "run_command"}}
	got := expandPlanToolHints([]string{"write_file", "run_command"}, definitions)
	if len(got) != 4 || got[0] != "write_file" || got[1] != "run_command" || got[2] != "append_file" || got[3] != "edit_file" {
		t.Fatalf("expanded hints = %#v", got)
	}
	got = expandPlanToolHints([]string{"write_file"}, []tool.Definition{{Name: "write_file"}})
	if len(got) != 1 {
		t.Fatalf("unavailable append_file was added: %#v", got)
	}
}

func TestReplaceStepUpdateToolHintsDoesNotInjectVerification(t *testing.T) {
	raw := json.RawMessage(`{"step_id":"build","status":"completed","tool_hints":["run_command"],"acceptance_criteria":[{"id":"syntax","status":"passed","evidence":"ok","evidence_call_ids":["call-1"]}]}`)
	normalized, err := replaceStepUpdateToolHints(raw, []string{"run_command", "edit_file"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(normalized), `"verification"`) {
		t.Fatalf("planner-owned verification leaked into compact mutation: %s", normalized)
	}
	if !strings.Contains(string(normalized), `"edit_file"`) || !strings.Contains(string(normalized), `"call-1"`) {
		t.Fatalf("normalized mutation lost caller fields: %s", normalized)
	}
}

func TestExpandPlanStepToolHintsAddsRequiredVerificationTool(t *testing.T) {
	definitions := []tool.Definition{{Name: "read_file"}, {Name: "search_files"}, {Name: "run_command"}, {Name: "edit_file"}, {Name: "install_dependency"}}
	step := taskplan.Step{
		ID: "verify", Status: taskplan.StatusInProgress, ToolHints: []string{"read_file", "search_files"},
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
			ID: "syntax", Description: "snake.py syntax is valid", Status: taskplan.CriterionPending,
			Verification: taskplan.VerificationSpec{Kind: "python_syntax", Target: "snake.py"},
		}},
	}
	got := expandPlanStepToolHints(step, definitions)
	if !reflect.DeepEqual(got, []string{"read_file", "search_files", "run_command", "edit_file", "install_dependency"}) {
		t.Fatalf("expanded verification hints = %#v", got)
	}
	if err := validatePlanToolOrder(taskplan.Plan{Steps: []taskplan.Step{step}}, tool.Call{Name: "run_command"}, definitions); err != nil {
		t.Fatalf("required verification tool was rejected by stale hints: %v", err)
	}
}

func TestValidateExecutableVerificationContractsRejectsShellTarget(t *testing.T) {
	update := taskplan.Update{Goal: "game", Steps: []taskplan.Step{{
		ID: "smoke", Description: "smoke test", Status: taskplan.StatusInProgress,
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
			ID: "runs", Description: "runs", Status: taskplan.CriterionPending,
			Verification: taskplan.VerificationSpec{Kind: "command_exit_zero", Target: "SDL_VIDEODRIVER=dummy timeout 2 python snake.py"},
		}},
	}}}
	definitions := []tool.Definition{{Name: "run_command", InputSchema: json.RawMessage(`{"type":"object","required":["command","args"],"properties":{"command":{"const":"python3"},"args":{"type":"array","minItems":1,"items":{"type":"string"}}},"additionalProperties":true}`)}}
	if err := validateExecutableVerificationContracts(update, definitions); err == nil || !strings.Contains(err.Error(), "unreachable") {
		t.Fatalf("unreachable shell target was accepted: %v", err)
	}
	update.Steps[0].AcceptanceCriteria[0].Verification = taskplan.VerificationSpec{
		Kind: "command_exit_zero", Arguments: json.RawMessage(`{"command":"python3","args":["smoke_run.py"]}`),
	}
	if err := validateExecutableVerificationContracts(update, definitions); err != nil {
		t.Fatalf("structured run_command contract was rejected: %v", err)
	}
	update.Steps[0].AcceptanceCriteria[0].Verification = taskplan.VerificationSpec{
		Kind: "tool_receipt", Tool: "run_command",
		Arguments:  json.RawMessage(`{"command":"python3","args":["-c","import pygame; print(pygame.version.ver)"]}`),
		Assertions: []taskplan.VerificationAssertion{{Path: "exit_code", Operator: "equals", Value: json.RawMessage(`0`)}},
	}
	if err := validateExecutableVerificationContracts(update, definitions); err != nil {
		t.Fatalf("restricted inline probe contract was rejected: %v", err)
	}
}

func TestCurrentExecutionStepPrefersInProgressOverBlocked(t *testing.T) {
	step, ok := currentExecutionStep(taskplan.Plan{Steps: []taskplan.Step{
		{ID: "waiting", Status: taskplan.StatusBlocked},
		{ID: "build", Status: taskplan.StatusInProgress},
	}})
	if !ok || step.ID != "build" {
		t.Fatalf("current execution step = %#v, %v", step, ok)
	}
}

func TestCompletionEvidenceBlockIsActionable(t *testing.T) {
	plan := taskplan.Plan{Steps: []taskplan.Step{{
		ID: "verify", Description: "validate snake.py", Status: taskplan.StatusCompleted,
		ToolHints: []string{"read_file", "search_files"},
	}}}
	evidenceErr := &taskplan.EvidenceValidationError{
		StepID: "verify", CriterionID: "syntax", Description: "snake.py syntax is valid",
		Verification: taskplan.VerificationSpec{Kind: "python_syntax", Target: "snake.py"},
		Cause:        errors.New("successful Tool receipts do not satisfy verification kind=python_syntax target=\"snake.py\""),
	}
	block := completionEvidenceBlock(plan, evidenceErr, []tool.Definition{{Name: "read_file"}, {Name: "search_files"}, {Name: "run_command"}})
	if block.Reason != "invalid_evidence" || block.CriterionID != "syntax" || !reflect.DeepEqual(block.RequiredTools, []string{"run_command"}) || !reflect.DeepEqual(block.AutoInjectedTools, []string{"run_command"}) {
		t.Fatalf("completion block = %+v", block)
	}
	for _, wanted := range []string{"python_syntax", "snake.py", "run_command", "py_compile", "automatically"} {
		if !strings.Contains(block.Instruction, wanted) {
			t.Fatalf("instruction missing %q: %s", wanted, block.Instruction)
		}
	}
}
