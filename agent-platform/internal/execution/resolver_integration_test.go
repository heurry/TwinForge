package execution

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/internal/persistence/postgres"
	runtimepkg "github.com/heurry/cloudnative-infra-platform/agent-platform/internal/runtime"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/event"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/resource"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func TestWorkerExecutesVersionPinnedModelAndHTTPTool(t *testing.T) {
	databaseURL := os.Getenv("TEST_AGENT_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("TEST_AGENT_DATABASE_URL is not set")
	}
	toolServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Idempotency-Key") == "" {
			t.Error("tool call has no idempotency key")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"name":"record-7"}`))
	}))
	defer toolServer.Close()

	var modelCalls atomic.Int32
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"mock-model","max_model_len":4096,"capabilities":["chat","tool_calling"]}]}`))
			return
		}
		var request struct {
			Messages []struct {
				Role       string `json:"role"`
				ToolCallID string `json:"tool_call_id"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode model request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		callNumber := modelCalls.Add(1)
		if callNumber == 1 {
			_, _ = w.Write([]byte(`{
				"model":"mock-model","choices":[{"finish_reason":"tool_calls","message":{
					"role":"assistant","tool_calls":[{"id":"plan-1","type":"function","function":{"name":"update_plan","arguments":"{\"goal\":\"Find record 7\",\"steps\":[{\"id\":\"lookup\",\"description\":\"Look up record 7\",\"status\":\"pending\",\"acceptance_criteria\":[{\"id\":\"record-found\",\"description\":\"Lookup succeeds\",\"status\":\"pending\",\"verification\":{\"kind\":\"tool_success\"}}],\"tool_hints\":[\"lookup_record\"]}]}"}}]
				}}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}
			}`))
			return
		}
		if callNumber == 2 {
			_, _ = w.Write([]byte(`{
				"model":"mock-model","choices":[{"finish_reason":"tool_calls","message":{
					"role":"assistant","tool_calls":[{"id":"lookup-1","type":"function","function":{"name":"lookup_record","arguments":"{\"id\":7}"}}]
				}}],"usage":{"prompt_tokens":14,"completion_tokens":2,"total_tokens":16}
			}`))
			return
		}
		foundToolResult := false
		for _, message := range request.Messages {
			if message.Role == "tool" && message.ToolCallID == "lookup-1" {
				foundToolResult = true
			}
		}
		if !foundToolResult {
			t.Error("final model request did not contain paired tool result")
		}
		_, _ = w.Write([]byte(`{
			"model":"mock-model","choices":[{"finish_reason":"stop","message":{
				"role":"assistant","content":"{\"answer\":\"record-7\"}"
			}}],"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}
		}`))
	}))
	defer modelServer.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	pool, err := postgres.Open(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	testLock, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := testLock.Exec(ctx, `SELECT pg_advisory_lock(7188617099)`); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_, _ = testLock.Exec(context.Background(), `SELECT pg_advisory_unlock(7188617099)`)
		testLock.Release()
	}()
	if err := postgres.Migrate(ctx, pool, "../../migrations"); err != nil {
		t.Fatal(err)
	}
	store := postgres.NewRunStore(pool)
	tenantID := "worker-e2e-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	prompt, err := store.CreatePromptVersion(ctx, resource.CreatePromptVersion{
		TenantID: tenantID, Key: "support-prompt", Name: "Support Prompt",
		Content: "Use tools and return JSON.",
	})
	if err != nil {
		t.Fatal(err)
	}
	toolVersion, err := store.CreateToolVersion(ctx, resource.CreateToolVersion{
		TenantID: tenantID, Key: "lookup-record", Name: "Lookup Record",
		Spec: resource.ToolSpec{
			Definition: tool.Definition{
				Name: "lookup_record", Description: "look up a record",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}`),
				Risk:        tool.RiskRead, ExecutionMode: tool.ExecutionSerial,
			},
			ProviderType: "http",
			HTTP:         &resource.HTTPProvider{Endpoint: toolServer.URL, Method: http.MethodPost},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	toolSet, err := store.CreateToolSetVersion(ctx, resource.CreateToolSetVersion{
		TenantID: tenantID, Key: "support-tools", Name: "Support Tools",
		Spec: resource.ToolSetSpec{Tools: []resource.VersionRef{{
			ID: toolVersion.ID, Version: strconv.Itoa(toolVersion.Version),
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	definition, err := store.CreateDefinition(ctx, agent.CreateDefinition{
		TenantID: tenantID, Key: "support-agent", Name: "Support Agent",
	})
	if err != nil {
		t.Fatal(err)
	}
	spec := agent.Spec{
		Name: "support-agent", Harness: agent.HarnessSpec{Name: "react-v1", MaxTurns: 1, MaxSteps: 4},
		Model: agent.ModelBinding{
			Provider: "openai-compatible", ServiceRef: "mock-service", ModelID: "mock-model",
		},
		PromptRef:   agent.VersionRef{ID: prompt.ID, Version: strconv.Itoa(prompt.Version)},
		ToolSetRef:  agent.VersionRef{ID: toolSet.ID, Version: strconv.Itoa(toolSet.Version)},
		InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: json.RawMessage(`{"type":"object"}`),
		Context: agent.ContextPolicy{MaxInputTokens: 4096, ReserveOutputTokens: 512},
		Runtime: agent.RuntimePolicy{
			RunTimeout: 5 * time.Second, ModelTimeout: 2 * time.Second, ToolTimeout: 2 * time.Second,
			MaxModelCalls: 4, MaxToolCalls: 4,
		},
	}
	version, err := store.CreateVersion(ctx, agent.CreateVersion{TenantID: tenantID, AgentID: definition.ID, Spec: spec})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.ReleaseVersion(ctx, tenantID, version.ID); err != nil {
		t.Fatal(err)
	}
	run, err := store.CreateRun(ctx, agent.CreateRun{
		TenantID: tenantID, AgentVersionID: version.ID, Input: json.RawMessage(`{"question":"find record 7"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	resolver, err := NewResolver(store, Config{
		ModelServices: map[string]ModelService{
			"mock-service": {Endpoint: modelServer.URL, Model: "mock-model"},
		},
		ToolAllowedHosts: []string{toolServer.Listener.Addr().String()},
		HTTPClient:       modelServer.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	processor, err := runtimepkg.NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := runtimepkg.NewWorker(store, processor, runtimepkg.Config{
		WorkerID: "e2e-worker", RuntimeVersion: "integration-test/1", ToolContractVersion: "tool-contract/v1",
		ProtocolVersion: "structured-tool-calls/v1", CapabilityHash: "integration-test/1:tool-contract/v1:structured-tool-calls/v1",
		Capabilities: []string{"react-v1", "tool-contract-v1", "structured-tool-calls-v1"}, PollInterval: 10 * time.Millisecond,
		LeaseDuration: 3 * time.Second, HeartbeatInterval: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	workerCtx, stopWorker := context.WithCancel(ctx)
	workerDone := make(chan error, 1)
	go func() { workerDone <- worker.Run(workerCtx) }()

	var completed agent.Run
	for {
		completed, err = store.GetRun(ctx, run.ID)
		if err != nil {
			t.Fatal(err)
		}
		if completed.Status.Terminal() {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatal("timed out waiting for Worker")
		case <-time.After(10 * time.Millisecond):
		}
	}
	stopWorker()
	if err := <-workerDone; err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("worker exit: %v", err)
	}
	if completed.Status != agent.RunCompleted || string(completed.Output) != `{"answer": "record-7"}` {
		var output map[string]string
		if err := json.Unmarshal(completed.Output, &output); err != nil || output["answer"] != "record-7" {
			t.Fatalf("completed run = %+v, output error = %v", completed, err)
		}
	}
	timeline, err := store.ListEventsForTenant(ctx, tenantID, run.ID, 0, 100)
	if err != nil {
		t.Fatal(err)
	}
	var called, toolCompleted, planSynchronized, runCompleted, checkpointCreated bool
	for _, committed := range timeline {
		switch committed.Type {
		case event.ToolCalled:
			called = called || committed.CallID == "lookup-1"
		case event.ToolCompleted:
			toolCompleted = toolCompleted || committed.CallID == "lookup-1"
		case event.PlanUpdated:
			var payload struct {
				MutationSource string `json:"mutation_source"`
			}
			if json.Unmarshal(committed.Payload, &payload) == nil && payload.MutationSource == "runtime_plan_sync" {
				planSynchronized = true
			}
		case event.RunCompleted:
			runCompleted = true
		case event.CheckpointCreated:
			checkpointCreated = true
		}
	}
	var checkpointCount, currentStateCount, toolExecutionCount int
	if err := pool.QueryRow(ctx,
		`SELECT count(*) FROM agent_platform.agent_checkpoints WHERE run_id=$1::uuid`, run.ID,
	).Scan(&checkpointCount); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx,
		`SELECT count(*) FROM agent_platform.agent_run_states WHERE run_id=$1::uuid`, run.ID,
	).Scan(&currentStateCount); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx,
		`SELECT count(*) FROM agent_platform.agent_tool_executions WHERE run_id=$1::uuid AND status='succeeded'`, run.ID,
	).Scan(&toolExecutionCount); err != nil {
		t.Fatal(err)
	}
	if !called || !toolCompleted || !planSynchronized || !runCompleted || !checkpointCreated || modelCalls.Load() != 3 ||
		checkpointCount != 1 || currentStateCount != 1 || toolExecutionCount != 1 {
		t.Fatalf("timeline/model calls: called=%v tool_completed=%v plan_synchronized=%v run=%v checkpoint=%v model_calls=%d terminal_checkpoints=%d current_states=%d tools=%d timeline=%s",
			called, toolCompleted, planSynchronized, runCompleted, checkpointCreated, modelCalls.Load(), checkpointCount, currentStateCount, toolExecutionCount, fmt.Sprint(timeline))
	}
}

func TestWorkerNormalizesEncodedReviewerDelegationAndResumesParent(t *testing.T) {
	databaseURL := os.Getenv("TEST_AGENT_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("TEST_AGENT_DATABASE_URL is not set")
	}
	var modelCalls atomic.Int32
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
			_, _ = w.Write([]byte(`{"data":[{"id":"review-model","max_model_len":8192,"capabilities":["chat","tool_calling"]}]}`))
			return
		}
		switch modelCalls.Add(1) {
		case 1:
			_, _ = w.Write([]byte(`{
				"model":"review-model","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"plan-review","type":"function","function":{"name":"update_plan","arguments":"{\"goal\":\"Obtain an independent review\",\"steps\":[{\"id\":\"review\",\"description\":\"Delegate the bounded review\",\"status\":\"in_progress\",\"acceptance_criteria\":[{\"id\":\"review-returned\",\"description\":\"Reviewer returns a result\",\"status\":\"pending\",\"verification\":{\"kind\":\"tool_success\",\"tool\":\"delegate_agent\"}}],\"tool_hints\":[\"delegate_agent\"]}]}"}}]}}],
				"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}
			}`))
		case 2:
			// This is the production failure shape: input is a JSON object encoded
			// one extra time as a string. Runtime normalization must repair it
			// before strict Tool Schema validation.
			_, _ = w.Write([]byte(`{
				"model":"review-model","choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"reviewer-call","type":"function","function":{"name":"delegate_agent","arguments":"{\"target_agent\":\"reviewer-agent\",\"mode\":\"sync\",\"input\":\"{\\\"task\\\":\\\"review db.py\\\",\\\"paths\\\":[\\\"db.py\\\"]}\"}"}}]}}],
				"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}
			}`))
		case 3:
			_, _ = w.Write([]byte(`{"model":"review-model","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"{\"answer\":\"review passed\"}"}}],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}`))
		default:
			_, _ = w.Write([]byte(`{"model":"review-model","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"{\"answer\":\"review completed\"}"}}],"usage":{"prompt_tokens":14,"completion_tokens":2,"total_tokens":16}}`))
		}
	}))
	defer modelServer.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	pool, err := postgres.Open(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	lock, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := lock.Exec(ctx, `SELECT pg_advisory_lock(7188617099)`); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_, _ = lock.Exec(context.Background(), `SELECT pg_advisory_unlock(7188617099)`)
		lock.Release()
	}()
	if err := postgres.Migrate(ctx, pool, "../../migrations"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `TRUNCATE agent_platform.agent_definitions CASCADE`); err != nil {
		t.Fatal(err)
	}
	store := postgres.NewRunStore(pool)
	tenantID := "review-delegation-e2e"
	prompt, err := store.CreatePromptVersion(ctx, resource.CreatePromptVersion{TenantID: tenantID, Key: "review-prompt", Name: "Review Prompt", Content: "Use the provided tools."})
	if err != nil {
		t.Fatal(err)
	}
	toolSet, err := store.CreateToolSetVersion(ctx, resource.CreateToolSetVersion{TenantID: tenantID, Key: "review-tools", Name: "Review Tools", Spec: resource.ToolSetSpec{}})
	if err != nil {
		t.Fatal(err)
	}
	baseSpec := agent.Spec{
		Name: "reviewer-agent", Planning: agent.PlanningPolicy{Policy: agent.PlanningPolicyDisabled},
		Harness:   agent.HarnessSpec{Name: "react-v1", MaxTurns: 1, MaxSteps: 4},
		Model:     agent.ModelBinding{Provider: "openai-compatible", ServiceRef: "review-service", ModelID: "review-model"},
		PromptRef: agent.VersionRef{ID: prompt.ID, Version: strconv.Itoa(prompt.Version)}, ToolSetRef: agent.VersionRef{ID: toolSet.ID, Version: strconv.Itoa(toolSet.Version)},
		InputSchema: json.RawMessage(`{"type":"object"}`), OutputSchema: json.RawMessage(`{"type":"object"}`),
		Context: agent.ContextPolicy{MaxInputTokens: 4096, ReserveOutputTokens: 512},
		Runtime: agent.RuntimePolicy{RunTimeout: 8 * time.Second, ModelTimeout: 2 * time.Second, ToolTimeout: 2 * time.Second, MaxModelCalls: 4, MaxToolCalls: 4},
	}
	reviewerDefinition, err := store.CreateDefinition(ctx, agent.CreateDefinition{TenantID: tenantID, Key: "reviewer-agent", Name: "Reviewer Agent"})
	if err != nil {
		t.Fatal(err)
	}
	reviewerVersion, err := store.CreateVersion(ctx, agent.CreateVersion{TenantID: tenantID, AgentID: reviewerDefinition.ID, Spec: baseSpec})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.ReleaseVersion(ctx, tenantID, reviewerVersion.ID); err != nil {
		t.Fatal(err)
	}
	parentDefinition, err := store.CreateDefinition(ctx, agent.CreateDefinition{TenantID: tenantID, Key: "parent-agent", Name: "Parent Agent"})
	if err != nil {
		t.Fatal(err)
	}
	parentSpec := baseSpec
	parentSpec.Name = "parent-agent"
	parentSpec.Planning = agent.PlanningPolicy{Policy: agent.PlanningPolicyRequired}
	parentSpec.Runtime.MaxModelCalls = 8
	parentSpec.Runtime.MaxToolCalls = 8
	parentSpec.Collaboration = agent.CollaborationPolicy{
		AllowedTargets: []agent.CollaborationTarget{{AgentID: reviewerDefinition.ID, AgentVersionID: reviewerVersion.ID, Modes: []string{"sync"}}},
		MaxDepth:       2, MaxFanOut: 1, MaxChildRuns: 2, ChildTimeout: 5 * time.Second,
		Budget: agent.CollaborationBudget{MaxModelCalls: 8, MaxToolCalls: 8, MaxTokens: 32768},
	}
	parentVersion, err := store.CreateVersion(ctx, agent.CreateVersion{TenantID: tenantID, AgentID: parentDefinition.ID, Spec: parentSpec})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.ReleaseVersion(ctx, tenantID, parentVersion.ID); err != nil {
		t.Fatal(err)
	}
	parentRun, err := store.CreateRun(ctx, agent.CreateRun{TenantID: tenantID, AgentVersionID: parentVersion.ID, Input: json.RawMessage(`{"task":"obtain an independent review"}`)})
	if err != nil {
		t.Fatal(err)
	}
	resolver, err := NewResolver(store, Config{ModelServices: map[string]ModelService{"review-service": {Endpoint: modelServer.URL, Model: "review-model"}}, HTTPClient: modelServer.Client()})
	if err != nil {
		t.Fatal(err)
	}
	processor, err := runtimepkg.NewReActProcessor(resolver)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := runtimepkg.NewWorker(store, processor, runtimepkg.Config{
		WorkerID: "review-e2e-worker", RuntimeVersion: "integration-test/1", ToolContractVersion: "tool-contract/v1", ProtocolVersion: "structured-tool-calls/v1",
		CapabilityHash: "review-e2e", Capabilities: []string{"react-v1", "tool-contract-v1", "structured-tool-calls-v1"},
		PollInterval: 10 * time.Millisecond, LeaseDuration: 3 * time.Second, HeartbeatInterval: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	workerCtx, stopWorker := context.WithCancel(ctx)
	workerDone := make(chan error, 1)
	go func() { workerDone <- worker.Run(workerCtx) }()
	var parent agent.Run
	for {
		parent, err = store.GetRun(ctx, parentRun.ID)
		if err != nil {
			t.Fatal(err)
		}
		if parent.Status.Terminal() {
			break
		}
		select {
		case <-ctx.Done():
			t.Fatalf("timed out waiting for delegated review; parent=%+v model_calls=%d", parent, modelCalls.Load())
		case <-time.After(20 * time.Millisecond):
		}
	}
	stopWorker()
	if err := <-workerDone; err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("worker exit: %v", err)
	}
	if parent.Status != agent.RunCompleted {
		t.Fatalf("parent status=%s error=%v/%v", parent.Status, parent.ErrorCode, parent.ErrorMessage)
	}
	children, err := store.ListChildRunsForTenant(ctx, tenantID, parent.ID)
	if err != nil || len(children) != 1 || children[0].Status != agent.RunCompleted {
		t.Fatalf("reviewer child runs=%+v error=%v", children, err)
	}
	var delegationCount int
	var storedInput json.RawMessage
	if err := pool.QueryRow(ctx, `SELECT count(*),COALESCE(min(input::text),'{}') FROM agent_platform.agent_delegations WHERE parent_run_id=$1::uuid`, parent.ID).Scan(&delegationCount, &storedInput); err != nil {
		t.Fatal(err)
	}
	var input map[string]any
	if err := json.Unmarshal(storedInput, &input); err != nil || delegationCount != 1 || input["task"] != "review db.py" {
		t.Fatalf("delegation count=%d input=%s decoded=%+v error=%v", delegationCount, storedInput, input, err)
	}
	plan, err := store.GetTaskPlanForWorkflow(ctx, tenantID, parent.WorkflowID)
	if err != nil || len(plan.Steps) != 1 || plan.Steps[0].Status != taskplan.StatusCompleted ||
		len(plan.Steps[0].AcceptanceCriteria) != 1 || plan.Steps[0].AcceptanceCriteria[0].Status != taskplan.CriterionPassed {
		t.Fatalf("receipt-backed plan progress was not persisted: plan=%+v error=%v", plan, err)
	}
	var stateRows, rootStateRows, childStateRows, eventEvidenceRows int
	if err := pool.QueryRow(ctx, `
		SELECT count(*),count(*) FILTER (WHERE scope_run_id IS NULL),
		       count(*) FILTER (WHERE scope_run_id=$2::uuid)
		FROM agent_platform.agent_run_states WHERE workflow_id=$1::uuid`, parent.WorkflowID, children[0].ID).
		Scan(&stateRows, &rootStateRows, &childStateRows); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `
		SELECT count(*) FROM agent_platform.agent_evidence_records
		WHERE run_id=$1::uuid AND tool_execution_id IS NULL AND verdict='passed'`, parent.ID).
		Scan(&eventEvidenceRows); err != nil {
		t.Fatal(err)
	}
	if stateRows != 2 || rootStateRows != 1 || childStateRows != 1 || eventEvidenceRows == 0 {
		t.Fatalf("checkpoint/evidence isolation mismatch: states=%d root=%d child=%d event_evidence=%d", stateRows, rootStateRows, childStateRows, eventEvidenceRows)
	}
	if modelCalls.Load() < 4 {
		t.Fatalf("model calls=%d, reviewer or resumed parent did not execute", modelCalls.Load())
	}
}
