package agent

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestSpecValidate(t *testing.T) {
	t.Parallel()

	spec := validSpec()
	if err := spec.Validate(); err != nil {
		t.Fatalf("valid spec rejected: %v", err)
	}

	spec.Model.ServiceRef = ""
	if err := spec.Validate(); err == nil {
		t.Fatal("spec without model service must be rejected")
	}
}

func TestApprovalPolicyRoundTripsSandboxCommandGrant(t *testing.T) {
	input := ApprovalPolicy{RequireFor: []string{"HIGH_RISK"}, AutoApproveSandboxCommand: true, ExpiresIn: 30 * time.Minute}
	encoded, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ApprovalPolicy
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !decoded.AutoApproveSandboxCommand || decoded.ExpiresIn != 30*time.Minute || len(decoded.RequireFor) != 1 || decoded.RequireFor[0] != "HIGH_RISK" {
		t.Fatalf("approval policy did not round trip: encoded=%s decoded=%+v", encoded, decoded)
	}
}

func TestSpecValidateRejectsUnsupportedHarnessAndCapabilities(t *testing.T) {
	t.Parallel()
	spec := validSpec()
	spec.Harness.Name = "imaginary-v9"
	if err := spec.Validate(); err == nil {
		t.Fatal("unsupported harness must be rejected")
	}
	spec = validSpec()
	spec.Harness.MaxTurns = 2
	if err := spec.Validate(); err != nil {
		t.Fatalf("react-v1 multi-turn continuation rejected: %v", err)
	}
}

func TestSpecValidateRejectsInvalidJSONSchema(t *testing.T) {
	t.Parallel()
	spec := validSpec()
	spec.InputSchema = json.RawMessage(`{"type":"not-a-valid-type"}`)
	if err := spec.Validate(); err == nil {
		t.Fatal("invalid JSON Schema must be rejected")
	}
}

func TestSpecValidateModelSelectionPolicies(t *testing.T) {
	t.Parallel()
	auto := validSpec()
	auto.Model.SelectionPolicy = ModelSelectionAuto
	auto.Model.ModelID = ""
	auto.Model.ModelCandidates = []string{"model-a", "model-b"}
	if err := auto.Validate(); err != nil {
		t.Fatalf("auto model selection rejected: %v", err)
	}
	invalid := validSpec()
	invalid.Model.SelectionPolicy = "random"
	if err := invalid.Validate(); err == nil {
		t.Fatal("unsupported selection policy must be rejected")
	}
}

func TestSpecValidateMemoryPolicy(t *testing.T) {
	t.Parallel()
	spec := validSpec()
	spec.Memory = MemoryPolicy{Enabled: true, ReadScopes: []string{MemoryScopeAgent, MemoryScopeSession}, MaxRecall: 5, MinimumScore: 0.2}
	spec.Context.MemoryTokens = 512
	if err := spec.Validate(); err != nil {
		t.Fatalf("valid memory policy rejected: %v", err)
	}
	spec.Context.MemoryTokens = 0
	if err := spec.Validate(); err == nil {
		t.Fatal("enabled memory without a token budget must be rejected")
	}
	spec.Context.MemoryTokens = 512
	spec.Memory.ReadScopes = []string{"global"}
	if err := spec.Validate(); err == nil {
		t.Fatal("unsupported memory scope must be rejected")
	}
}

func TestIdentityCompileAndValidate(t *testing.T) {
	t.Parallel()
	spec := validSpec()
	spec.Identity = Identity{
		DisplayName: "Release Reviewer", Role: "Review releases", Goal: "Produce an evidence-backed gate decision",
		Responsibilities: []string{"Inspect health", "Run benchmarks"}, Boundaries: []string{"Do not change traffic"}, CommunicationStyle: "Concise",
	}
	if err := spec.Validate(); err != nil {
		t.Fatalf("valid identity rejected: %v", err)
	}
	compiled := CompileIdentity(spec.Identity)
	for _, expected := range []string{"# Agent Identity", "Role: Review releases", "- Inspect health", "- Do not change traffic", "Communication style: Concise"} {
		if !bytes.Contains([]byte(compiled), []byte(expected)) {
			t.Fatalf("compiled identity missing %q: %s", expected, compiled)
		}
	}
	if IdentityDigest(spec.Identity) == "" || IdentityDigest(spec.Identity) != IdentityDigest(spec.Identity) {
		t.Fatal("identity digest must be stable")
	}
	spec.Identity.Goal = ""
	if err := spec.Validate(); err == nil {
		t.Fatal("configured identity without goal must be rejected")
	}
}

func TestRuntimePolicyDurationJSON(t *testing.T) {
	t.Parallel()
	var policy RuntimePolicy
	if err := json.Unmarshal([]byte(`{"run_timeout":"5m","model_timeout":"30s","tool_timeout":"2s","max_model_calls":4,"max_tool_calls":8}`), &policy); err != nil {
		t.Fatal(err)
	}
	if policy.RunTimeout != 5*time.Minute || policy.ToolTimeout != 2*time.Second {
		t.Fatalf("policy = %+v", policy)
	}
	encoded, err := json.Marshal(policy)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded, []byte(`"run_timeout":"5m0s"`)) {
		t.Fatalf("encoded policy = %s", encoded)
	}
}

func TestRunStatusTransitions(t *testing.T) {
	t.Parallel()

	for _, transition := range [][2]RunStatus{
		{RunQueued, RunRunning},
		{RunRunning, RunWaitingApproval},
		{RunWaitingApproval, RunRunning},
		{RunRunning, RunCompleted},
	} {
		if err := ValidateTransition(transition[0], transition[1]); err != nil {
			t.Fatalf("valid transition rejected: %v", err)
		}
	}
	if err := ValidateTransition(RunCompleted, RunRunning); err == nil {
		t.Fatal("terminal run must not return to running")
	}
	if err := ValidateTransition(RunFailed, RunQueued); err == nil {
		t.Fatal("failed run must be retried as a new run")
	}
}

func validSpec() Spec {
	return Spec{
		Name:         "customer-support",
		Harness:      HarnessSpec{Name: "react-v1", MaxTurns: 1, MaxSteps: 8},
		Model:        ModelBinding{Provider: "openai", ServiceRef: "qwen-customer", Capability: "tool_calling", ModelID: "qwen-customer-v1"},
		PromptRef:    VersionRef{ID: "customer-prompt", Version: "1"},
		ToolSetRef:   VersionRef{ID: "customer-tools", Version: "1"},
		InputSchema:  json.RawMessage(`{"type":"object"}`),
		OutputSchema: json.RawMessage(`{"type":"object"}`),
		Context:      ContextPolicy{MaxInputTokens: 8192, ReserveOutputTokens: 1024},
		Runtime: RuntimePolicy{
			RunTimeout: 5 * time.Minute, ModelTimeout: time.Minute, ToolTimeout: 30 * time.Second,
			MaxModelCalls: 8, MaxToolCalls: 16,
		},
	}
}

func TestPlanningPolicyDefaultsToAutoAndRejectsUnknownPolicy(t *testing.T) {
	spec := validSpec()
	if got := spec.Planning.EffectivePolicy(); got != PlanningPolicyAuto {
		t.Fatalf("effective planning policy = %q, want %q", got, PlanningPolicyAuto)
	}
	spec.Planning.Policy = "sometimes"
	if err := spec.Validate(); err == nil || !strings.Contains(err.Error(), "planning.policy") {
		t.Fatalf("expected planning policy validation error, got %v", err)
	}
}

func TestContextWindowMayBeInheritedFromResolvedModel(t *testing.T) {
	spec := validSpec()
	spec.Context.MaxInputTokens = 0
	if err := spec.Validate(); err != nil {
		t.Fatalf("model-inherited context window was rejected: %v", err)
	}
	spec.Context.MaxInputTokens = -1
	if err := spec.Validate(); err == nil || !strings.Contains(err.Error(), "max_input_tokens") {
		t.Fatalf("negative context cap must be rejected, got %v", err)
	}
}
