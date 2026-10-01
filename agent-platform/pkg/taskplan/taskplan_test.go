package taskplan

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestNextDecisionUsesStructuredTestFacts(t *testing.T) {
	plan := Plan{Steps: []Step{
		{ID: "build", Status: StatusCompleted, State: NodeState{Tests: []NodeTestResult{{Name: "compile", Status: "failed"}}}},
		{ID: "verify", Status: StatusPending, DependsOn: []string{"build"}},
	}}
	if node, action := plan.NextDecision(); node != "build" || action != NodeActionRetry {
		t.Fatalf("failed test must route to retry, got node=%q action=%q", node, action)
	}
}

func TestNextDecisionRunsOnlyDependencyReadyNode(t *testing.T) {
	plan := Plan{Steps: []Step{
		{ID: "build", Status: StatusCompleted, State: NodeState{Tests: []NodeTestResult{{Name: "compile", Status: "passed"}}}},
		{ID: "verify", Status: StatusPending, DependsOn: []string{"build"}},
	}}
	if node, action := plan.NextDecision(); node != "verify" || action != NodeActionRun {
		t.Fatalf("ready dependent node must run, got node=%q action=%q", node, action)
	}
}

func TestDeriveGraphStateExposesDeterministicRetryAndReadyNodes(t *testing.T) {
	plan := Plan{Steps: []Step{
		{ID: "build", Status: StatusCompleted, State: NodeState{Tests: []NodeTestResult{{Name: "compile", Status: "failed"}}}},
		{ID: "verify", Status: StatusPending, DependsOn: []string{"build"}},
	}}
	state := plan.DeriveGraphState()
	if state.Status != "retry_required" || state.RetryNodeID != "build" || state.NextAction != NodeActionRetry {
		t.Fatalf("graph state = %#v", state)
	}
	if len(state.ReadyNodeIDs) != 0 {
		t.Fatalf("failed dependency must not be ready: %#v", state.ReadyNodeIDs)
	}
}

func TestNormalizePlanComputesGraphStateForAPIProjection(t *testing.T) {
	plan, err := NormalizePlan(Plan{Goal: "build", Steps: []Step{{ID: "build", Status: StatusCompleted}}})
	if err != nil {
		t.Fatal(err)
	}
	if plan.GraphState.Status != "completed" || plan.GraphState.NextAction != NodeActionComplete {
		t.Fatalf("normalized graph state = %#v", plan.GraphState)
	}
}

func TestPlanReadyNodesReturnsIndependentPendingNodes(t *testing.T) {
	plan := Plan{Steps: []Step{
		{ID: "inspect", Status: StatusCompleted},
		{ID: "backend", Status: StatusPending, DependsOn: []string{"inspect"}},
		{ID: "frontend", Status: StatusPending, DependsOn: []string{"inspect"}},
		{ID: "release", Status: StatusPending, DependsOn: []string{"backend", "frontend"}},
	}}
	ready := plan.ReadyNodes()
	if len(ready) != 2 || ready[0].ID != "backend" || ready[1].ID != "frontend" {
		t.Fatalf("ready nodes = %#v, want backend and frontend", ready)
	}
}

func TestReconcileStepsActivatesFirstReadyStep(t *testing.T) {
	steps := ReconcileSteps(nil, []Step{
		{ID: "inspect", Description: "inspect", Status: StatusPending},
		{ID: "report", Description: "report", Status: StatusPending, DependsOn: []string{"inspect"}},
	})
	if steps[0].Status != StatusInProgress || steps[1].Status != StatusPending {
		t.Fatalf("steps = %#v", steps)
	}
}

func TestReconcileStepsPreservesProgressAndAdvances(t *testing.T) {
	previous := []Step{
		{ID: "inspect", Description: "inspect", Status: StatusInProgress, Result: "read 20 files"},
		{ID: "report", Description: "report", Status: StatusPending, DependsOn: []string{"inspect"}},
	}
	steps := ReconcileSteps(previous, []Step{
		{ID: "inspect", Description: "inspect", Status: StatusCompleted},
		{ID: "report", Description: "report", Status: StatusPending, DependsOn: []string{"inspect"}},
	})
	if steps[0].Status != StatusCompleted || steps[0].Result != "read 20 files" {
		t.Fatalf("completed step = %#v", steps[0])
	}
	if steps[1].Status != StatusInProgress {
		t.Fatalf("next step = %#v", steps[1])
	}
}

func TestReconcileStepsTreatsUpdateAsFullRevision(t *testing.T) {
	previous := []Step{
		{ID: "explore", Description: "explore", Status: StatusInProgress},
		{ID: "build", Description: "build", Status: StatusPending},
		{ID: "verify", Description: "verify", Status: StatusPending, DependsOn: []string{"build"}},
	}
	// update_plan is a complete revision. Callers use update_plan_step for a
	// compact delta, so omitted nodes must not silently accumulate forever.
	steps := ReconcileSteps(previous, []Step{
		{ID: "explore", Description: "explore", Status: StatusCompleted},
	})
	if len(steps) != 1 || steps[0].ID != "explore" || steps[0].Status != StatusCompleted {
		t.Fatalf("full revision retained omitted nodes: %#v", steps)
	}
	if (Plan{Steps: steps}).HasOpenWork() {
		t.Fatalf("replacement plan unexpectedly remains open: %#v", steps)
	}
}

func TestReconcileStepsDoesNotReopenCompletedStep(t *testing.T) {
	steps := ReconcileSteps(
		[]Step{{ID: "done", Description: "done", Status: StatusCompleted}},
		[]Step{{ID: "done", Description: "done", Status: StatusPending}},
	)
	if steps[0].Status != StatusCompleted {
		t.Fatalf("status = %q", steps[0].Status)
	}
	if (Plan{Steps: steps}).HasOpenWork() {
		t.Fatal("completed plan reported open work")
	}
}

func TestReconcileStepsExplicitlyReopensCompletedStepAfterEvidenceReset(t *testing.T) {
	previous := []Step{{
		ID: "build", Description: "build", Status: StatusCompleted, Result: "old result",
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "run", Description: "runs", Status: CriterionPassed, Evidence: "old", EvidenceCallIDs: []string{"call-old"}}},
	}}
	steps := ReconcileSteps(previous, []Step{{
		ID: "build", Description: "build", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "run", Description: "runs", Status: CriterionPending}},
	}})
	if steps[0].Status != StatusInProgress || steps[0].AcceptanceCriteria[0].Status != CriterionPending {
		t.Fatalf("step was not reopened: %#v", steps[0])
	}
	if steps[0].Result != "" || len(steps[0].AcceptanceCriteria[0].EvidenceCallIDs) != 0 {
		t.Fatalf("stale completion data survived reset: %#v", steps[0])
	}
}

func TestReconcileStepsDoesNotReopenCompletedStepWithStaleEvidence(t *testing.T) {
	previous := []Step{{
		ID: "build", Description: "build", Status: StatusCompleted,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "run", Description: "runs", Status: CriterionPassed, Evidence: "old", EvidenceCallIDs: []string{"call-old"}}},
	}}
	steps := ReconcileSteps(previous, []Step{{
		ID: "build", Description: "build", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "run", Description: "runs", Status: CriterionPassed, Evidence: "old", EvidenceCallIDs: []string{"call-old"}}},
	}})
	if steps[0].Status != StatusCompleted || steps[0].AcceptanceCriteria[0].Status != CriterionPassed {
		t.Fatalf("stale plan echo reopened terminal work: %#v", steps[0])
	}
}

func TestPlanKeepsOpenAcceptanceCriterionAndPreservesEvidence(t *testing.T) {
	previous := []Step{{
		ID: "build", Description: "build", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "syntax", Description: "syntax passes", Status: CriterionPassed, Evidence: "python3 -m py_compile game.py: exit 0", EvidenceCallIDs: []string{"compile-1"}}},
	}}
	steps := ReconcileSteps(previous, []Step{{
		ID: "build", Description: "build", Status: StatusCompleted,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "syntax", Description: "syntax passes", Status: CriterionPending}},
	}})
	criterion := steps[0].AcceptanceCriteria[0]
	if criterion.Status != CriterionPassed || criterion.Evidence == "" {
		t.Fatalf("criterion = %#v", criterion)
	}
	if (Plan{Steps: steps}).HasOpenWork() {
		t.Fatal("verified plan reported open work")
	}
}

func TestCompletedStepSeparatesAdvisoryFromRequiredVerification(t *testing.T) {
	update := Update{Goal: "build", Steps: []Step{{
		ID: "build", Description: "build", Status: StatusCompleted,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "syntax", Description: "syntax passes", Status: CriterionPending}},
	}}}
	update, err := NormalizeUpdate(update)
	if err != nil {
		t.Fatal(err)
	}
	if err := update.Validate(); err != nil {
		t.Fatalf("advisory verification blocked completed work: %v", err)
	}
	update.Steps[0].AcceptanceCriteria[0].Enforcement = EnforcementRequired
	if err := update.Validate(); err == nil {
		t.Fatal("expected completed step with pending criterion to be rejected")
	}
}

func TestNormalizeStructuredPyCompileUsesInvocationSubject(t *testing.T) {
	update, err := NormalizeUpdate(Update{Goal: "game", Steps: []Step{{
		ID: "write", Description: "write game", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{
			ID: "syntax", Description: "snake_game.py syntax compiles", Status: CriterionPending,
			Verification: VerificationSpec{Kind: "command_exit_zero", Target: "smoke_snake.py", Arguments: json.RawMessage(`{"command":"python3","args":["-m","py_compile","snake_game.py"]}`)},
		}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	criterion := update.Steps[0].AcceptanceCriteria[0]
	if criterion.Verification.Kind != "python_syntax" || criterion.Verification.Target != "snake_game.py" || len(criterion.Verification.Arguments) != 0 {
		t.Fatalf("compiled verification = %#v", criterion.Verification)
	}
	if criterion.Enforcement != EnforcementAdvisory || criterion.Origin != OriginAgentInferred {
		t.Fatalf("criterion policy = %#v", criterion)
	}
}

func TestPassedCriterionRequiresEvidence(t *testing.T) {
	update := Update{Goal: "build", Steps: []Step{{
		ID: "build", Description: "build", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "syntax", Description: "syntax passes", Status: CriterionPassed}},
	}}}
	if err := update.Validate(); err == nil {
		t.Fatal("expected passed criterion without evidence to be rejected")
	}
}

func TestPassedCriterionRequiresSuccessfulToolReceipt(t *testing.T) {
	update := Update{Goal: "build", Steps: []Step{{
		ID: "build", Description: "build", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "syntax", Description: "syntax passes", Status: CriterionPassed, Evidence: "claimed without receipt"}},
	}}}
	if err := update.Validate(); err == nil {
		t.Fatal("expected passed criterion without evidence_call_ids to be rejected")
	}
}

func TestVerificationSpecRejectsUnknownKindAndMissingTarget(t *testing.T) {
	if err := (VerificationSpec{Kind: "made_up", Target: "game.py"}).Validate(); err == nil {
		t.Fatal("unknown verification kind was accepted")
	}
	if err := (VerificationSpec{Kind: "python_syntax"}).Validate(); err == nil {
		t.Fatal("targetless syntax verification was accepted")
	}
	if err := (VerificationSpec{Kind: "file_contains", Target: "game.py"}).Validate(); err == nil {
		t.Fatal("file_contains without match text was accepted")
	}
}

func TestVerificationSpecEvidenceTools(t *testing.T) {
	tests := []struct {
		kind string
		want []string
	}{
		{kind: "python_syntax", want: []string{"run_command"}},
		{kind: "file_contains", want: []string{"read_file"}},
		{kind: "list_nonempty", want: []string{"list_files"}},
		{kind: "tool_success", want: nil},
	}
	for _, test := range tests {
		if got := (VerificationSpec{Kind: test.kind}).EvidenceTools(); !reflect.DeepEqual(got, test.want) {
			t.Fatalf("kind %s tools = %#v, want %#v", test.kind, got, test.want)
		}
	}
}

func TestVerificationCandidateMatchingSeparatesSubjectFromVerdict(t *testing.T) {
	fileSpec := VerificationSpec{Kind: "file_contains", Target: "pkg/__init__.py", Match: "version"}
	unrelated := EvidenceReceipt{Tool: "read_file", Arguments: json.RawMessage(`{"path":"pkg/db.py"}`), Result: json.RawMessage(`{"content":"version"}`)}
	if VerificationCandidateMatches(fileSpec, unrelated) {
		t.Fatal("unrelated file receipt entered the formal verification candidate set")
	}
	exactButFailing := EvidenceReceipt{Tool: "read_file", Arguments: json.RawMessage(`{"path":"./pkg/__init__.py"}`), Result: json.RawMessage(`{"content":"missing marker"}`)}
	if !VerificationCandidateMatches(fileSpec, exactButFailing) {
		t.Fatal("exact file receipt was rejected before verdict evaluation")
	}
	if VerifyEvidence(fileSpec, []EvidenceReceipt{exactButFailing}) == nil {
		t.Fatal("candidate matching incorrectly treated a failing result as passed")
	}
	promoted := EvidenceReceipt{Tool: "promote_file", Arguments: json.RawMessage(`{"source_path":"stage.tmp","target_path":"pkg/__init__.py"}`)}
	if !VerificationCandidateMatches(VerificationSpec{Kind: "file_exists", Target: "pkg/__init__.py"}, promoted) {
		t.Fatal("promote_file target_path was not recognized as exact file evidence")
	}
}

func TestCommandCandidateMatchingIgnoresVerdictButRequiresExactInvocation(t *testing.T) {
	spec := VerificationSpec{Kind: "command_exit_zero", Arguments: json.RawMessage(`{"command":"python3","args":["smoke.py"]}`)}
	exactFailure := EvidenceReceipt{Tool: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["smoke.py"]}`), Result: json.RawMessage(`{"exit_code":1}`)}
	if !VerificationCandidateMatches(spec, exactFailure) {
		t.Fatal("exact command failure must become one formal verification attempt")
	}
	if VerifyEvidence(spec, []EvidenceReceipt{exactFailure}) == nil {
		t.Fatal("non-zero command result was accepted")
	}
	unrelated := EvidenceReceipt{Tool: "run_command", Arguments: json.RawMessage(`{"command":"python3","args":["other.py"]}`), Result: json.RawMessage(`{"exit_code":0}`)}
	if VerificationCandidateMatches(spec, unrelated) {
		t.Fatal("unrelated successful command entered the formal attempt")
	}
}

func TestNormalizeUpdateConvertsLegacyPyCompileSyntaxContract(t *testing.T) {
	update, err := NormalizeUpdate(Update{Goal: "build", Steps: []Step{{
		ID: "verify", Description: "verify", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{
			ID: "syntax", Description: "snake.py 语法正确，无错误", Status: CriterionPending,
			Verification: VerificationSpec{Kind: "command_exit_zero", Target: "python -m py_compile snake.py"},
		}},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	got := update.Steps[0].AcceptanceCriteria[0].Verification
	if got.Kind != "python_syntax" || got.Target != "snake.py" {
		t.Fatalf("verification = %#v", got)
	}
}

func TestNormalizeUpdateSeparatesInvalidAdvisoryFromRequiredRuntimeProof(t *testing.T) {
	input := Update{Goal: "build", Steps: []Step{{
		ID: "verify", Description: "verify", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{
			ID: "runs", Description: "游戏可以直接运行", Status: CriterionPending,
			Verification: VerificationSpec{Kind: "command_exit_zero", Target: "python -m py_compile snake.py"},
		}},
	}}}
	update, err := NormalizeUpdate(input)
	if err != nil {
		t.Fatal(err)
	}
	criterion := update.Steps[0].AcceptanceCriteria[0]
	if criterion.Status != CriterionInvalid || criterion.VerificationReason != VerificationReasonSpecInvalid {
		t.Fatalf("invalid advisory criterion was not classified: %#v", criterion)
	}
	input.Steps[0].AcceptanceCriteria[0].Enforcement = EnforcementRequired
	if _, err := NormalizeUpdate(input); err == nil {
		t.Fatal("invalid required runtime proof was accepted")
	}
}

func TestGenericToolReceiptProviderUsesArgumentSubsetAndAssertions(t *testing.T) {
	spec := VerificationSpec{
		Kind: "tool_receipt", Tool: "build_project",
		Arguments: json.RawMessage(`{"language":"cpp"}`),
		Assertions: []VerificationAssertion{
			{Path: "exit_code", Operator: "equals", Value: json.RawMessage(`0`)},
			{Path: "artifact", Operator: "nonempty"},
		},
	}
	if err := spec.Validate(); err != nil {
		t.Fatal(err)
	}
	err := VerifyEvidence(spec, []EvidenceReceipt{{
		Tool:      "build_project",
		Arguments: json.RawMessage(`{"language":"cpp","target":"game"}`),
		Result:    json.RawMessage(`{"exit_code":0,"artifact":"build/game"}`),
	}})
	if err != nil {
		t.Fatalf("generic receipt rejected: %v", err)
	}
}

func TestToolSuccessProviderDoesNotAcceptUnrelatedReceipt(t *testing.T) {
	spec := VerificationSpec{Kind: "tool_success", Tool: "read_file", Arguments: json.RawMessage(`{"path":"db.py"}`)}
	if err := spec.Validate(); err != nil {
		t.Fatal(err)
	}
	if VerificationCandidateMatches(spec, EvidenceReceipt{Tool: "list_files", Arguments: json.RawMessage(`{"path":"."}`)}) {
		t.Fatal("unrelated tool was accepted")
	}
	if VerificationCandidateMatches(spec, EvidenceReceipt{Tool: "read_file", Arguments: json.RawMessage(`{"path":"models.py"}`)}) {
		t.Fatal("unrelated arguments were accepted")
	}
	exact := EvidenceReceipt{Tool: "read_file", Arguments: json.RawMessage(`{"path":"db.py","line_count":80}`)}
	if !VerificationCandidateMatches(spec, exact) || VerifyEvidence(spec, []EvidenceReceipt{exact}) != nil {
		t.Fatal("matching successful receipt was rejected")
	}
}

func TestCommandExitZeroMatchesStructuredArguments(t *testing.T) {
	spec := VerificationSpec{Kind: "command_exit_zero", Arguments: json.RawMessage(`{"command":"python3","args":["smoke_run.py"]}`)}
	receipts := []EvidenceReceipt{{
		Tool:      "run_command",
		Arguments: json.RawMessage(`{"command":"python3","args":["smoke_run.py"],"timeout_seconds":5}`),
		Result:    json.RawMessage(`{"exit_code":0,"timed_out":false}`),
	}}
	if err := VerifyEvidence(spec, receipts); err != nil {
		t.Fatalf("structured command receipt did not satisfy criterion: %v", err)
	}
}

func TestTestPassRecognizesFrameworkAndStandaloneTestEntrypoints(t *testing.T) {
	tests := []struct {
		name      string
		arguments string
		want      bool
	}{
		{name: "unittest", arguments: `{"command":"python3","args":["-m","unittest","discover"]}`, want: true},
		{name: "pytest module", arguments: `{"command":"python3","args":["-m","pytest","-q"]}`, want: true},
		{name: "go test", arguments: `{"command":"go","args":["test","./..."]}`, want: true},
		{name: "standalone smoke", arguments: `{"command":"python3","args":["tools/smoke_e2e.py"]}`, want: true},
		{name: "standalone test", arguments: `{"command":"python3","args":["tests/test_smoke.py"]}`, want: true},
		{name: "ordinary script", arguments: `{"command":"python3","args":["app.py"]}`, want: false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := VerificationSpec{Kind: "test_pass", Arguments: json.RawMessage(test.arguments)}
			receipt := EvidenceReceipt{Tool: "run_command", Arguments: json.RawMessage(test.arguments), Result: json.RawMessage(`{"exit_code":0}`)}
			err := VerifyEvidence(spec, []EvidenceReceipt{receipt})
			if (err == nil) != test.want {
				t.Fatalf("VerifyEvidence() error=%v want_success=%v", err, test.want)
			}
		})
	}
}

func TestNormalizeRunCommandAssertionTypes(t *testing.T) {
	spec, err := NormalizeCriterionVerification("smoke", VerificationSpec{
		Kind: "tool_receipt", Tool: "run_command",
		Assertions: []VerificationAssertion{{Path: "exit_code", Operator: "equals", Value: json.RawMessage(`"0"`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(spec.Assertions[0].Value) != "0" {
		t.Fatalf("exit_code assertion was not normalized to a number: %s", spec.Assertions[0].Value)
	}
}

func TestGenericToolReceiptProviderRejectsWorkspaceMutationAsOutcome(t *testing.T) {
	spec := VerificationSpec{
		Kind: "tool_receipt", Tool: "write_file", Arguments: json.RawMessage(`{"path":"snake_game.py"}`),
		Assertions: []VerificationAssertion{{Path: "artifact", Operator: "nonempty"}},
	}
	if err := spec.Validate(); err == nil || !strings.Contains(err.Error(), "not delivery outcomes") {
		t.Fatalf("workspace mutation receipt validation error = %v", err)
	}
}

func TestNormalizePlanReopensCompletedDependentOfOpenStep(t *testing.T) {
	plan, err := NormalizePlan(Plan{Goal: "build", Steps: []Step{
		{ID: "write", Description: "write", Status: StatusInProgress, AcceptanceCriteria: []AcceptanceCriterion{{ID: "exists", Description: "exists", Status: CriterionPending, Verification: VerificationSpec{Kind: "file_exists", Target: "game.py"}}}},
		{ID: "verify", Description: "verify", Status: StatusCompleted, DependsOn: []string{"write"}, Result: "old", AcceptanceCriteria: []AcceptanceCriterion{{ID: "syntax", Description: "syntax", Status: CriterionPassed, Evidence: "old", EvidenceCallIDs: []string{"old-call"}, Verification: VerificationSpec{Kind: "python_syntax", Target: "game.py"}}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	dependent := plan.Steps[1]
	if dependent.Status != StatusPending || dependent.Result != "" || dependent.AcceptanceCriteria[0].Status != CriterionPending || len(dependent.AcceptanceCriteria[0].EvidenceCallIDs) != 0 {
		t.Fatalf("dependent progress was not invalidated: %#v", dependent)
	}
}

func TestBlockedPlanRemainsOpen(t *testing.T) {
	plan := Plan{Steps: []Step{{
		ID: "build", Description: "build", Status: StatusBlocked,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "runtime", Description: "runtime works", Status: CriterionPending}},
	}}}
	if !plan.HasOpenWork() {
		t.Fatal("blocked plan must not pass the Run completion guard")
	}
}

func TestStalePlanRequiresDeterministicRetry(t *testing.T) {
	plan := Plan{Goal: "build", Steps: []Step{{
		ID: "verify", Description: "verify", Status: StatusStale,
		AcceptanceCriteria: []AcceptanceCriterion{{
			ID: "syntax", Description: "syntax", Status: CriterionStale,
			Verification: VerificationSpec{Kind: "python_syntax", Target: "main.py"},
		}},
	}}}
	if !plan.HasOpenWork() {
		t.Fatal("stale plan must remain open")
	}
	if nodeID, action := plan.NextDecision(); nodeID != "verify" || action != NodeActionRetry {
		t.Fatalf("next decision = %q/%q, want verify/retry", nodeID, action)
	}
	update, err := ApplyStepUpdate(plan, StepUpdate{StepID: "verify", Status: StatusInProgress})
	if err != nil {
		t.Fatal(err)
	}
	if update.Steps[0].Status != StatusInProgress || update.Steps[0].AcceptanceCriteria[0].Status != CriterionStale {
		t.Fatalf("reopened stale step = %#v", update.Steps[0])
	}
}

func TestApplyStepUpdateAdvancesNextTodo(t *testing.T) {
	plan := Plan{Goal: "build", Steps: []Step{
		{ID: "inspect", Description: "inspect", Status: StatusInProgress, AcceptanceCriteria: []AcceptanceCriterion{{ID: "seen", Description: "seen", Status: CriterionPending}}},
		{ID: "build", Description: "build", Status: StatusPending, DependsOn: []string{"inspect"}, AcceptanceCriteria: []AcceptanceCriterion{{ID: "exists", Description: "exists", Status: CriterionPending}}},
	}}
	update, err := ApplyStepUpdate(plan, StepUpdate{
		StepID: "inspect", Status: StatusCompleted, Result: "listed workspace",
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "seen", Status: CriterionPassed, Evidence: "list_files succeeded", EvidenceCallIDs: []string{"list-1"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if update.Steps[0].Status != StatusCompleted || update.Steps[1].Status != StatusInProgress {
		t.Fatalf("steps=%#v", update.Steps)
	}
	if update.Steps[0].AcceptanceCriteria[0].Evidence != "list_files succeeded" {
		t.Fatalf("criterion=%#v", update.Steps[0].AcceptanceCriteria[0])
	}
}

func TestApplyStepUpdateRejectsLaterPendingTodo(t *testing.T) {
	plan := Plan{Goal: "build", Steps: []Step{
		{ID: "inspect", Description: "inspect", Status: StatusInProgress},
		{ID: "verify", Description: "verify", Status: StatusPending},
	}}
	if _, err := ApplyStepUpdate(plan, StepUpdate{StepID: "verify", Status: StatusCompleted}); err == nil {
		t.Fatal("expected out-of-order pending Todo update to be rejected")
	}
}

func TestApplyStepUpdateCanRepairActiveToolHints(t *testing.T) {
	plan := Plan{Goal: "build", Steps: []Step{{
		ID: "inspect", Description: "inspect", Status: StatusInProgress,
		ToolHints:          []string{"run_command"},
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "seen", Description: "seen", Status: CriterionPending}},
	}}}
	hints := []string{"write_file", "run_command"}
	update, err := ApplyStepUpdate(plan, StepUpdate{StepID: "inspect", Status: StatusInProgress, ToolHints: &hints})
	if err != nil {
		t.Fatal(err)
	}
	if got := update.Steps[0].ToolHints; len(got) != 2 || got[0] != "write_file" || got[1] != "run_command" {
		t.Fatalf("tool hints = %#v", got)
	}
}

func TestApplyStepUpdateReopensFailedTerminalCriterion(t *testing.T) {
	plan := Plan{Goal: "build", Steps: []Step{
		{ID: "write", Description: "write", Status: StatusCompleted, Result: "old", AcceptanceCriteria: []AcceptanceCriterion{{ID: "exists", Description: "exists", Status: CriterionPassed, Evidence: "old", EvidenceCallIDs: []string{"write-1"}, Verification: VerificationSpec{Kind: "file_exists", Target: "game.py"}}}},
		{ID: "verify", Description: "verify", Status: StatusCompleted, DependsOn: []string{"write"}, Result: "old verify", AcceptanceCriteria: []AcceptanceCriterion{{ID: "syntax", Description: "syntax", Status: CriterionPassed, Evidence: "old", EvidenceCallIDs: []string{"compile-1"}, Verification: VerificationSpec{Kind: "python_syntax", Target: "game.py"}}}},
	}}
	update, err := ApplyStepUpdate(plan, StepUpdate{
		StepID: "write", Status: StatusInProgress,
		AcceptanceCriteria: []AcceptanceCriterion{{ID: "exists", Status: CriterionPending}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if update.Steps[0].Status != StatusInProgress || update.Steps[0].Result != "" || update.Steps[0].AcceptanceCriteria[0].Status != CriterionPending {
		t.Fatalf("reopened step = %#v", update.Steps[0])
	}
	if update.Steps[1].Status != StatusPending || update.Steps[1].AcceptanceCriteria[0].Evidence != "" {
		t.Fatalf("dependent step retained stale completion: %#v", update.Steps[1])
	}
}

func TestApplyCriterionRevisionRepairsOnlyNamedContract(t *testing.T) {
	plan := Plan{Goal: "build", Steps: []Step{{ID: "verify", Description: "verify", Status: StatusInProgress, AcceptanceCriteria: []AcceptanceCriterion{
		{ID: "syntax", Description: "syntax", Status: CriterionInvalid, Enforcement: EnforcementAdvisory, Origin: OriginAgentInferred, Verification: VerificationSpec{Kind: "python_syntax"}},
		{ID: "exists", Description: "exists", Status: CriterionPassed, Evidence: "kept", EvidenceCallIDs: []string{"call-kept"}, Verification: VerificationSpec{Kind: "file_exists", Target: "game.py"}},
	}}}}
	update, err := ApplyCriterionRevision(plan, CriterionRevision{StepID: "verify", CriterionID: "syntax", Action: "replace", Reason: "target was missing", Verification: VerificationSpec{Kind: "python_syntax", Target: "game.py"}})
	if err != nil {
		t.Fatal(err)
	}
	repaired, untouched := update.Steps[0].AcceptanceCriteria[0], update.Steps[0].AcceptanceCriteria[1]
	if repaired.Status != CriterionPending || repaired.Verification.Target != "game.py" || repaired.Enforcement != EnforcementAdvisory || repaired.Origin != OriginAgentInferred {
		t.Fatalf("repaired criterion = %+v", repaired)
	}
	if untouched.Status != CriterionPassed || untouched.Evidence != "kept" || len(untouched.EvidenceCallIDs) != 1 {
		t.Fatalf("unrelated criterion changed = %+v", untouched)
	}
}

func TestApplyCriterionRevisionCannotSkipRequired(t *testing.T) {
	plan := Plan{Goal: "release", Steps: []Step{{ID: "gate", Description: "gate", Status: StatusInProgress, AcceptanceCriteria: []AcceptanceCriterion{{ID: "health", Description: "health", Status: CriterionUnsupported, Enforcement: EnforcementReleaseGate, Verification: VerificationSpec{Kind: "unknown"}}}}}}
	if _, err := ApplyCriterionRevision(plan, CriterionRevision{StepID: "gate", CriterionID: "health", Action: "skip_advisory", Reason: "not available"}); err == nil {
		t.Fatal("expected release gate skip to be rejected")
	}
}
