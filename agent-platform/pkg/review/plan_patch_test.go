package review

import (
	"errors"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
)

func TestCompilePlanPatchModifiesAndRetiresOnlyNamedOpenNodes(t *testing.T) {
	plan := taskplan.Plan{Revision: 7, Goal: "ship", Steps: []taskplan.Step{
		patchTestStep("done", taskplan.StatusCompleted, nil),
		patchTestStep("build", taskplan.StatusInProgress, []string{"done"}),
		patchTestStep("obsolete", taskplan.StatusPending, []string{"build"}),
		patchTestStep("verify", taskplan.StatusPending, []string{"build"}),
	}}
	result := Result{Verdict: VerdictChangesRequired, Summary: "split obsolete work", Findings: []Finding{{Severity: "medium", Summary: "plan is stale", Evidence: "workspace differs"}}, RecommendedPlanChanges: []PlanChange{
		{Operation: "modify_step", StepID: "build", Description: "build the corrected module", Reason: "interface mismatch", DependsOn: []string{}},
		{Operation: "retire_step", StepID: "obsolete", Reason: "no longer in scope"},
	}}
	update, err := CompilePlanPatch(plan, result)
	if err != nil {
		t.Fatal(err)
	}
	if update.BaseRevision == nil || *update.BaseRevision != 7 || update.ChangeMode != "replan" || len(update.Steps) != 2 || len(update.RetiredSteps) != 1 {
		t.Fatalf("compiled patch = %+v", update)
	}
	if update.Steps[0].ID != "build" || update.Steps[0].Description != "build the corrected module" || len(update.Steps[0].DependsOn) != 0 {
		t.Fatalf("modified node = %+v", update.Steps[0])
	}
	if update.Steps[1].ID != "verify" || len(update.Steps[1].DependsOn) != 1 || update.Steps[1].DependsOn[0] != "build" {
		t.Fatalf("unrelated node changed = %+v", update.Steps[1])
	}
}

func TestCompilePlanPatchAddsFullySpecifiedStep(t *testing.T) {
	plan := taskplan.Plan{Revision: 2, Goal: "ship", Steps: []taskplan.Step{patchTestStep("build", taskplan.StatusInProgress, nil)}}
	change := PlanChange{
		Operation: "add_step", StepID: "verify", Description: "verify the current build", Reason: "missing verification phase",
		DependsOn: []string{"build"}, ToolHints: []string{"read_file"},
		AcceptanceCriteria: []PlanCriterion{{ID: "exists", Description: "artifact exists", Verification: taskplan.VerificationSpec{Kind: "file_exists", Target: "artifact.bin"}}},
	}
	update, err := CompilePlanPatch(plan, Result{Verdict: VerdictChangesRequired, Summary: "add verification", Findings: []Finding{{Severity: "medium", Summary: "verification missing", Evidence: "Plan ends after build"}}, RecommendedPlanChanges: []PlanChange{change}})
	if err != nil {
		t.Fatal(err)
	}
	if len(update.Steps) != 2 || update.Steps[1].ID != "verify" || update.Steps[1].AcceptanceCriteria[0].Verification.Target != "artifact.bin" {
		t.Fatalf("compiled addition = %+v", update)
	}
}

func TestCompileVerificationRevisionUsesNarrowCASMutation(t *testing.T) {
	plan := taskplan.Plan{Revision: 5, Goal: "ship", Steps: []taskplan.Step{patchTestStep("build", taskplan.StatusInProgress, nil)}}
	result := Result{Verdict: VerdictChangesRequired, Summary: "repair verification", Findings: []Finding{{Severity: "medium", Summary: "wrong target", Evidence: "actual file is output.go"}}, RecommendedPlanChanges: []PlanChange{{
		Operation: "revise_verification", StepID: "build", CriterionID: "exists", Reason: "target moved",
		Verification: taskplan.VerificationSpec{Kind: "file_exists", Target: "output.go"},
	}}}
	revision, err := CompileVerificationRevision(plan, result)
	if err != nil {
		t.Fatal(err)
	}
	if revision.BaseRevision == nil || *revision.BaseRevision != 5 || revision.Action != "replace" || revision.Verification.Target != "output.go" {
		t.Fatalf("compiled verification revision = %+v", revision)
	}
	if _, err := CompilePlanPatch(plan, result); !errors.Is(err, ErrPlanPatchRequiresPlanner) {
		t.Fatalf("verification change was incorrectly compiled as graph replacement: %v", err)
	}
}

func TestCompilePlanPatchRejectsDanglingDependency(t *testing.T) {
	plan := taskplan.Plan{Revision: 3, Goal: "ship", Steps: []taskplan.Step{
		patchTestStep("build", taskplan.StatusInProgress, nil),
		patchTestStep("verify", taskplan.StatusPending, []string{"build"}),
	}}
	_, err := CompilePlanPatch(plan, Result{Verdict: VerdictChangesRequired, Summary: "remove build", Findings: []Finding{{Severity: "medium", Summary: "build is obsolete", Evidence: "workspace differs"}}, RecommendedPlanChanges: []PlanChange{{Operation: "retire_step", StepID: "build", Reason: "replace it"}}})
	if err == nil {
		t.Fatal("retirement with a dangling dependency was accepted")
	}
}

func patchTestStep(id, status string, dependencies []string) taskplan.Step {
	return taskplan.Step{
		ID: id, Description: id, Status: status, DependsOn: dependencies,
		AcceptanceCriteria: []taskplan.AcceptanceCriterion{{
			ID: "exists", Description: id + " exists", Status: taskplan.CriterionPending,
			Verification: taskplan.VerificationSpec{Kind: "file_exists", Target: id + ".go"},
		}},
	}
}
