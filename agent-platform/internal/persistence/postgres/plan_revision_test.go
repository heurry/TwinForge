package postgres

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/agent"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

func revisionStep(id, status string) taskplan.Step {
	return taskplan.Step{ID: id, Description: id, Status: status, AcceptanceCriteria: []taskplan.AcceptanceCriterion{{ID: id + "-ok", Description: "verified", Status: taskplan.CriterionPending}}}
}

func contractCode(err error) string {
	var contract *tool.ContractError
	if errors.As(err, &contract) {
		return contract.Code
	}
	return ""
}

func TestPreparePlanRevisionRequiresCASForExistingPlan(t *testing.T) {
	_, err := preparePlanRevision("update_plan", 3, "goal", "", []taskplan.Step{revisionStep("a", taskplan.StatusInProgress)}, taskplan.Update{Goal: "goal", ChangeMode: "replan", ReplanReason: "changed", Steps: []taskplan.Step{revisionStep("a", taskplan.StatusInProgress)}})
	if contractCode(err) != "PLAN_REVISION_CONFLICT" {
		t.Fatalf("error=%v", err)
	}
}

func TestPreparePlanRevisionRejectsSilentOpenNodeDeletion(t *testing.T) {
	base := 3
	_, err := preparePlanRevision("update_plan", 3, "goal", "", []taskplan.Step{revisionStep("a", taskplan.StatusInProgress), revisionStep("b", taskplan.StatusPending)}, taskplan.Update{Goal: "goal", ChangeMode: "replan", BaseRevision: &base, ReplanReason: "dependencies changed", Steps: []taskplan.Step{revisionStep("a", taskplan.StatusInProgress)}})
	if contractCode(err) != "PLAN_OPEN_NODE_OMITTED" {
		t.Fatalf("error=%v", err)
	}
}

func TestPreparePlanRevisionAllowsExplicitRetirement(t *testing.T) {
	base := 3
	update, err := preparePlanRevision("update_plan", 3, "goal", "", []taskplan.Step{revisionStep("a", taskplan.StatusInProgress), revisionStep("b", taskplan.StatusPending)}, taskplan.Update{Goal: "goal", ChangeMode: "replan", BaseRevision: &base, ReplanReason: "scope removed", RetiredSteps: []taskplan.RetiredStep{{ID: "b", Reason: "user removed scope"}}, Steps: []taskplan.Step{revisionStep("a", taskplan.StatusInProgress)}})
	if err != nil || len(update.Steps) != 1 || update.Steps[0].ID != "a" {
		t.Fatalf("update=%+v err=%v", update, err)
	}
}

func TestPreparePlanRevisionExtendPreservesExistingGraph(t *testing.T) {
	base := 2
	update, err := preparePlanRevision("update_plan", 2, "original", "", []taskplan.Step{revisionStep("a", taskplan.StatusInProgress)}, taskplan.Update{Goal: "replacement", ChangeMode: "extend", BaseRevision: &base, ReplanReason: "user added delivery", Steps: []taskplan.Step{revisionStep("b", taskplan.StatusPending)}})
	if err != nil || len(update.Steps) != 2 || update.Goal != "original" {
		t.Fatalf("update=%+v err=%v", update, err)
	}
}

func TestValidateCriterionRevisionCAS(t *testing.T) {
	if err := validateCriterionRevisionCAS("revise_verification", 4, json.RawMessage(`{"step_id":"build","criterion_id":"exists"}`)); err != nil {
		t.Fatalf("ordinary revision without CAS was rejected: %v", err)
	}
	if err := validateCriterionRevisionCAS("revise_verification", 4, json.RawMessage(`{"base_revision":4}`)); err != nil {
		t.Fatalf("matching revision was rejected: %v", err)
	}
	err := validateCriterionRevisionCAS("revise_verification", 4, json.RawMessage(`{"base_revision":3}`))
	var contractErr *tool.ContractError
	if !errors.As(err, &contractErr) || contractErr.Code != "PLAN_REVISION_CONFLICT" {
		t.Fatalf("stale revision error = %v", err)
	}
}

func TestNormalizeWorkflowRoutingIntentSeparatesContinuationAndNewTask(t *testing.T) {
	for input, want := range map[string]string{"resume": "continue", "new_turn": "continue", "extend": "extend", "replan": "replan", "new_workflow": "new_task"} {
		got, err := normalizeWorkflowRoutingIntent(input, false)
		if err != nil || got != want {
			t.Fatalf("intent %q = %q, err=%v; want %q", input, got, err, want)
		}
	}
	if _, err := normalizeWorkflowRoutingIntent("guess", false); !errors.Is(err, agent.ErrWorkflowRoutingInvalid) {
		t.Fatalf("unknown intent error=%v", err)
	}
}
