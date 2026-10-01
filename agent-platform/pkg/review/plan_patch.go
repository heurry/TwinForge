package review

import (
	"errors"
	"fmt"
	"strings"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/taskplan"
)

// ErrPlanPatchRequiresPlanner means the Reviewer recommendation is useful but
// cannot be applied without inventing Plan fields that the Reviewer did not
// supply. The parent planner may translate it into a complete Plan revision;
// the deterministic scheduler must not guess acceptance or verification rules.
var ErrPlanPatchRequiresPlanner = errors.New("Reviewer Plan change requires planner translation")

// CompilePlanPatch converts the safe subset of Reviewer recommendations into
// one complete, CAS-fenced Plan revision. It intentionally supports only
// modifications, retirements, and fully specified additions. Verification-only
// revisions are compiled separately so they use the narrower criterion API and
// cannot accidentally replace the graph.
func CompilePlanPatch(plan taskplan.Plan, result Result) (taskplan.Update, error) {
	if err := result.Validate(); err != nil {
		return taskplan.Update{}, fmt.Errorf("invalid Reviewer result: %w", err)
	}
	if result.Verdict != VerdictChangesRequired {
		return taskplan.Update{}, fmt.Errorf("%w: verdict %q is not auto-applicable", ErrPlanPatchRequiresPlanner, result.Verdict)
	}
	if len(result.RecommendedPlanChanges) == 0 {
		return taskplan.Update{}, fmt.Errorf("%w: no recommended changes", ErrPlanPatchRequiresPlanner)
	}

	terminal := make(map[string]bool, len(plan.Steps))
	steps := make([]taskplan.Step, 0, len(plan.Steps))
	for _, existing := range plan.Steps {
		closed := existing.Status == taskplan.StatusCompleted || existing.Status == taskplan.StatusSkipped
		terminal[existing.ID] = closed
		if closed {
			continue
		}
		steps = append(steps, planPatchStep(existing, terminal))
	}
	// Dependencies are normalized after the complete terminal set is known.
	for index := range steps {
		steps[index].DependsOn = activeDependencies(steps[index].DependsOn, terminal)
	}
	byID := make(map[string]int, len(steps))
	allIDs := make(map[string]struct{}, len(plan.Steps))
	for _, existing := range plan.Steps {
		allIDs[existing.ID] = struct{}{}
	}
	for index := range steps {
		byID[steps[index].ID] = index
	}

	retired := make(map[string]string)
	changed := make(map[string]string)
	for _, change := range result.RecommendedPlanChanges {
		id := strings.TrimSpace(change.StepID)
		switch change.Operation {
		case "add_step":
			if _, exists := allIDs[id]; exists {
				return taskplan.Update{}, fmt.Errorf("Reviewer add_step target %q already exists", id)
			}
			criteria := make([]taskplan.AcceptanceCriterion, 0, len(change.AcceptanceCriteria))
			for _, criterion := range change.AcceptanceCriteria {
				criteria = append(criteria, taskplan.AcceptanceCriterion{
					ID: strings.TrimSpace(criterion.ID), Description: strings.TrimSpace(criterion.Description),
					Status: taskplan.CriterionPending, Verification: criterion.Verification,
				})
			}
			steps = append(steps, taskplan.Step{
				ID: id, Description: strings.TrimSpace(change.Description), Status: taskplan.StatusPending,
				DependsOn: append([]string(nil), change.DependsOn...), ToolHints: append([]string(nil), change.ToolHints...),
				AcceptanceCriteria: criteria,
			})
			byID[id] = len(steps) - 1
			allIDs[id] = struct{}{}
			changed[id] = change.Operation
		case "revise_verification":
			return taskplan.Update{}, fmt.Errorf("%w: operation %q must use the criterion revision compiler", ErrPlanPatchRequiresPlanner, change.Operation)
		case "modify_step":
			stepIndex, exists := byID[id]
			if id == "" || !exists {
				return taskplan.Update{}, fmt.Errorf("Reviewer modify_step target %q is not an open Plan node", id)
			}
			if previous, duplicate := changed[id]; duplicate {
				return taskplan.Update{}, fmt.Errorf("Reviewer Plan patch targets step %q more than once (%s, modify_step)", id, previous)
			}
			if strings.TrimSpace(change.Description) == "" && change.DependsOn == nil && change.ToolHints == nil {
				return taskplan.Update{}, fmt.Errorf("Reviewer modify_step %q changes no fields", id)
			}
			if strings.TrimSpace(change.Description) != "" {
				steps[stepIndex].Description = strings.TrimSpace(change.Description)
			}
			if change.DependsOn != nil {
				steps[stepIndex].DependsOn = append([]string(nil), change.DependsOn...)
			}
			if change.ToolHints != nil {
				steps[stepIndex].ToolHints = append([]string(nil), change.ToolHints...)
			}
			changed[id] = change.Operation
		case "retire_step":
			if id == "" {
				return taskplan.Update{}, errors.New("Reviewer retire_step requires step_id")
			}
			if _, exists := byID[id]; !exists {
				return taskplan.Update{}, fmt.Errorf("Reviewer retire_step target %q is not an open Plan node", id)
			}
			if previous, duplicate := changed[id]; duplicate {
				return taskplan.Update{}, fmt.Errorf("Reviewer Plan patch targets step %q more than once (%s, retire_step)", id, previous)
			}
			retired[id] = strings.TrimSpace(change.Reason)
			changed[id] = change.Operation
		default:
			return taskplan.Update{}, fmt.Errorf("Reviewer Plan patch operation %q is unsupported", change.Operation)
		}
	}

	kept := steps[:0]
	for _, step := range steps {
		if _, remove := retired[step.ID]; !remove {
			kept = append(kept, step)
		}
	}
	steps = kept
	for index := range steps {
		steps[index].DependsOn = activeDependencies(steps[index].DependsOn, terminal)
	}
	if len(steps) == 0 {
		return taskplan.Update{}, fmt.Errorf("%w: patch would retire every open node", ErrPlanPatchRequiresPlanner)
	}
	if len(steps) > 8 {
		return taskplan.Update{}, fmt.Errorf("%w: %d open nodes exceed update_plan limit", ErrPlanPatchRequiresPlanner, len(steps))
	}
	remaining := make(map[string]struct{}, len(steps))
	for _, step := range steps {
		remaining[step.ID] = struct{}{}
	}
	for _, step := range steps {
		for _, dependency := range step.DependsOn {
			if _, exists := remaining[dependency]; !exists {
				return taskplan.Update{}, fmt.Errorf("Reviewer Plan patch leaves step %q depending on retired or unknown node %q", step.ID, dependency)
			}
		}
	}

	base := plan.Revision
	retiredSteps := make([]taskplan.RetiredStep, 0, len(retired))
	for _, existing := range plan.Steps {
		if reason, ok := retired[existing.ID]; ok {
			retiredSteps = append(retiredSteps, taskplan.RetiredStep{ID: existing.ID, Reason: reason})
		}
	}
	reason := strings.TrimSpace(result.Summary)
	if len([]rune(reason)) > 1800 {
		reason = string([]rune(reason)[:1800])
	}
	update := taskplan.Update{
		Goal: plan.Goal, Explanation: plan.Explanation, Steps: steps,
		ChangeMode: "replan", BaseRevision: &base,
		ReplanReason: reason, RetiredSteps: retiredSteps,
	}
	if update.ReplanReason == "" {
		update.ReplanReason = "Apply the structured Reviewer findings"
	}
	normalized, err := taskplan.NormalizeUpdate(update)
	if err != nil {
		return taskplan.Update{}, fmt.Errorf("normalize Reviewer Plan patch: %w", err)
	}
	if err := normalized.Validate(); err != nil {
		return taskplan.Update{}, fmt.Errorf("validate Reviewer Plan patch: %w", err)
	}
	return normalized, nil
}

// CompileVerificationRevision compiles exactly one Reviewer recommendation to
// the narrow criterion mutation API. Mixing it with graph operations would
// require a multi-write transaction across Tool calls, so mixed batches remain
// planner-owned until a dedicated atomic store operation exists.
func CompileVerificationRevision(plan taskplan.Plan, result Result) (taskplan.CriterionRevision, error) {
	if err := result.Validate(); err != nil {
		return taskplan.CriterionRevision{}, fmt.Errorf("invalid Reviewer result: %w", err)
	}
	if result.Verdict != VerdictChangesRequired || len(result.RecommendedPlanChanges) != 1 {
		return taskplan.CriterionRevision{}, fmt.Errorf("%w: verification auto-repair requires exactly one change", ErrPlanPatchRequiresPlanner)
	}
	change := result.RecommendedPlanChanges[0]
	if change.Operation != "revise_verification" {
		return taskplan.CriterionRevision{}, fmt.Errorf("%w: operation %q is not a verification revision", ErrPlanPatchRequiresPlanner, change.Operation)
	}
	stepFound, criterionFound := false, false
	for _, step := range plan.Steps {
		if step.ID != change.StepID {
			continue
		}
		stepFound = true
		for _, criterion := range step.AcceptanceCriteria {
			if criterion.ID == change.CriterionID {
				criterionFound = true
				break
			}
		}
		break
	}
	if !stepFound || !criterionFound {
		return taskplan.CriterionRevision{}, fmt.Errorf("Reviewer verification target step=%q criterion=%q does not exist", change.StepID, change.CriterionID)
	}
	base := plan.Revision
	return taskplan.CriterionRevision{
		StepID: change.StepID, CriterionID: change.CriterionID, Action: "replace",
		Verification: change.Verification, Reason: change.Reason, BaseRevision: &base,
	}, nil
}

func planPatchStep(existing taskplan.Step, terminal map[string]bool) taskplan.Step {
	status := existing.Status
	if status != taskplan.StatusInProgress {
		status = taskplan.StatusPending
	}
	criteria := make([]taskplan.AcceptanceCriterion, 0, len(existing.AcceptanceCriteria))
	for _, current := range existing.AcceptanceCriteria {
		criteria = append(criteria, taskplan.AcceptanceCriterion{
			ID: current.ID, Description: current.Description, Status: taskplan.CriterionPending,
			Verification: current.Verification,
		})
	}
	return taskplan.Step{
		ID: existing.ID, Description: existing.Description, Status: status,
		Assignee: existing.Assignee, AgentVersionID: existing.AgentVersionID,
		DependsOn:          activeDependencies(existing.DependsOn, terminal),
		ToolHints:          append([]string(nil), existing.ToolHints...),
		AcceptanceCriteria: criteria,
	}
}

func activeDependencies(dependencies []string, terminal map[string]bool) []string {
	result := make([]string, 0, len(dependencies))
	for _, dependency := range dependencies {
		if !terminal[dependency] {
			result = append(result, dependency)
		}
	}
	return result
}
