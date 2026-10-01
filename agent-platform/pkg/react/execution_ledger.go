package react

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	contextpkg "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/context"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
	reviewcontract "github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/review"
	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/tool"
)

const (
	runtimeExecutionLedgerStart = "\n\n<RUNTIME_EXECUTION_LEDGER>\n"
	runtimeExecutionLedgerEnd   = "\n</RUNTIME_EXECUTION_LEDGER>"
	executionLedgerStatePrefix  = "STATE="
	executionGuidanceMetadata   = "runtime.execution_guidance"
	// The complete ledger remains in the checkpoint. This is only the hard
	// model-facing ceiling for the regenerated system projection. Keeping the
	// ceiling in characters makes the bound deterministic before provider
	// tokenization (and leaves room for the surrounding runtime prompt).
	modelLedgerMaxChars = 4200
)

type executionLedger struct {
	ByTool             map[string]toolProgress        `json:"by_tool"`
	ReadFiles          map[string]readProgress        `json:"read_file,omitempty"`
	Writes             []string                       `json:"write_file,omitempty"`
	Appends            []string                       `json:"append_file,omitempty"`
	Edits              []string                       `json:"edit_file,omitempty"`
	Promotes           []string                       `json:"promote_file,omitempty"`
	Errors             []ledgerEntry                  `json:"recent_errors,omitempty"`
	Other              []ledgerEntry                  `json:"recent_other,omitempty"`
	SuccessfulEvidence []ledgerEntry                  `json:"successful_evidence_receipts,omitempty"`
	ToolCalls          int                            `json:"all_tool_calls"`
	Succeeded          int                            `json:"all_tools_succeeded"`
	Failed             int                            `json:"all_tools_failed"`
	ListsSinceMutation int                            `json:"list_observations_since_mutation,omitempty"`
	FailureStreakTool  string                         `json:"failure_streak_tool,omitempty"`
	FailureStreakCount int                            `json:"failure_streak_count,omitempty"`
	FileFocusPath      string                         `json:"file_focus_path,omitempty"`
	FileFocusDecisions int                            `json:"file_focus_decisions,omitempty"`
	FileFocusMutations int                            `json:"file_focus_mutations,omitempty"`
	FileFocusFailures  int                            `json:"file_focus_failures,omitempty"`
	LastAction         *ledgerEntry                   `json:"last_action,omitempty"`
	NextAction         string                         `json:"next_action,omitempty"`
	Files              map[string]fileMutation        `json:"files,omitempty"`
	WorkspaceRevision  int64                          `json:"workspace_revision,omitempty"`
	CommandFailures    map[string]commandFailureState `json:"deterministic_command_failures,omitempty"`
	ReadObservations   map[string]readObservation     `json:"read_observations,omitempty"`
	// ProgressReview is the active model-facing recovery instruction. The
	// baseline is durable bookkeeping: a successful mutation may clear the
	// instruction, but must not make all historical failures look new again.
	ProgressReview         *progressReview   `json:"progress_review,omitempty"`
	ProgressReviewBaseline *progressReview   `json:"progress_review_baseline,omitempty"`
	ReviewerDecision       *reviewerDecision `json:"reviewer_decision,omitempty"`
}

type reviewerDecision struct {
	CallID                 string                      `json:"call_id"`
	ChildRunID             string                      `json:"child_run_id,omitempty"`
	BasePlanRevision       int                         `json:"base_plan_revision,omitempty"`
	Verdict                string                      `json:"verdict"`
	Summary                string                      `json:"summary"`
	Findings               []reviewcontract.Finding    `json:"findings,omitempty"`
	RecommendedPlanChanges []reviewcontract.PlanChange `json:"recommended_plan_changes,omitempty"`
	ParseError             string                      `json:"parse_error,omitempty"`
}

type commandFailureState struct {
	WorkspaceRevision int64  `json:"workspace_revision"`
	FailureKind       string `json:"failure_kind"`
	Diagnostic        string `json:"diagnostic,omitempty"`
}

type readObservation struct {
	WorkspaceRevision int64           `json:"workspace_revision"`
	Content           json.RawMessage `json:"content"`
}

// progressReview is a bounded, fact-only strategy checkpoint. It deliberately
// contains no model reasoning: the model receives the observable stagnation
// facts and the Runtime selects a bounded scheduler action from them. Keeping
// it inside the durable execution ledger means it survives checkpoint rollover
// while its compact projection remains rebuildable after Context Collapse.
type progressReview struct {
	Generation         int    `json:"generation"`
	Phase              string `json:"phase"`
	Action             string `json:"action"`
	Risk               string `json:"risk,omitempty"`
	Confidence         string `json:"confidence,omitempty"`
	Reason             string `json:"reason"`
	ToolCalls          int    `json:"tool_calls"`
	FailedToolCalls    int    `json:"failed_tool_calls"`
	WorkspaceMutations int    `json:"workspace_mutations"`
	PlanControlCalls   int    `json:"plan_control_calls"`
}

// fileMutation is the bounded manifest of the latest workspace state. The
// file hash identifies bytes at the workspace path; the Artifact fields
// identify persisted evidence and are deliberately separate.
type fileMutation struct {
	Path              string `json:"path"`
	Revision          int    `json:"revision"`
	LastOperation     string `json:"last_operation"`
	LastCallID        string `json:"last_call_id,omitempty"`
	FileSHA256        string `json:"file_sha256,omitempty"`
	ArtifactID        string `json:"artifact_id,omitempty"`
	ArtifactSHA256    string `json:"artifact_sha256,omitempty"`
	Bytes             int    `json:"bytes,omitempty"`
	LineCount         int    `json:"line_count,omitempty"`
	SyntaxStatus      string `json:"syntax_status,omitempty"`
	VerificationError string `json:"verification_error,omitempty"`
}

type toolProgress struct {
	Succeeded int `json:"succeeded,omitempty"`
	Failed    int `json:"failed,omitempty"`
}

type readProgress struct {
	Succeeded []string `json:"succeeded_ranges,omitempty"`
	Failed    []string `json:"failed_ranges,omitempty"`
}

type ledgerEntry struct {
	Tool      string `json:"tool"`
	Arguments string `json:"arguments,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Error     string `json:"error,omitempty"`
}

func updateExecutionLedger(messages []model.Message, call model.ToolCall, result tool.Result) []model.Message {
	ledger := loadExecutionLedger(messages)
	ensureToolCounters(&ledger)
	ledger.ToolCalls++
	failed := result.IsError
	progress := ledger.ByTool[call.Name]
	if failed {
		ledger.Failed++
		progress.Failed++
	} else {
		ledger.Succeeded++
		progress.Succeeded++
	}
	ledger.ByTool[call.Name] = progress
	arguments := compactLedgerArguments(call.Name, call.Arguments)
	trackFileFocus(&ledger, call, failed)
	previousAction := ledger.LastAction
	ledger.LastAction = &ledgerEntry{Tool: call.Name, Arguments: arguments, CallID: call.ID}
	if failed {
		ledger.LastAction.Error = result.Error
		if previousAction != nil && previousAction.Tool == call.Name && previousAction.Error != "" {
			ledger.FailureStreakCount++
		} else {
			ledger.FailureStreakTool = call.Name
			ledger.FailureStreakCount = 1
		}
	} else if call.Name != "update_plan" && call.Name != "update_plan_step" && call.Name != "ask_user" {
		ledger.FailureStreakTool = ""
		ledger.FailureStreakCount = 0
		// Compaction may remove the original Tool message, but acceptance gates
		// still require its exact immutable call ID. Keep a bounded receipt list
		// in the durable ledger so the model never has to guess evidence IDs.
		ledger.SuccessfulEvidence = appendBoundedEntry(ledger.SuccessfulEvidence, ledgerEntry{
			Tool: call.Name, Arguments: arguments, CallID: call.ID,
		}, 12)
	} else {
		ledger.FailureStreakTool = ""
		ledger.FailureStreakCount = 0
	}
	var resultEnvelope struct {
		NextAction string `json:"next_action"`
	}
	if json.Unmarshal(result.Content, &resultEnvelope) == nil && strings.TrimSpace(resultEnvelope.NextAction) != "" {
		ledger.NextAction = strings.TrimSpace(resultEnvelope.NextAction)
	} else if !failed {
		ledger.NextAction = ""
	}
	switch call.Name {
	case "read_file":
		var input struct {
			Path      string `json:"path"`
			StartLine int    `json:"start_line"`
			LineCount int    `json:"line_count"`
		}
		if json.Unmarshal(call.Arguments, &input) == nil && input.Path != "" {
			if input.LineCount > 0 && input.StartLine < 1 {
				input.StartLine = 1
			}
			if ledger.ReadFiles == nil {
				ledger.ReadFiles = make(map[string]readProgress)
			}
			progress := ledger.ReadFiles[input.Path]
			rangeLabel := fmt.Sprintf("%d+%d", input.StartLine, input.LineCount)
			if !failed && input.StartLine <= 1 && input.LineCount > 0 {
				var output struct {
					LineCount *int `json:"line_count"`
				}
				if json.Unmarshal(result.Content, &output) == nil && output.LineCount != nil && *output.LineCount < input.LineCount {
					rangeLabel = "0+0"
				}
			}
			if failed {
				progress.Failed = appendBounded(progress.Failed, rangeLabel, 32)
			} else {
				progress.Succeeded = appendBounded(progress.Succeeded, rangeLabel, 64)
				if ledger.ReadObservations == nil {
					ledger.ReadObservations = make(map[string]readObservation)
				}
				if len(ledger.ReadObservations) >= 4 {
					keys := make([]string, 0, len(ledger.ReadObservations))
					for key := range ledger.ReadObservations {
						keys = append(keys, key)
					}
					sort.Strings(keys)
					delete(ledger.ReadObservations, keys[0])
				}
				ledger.ReadObservations[toolArgumentsHash(call.Arguments)] = readObservation{WorkspaceRevision: ledger.WorkspaceRevision, Content: append(json.RawMessage(nil), result.ModelVisible().Content...)}
			}
			ledger.ReadFiles[input.Path] = progress
		}
	case "write_file":
		ledger.Writes = appendBounded(ledger.Writes, ledgerPathStatus(call.Arguments, failed), 12)
		if !failed {
			ledger.ListsSinceMutation = 0
			delete(ledger.ReadFiles, ledgerPath(call.Arguments))
		}
	case "append_file":
		ledger.Appends = appendBounded(ledger.Appends, ledgerPathStatus(call.Arguments, failed), 24)
		if !failed {
			ledger.ListsSinceMutation = 0
			delete(ledger.ReadFiles, ledgerPath(call.Arguments))
		}
	case "edit_file":
		ledger.Edits = appendBounded(ledger.Edits, ledgerPathStatus(call.Arguments, failed), 12)
		if !failed {
			ledger.ListsSinceMutation = 0
			delete(ledger.ReadFiles, ledgerPath(call.Arguments))
		}
	case "promote_file":
		ledger.Promotes = appendBounded(ledger.Promotes, ledgerPathStatus(call.Arguments, failed), 12)
		if !failed {
			ledger.ListsSinceMutation = 0
			delete(ledger.ReadFiles, ledgerPath(call.Arguments))
		}
	case "list_files":
		if !failed {
			ledger.ListsSinceMutation++
			ledger.Other = appendBoundedEntry(ledger.Other, ledgerEntry{Tool: call.Name, Arguments: arguments}, 4)
		}
	case "update_plan", "update_plan_step", "revise_verification":
		// The durable Plan directive already carries its latest revision.
	default:
		if !failed {
			ledger.Other = appendBoundedEntry(ledger.Other, ledgerEntry{Tool: call.Name, Arguments: arguments}, 4)
		}
	}
	if !failed {
		if mutation, ok := parseFileMutation(call, result); ok {
			if ledger.Files == nil {
				ledger.Files = make(map[string]fileMutation)
			}
			mutation.Revision = ledger.Files[mutation.Path].Revision + 1
			mutation.LastCallID = call.ID
			ledger.Files[mutation.Path] = mutation
			ledger.WorkspaceRevision++
			ledger.ReadObservations = nil
		}
	}
	if call.Name == "run_command" {
		updateCommandFailureState(&ledger, call, result)
		recordSyntaxVerification(&ledger, call, result)
		if !failed {
			clearFileFocus(&ledger)
		}
	}
	if call.Name == "update_plan" && !failed {
		// A successful full Plan mutation is an explicit decomposition/replan
		// boundary. No-op Plan rewrites are rejected before reaching this point.
		// Completing the requested replan must also release the deterministic
		// tool projection; otherwise the next cycle remains permanently limited
		// to the Plan tool even though the durable Plan revision committed.
		clearFileFocus(&ledger)
		if ledger.ProgressReview != nil && ledger.ProgressReview.Action == "replan" {
			ledger.ProgressReview = nil
		}
		ledger.ReviewerDecision = nil
	}
	if call.Name == "revise_verification" && !failed && ledger.ReviewerDecision != nil {
		if ledger.ProgressReview != nil && ledger.ProgressReview.Action == "replan" {
			ledger.ProgressReview = nil
		}
		ledger.ReviewerDecision = nil
	}
	reviewerHandled := false
	if !failed && isAutomaticReviewerCall(messages, call) {
		reviewerHandled = true
		decision := parseReviewerDecision(call, result)
		ledger.ReviewerDecision = &decision
		if decision.Verdict == reviewcontract.VerdictPass && decision.ParseError == "" {
			ledger.ProgressReview = nil
		} else {
			reason := fmt.Sprintf("Reviewer verdict=%s for Plan revision %d: %s", decision.Verdict, decision.BasePlanRevision, decision.Summary)
			if decision.ParseError != "" {
				reason = fmt.Sprintf("Reviewer result was invalid for Plan revision %d: %s", decision.BasePlanRevision, decision.ParseError)
			}
			if decision.Verdict == reviewcontract.VerdictBlocked && decision.ParseError == "" {
				applyReviewerUserEscalation(&ledger, reason)
			} else {
				applyReviewerFallback(&ledger, reason)
			}
		}
	}
	if !failed {
		switch call.Name {
		case "read_file", "list_files", "update_plan", "update_plan_step", "revise_verification":
			// Observation and Plan bookkeeping do not prove implementation
			// progress, so an active recovery phase remains in force.
		case "ask_user":
			if ledger.ProgressReview != nil && ledger.ProgressReview.Action == "ask_user" {
				// A real user response changes the available facts and starts a new
				// bounded recovery budget. Keep the current counters as the next
				// comparison baseline, but do not immediately escalate generation 5
				// again just because the workspace has not changed yet.
				baseline := *ledger.ProgressReview
				baseline.Generation = 0
				baseline.Action = ""
				baseline.Phase = ""
				baseline.Reason = "user decision received"
				baseline.ToolCalls = ledger.ToolCalls
				baseline.FailedToolCalls = ledger.Failed
				baseline.WorkspaceMutations = workspaceMutationCount(ledger)
				baseline.PlanControlCalls = planControlCallCount(ledger)
				ledger.ProgressReview = nil
				ledger.ProgressReviewBaseline = &baseline
			}
		default:
			if !reviewerHandled {
				ledger.ProgressReview = nil
			}
		}
	}
	if failed {
		ledger.Errors = appendBoundedEntry(ledger.Errors, ledgerEntry{Tool: call.Name, Arguments: arguments, CallID: call.ID, Error: result.Error}, 3)
		if (call.Name == "update_plan" || call.Name == "revise_verification") && ledger.ReviewerDecision != nil && toolResultErrorCode(result) == "PLAN_REVISION_CONFLICT" {
			staleRevision := ledger.ReviewerDecision.BasePlanRevision
			ledger.ReviewerDecision = nil
			if ledger.ProgressReview != nil {
				ledger.ProgressReview.Reason = fmt.Sprintf("Reviewer recommendations for Plan revision %d are stale because the durable Plan changed; ignore them and replan from the latest Plan projection", staleRevision)
				baseline := *ledger.ProgressReview
				ledger.ProgressReviewBaseline = &baseline
			}
		}
	}
	updated := replaceExecutionLedger(messages, ledger)
	switch call.Name {
	case "update_plan", "update_plan_step", "revise_verification", "read_file", "list_files", "write_file", "append_file", "edit_file", "promote_file", "run_command":
		return appendExecutionGuidance(updated, ledger)
	default:
		return updated
	}
}

func toolResultErrorCode(result tool.Result) string {
	var payload struct {
		ErrorCode string `json:"error_code"`
	}
	if json.Unmarshal(result.Content, &payload) != nil {
		return ""
	}
	return strings.TrimSpace(payload.ErrorCode)
}

func cachedReadObservation(messages []model.Message, call model.ToolCall) (tool.Result, bool) {
	if call.Name != "read_file" {
		return tool.Result{}, false
	}
	ledger := loadExecutionLedger(messages)
	observation, ok := ledger.ReadObservations[toolArgumentsHash(call.Arguments)]
	if !ok || observation.WorkspaceRevision != ledger.WorkspaceRevision || len(observation.Content) == 0 {
		return tool.Result{}, false
	}
	return tool.Result{Content: append(json.RawMessage(nil), observation.Content...), Meta: map[string]string{
		"cache_hit": "true", "observation_reused": "true", "event_ledger_ref": toolArgumentsHash(call.Arguments),
	}}, true
}

func validateFileFocusBudget(messages []model.Message, call model.ToolCall) error {
	switch call.Name {
	case "read_file", "write_file", "append_file", "edit_file", "promote_file":
	default:
		return nil
	}
	ledger := loadExecutionLedger(messages)
	path := ledgerPath(call.Arguments)
	if path == "" || path != ledger.FileFocusPath || ledger.FileFocusDecisions < 14 {
		return nil
	}
	return tool.NewContractErrorWithRepair(
		"FILE_FOCUS_BUDGET_EXCEEDED", call.Name, "/path",
		"a validation boundary, an actual Plan decomposition, or a different dependency-ready file",
		fmt.Sprintf("%s after %d consecutive decisions", path, ledger.FileFocusDecisions),
		fmt.Sprintf("continued work on %q is paused because the trajectory is stuck in one-file local patching", path),
		"Run the relevant verification command; if the file still contains multiple responsibilities, use update_plan with an explicit replan/extend reason to split the remaining work, or continue a dependency-ready module. Then retry only the necessary changed operation.",
		nil, true,
	)
}

func trackFileFocus(ledger *executionLedger, call model.ToolCall, failed bool) {
	if ledger == nil {
		return
	}
	switch call.Name {
	case "read_file", "write_file", "append_file", "edit_file", "promote_file":
	default:
		return
	}
	path := ledgerPath(call.Arguments)
	if strings.TrimSpace(path) == "" {
		return
	}
	if ledger.FileFocusPath != path {
		ledger.FileFocusPath = path
		ledger.FileFocusDecisions = 0
		ledger.FileFocusMutations = 0
		ledger.FileFocusFailures = 0
	}
	ledger.FileFocusDecisions++
	if failed {
		ledger.FileFocusFailures++
	}
	if !failed && call.Name != "read_file" {
		ledger.FileFocusMutations++
	}
}

func clearFileFocus(ledger *executionLedger) {
	ledger.FileFocusPath = ""
	ledger.FileFocusDecisions = 0
	ledger.FileFocusMutations = 0
	ledger.FileFocusFailures = 0
}

func updateCommandFailureState(ledger *executionLedger, call model.ToolCall, result tool.Result) {
	if ledger == nil {
		return
	}
	signature := toolArgumentsHash(call.Arguments)
	if !result.IsError {
		delete(ledger.CommandFailures, signature)
		return
	}
	var payload struct {
		FailureKind string `json:"failure_kind"`
		Diagnostic  string `json:"diagnostic"`
	}
	if json.Unmarshal(result.Content, &payload) != nil || (payload.FailureKind != "process_exit" && payload.FailureKind != "timeout") {
		return
	}
	if ledger.CommandFailures == nil {
		ledger.CommandFailures = make(map[string]commandFailureState)
	}
	// Keep the durable gate bounded. The most recent command is sufficient for
	// normal recovery; retaining a few alternates also catches A/B retry loops.
	if len(ledger.CommandFailures) >= 8 {
		keys := make([]string, 0, len(ledger.CommandFailures))
		for key := range ledger.CommandFailures {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		delete(ledger.CommandFailures, keys[0])
	}
	ledger.CommandFailures[signature] = commandFailureState{
		WorkspaceRevision: ledger.WorkspaceRevision,
		FailureKind:       payload.FailureKind,
		Diagnostic:        compactLedgerText(payload.Diagnostic, 240),
	}
}

func appendExecutionGuidance(messages []model.Message, ledger executionLedger) []model.Message {
	updated := make([]model.Message, 0, len(messages)+1)
	for _, message := range messages {
		if message.Metadata[executionGuidanceMetadata] != "true" {
			updated = append(updated, message)
		}
	}
	reads := ledger.ByTool["read_file"]
	writes := ledger.ByTool["write_file"]
	appends := ledger.ByTool["append_file"]
	edits := ledger.ByTool["edit_file"]
	promotes := ledger.ByTool["promote_file"]
	ranges, _ := json.Marshal(ledger.ReadFiles)
	runs := ledger.ByTool["run_command"]
	guidance := fmt.Sprintf("Verified progress: read=%d/%d ranges=%s; write=%d; append=%d; edit=%d; promote=%d; command=%d/%d (success/failure where paired). Use Plan evidence; do not repeat successful observations.", reads.Succeeded, reads.Failed, ranges, writes.Succeeded, appends.Succeeded, edits.Succeeded, promotes.Succeeded, runs.Succeeded, runs.Failed)
	if ledger.ListsSinceMutation > 0 {
		guidance += fmt.Sprintf(" Directory observations since the last mutation=%d; act on them instead of listing again.", ledger.ListsSinceMutation)
	}
	if path := latestSuccessfulPath(ledger.Promotes, ledger.Appends, ledger.Writes); path != "" {
		guidance += fmt.Sprintf(" Latest changed file=%q; use promote_file only after a complete staged file has a verified source hash, and use edit_file for a precise repair.", path)
	}
	if ledger.NextAction != "" {
		guidance += " Tool recovery hint: " + ledger.NextAction
	}
	if review := ledger.ProgressReview; review != nil {
		guidance += fmt.Sprintf(" STRATEGY CHECKPOINT #%d phase=%s action=%s risk=%s confidence=%s: %s (tool_calls=%d failed=%d workspace_mutations=%d plan_controls=%d). Follow the runtime-selected action using the currently offered tools. execute=perform one concrete workspace/verification action; retry=repair the reported cause with changed arguments or workspace state; replan=update the Plan DAG before further mutation; review=delegate a bounded read-only review when available, otherwise replan; ask_user=ask one focused decision question because the bounded autonomous recovery budget is exhausted. Do not repeat a failed payload, duplicate an unchanged read, or keep revising the Plan without execution.", review.Generation, review.Phase, review.Action, review.Risk, review.Confidence, review.Reason, review.ToolCalls, review.FailedToolCalls, review.WorkspaceMutations, review.PlanControlCalls)
	}
	if ledger.FileFocusPath != "" {
		guidance += fmt.Sprintf(" Current file focus: path=%q decisions=%d mutations=%d failures=%d. Validate the current slice, split the Plan/module, or move to another dependency-ready file before the hard focus limit.", ledger.FileFocusPath, ledger.FileFocusDecisions, ledger.FileFocusMutations, ledger.FileFocusFailures)
	}
	for _, file := range latestFileMutations(ledger.Files, 4) {
		guidance += fmt.Sprintf(" File state: %s revision=%d operation=%s file_sha256=%s line_count=%d syntax_status=%s.", file.Path, file.Revision, file.LastOperation, file.FileSHA256, file.LineCount, file.SyntaxStatus)
		if file.VerificationError != "" {
			guidance += " Repair the reported range instead of rewriting the whole file: " + file.VerificationError
		}
	}
	message := model.TextMessage(model.RoleUser, guidance)
	message.Metadata = map[string]string{
		executionGuidanceMetadata:            "true",
		contextpkg.ContextSectionMetadataKey: contextpkg.ContextSectionRuntime,
		contextpkg.RuntimeControlMetadataKey: "true",
	}
	return append(updated, message)
}

// ensureProgressReview raises a durable strategy checkpoint when the current
// trajectory is demonstrably stalled. Exact-failure streaks alone miss loops
// that alternate read_file, Plan controls, and verification tools, so this uses
// several independent, observable progress signals.
func ensureProgressReview(messages []model.Message, force bool) ([]model.Message, *progressReview) {
	ledger := loadExecutionLedger(messages)
	mutations := workspaceMutationCount(ledger)
	planControls := planControlCallCount(ledger)
	previous := ledger.ProgressReviewBaseline
	if previous == nil {
		// Backward compatibility for checkpoints written before the baseline was
		// separated from the active model-facing review.
		previous = ledger.ProgressReview
	}
	previousCalls := 0
	previousFailures := 0
	previousMutations := 0
	if previous != nil {
		previousCalls = previous.ToolCalls
		previousFailures = previous.FailedToolCalls
		previousMutations = previous.WorkspaceMutations
	}
	// A review is a checkpoint for observations up to ToolCalls. Do not emit it
	// again on every following model request until a new tool observation has
	// actually changed the trajectory.
	if previous != nil {
		if ledger.ToolCalls <= previousCalls {
			return messages, nil
		}
		// Give the selected recovery phase a bounded execution window. The old
		// absolute failure/list counters otherwise emitted another review after
		// every single call and consumed the decision loop without new evidence.
		if mutations == previousMutations && ledger.ToolCalls-previousCalls < 3 {
			return messages, nil
		}
	}
	if ledger.ToolCalls == 0 || (!force && !progressReviewNeeded(ledger, previousCalls, previousFailures, previousMutations, mutations)) {
		return messages, nil
	}
	reason := progressReviewReason(ledger, previousCalls, previousFailures, previousMutations, mutations, force)
	if reason == "" {
		return messages, nil
	}
	generation := 1
	if previous != nil {
		generation = previous.Generation + 1
	}
	phase := progressReviewPhase(ledger, force)
	risk, confidence := progressReviewAssessment(ledger, generation, previousMutations, mutations)
	review := &progressReview{
		Generation: generation, Phase: phase, Action: progressReviewAction(phase, generation), Risk: risk, Confidence: confidence, Reason: reason, ToolCalls: ledger.ToolCalls,
		FailedToolCalls: ledger.Failed, WorkspaceMutations: mutations, PlanControlCalls: planControls,
	}
	ledger.ProgressReview = review
	baseline := *review
	ledger.ProgressReviewBaseline = &baseline
	return appendExecutionGuidance(replaceExecutionLedger(messages, ledger), ledger), review
}

func progressReviewAction(phase string, generation int) string {
	// A repeated recovery checkpoint with no intervening workspace progress is
	// an escalation, not another prose reminder. Generation two forces a Plan
	// change; generation three asks for an independent Reviewer boundary. Two
	// further failed generations exhaust autonomous recovery.
	if generation >= 5 {
		return "ask_user"
	}
	if generation >= 3 {
		return "review"
	}
	if generation >= 2 {
		return "replan"
	}
	switch phase {
	case "decompose", "execute":
		return "replan"
	case "repair":
		return "retry"
	case "implement":
		return "execute"
	default:
		return "execute"
	}
}

func progressReviewAssessment(ledger executionLedger, generation, previousMutations, mutations int) (risk, confidence string) {
	risk, confidence = "low", "medium"
	if ledger.FailureStreakCount >= 2 || ledger.Failed >= 3 || ledger.FileFocusDecisions >= 8 {
		risk = "high"
	} else if generation >= 2 || ledger.Failed > 0 {
		risk = "medium"
	}
	if generation >= 3 && mutations == previousMutations {
		confidence = "low"
	} else if mutations > previousMutations {
		confidence = "high"
	}
	return risk, confidence
}

func fallbackReviewerToReplan(messages []model.Message, reason string) ([]model.Message, *progressReview) {
	ledger := loadExecutionLedger(messages)
	if ledger.ProgressReview == nil || ledger.ProgressReview.Action != "review" {
		return messages, nil
	}
	fallback := applyReviewerFallback(&ledger, reason)
	return appendExecutionGuidance(replaceExecutionLedger(messages, ledger), ledger), fallback
}

func applyReviewerFallback(ledger *executionLedger, reason string) *progressReview {
	if ledger == nil {
		return nil
	}
	fallback := progressReview{Generation: 1, Phase: "decompose", Action: "replan", Reason: strings.TrimSpace(reason), ToolCalls: ledger.ToolCalls, FailedToolCalls: ledger.Failed, WorkspaceMutations: workspaceMutationCount(*ledger)}
	if ledger.ProgressReview != nil {
		fallback = *ledger.ProgressReview
		fallback.Generation++
		fallback.Phase = "decompose"
		fallback.Action = "replan"
		fallback.Reason = strings.TrimSpace(reason)
	}
	ledger.ProgressReview = &fallback
	baseline := fallback
	ledger.ProgressReviewBaseline = &baseline
	return &fallback
}

func applyReviewerUserEscalation(ledger *executionLedger, reason string) *progressReview {
	if ledger == nil {
		return nil
	}
	escalation := progressReview{
		Generation: 1, Phase: "clarify", Action: "ask_user", Risk: "high", Confidence: "low",
		Reason: strings.TrimSpace(reason), ToolCalls: ledger.ToolCalls, FailedToolCalls: ledger.Failed,
		WorkspaceMutations: workspaceMutationCount(*ledger), PlanControlCalls: planControlCallCount(*ledger),
	}
	if ledger.ProgressReview != nil {
		escalation.Generation = ledger.ProgressReview.Generation + 1
	}
	ledger.ProgressReview = &escalation
	baseline := escalation
	ledger.ProgressReviewBaseline = &baseline
	return &escalation
}

func parseReviewerDecision(call model.ToolCall, result tool.Result) reviewerDecision {
	decision := reviewerDecision{CallID: call.ID, Verdict: reviewcontract.VerdictBlocked}
	var request struct {
		Input struct {
			BasePlanRevision int `json:"base_plan_revision"`
		} `json:"input"`
	}
	_ = json.Unmarshal(call.Arguments, &request)
	decision.BasePlanRevision = request.Input.BasePlanRevision
	var outcome struct {
		ChildRunID string          `json:"child_run_id"`
		Output     json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(result.Content, &outcome); err != nil {
		decision.Summary = "Reviewer result could not be decoded"
		decision.ParseError = compactLedgerText(err.Error(), 400)
		return decision
	}
	decision.ChildRunID = outcome.ChildRunID
	parsed, err := reviewcontract.Parse(outcome.Output)
	if err != nil {
		decision.Summary = "Reviewer returned an invalid structured result"
		decision.ParseError = compactLedgerText(err.Error(), 400)
		return decision
	}
	decision.Verdict = parsed.Verdict
	decision.Summary = compactLedgerText(parsed.Summary, 600)
	for _, finding := range parsed.Findings {
		if len(decision.Findings) >= 3 {
			break
		}
		finding.Summary = compactLedgerText(finding.Summary, 240)
		finding.Evidence = compactLedgerText(finding.Evidence, 420)
		finding.Path = compactLedgerText(finding.Path, 200)
		finding.Recommendation = compactLedgerText(finding.Recommendation, 320)
		decision.Findings = append(decision.Findings, finding)
	}
	for _, change := range parsed.RecommendedPlanChanges {
		// The Reviewer contract is bounded to eight changes. Preserve the whole
		// set in the durable ledger because the deterministic Plan compiler must
		// either apply the complete recommendation atomically or apply none of it.
		// Model-facing projection is compacted separately by projectReviewerDecision.
		if len(decision.RecommendedPlanChanges) >= 8 {
			break
		}
		change.StepID = compactLedgerText(change.StepID, 96)
		change.CriterionID = compactLedgerText(change.CriterionID, 96)
		change.Description = compactLedgerText(change.Description, 320)
		change.Reason = compactLedgerText(change.Reason, 320)
		if len(change.DependsOn) > 16 {
			change.DependsOn = append([]string(nil), change.DependsOn[:16]...)
		}
		if len(change.ToolHints) > 12 {
			change.ToolHints = append([]string(nil), change.ToolHints[:12]...)
		}
		if len(change.AcceptanceCriteria) > 8 {
			change.AcceptanceCriteria = append([]reviewcontract.PlanCriterion(nil), change.AcceptanceCriteria[:8]...)
		}
		for criterionIndex := range change.AcceptanceCriteria {
			change.AcceptanceCriteria[criterionIndex].ID = compactLedgerText(change.AcceptanceCriteria[criterionIndex].ID, 128)
			change.AcceptanceCriteria[criterionIndex].Description = compactLedgerText(change.AcceptanceCriteria[criterionIndex].Description, 1000)
		}
		decision.RecommendedPlanChanges = append(decision.RecommendedPlanChanges, change)
	}
	return decision
}

func progressReviewPhase(ledger executionLedger, force bool) string {
	switch {
	case ledger.FileFocusDecisions >= 8:
		return "decompose"
	case ledger.FailureStreakCount >= 2 || ledger.Failed > 0:
		return "repair"
	case ledger.ListsSinceMutation >= 3:
		return "implement"
	case force:
		return "resume"
	default:
		return "execute"
	}
}

func progressReviewNeeded(ledger executionLedger, previousCalls, previousFailures, previousMutations, mutations int) bool {
	if ledger.FailureStreakCount >= 2 || ledger.ListsSinceMutation >= 3 || ledger.FileFocusDecisions >= 8 {
		return true
	}
	// Three more failures after the last review are enough even when the model
	// alternates tools and avoids the same-tool streak guard.
	if ledger.Failed-previousFailures >= 3 {
		return true
	}
	// Eight tool decisions with no new workspace state catches Plan and
	// verification churn without mistaking a normal read-then-write sequence
	// for a loop.
	return ledger.ToolCalls-previousCalls >= 8 && mutations == previousMutations
}

func progressReviewReason(ledger executionLedger, previousCalls, previousFailures, previousMutations, mutations int, force bool) string {
	switch {
	case ledger.FileFocusDecisions >= 8:
		return fmt.Sprintf("%d consecutive decisions focused on %q without a validation or Plan decomposition boundary", ledger.FileFocusDecisions, ledger.FileFocusPath)
	case ledger.FailureStreakCount >= 2:
		return fmt.Sprintf("%d consecutive %s failures", ledger.FailureStreakCount, ledger.FailureStreakTool)
	case ledger.ListsSinceMutation >= 3:
		return fmt.Sprintf("%d directory observations without a workspace mutation", ledger.ListsSinceMutation)
	case ledger.Failed-previousFailures >= 3:
		return fmt.Sprintf("%d additional tool failures since the previous strategy checkpoint", ledger.Failed-previousFailures)
	case ledger.ToolCalls-previousCalls >= 8 && mutations == previousMutations:
		return fmt.Sprintf("%d tool decisions without a new workspace mutation", ledger.ToolCalls-previousCalls)
	case force && ledger.ToolCalls > previousCalls:
		return "turn rollover requires a progress review before continuation"
	default:
		return ""
	}
}

func workspaceMutationCount(ledger executionLedger) int {
	countSuccessful := func(values []string) int {
		count := 0
		for _, value := range values {
			if !strings.HasSuffix(value, ":failed") {
				count++
			}
		}
		return count
	}
	return countSuccessful(ledger.Writes) + countSuccessful(ledger.Appends) + countSuccessful(ledger.Edits) + countSuccessful(ledger.Promotes)
}

func planControlCallCount(ledger executionLedger) int {
	return ledger.ByTool["update_plan"].Succeeded + ledger.ByTool["update_plan"].Failed +
		ledger.ByTool["update_plan_step"].Succeeded + ledger.ByTool["update_plan_step"].Failed +
		ledger.ByTool["revise_verification"].Succeeded + ledger.ByTool["revise_verification"].Failed
}

func loadExecutionLedger(messages []model.Message) executionLedger {
	ledger := executionLedger{}
	if len(messages) == 0 || messages[0].Role != model.RoleSystem {
		return ledger
	}
	content := messages[0].TextContent()
	start := strings.Index(content, runtimeExecutionLedgerStart)
	if start < 0 {
		return ledger
	}
	start += len(runtimeExecutionLedgerStart)
	end := strings.Index(content[start:], runtimeExecutionLedgerEnd)
	if end < 0 {
		return ledger
	}
	body := content[start : start+end]
	state := strings.Index(body, executionLedgerStatePrefix)
	if state < 0 {
		return ledger
	}
	_ = json.Unmarshal([]byte(strings.TrimSpace(body[state+len(executionLedgerStatePrefix):])), &ledger)
	ensureToolCounters(&ledger)
	return ledger
}

func replaceExecutionLedger(messages []model.Message, ledger executionLedger) []model.Message {
	updated := append([]model.Message(nil), messages...)
	ensureToolCounters(&ledger)
	encoded, _ := json.Marshal(ledger)
	directive := "Verified Tool ledger. Missing counters mean zero. Use exact call_id values from successful_evidence_receipts for Plan evidence; never invent a receipt. It records observations, not a prescribed workflow.\n" + executionLedgerStatePrefix + string(encoded)
	if len(updated) == 0 || updated[0].Role != model.RoleSystem {
		return append([]model.Message{model.TextMessage(model.RoleSystem, runtimeExecutionLedgerStart+directive+runtimeExecutionLedgerEnd)}, updated...)
	}
	content := updated[0].TextContent()
	if start := strings.Index(content, runtimeExecutionLedgerStart); start >= 0 {
		if end := strings.Index(content[start+len(runtimeExecutionLedgerStart):], runtimeExecutionLedgerEnd); end >= 0 {
			end += start + len(runtimeExecutionLedgerStart) + len(runtimeExecutionLedgerEnd)
			content = content[:start] + content[end:]
		}
	}
	content += runtimeExecutionLedgerStart + directive + runtimeExecutionLedgerEnd
	updated[0] = model.TextMessage(model.RoleSystem, content)
	return updated
}

// projectExecutionLedgerForModel keeps the complete durable ledger in the
// checkpoint while giving the model only the state needed for its next
// decision. Tool arguments and repeated failures otherwise grow a mandatory
// system message until a small-context model can no longer compact history.
func projectExecutionLedgerForModel(messages []model.Message) []model.Message {
	if len(messages) == 0 || messages[0].Role != model.RoleSystem ||
		!strings.Contains(messages[0].TextContent(), runtimeExecutionLedgerStart) {
		return append([]model.Message(nil), messages...)
	}
	ledger := loadExecutionLedger(messages)
	projection := ledger
	projection.Writes = tailStrings(projection.Writes, 4)
	projection.Appends = tailStrings(projection.Appends, 4)
	projection.Edits = tailStrings(projection.Edits, 4)
	projection.Promotes = tailStrings(projection.Promotes, 4)
	projection.Other = nil
	projection.Errors = projectLedgerEntries(projection.Errors, 2, true)
	projection.SuccessfulEvidence = projectLedgerEntries(projection.SuccessfulEvidence, 8, false)
	if projection.LastAction != nil {
		entry := projectLedgerEntry(*projection.LastAction, true)
		entry.Arguments = ""
		projection.LastAction = &entry
	}
	projection.NextAction = compactLedgerText(projection.NextAction, 120)
	// Copy maps before trimming values. The durable ledger and the model
	// projection must never share mutable map storage.
	projection.ReadFiles = projectReadProgress(projection.ReadFiles, 4, 4, 3)
	// Cached bodies remain in the durable checkpoint ledger. The provider sees
	// only references/counters; replay happens inside Runtime before execution.
	projection.ReadObservations = nil
	projection.Files = projectFileMutations(projection.Files, 6)
	// The baseline is control-plane bookkeeping, not guidance. Only the active
	// ProgressReview is useful to the model; exposing both duplicates counters
	// and lets a growing recovery history consume protected context budget.
	projection.ProgressReviewBaseline = nil
	projection.ReviewerDecision = projectReviewerDecision(projection.ReviewerDecision, 2)

	// Reduce optional evidence in a deterministic order until the complete
	// serialized runtime block fits the model-facing ceiling. Counters,
	// failure state, and the latest action are retained as the recovery core.
	for _, shrink := range []func(){
		func() { projection.SuccessfulEvidence = projectLedgerEntries(projection.SuccessfulEvidence, 2, false) },
		func() { projection.Errors = projectLedgerEntries(projection.Errors, 1, true) },
		func() { projection.ReadFiles = projectReadProgress(projection.ReadFiles, 3, 2, 2) },
		func() { projection.Files = projectFileMutations(projection.Files, 4) },
		func() {
			projection.Writes = tailStrings(projection.Writes, 2)
			projection.Appends = tailStrings(projection.Appends, 2)
			projection.Edits = tailStrings(projection.Edits, 2)
			projection.Promotes = tailStrings(projection.Promotes, 2)
		},
		func() { projection.SuccessfulEvidence = nil; projection.ReadFiles = nil },
		func() { projection.Files = nil; projection.NextAction = compactLedgerText(projection.NextAction, 72) },
		func() { projection.ReviewerDecision = projectReviewerDecision(projection.ReviewerDecision, 1) },
	} {
		if encoded, _ := json.Marshal(projection); len(encoded) <= modelLedgerMaxChars {
			break
		}
		shrink()
	}
	return replaceExecutionLedger(messages, projection)
}

func projectReviewerDecision(decision *reviewerDecision, limit int) *reviewerDecision {
	if decision == nil {
		return nil
	}
	projected := *decision
	projected.Summary = compactLedgerText(projected.Summary, 320)
	projected.ParseError = compactLedgerText(projected.ParseError, 240)
	if limit < 0 {
		limit = 0
	}
	if len(projected.Findings) > limit {
		projected.Findings = append([]reviewcontract.Finding(nil), projected.Findings[:limit]...)
	}
	if len(projected.RecommendedPlanChanges) > limit {
		projected.RecommendedPlanChanges = append([]reviewcontract.PlanChange(nil), projected.RecommendedPlanChanges[:limit]...)
	}
	return &projected
}

func projectReadProgress(values map[string]readProgress, pathLimit, successLimit, failureLimit int) map[string]readProgress {
	if len(values) == 0 || pathLimit <= 0 {
		return nil
	}
	paths := make([]string, 0, len(values))
	for path := range values {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	if len(paths) > pathLimit {
		paths = paths[len(paths)-pathLimit:]
	}
	projected := make(map[string]readProgress, len(paths))
	for _, path := range paths {
		progress := values[path]
		projected[path] = readProgress{
			Succeeded: tailStrings(progress.Succeeded, successLimit),
			Failed:    tailStrings(progress.Failed, failureLimit),
		}
	}
	return projected
}

func parseFileMutation(call model.ToolCall, result tool.Result) (fileMutation, bool) {
	if call.Name != "write_file" && call.Name != "append_file" && call.Name != "edit_file" && call.Name != "promote_file" {
		return fileMutation{}, false
	}
	var payload struct {
		Path          string `json:"path"`
		Bytes         int    `json:"bytes"`
		LineCount     int    `json:"line_count"`
		FileSHA256    string `json:"file_sha256"`
		ContentSHA256 string `json:"content_sha256"`
		LegacySHA256  string `json:"sha256"`
		SyntaxStatus  string `json:"syntax_status"`
	}
	if json.Unmarshal(result.Content, &payload) != nil || strings.TrimSpace(payload.Path) == "" {
		return fileMutation{}, false
	}
	fileHash := strings.TrimSpace(payload.FileSHA256)
	if fileHash == "" {
		fileHash = strings.TrimSpace(payload.ContentSHA256)
	}
	if fileHash == "" {
		fileHash = strings.TrimSpace(payload.LegacySHA256)
	}
	mutation := fileMutation{Path: payload.Path, LastOperation: call.Name, FileSHA256: fileHash, Bytes: payload.Bytes, LineCount: payload.LineCount, SyntaxStatus: payload.SyntaxStatus}
	for key, value := range result.Meta {
		if strings.HasSuffix(key, "_kind") && value == "workspace_file" {
			prefix := strings.TrimSuffix(key, "_kind")
			mutation.ArtifactID = strings.TrimSpace(result.Meta[prefix])
			mutation.ArtifactSHA256 = strings.TrimSpace(result.Meta[prefix+"_sha256"])
			break
		}
	}
	return mutation, true
}

func recordSyntaxVerification(ledger *executionLedger, call model.ToolCall, result tool.Result) {
	var input struct {
		Command string   `json:"command"`
		Args    []string `json:"args"`
	}
	if json.Unmarshal(call.Arguments, &input) != nil || len(input.Args) < 3 {
		return
	}
	module := ""
	for index := 0; index+1 < len(input.Args); index++ {
		if input.Args[index] == "-m" && (input.Args[index+1] == "py_compile" || input.Args[index+1] == "compileall") {
			module = input.Args[index+1]
			break
		}
	}
	if module == "" {
		return
	}
	target := strings.TrimSpace(input.Args[len(input.Args)-1])
	if target == "" || strings.HasPrefix(target, "-") {
		return
	}
	var output struct {
		ExitCode int    `json:"exit_code"`
		Stderr   string `json:"stderr"`
	}
	if json.Unmarshal(result.Content, &output) != nil {
		return
	}
	status := "passed"
	if result.IsError || output.ExitCode != 0 {
		status = "failed"
	}
	if ledger.Files == nil {
		ledger.Files = make(map[string]fileMutation)
	}
	path := target
	for knownPath := range ledger.Files {
		if knownPath == target || strings.HasSuffix(knownPath, "/"+target) {
			path = knownPath
			break
		}
	}
	file := ledger.Files[path]
	file.Path = path
	file.SyntaxStatus = status
	if status == "failed" {
		errorText := strings.TrimSpace(output.Stderr)
		if errorText == "" {
			errorText = strings.TrimSpace(result.Error)
		}
		file.VerificationError = compactLedgerText(errorText, 240)
	} else {
		file.VerificationError = ""
	}
	file.LastOperation = module
	ledger.Files[path] = file
}

func projectFileMutations(values map[string]fileMutation, limit int) map[string]fileMutation {
	if len(values) <= limit {
		return values
	}
	ordered := latestFileMutations(values, len(values))
	projected := make(map[string]fileMutation, limit)
	for _, value := range ordered[:limit] {
		projected[value.Path] = value
	}
	return projected
}

func latestFileMutations(values map[string]fileMutation, limit int) []fileMutation {
	if limit <= 0 || len(values) == 0 {
		return nil
	}
	ordered := make([]fileMutation, 0, len(values))
	for _, value := range values {
		ordered = append(ordered, value)
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Revision != ordered[j].Revision {
			return ordered[i].Revision > ordered[j].Revision
		}
		return ordered[i].Path < ordered[j].Path
	})
	if limit < len(ordered) {
		ordered = ordered[:limit]
	}
	return ordered
}

func projectLedgerEntries(values []ledgerEntry, limit int, keepError bool) []ledgerEntry {
	if len(values) > limit {
		values = values[len(values)-limit:]
	}
	projected := make([]ledgerEntry, 0, len(values))
	for _, value := range values {
		projected = append(projected, projectLedgerEntry(value, keepError))
	}
	return projected
}

func projectLedgerEntry(value ledgerEntry, keepError bool) ledgerEntry {
	value.Arguments = compactEvidenceArguments(value.Tool, value.Arguments)
	if keepError {
		value.Error = compactLedgerText(value.Error, 180)
	} else {
		value.Error = ""
	}
	return value
}

func compactEvidenceArguments(name, arguments string) string {
	var object map[string]any
	if json.Unmarshal([]byte(arguments), &object) != nil {
		if path := partialJSONStringField(arguments, "path"); path != "" {
			compact := map[string]any{"path": path}
			encoded, _ := json.Marshal(compact)
			return string(encoded)
		}
		return compactLedgerText(arguments, 72)
	}
	keys := []string{"path"}
	if name == "promote_file" {
		keys = []string{"source_path", "target_path", "expected_source_sha256", "expected_target_sha256"}
	}
	if name == "read_file" {
		keys = append(keys, "start_line", "line_count")
	}
	compact := make(map[string]any, len(keys))
	for _, key := range keys {
		if value, ok := object[key]; ok {
			compact[key] = value
		}
	}
	if len(compact) == 0 {
		return compactLedgerText(arguments, 72)
	}
	encoded, _ := json.Marshal(compact)
	return string(encoded)
}

func partialJSONStringField(value, key string) string {
	marker := `"` + key + `":`
	index := strings.Index(value, marker)
	if index < 0 {
		return ""
	}
	var result string
	decoder := json.NewDecoder(strings.NewReader(strings.TrimSpace(value[index+len(marker):])))
	if decoder.Decode(&result) != nil {
		return ""
	}
	return result
}

func compactLedgerText(value string, limit int) string {
	runes := []rune(strings.TrimSpace(value))
	if len(runes) <= limit {
		return string(runes)
	}
	return string(runes[:limit]) + "…"
}

func tailStrings(values []string, limit int) []string {
	if len(values) > limit {
		values = values[len(values)-limit:]
	}
	return append([]string(nil), values...)
}

func executionLedgerJSON(messages []model.Message) json.RawMessage {
	ledger := loadExecutionLedger(messages)
	encoded, _ := json.Marshal(ledger)
	return encoded
}

func restoreExecutionLedger(messages []model.Message, raw json.RawMessage) []model.Message {
	if len(raw) == 0 {
		return messages
	}
	var ledger executionLedger
	if json.Unmarshal(raw, &ledger) != nil {
		return messages
	}
	ensureToolCounters(&ledger)
	return replaceExecutionLedger(messages, ledger)
}

func ensureToolCounters(ledger *executionLedger) {
	if ledger.ByTool == nil {
		ledger.ByTool = make(map[string]toolProgress)
	}
}

func compactLedgerArguments(name string, arguments json.RawMessage) string {
	var object map[string]any
	if json.Unmarshal(arguments, &object) == nil {
		// File bodies are already immutable in TOOL_CALLED events and Artifacts;
		// duplicating them in every checkpoint ledger has no decision value.
		for _, key := range []string{"content", "text"} {
			if value, ok := object[key].(string); ok {
				object[key] = fmt.Sprintf("<omitted:%d chars>", len([]rune(value)))
			}
		}
		for _, key := range []string{"old_text", "new_text"} {
			if value, ok := object[key].(string); ok {
				object[key] = compactLedgerText(value, 48)
			}
		}
		encoded, _ := json.Marshal(object)
		text := string(encoded)
		if len([]rune(text)) <= 180 {
			return text
		}
		if compact := compactEvidenceArguments(name, text); compact != "" {
			return compact
		}
	}
	return compactLedgerText(string(arguments), 180)
}

func ledgerPathStatus(arguments json.RawMessage, failed bool) string {
	path := ledgerPath(arguments)
	status := "ok"
	if failed {
		status = "failed"
	}
	return path + ":" + status
}

func ledgerPath(arguments json.RawMessage) string {
	var input struct {
		Path       string `json:"path"`
		FilePath   string `json:"file_path"`
		TargetPath string `json:"target_path"`
	}
	_ = json.Unmarshal(arguments, &input)
	if input.Path == "" {
		input.Path = input.FilePath
	}
	if input.Path == "" {
		input.Path = input.TargetPath
	}
	return input.Path
}

func appendBounded(values []string, value string, limit int) []string {
	values = append(values, value)
	if len(values) > limit {
		values = values[len(values)-limit:]
	}
	return values
}

func appendBoundedEntry(values []ledgerEntry, value ledgerEntry, limit int) []ledgerEntry {
	if len(values) > 0 && values[len(values)-1] == value {
		return values
	}
	values = append(values, value)
	if len(values) > limit {
		values = values[len(values)-limit:]
	}
	return values
}

func latestSuccessfulPath(groups ...[]string) string {
	for _, values := range groups {
		for index := len(values) - 1; index >= 0; index-- {
			if strings.HasSuffix(values[index], ":ok") {
				return strings.TrimSuffix(values[index], ":ok")
			}
		}
	}
	return ""
}
