// Package taskplan defines the durable, user-visible plan for one Workflow.
package taskplan

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

var ErrNotFound = errors.New("task plan not found")

const (
	StatusPending        = "pending"
	StatusInProgress     = "in_progress"
	StatusCompleted      = "completed"
	StatusBlocked        = "blocked"
	StatusStale          = "stale"
	StatusSkipped        = "skipped"
	CriterionPending     = "pending"
	CriterionPassed      = "passed"
	CriterionFailed      = "failed"
	CriterionSkipped     = "skipped"
	CriterionInvalid     = "invalid"
	CriterionUnsupported = "unsupported"
	CriterionStale       = "stale"

	EnforcementInformational = "informational"
	EnforcementAdvisory      = "advisory"
	EnforcementRequired      = "required"
	EnforcementReleaseGate   = "release_gate"

	OriginUserExplicit     = "user_explicit"
	OriginDeploymentPolicy = "deployment_policy"
	OriginAgentInferred    = "agent_inferred"
	OriginProviderRequired = "provider_required"

	VerificationReasonSpecInvalid         = "spec_invalid"
	VerificationReasonProviderUnavailable = "provider_unavailable"
	VerificationReasonEvidenceMissing     = "evidence_missing"
	VerificationReasonAssertionFailed     = "assertion_failed"
	VerificationReasonEvidenceStale       = "evidence_stale"
	VerificationReasonPolicyOverridden    = "policy_overridden"
)

// AcceptanceCriterion makes completion observable. The planner defines the
// condition up front; the runtime marks it passed only after resolving a
// matching immutable Tool receipt. Evidence fields are platform-owned output,
// not values the model is expected to copy or invent.
type AcceptanceCriterion struct {
	ID          string `json:"id"`
	Description string `json:"description"`
	Status      string `json:"status"`
	// Enforcement is platform-owned policy. Criteria inferred by a planner are
	// advisory by default; only explicit policy criteria may block completion.
	Enforcement string `json:"enforcement,omitempty"`
	// Origin records who introduced the criterion. A model-authored Plan never
	// gains user_explicit or release_gate authority merely by naming it.
	Origin       string           `json:"origin,omitempty"`
	Verification VerificationSpec `json:"verification,omitempty"`
	// VerificationReason and VerificationMessage are platform diagnostics,
	// separate from the model-owned description and step result.
	VerificationReason  string   `json:"verification_reason,omitempty"`
	VerificationMessage string   `json:"verification_message,omitempty"`
	Evidence            string   `json:"evidence,omitempty"`
	EvidenceCallIDs     []string `json:"evidence_call_ids,omitempty"`
}

// BlocksCompletion reports whether policy requires a positive verification
// verdict before the owning Step may close.
func (c AcceptanceCriterion) BlocksCompletion() bool {
	switch effectiveEnforcement(c.Enforcement) {
	case EnforcementRequired, EnforcementReleaseGate:
		return true
	default:
		return false
	}
}

func effectiveEnforcement(value string) string {
	if strings.TrimSpace(value) == "" {
		return EnforcementAdvisory
	}
	return strings.TrimSpace(value)
}

func effectiveOrigin(value string) string {
	if strings.TrimSpace(value) == "" {
		return OriginAgentInferred
	}
	return strings.TrimSpace(value)
}

// VerificationSpec gives the runtime a machine-checkable meaning for a
// criterion. Description and evidence remain user-facing; neither is trusted
// to decide whether an unrelated successful Tool call proves completion.
type VerificationSpec struct {
	Kind       string                  `json:"kind,omitempty"`
	Target     string                  `json:"target,omitempty"`
	Match      string                  `json:"match,omitempty"`
	Tool       string                  `json:"tool,omitempty"`
	Arguments  json.RawMessage         `json:"arguments,omitempty"`
	Assertions []VerificationAssertion `json:"assertions,omitempty"`
}

// EvidenceTools returns the action tools whose successful receipts can satisfy
// this verification contract. The order is also the runtime's preferred
// recovery order when a model under-specifies a step's tool_hints.
func (v VerificationSpec) EvidenceTools() []string {
	return VerificationEvidenceTools(v)
}

// EvidenceValidationError preserves the exact failed Plan criterion so the
// runtime can give the model and UI an actionable recovery contract instead of
// flattening every failure into "open plan steps".
type EvidenceValidationError struct {
	StepID       string
	CriterionID  string
	Description  string
	Verification VerificationSpec
	Cause        error
}

func (e *EvidenceValidationError) Error() string {
	if e == nil {
		return "invalid completion evidence"
	}
	prefix := fmt.Sprintf("criterion %q", e.CriterionID)
	if e.StepID != "" {
		prefix = fmt.Sprintf("step %q %s", e.StepID, prefix)
	}
	if e.Cause == nil {
		return prefix + ": invalid completion evidence"
	}
	return prefix + ": " + e.Cause.Error()
}

func (e *EvidenceValidationError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

type Step struct {
	ID             string   `json:"id"`
	Description    string   `json:"description"`
	Status         string   `json:"status"`
	Assignee       string   `json:"assignee,omitempty"`
	AgentVersionID string   `json:"agent_version_id,omitempty"`
	DependsOn      []string `json:"depends_on,omitempty"`
	ToolHints      []string `json:"tool_hints,omitempty"`
	Result         string   `json:"result,omitempty"`
	// State is the platform-owned structured projection. Legacy fields above
	// remain for model/API compatibility and are synchronized during normalize.
	State              NodeState             `json:"state,omitempty"`
	AcceptanceCriteria []AcceptanceCriterion `json:"acceptance_criteria,omitempty"`
}

// PlanNode is the canonical name for a durable Todo in the plan DAG. Step is
// retained as a source/API compatibility alias; it must not be confused with
// a ReAct execution step or an ActionAttempt.
type PlanNode = Step

type Update struct {
	Goal         string        `json:"goal"`
	Explanation  string        `json:"explanation,omitempty"`
	Steps        []Step        `json:"steps"`
	ChangeMode   string        `json:"change_mode,omitempty"`
	BaseRevision *int          `json:"base_revision,omitempty"`
	ReplanReason string        `json:"replan_reason,omitempty"`
	RetiredSteps []RetiredStep `json:"retired_steps,omitempty"`
}

// RetiredStep makes deletion of an unfinished Plan node explicit and
// auditable. Omitting an open node from a replan is never interpreted as an
// implicit delete.
type RetiredStep struct {
	ID     string `json:"id"`
	Reason string `json:"reason"`
}

// StepUpdate is the compact mutation used while executing an existing Plan.
// It avoids resending a large checklist after every observation.
type StepUpdate struct {
	StepID string `json:"step_id"`
	Status string `json:"status"`
	Result string `json:"result,omitempty"`
	// ToolHints is a pointer so an omitted field preserves the current allowlist,
	// while an explicitly empty list clears it. This lets the Agent repair an
	// under-specified active step without replacing the full durable plan.
	ToolHints          *[]string             `json:"tool_hints,omitempty"`
	AcceptanceCriteria []AcceptanceCriterion `json:"acceptance_criteria,omitempty"`
}

// CriterionRevision repairs one verification contract without allowing the
// model to replace the goal, reorder Todos, or erase unrelated progress.
type CriterionRevision struct {
	StepID       string           `json:"step_id"`
	CriterionID  string           `json:"criterion_id"`
	Action       string           `json:"action"`
	Verification VerificationSpec `json:"verification,omitempty"`
	Reason       string           `json:"reason"`
	// BaseRevision is optional for ordinary model-authored local repairs and
	// required by runtime-compiled Reviewer patches. PostgreSQL checks it under
	// the same Plan row lock used to commit the new revision.
	BaseRevision *int `json:"base_revision,omitempty"`
}

// NodeUsage is a machine-readable cost/latency projection for one Plan node.
// It is populated by the runtime projector from committed model/tool events;
// it is never inferred from the model's prose result.
type NodeUsage struct {
	InputTokens  int64   `json:"input_tokens,omitempty"`
	OutputTokens int64   `json:"output_tokens,omitempty"`
	TotalTokens  int64   `json:"total_tokens,omitempty"`
	CostUSD      float64 `json:"cost_usd,omitempty"`
	DurationMS   int64   `json:"duration_ms,omitempty"`
}

// NodeTestResult is the normalized result of a platform-recorded validation.
// ExitCode is optional because HTTP/MCP/approval checks do not have a process
// exit code. ToolCallID/EventSequence link back to immutable evidence.
type NodeTestResult struct {
	Name          string `json:"name"`
	Kind          string `json:"kind,omitempty"`
	Status        string `json:"status"`
	ExitCode      *int   `json:"exit_code,omitempty"`
	Message       string `json:"message,omitempty"`
	ToolCallID    string `json:"tool_call_id,omitempty"`
	EventSequence int64  `json:"event_sequence,omitempty"`
}

// NodeState is the structured execution state of a Plan node. Output and
// tests are deliberately separate: a model can describe an output, but only
// a committed Artifact/Tool event can produce a passing test result.
type NodeState struct {
	Revision        int64            `json:"revision,omitempty"`
	Status          string           `json:"status,omitempty"`
	Attempts        int              `json:"attempts,omitempty"`
	LastRunID       string           `json:"last_run_id,omitempty"`
	LastEventSeq    int64            `json:"last_event_sequence,omitempty"`
	Output          string           `json:"output,omitempty"`
	ArtifactIDs     []string         `json:"artifact_ids,omitempty"`
	Tests           []NodeTestResult `json:"tests,omitempty"`
	Usage           NodeUsage        `json:"usage,omitempty"`
	BlockedReason   string           `json:"blocked_reason,omitempty"`
	RetryFromNodeID string           `json:"retry_from_node_id,omitempty"`
	NextNodeIDs     []string         `json:"next_node_ids,omitempty"`
}

// NodeAction is the platform's deterministic scheduling decision for a Plan
// node. The model may propose a graph, but the runtime decides whether a node
// is executable from persisted status, dependencies and recorded test facts.
type NodeAction string

const (
	NodeActionRun      NodeAction = "run"
	NodeActionRetry    NodeAction = "retry"
	NodeActionWait     NodeAction = "wait"
	NodeActionComplete NodeAction = "complete"
	NodeActionNone     NodeAction = "none"
)

// GraphState is the platform-owned aggregate projection of a Plan DAG. The
// individual NodeState values remain the source facts; GraphState is the
// structured summary consumed by the scheduler and UI.
type GraphState struct {
	Status         string     `json:"status,omitempty"`
	ActiveNodeIDs  []string   `json:"active_node_ids,omitempty"`
	ReadyNodeIDs   []string   `json:"ready_node_ids,omitempty"`
	BlockedNodeIDs []string   `json:"blocked_node_ids,omitempty"`
	RetryNodeID    string     `json:"retry_node_id,omitempty"`
	NextNodeID     string     `json:"next_node_id,omitempty"`
	NextAction     NodeAction `json:"next_action,omitempty"`
}

// NextDecision returns the first deterministic action the scheduler should
// take. A failed platform-recorded test always wins over the model's prose
// result and routes the node back to retry. This is intentionally a pure
// projection over the structured Plan, so checkpoint replay and UI previews
// produce the same answer without another model call.
func (p Plan) NextDecision() (nodeID string, action NodeAction) {
	terminal := make(map[string]bool, len(p.Steps))
	for _, step := range p.Steps {
		terminal[step.ID] = (step.Status == StatusCompleted || step.Status == StatusSkipped) && !hasFailedNodeTest(step.State.Tests)
	}
	for _, step := range p.Steps {
		if hasFailedNodeTest(step.State.Tests) {
			return step.ID, NodeActionRetry
		}
		if step.Status == StatusStale {
			return step.ID, NodeActionRetry
		}
		if step.Status == StatusBlocked {
			return step.ID, NodeActionWait
		}
	}
	for _, step := range p.Steps {
		switch step.Status {
		case StatusInProgress:
			return step.ID, NodeActionWait
		case StatusPending:
			ready := true
			for _, dependency := range step.DependsOn {
				if !terminal[dependency] {
					ready = false
					break
				}
			}
			if ready {
				return step.ID, NodeActionRun
			}
		}
	}
	if len(p.Steps) > 0 {
		for _, step := range p.Steps {
			if step.Status != StatusCompleted && step.Status != StatusSkipped {
				return step.ID, NodeActionWait
			}
		}
		return "", NodeActionComplete
	}
	return "", NodeActionNone
}

// DeriveGraphState creates the deterministic aggregate used by API/UI
// projections. It delegates the final decision to NextDecision so scheduler
// behavior and displayed state cannot disagree about retry/ready/wait.
func (p Plan) DeriveGraphState() GraphState {
	state := GraphState{Status: "idle"}
	for _, step := range p.Steps {
		switch step.Status {
		case StatusInProgress:
			state.ActiveNodeIDs = append(state.ActiveNodeIDs, step.ID)
		case StatusBlocked:
			state.BlockedNodeIDs = append(state.BlockedNodeIDs, step.ID)
		case StatusStale:
			state.RetryNodeID = step.ID
		}
	}
	for _, step := range p.ReadyNodes() {
		state.ReadyNodeIDs = append(state.ReadyNodeIDs, step.ID)
	}
	nodeID, action := p.NextDecision()
	state.NextNodeID, state.NextAction = nodeID, action
	switch action {
	case NodeActionRetry:
		state.Status, state.RetryNodeID = "retry_required", nodeID
	case NodeActionComplete:
		state.Status = "completed"
	case NodeActionWait:
		if len(state.ActiveNodeIDs) > 0 {
			state.Status = "running"
		} else {
			state.Status = "waiting"
		}
	case NodeActionRun:
		state.Status = "ready"
	}
	if state.Status == "idle" && len(state.ActiveNodeIDs) > 0 {
		state.Status = "running"
	}
	return state
}

func hasFailedNodeTest(tests []NodeTestResult) bool {
	for _, test := range tests {
		if test.Status == CriterionFailed || test.Status == "failed" || test.Status == "error" {
			return true
		}
	}
	return false
}

type Plan struct {
	// PlanID is the stable identity of the Workflow-owned plan.
	PlanID string `json:"plan_id"`
	// RunID is the Run that created the Plan and is immutable. Later attempts
	// are recorded separately in LastModifiedRunID and Plan events.
	RunID               string     `json:"run_id"`
	LastModifiedRunID   string     `json:"last_modified_run_id,omitempty"`
	WorkflowID          string     `json:"workflow_id,omitempty"`
	TenantID            string     `json:"tenant_id"`
	Revision            int        `json:"revision"`
	OriginalGoal        string     `json:"original_goal,omitempty"`
	Goal                string     `json:"goal"`
	Explanation         string     `json:"explanation,omitempty"`
	Steps               []Step     `json:"steps"`
	GraphState          GraphState `json:"graph_state"`
	ExecutionOutcome    string     `json:"execution_outcome"`
	VerificationOutcome string     `json:"verification_outcome"`
	CreatedAt           time.Time  `json:"created_at"`
	UpdatedAt           time.Time  `json:"updated_at"`
}

// EffectiveRunID is the attempt that produced the current revision. It keeps
// verification/evidence joins correct without changing Plan ownership.
func (p Plan) EffectiveRunID() string {
	if p.LastModifiedRunID != "" {
		return p.LastModifiedRunID
	}
	return p.RunID
}

// ReadyNodes returns every pending node whose dependencies are terminal. The
// legacy reconciler activates only one node for backward compatibility; new
// schedulers should use this function to support independent parallel work.
func (p Plan) ReadyNodes() []PlanNode {
	terminal := make(map[string]bool, len(p.Steps))
	for _, step := range p.Steps {
		terminal[step.ID] = (step.Status == StatusCompleted || step.Status == StatusSkipped) && !hasFailedNodeTest(step.State.Tests)
	}
	ready := make([]PlanNode, 0)
	for _, step := range p.Steps {
		if step.Status != StatusPending {
			continue
		}
		ok := true
		for _, dependency := range step.DependsOn {
			if !terminal[dependency] {
				ok = false
				break
			}
		}
		if ok {
			ready = append(ready, step)
		}
	}
	return ready
}

// DeriveOutcomes projects execution progress and verification confidence
// independently. It is API/UI output derived from the durable Plan, not a
// second persisted source of truth.
func (p Plan) DeriveOutcomes() (execution, verification string) {
	execution = "completed"
	for _, step := range p.Steps {
		switch step.Status {
		case StatusBlocked, StatusStale:
			execution = "blocked"
		case StatusPending, StatusInProgress:
			if execution != "blocked" {
				execution = "running"
			}
		}
	}
	total, passed, mandatoryOpen := 0, 0, 0
	for _, step := range p.Steps {
		for _, criterion := range step.AcceptanceCriteria {
			total++
			if criterion.Status == CriterionPassed {
				passed++
			} else if criterion.BlocksCompletion() {
				mandatoryOpen++
			}
		}
	}
	switch {
	case mandatoryOpen > 0:
		verification = "rejected"
	case total == 0 || passed == 0:
		verification = "unverified"
	case passed == total:
		verification = "verified"
	default:
		verification = "partially_verified"
	}
	return execution, verification
}

// ReconcileSteps treats proposed as the complete next plan revision, preserves
// durable progress for matching node IDs, and selects the next dependency-ready
// step. Omitted nodes leave the active graph instead of accumulating across
// revisions. update_plan_step is the delta API for ordinary execution progress.
func ReconcileSteps(previous, proposed []Step) []Step {
	previousByID := make(map[string]Step, len(previous))
	for _, step := range previous {
		previousByID[step.ID] = step
	}
	steps := make([]Step, 0, len(proposed))
	for _, proposedStep := range proposed {
		candidate := proposedStep
		old, exists := previousByID[candidate.ID]
		if !exists {
			steps = append(steps, candidate)
			continue
		}
		reopened := false
		switch old.Status {
		case StatusCompleted, StatusSkipped:
			// A closed step may be reopened only by an explicit fresh plan whose
			// criteria have all been reset. update_plan is hidden during normal
			// execution and exposed again only when completion evidence becomes
			// invalid, so stale checklist echoes still preserve terminal progress.
			if candidate.Status == StatusInProgress && criteriaReset(candidate.AcceptanceCriteria) {
				reopened = true
			} else {
				candidate.Status = old.Status
			}
		case StatusInProgress:
			if candidate.Status == StatusPending {
				candidate.Status = StatusInProgress
			}
		}
		if reopened {
			// An explicit evidence reset starts a fresh attempt. Do not carry
			// terminal output, receipts, or passed criteria into the retry.
			steps = append(steps, candidate)
			continue
		}
		if candidate.Result == "" && old.Result != "" {
			candidate.Result = old.Result
		}
		// State is platform-owned. Preserve all receipts/usage from the
		// previous projection and only refresh the model-facing status/output.
		previousState := old.State
		previousState.Status = candidate.Status
		if candidate.Result != "" {
			previousState.Output = candidate.Result
		}
		candidate.State = previousState
		candidate.AcceptanceCriteria = reconcileCriteria(old.AcceptanceCriteria, candidate.AcceptanceCriteria)
		steps = append(steps, candidate)
	}
	for _, step := range steps {
		if step.Status == StatusInProgress {
			return steps
		}
	}
	completed := make(map[string]bool, len(steps))
	for _, step := range steps {
		completed[step.ID] = step.Status == StatusCompleted || step.Status == StatusSkipped
	}
	for index := range steps {
		if steps[index].Status != StatusPending {
			continue
		}
		ready := true
		for _, dependency := range steps[index].DependsOn {
			if !completed[dependency] {
				ready = false
				break
			}
		}
		if ready {
			steps[index].Status = StatusInProgress
			break
		}
	}
	return steps
}

// NormalizeNodeState keeps the legacy status/result fields and the structured
// projection coherent across old plans, model updates, and API reads.
func NormalizeNodeState(step *Step) {
	if step == nil {
		return
	}
	if strings.TrimSpace(step.Status) == "" {
		step.Status = step.State.Status
	}
	if strings.TrimSpace(step.Status) == "" {
		step.Status = StatusPending
	}
	step.State.Status = step.Status
	if step.State.Output == "" {
		step.State.Output = step.Result
	}
	if step.Result == "" {
		step.Result = step.State.Output
	}
	if step.State.Attempts < 0 {
		step.State.Attempts = 0
	}
}

func criteriaReset(criteria []AcceptanceCriterion) bool {
	if len(criteria) == 0 {
		return false
	}
	for _, criterion := range criteria {
		if criterion.Status != CriterionPending || strings.TrimSpace(criterion.Evidence) != "" || len(criterion.EvidenceCallIDs) != 0 {
			return false
		}
	}
	return true
}

// HasOpenWork reports whether a plan still contains runnable or waiting work.
func (p Plan) HasOpenWork() bool {
	for _, step := range p.Steps {
		if step.Status == StatusPending || step.Status == StatusInProgress || step.Status == StatusBlocked || step.Status == StatusStale {
			return true
		}
		for _, criterion := range step.AcceptanceCriteria {
			if criterion.BlocksCompletion() && criterion.Status != CriterionPassed {
				return true
			}
		}
	}
	return false
}

func reconcileCriteria(previous, proposed []AcceptanceCriterion) []AcceptanceCriterion {
	criteria := append([]AcceptanceCriterion(nil), proposed...)
	previousByID := make(map[string]AcceptanceCriterion, len(previous))
	for _, criterion := range previous {
		previousByID[criterion.ID] = criterion
	}
	for index := range criteria {
		old, exists := previousByID[criteria[index].ID]
		if !exists {
			continue
		}
		if old.Status == CriterionPassed || old.Status == CriterionSkipped {
			criteria[index].Status = old.Status
			if criteria[index].Evidence == "" {
				criteria[index].Evidence = old.Evidence
			}
			if len(criteria[index].EvidenceCallIDs) == 0 {
				criteria[index].EvidenceCallIDs = append([]string(nil), old.EvidenceCallIDs...)
			}
		}
		if criteria[index].Verification.Kind == "" {
			criteria[index].Verification = old.Verification
		}
		if criteria[index].Enforcement == "" {
			criteria[index].Enforcement = old.Enforcement
		}
		if criteria[index].Origin == "" {
			criteria[index].Origin = old.Origin
		}
		if criteria[index].VerificationReason == "" {
			criteria[index].VerificationReason = old.VerificationReason
		}
		if criteria[index].VerificationMessage == "" {
			criteria[index].VerificationMessage = old.VerificationMessage
		}
	}
	return criteria
}

func (u Update) Validate() error {
	if strings.TrimSpace(u.Goal) == "" || len(u.Steps) == 0 || len(u.Steps) > 8 {
		return errors.New("plan goal and 1-8 steps are required")
	}
	if len([]rune(u.Goal)) > 2000 || len([]rune(u.Explanation)) > 8000 {
		return errors.New("plan goal or explanation is too long")
	}
	ids := make(map[string]struct{}, len(u.Steps))
	active := 0
	for _, step := range u.Steps {
		if strings.TrimSpace(step.ID) == "" || strings.TrimSpace(step.Description) == "" {
			return errors.New("every plan step requires id and description")
		}
		if len(step.AcceptanceCriteria) == 0 {
			return errors.New("every plan step requires at least one acceptance criterion")
		}
		if len([]rune(step.ID)) > 128 || len([]rune(step.Description)) > 2000 || len([]rune(step.Result)) > 8000 {
			return errors.New("plan step field is too long")
		}
		criterionIDs := make(map[string]struct{}, len(step.AcceptanceCriteria))
		if len(step.ToolHints) > 12 {
			return errors.New("plan step tool_hints must contain at most 12 tools")
		}
		toolHints := make(map[string]struct{}, len(step.ToolHints))
		for _, name := range step.ToolHints {
			name = strings.TrimSpace(name)
			if name == "" {
				return errors.New("plan step tool_hint must not be empty")
			}
			if _, exists := toolHints[name]; exists {
				return errors.New("plan step tool_hints must be unique")
			}
			toolHints[name] = struct{}{}
		}
		for _, criterion := range step.AcceptanceCriteria {
			if strings.TrimSpace(criterion.ID) == "" || strings.TrimSpace(criterion.Description) == "" {
				return errors.New("every acceptance criterion requires id and description")
			}
			if len([]rune(criterion.ID)) > 128 || len([]rune(criterion.Description)) > 1000 || len([]rune(criterion.Evidence)) > 4000 || len([]rune(criterion.VerificationMessage)) > 4000 {
				return errors.New("acceptance criterion field is too long")
			}
			if _, exists := criterionIDs[criterion.ID]; exists {
				return errors.New("acceptance criterion ids must be unique within a step")
			}
			criterionIDs[criterion.ID] = struct{}{}
			switch effectiveEnforcement(criterion.Enforcement) {
			case EnforcementInformational, EnforcementAdvisory, EnforcementRequired, EnforcementReleaseGate:
			default:
				return errors.New("invalid acceptance criterion enforcement")
			}
			switch effectiveOrigin(criterion.Origin) {
			case OriginUserExplicit, OriginDeploymentPolicy, OriginAgentInferred, OriginProviderRequired:
			default:
				return errors.New("invalid acceptance criterion origin")
			}
			if criterion.Status != CriterionInvalid && criterion.Status != CriterionUnsupported {
				if err := criterion.Verification.Validate(); err != nil {
					return err
				}
			}
			callIDs := make(map[string]struct{}, len(criterion.EvidenceCallIDs))
			for _, callID := range criterion.EvidenceCallIDs {
				if strings.TrimSpace(callID) == "" || len([]rune(callID)) > 256 {
					return errors.New("acceptance evidence call id is invalid")
				}
				if _, exists := callIDs[callID]; exists {
					return errors.New("acceptance evidence call ids must be unique")
				}
				callIDs[callID] = struct{}{}
			}
			switch criterion.Status {
			case CriterionPending, CriterionFailed, CriterionSkipped, CriterionStale:
			case CriterionInvalid, CriterionUnsupported:
				if strings.TrimSpace(criterion.VerificationReason) == "" {
					return errors.New("invalid or unsupported acceptance criterion requires verification_reason")
				}
			case CriterionPassed:
				if strings.TrimSpace(criterion.Evidence) == "" {
					return errors.New("passed acceptance criterion requires evidence")
				}
				if len(criterion.EvidenceCallIDs) == 0 {
					return errors.New("passed acceptance criterion requires evidence_call_ids")
				}
			default:
				return errors.New("invalid acceptance criterion status")
			}
			if step.Status == StatusCompleted && criterion.BlocksCompletion() && criterion.Status != CriterionPassed {
				return errors.New("completed plan step requires every required acceptance criterion to pass")
			}
		}
		if (step.Status == StatusBlocked || step.Status == StatusSkipped) && strings.TrimSpace(step.Result) == "" {
			return errors.New("blocked or skipped plan step requires a result explaining why")
		}
		if step.Status == StatusSkipped {
			for _, criterion := range step.AcceptanceCriteria {
				if criterion.BlocksCompletion() && criterion.Status != CriterionPassed && criterion.Status != CriterionSkipped {
					return errors.New("skipped plan step requires every required acceptance criterion to be passed or skipped")
				}
			}
		}
		if _, exists := ids[step.ID]; exists {
			return errors.New("plan step ids must be unique")
		}
		ids[step.ID] = struct{}{}
		switch step.Status {
		case StatusPending, StatusCompleted, StatusBlocked, StatusStale, StatusSkipped:
		case StatusInProgress:
			active++
		default:
			return errors.New("invalid plan step status")
		}
	}
	if active > 1 {
		return errors.New("only one plan step may be in_progress")
	}
	for _, step := range u.Steps {
		for _, dependency := range step.DependsOn {
			if _, exists := ids[dependency]; !exists || dependency == step.ID {
				return errors.New("plan step dependency must reference another step")
			}
		}
	}
	if hasDependencyCycle(u.Steps) {
		return errors.New("plan step dependencies must be acyclic")
	}
	return nil
}

func (v VerificationSpec) Validate() error {
	return ValidateVerification(v)
}

// ApplyStepUpdate applies a bounded status/evidence mutation and lets the
// normal reconciler select the next dependency-ready step.
func ApplyStepUpdate(plan Plan, mutation StepUpdate) (Update, error) {
	if strings.TrimSpace(mutation.StepID) == "" {
		return Update{}, errors.New("step_id is required")
	}
	steps := append([]Step(nil), plan.Steps...)
	index := -1
	for candidate := range steps {
		if steps[candidate].ID == mutation.StepID {
			index = candidate
			break
		}
	}
	if index < 0 {
		return Update{}, errors.New("plan step was not found")
	}
	reopening := (steps[index].Status == StatusCompleted || steps[index].Status == StatusStale) && mutation.Status == StatusInProgress
	if steps[index].Status != StatusInProgress && steps[index].Status != StatusBlocked && !reopening {
		return Update{}, errors.New("only the current in_progress/blocked/stale step or an explicitly reopened completed step may be updated")
	}
	steps[index].Status = mutation.Status
	if reopening {
		steps[index].Result = ""
	}
	if mutation.Result != "" {
		steps[index].Result = mutation.Result
	}
	if mutation.ToolHints != nil {
		steps[index].ToolHints = append([]string(nil), (*mutation.ToolHints)...)
	}
	criteriaByID := make(map[string]int, len(steps[index].AcceptanceCriteria))
	for candidate, criterion := range steps[index].AcceptanceCriteria {
		criteriaByID[criterion.ID] = candidate
	}
	for _, changed := range mutation.AcceptanceCriteria {
		criterionIndex, exists := criteriaByID[changed.ID]
		if !exists {
			return Update{}, errors.New("acceptance criterion was not found")
		}
		steps[index].AcceptanceCriteria[criterionIndex].Status = changed.Status
		if changed.Status == CriterionPending || changed.Status == CriterionFailed || changed.Status == CriterionInvalid || changed.Status == CriterionUnsupported || changed.Status == CriterionStale {
			steps[index].AcceptanceCriteria[criterionIndex].Evidence = ""
			steps[index].AcceptanceCriteria[criterionIndex].EvidenceCallIDs = nil
		}
		if changed.Status == CriterionPassed {
			steps[index].AcceptanceCriteria[criterionIndex].VerificationReason = ""
			steps[index].AcceptanceCriteria[criterionIndex].VerificationMessage = ""
		} else if changed.VerificationReason != "" || changed.VerificationMessage != "" {
			steps[index].AcceptanceCriteria[criterionIndex].VerificationReason = changed.VerificationReason
			steps[index].AcceptanceCriteria[criterionIndex].VerificationMessage = changed.VerificationMessage
		}
		if changed.Evidence != "" {
			steps[index].AcceptanceCriteria[criterionIndex].Evidence = changed.Evidence
		}
		if len(changed.EvidenceCallIDs) != 0 {
			steps[index].AcceptanceCriteria[criterionIndex].EvidenceCallIDs = append([]string(nil), changed.EvidenceCallIDs...)
		}
	}
	nextSteps := ReconcileSteps(plan.Steps, steps)
	if reopening {
		// ReconcileSteps normally protects terminal progress from stale model
		// echoes. This is an explicit evidence-repair transition, so preserve the
		// reopened step and invalidate completed dependants instead.
		nextSteps = repairDependencyProgress(steps)
	}
	update := Update{Goal: plan.Goal, Explanation: plan.Explanation, Steps: nextSteps}
	if err := update.Validate(); err != nil {
		return Update{}, err
	}
	return update, nil
}

// ApplyCriterionRevision applies a bounded contract repair. Model-originated
// revisions retain the criterion's platform-owned enforcement and origin.
func ApplyCriterionRevision(plan Plan, revision CriterionRevision) (Update, error) {
	if strings.TrimSpace(revision.StepID) == "" || strings.TrimSpace(revision.CriterionID) == "" {
		return Update{}, errors.New("step_id and criterion_id are required")
	}
	if strings.TrimSpace(revision.Reason) == "" {
		return Update{}, errors.New("verification revision reason is required")
	}
	steps := append([]Step(nil), plan.Steps...)
	stepIndex, criterionIndex := -1, -1
	for index := range steps {
		if steps[index].ID != revision.StepID {
			continue
		}
		stepIndex = index
		steps[index].AcceptanceCriteria = append([]AcceptanceCriterion(nil), steps[index].AcceptanceCriteria...)
		for candidate := range steps[index].AcceptanceCriteria {
			if steps[index].AcceptanceCriteria[candidate].ID == revision.CriterionID {
				criterionIndex = candidate
				break
			}
		}
		break
	}
	if stepIndex < 0 || criterionIndex < 0 {
		return Update{}, errors.New("acceptance criterion was not found")
	}
	criterion := &steps[stepIndex].AcceptanceCriteria[criterionIndex]
	if criterion.Status == CriterionPassed {
		return Update{}, errors.New("a passed acceptance criterion cannot be revised until its evidence becomes stale")
	}
	switch revision.Action {
	case "replace":
		if strings.TrimSpace(revision.Verification.Kind) == "" {
			return Update{}, errors.New("replacement verification kind is required")
		}
		criterion.Verification = revision.Verification
		criterion.Status = CriterionPending
		criterion.VerificationReason = ""
		criterion.VerificationMessage = ""
		criterion.Evidence = ""
		criterion.EvidenceCallIDs = nil
	case "skip_advisory":
		if criterion.BlocksCompletion() {
			return Update{}, errors.New("required or release-gate verification cannot be skipped by the Agent")
		}
		criterion.Status = CriterionSkipped
		criterion.VerificationReason = VerificationReasonPolicyOverridden
		criterion.VerificationMessage = revision.Reason
		criterion.Evidence = ""
		criterion.EvidenceCallIDs = nil
	default:
		return Update{}, errors.New("verification revision action must be replace or skip_advisory")
	}
	if steps[stepIndex].Status == StatusCompleted && criterion.BlocksCompletion() && revision.Action == "replace" {
		steps[stepIndex].Status = StatusInProgress
		steps[stepIndex].Result = ""
		steps = repairDependencyProgress(steps)
	}
	update, err := NormalizeUpdate(Update{Goal: plan.Goal, Explanation: plan.Explanation, Steps: steps})
	if err != nil {
		return Update{}, err
	}
	if err := update.Validate(); err != nil {
		return Update{}, err
	}
	update.BaseRevision = revision.BaseRevision
	return update, nil
}

func hasDependencyCycle(steps []Step) bool {
	dependencies := make(map[string][]string, len(steps))
	for _, step := range steps {
		dependencies[step.ID] = step.DependsOn
	}
	state := make(map[string]uint8, len(steps))
	var visit func(string) bool
	visit = func(id string) bool {
		if state[id] == 1 {
			return true
		}
		if state[id] == 2 {
			return false
		}
		state[id] = 1
		for _, dependency := range dependencies[id] {
			if visit(dependency) {
				return true
			}
		}
		state[id] = 2
		return false
	}
	for id := range dependencies {
		if visit(id) {
			return true
		}
	}
	return false
}
