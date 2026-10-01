// Package event defines append-only facts emitted by an Agent Workflow. Run,
// Turn and Step fields remain compatibility aliases for existing consumers.
package event

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Type identifies a versioned Run fact.
type Type string

const (
	RunCreated         Type = "RUN_CREATED"
	RunClaimed         Type = "RUN_CLAIMED"
	RunResumed         Type = "RUN_RESUMED"
	RunSuspended       Type = "RUN_SUSPENDED"
	RunCancelRequested Type = "RUN_CANCEL_REQUESTED"
	RunCompleted       Type = "RUN_COMPLETED"
	RunFailed          Type = "RUN_FAILED"
	RunCancelled       Type = "RUN_CANCELLED"
	// Workflow lifecycle facts make the long-lived task boundary explicit;
	// Run* events remain attempt-level compatibility facts.
	WorkflowCreated              Type = "WORKFLOW_CREATED"
	WorkflowRoutingDecided       Type = "WORKFLOW_ROUTING_DECIDED"
	RunAttemptCreated            Type = "RUN_ATTEMPT_CREATED"
	WorkflowResumed              Type = "WORKFLOW_RESUMED"
	WorkflowSuspended            Type = "WORKFLOW_SUSPENDED"
	WorkflowCompleted            Type = "WORKFLOW_COMPLETED"
	TurnCreated                  Type = "TURN_CREATED"
	TurnStarted                  Type = "TURN_STARTED"
	TurnCompleted                Type = "TURN_COMPLETED"
	StepStarted                  Type = "STEP_STARTED"
	StepCompleted                Type = "STEP_COMPLETED"
	StepFailed                   Type = "STEP_FAILED"
	ContextBuilt                 Type = "CONTEXT_BUILT"
	ContextCompacted             Type = "CONTEXT_COMPACTED"
	MemoryRetrieved              Type = "MEMORY_RETRIEVED"
	MemoryExtractionRequested    Type = "MEMORY_EXTRACTION_REQUESTED"
	MemoryExtractionStarted      Type = "MEMORY_EXTRACTION_STARTED"
	MemoryCandidateProposed      Type = "MEMORY_CANDIDATE_PROPOSED"
	MemoryCreated                Type = "MEMORY_CREATED"
	MemoryUpdated                Type = "MEMORY_UPDATED"
	MemoryDeleted                Type = "MEMORY_DELETED"
	MemoryMerged                 Type = "MEMORY_MERGED"
	MemorySuperseded             Type = "MEMORY_SUPERSEDED"
	MemoryReviewRequired         Type = "MEMORY_REVIEW_REQUIRED"
	MemoryExtractionCompleted    Type = "MEMORY_EXTRACTION_COMPLETED"
	MemoryExtractionFailed       Type = "MEMORY_EXTRACTION_FAILED"
	MemoryRouted                 Type = "MEMORY_ROUTED"
	MemoryInjected               Type = "MEMORY_INJECTED"
	MemorySuppressed             Type = "MEMORY_SUPPRESSED"
	MemoryVerified               Type = "MEMORY_VERIFIED"
	MemoryContradicted           Type = "MEMORY_CONTRADICTED"
	MemoryTeamPromotionRequested Type = "MEMORY_TEAM_PROMOTION_REQUESTED"
	MemoryTeamPromoted           Type = "MEMORY_TEAM_PROMOTED"
	MemorySourceRevisionObserved Type = "MEMORY_SOURCE_REVISION_OBSERVED"
	IdentityCompiled             Type = "IDENTITY_COMPILED"
	SkillActivated               Type = "SKILL_ACTIVATED"
	ModelResolved                Type = "MODEL_RESOLVED"
	ModelRequested               Type = "MODEL_REQUESTED"
	// ToolSchemaProjected records the visibility/authorization projection for
	// one model request. MODEL_REQUESTED stores an exact schema once per digest
	// and references the preceding identical digest on later decision cycles.
	ToolSchemaProjected   Type = "TOOL_SCHEMA_PROJECTED"
	ModelCompleted        Type = "MODEL_COMPLETED"
	ModelFailed           Type = "MODEL_FAILED"
	ToolCalled            Type = "TOOL_CALLED"
	ToolApprovalRequested Type = "TOOL_APPROVAL_REQUESTED"
	ToolApprovalResolved  Type = "TOOL_APPROVAL_RESOLVED"
	ToolCompleted         Type = "TOOL_COMPLETED"
	ToolFailed            Type = "TOOL_FAILED"
	CheckpointCreated     Type = "CHECKPOINT_CREATED"
	EvalCompleted         Type = "EVAL_COMPLETED"
	DelegationRequested   Type = "DELEGATION_REQUESTED"
	DelegationCompleted   Type = "DELEGATION_COMPLETED"
	DelegationFailed      Type = "DELEGATION_FAILED"
	ExecutionModeSelected Type = "EXECUTION_MODE_SELECTED"
	PlanCreated           Type = "PLAN_CREATED"
	PlanUpdated           Type = "PLAN_UPDATED"
	// ProgressReviewCreated is a deterministic, durable strategy checkpoint.
	// It records that the runtime surfaced repeated failures or lack of
	// workspace progress to the next ReAct decision; it never contains hidden
	// chain-of-thought.
	ProgressReviewCreated Type = "PROGRESS_REVIEW_CREATED"
	// ReviewDecisionRecorded is the normalized, structured result of a
	// runtime-scheduled Reviewer delegation. The original delegation/tool
	// result remains the evidence source; this event gives audit readers a
	// bounded semantic fact without requiring them to decode tool payloads.
	ReviewDecisionRecorded    Type = "REVIEW_DECISION_RECORDED"
	PlanCompletionBlocked     Type = "PLAN_COMPLETION_BLOCKED"
	FinalOutputRejected       Type = "FINAL_OUTPUT_REJECTED"
	VerificationIntentCreated Type = "VERIFICATION_INTENT_CREATED"
	VerificationSpecCompiled  Type = "VERIFICATION_SPEC_COMPILED"
	VerificationSpecRevised   Type = "VERIFICATION_SPEC_REVISED"
	VerificationCompleted     Type = "VERIFICATION_COMPLETED"
	VerificationFailed        Type = "VERIFICATION_FAILED"
	VerificationLoopDetected  Type = "VERIFICATION_LOOP_DETECTED"
	UserInputRequested        Type = "USER_INPUT_REQUESTED"
	UserInputReceived         Type = "USER_INPUT_RECEIVED"
)

// Input is an unsequenced event proposed by a runtime component.
type Input struct {
	RunID string `json:"run_id"`
	// SessionID is the user-facing thread identity. It is carried in the
	// semantic envelope so cross-Run projections can be queried without
	// guessing from an event's parent Run.
	SessionID string `json:"session_id,omitempty"`
	// WorkflowID is the stable long-running task identity. During the
	// migration it defaults to RunID, so old writers remain compatible.
	WorkflowID    string `json:"workflow_id,omitempty"`
	Type          Type   `json:"type"`
	SchemaVersion int    `json:"schema_version,omitempty"`
	Turn          int    `json:"turn,omitempty"`
	Step          int    `json:"step,omitempty"`
	// PlanNodeID identifies a durable Todo/plan node. It is deliberately
	// separate from Step: a plan node can have many action attempts.
	PlanNodeID string `json:"plan_node_id,omitempty"`
	// DecisionCycle is the internal model-decision iteration. Turn is retained
	// as a wire compatibility alias for existing clients.
	DecisionCycle int `json:"decision_cycle,omitempty"`
	// ActionID identifies one executable activity (tool, MCP, A2A or child
	// workflow). CallID remains the legacy tool-call alias.
	ActionID      string          `json:"action_id,omitempty"`
	CallID        string          `json:"call_id,omitempty"`
	TurnID        string          `json:"turn_id,omitempty"`
	CheckpointSeq int64           `json:"checkpoint_seq,omitempty"`
	ParentEventID string          `json:"parent_event_id,omitempty"`
	Payload       json.RawMessage `json:"payload,omitempty"`
}

// Event is a committed append-only Run fact.
type Event struct {
	Input
	Sequence         int64     `json:"sequence"`
	WorkflowSequence int64     `json:"workflow_sequence"`
	CreatedAt        time.Time `json:"created_at"`
}

// Sink atomically validates, sequences, and persists an event.
type Sink interface {
	Append(ctx context.Context, input Input) (Event, error)
}

// MemoryStore is a concurrency-safe reference store used by unit tests and
// embedded callers. Production workers use the PostgreSQL implementation.
type MemoryStore struct {
	mu                sync.Mutex
	events            map[string][]Event
	traces            map[string]trace
	workflowSequences map[string]int64
	nowFunc           func() time.Time
}

type trace struct {
	created      bool
	terminal     bool
	openTurn     int
	openStep     int
	nextTurn     int
	nextStep     int
	pendingCalls map[string]struct{}
}

// NewMemoryStore creates an empty reference event store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		events:            make(map[string][]Event),
		traces:            make(map[string]trace),
		workflowSequences: make(map[string]int64),
		nowFunc:           time.Now,
	}
}

// Append validates lifecycle relationships and assigns a per-Run sequence.
func (s *MemoryStore) Append(_ context.Context, input Input) (Event, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	input = NormalizeSemantic(input)

	if input.RunID == "" {
		return Event{}, fmt.Errorf("event run_id is required")
	}
	if input.Type == "" {
		return Event{}, fmt.Errorf("event type is required")
	}
	if input.SchemaVersion == 0 {
		input.SchemaVersion = 1
	}
	current := s.traces[input.RunID]
	if current.nextTurn == 0 {
		current.nextTurn = 1
	}
	if current.pendingCalls == nil {
		current.pendingCalls = make(map[string]struct{})
	}
	if err := current.apply(input); err != nil {
		return Event{}, err
	}
	s.workflowSequences[input.WorkflowID]++
	event := Event{
		Input:            input,
		Sequence:         int64(len(s.events[input.RunID]) + 1),
		WorkflowSequence: s.workflowSequences[input.WorkflowID],
		CreatedAt:        s.nowFunc().UTC(),
	}
	s.events[input.RunID] = append(s.events[input.RunID], event)
	s.traces[input.RunID] = current
	return event, nil
}

// Events returns an isolated copy of one Run's committed events.
func (s *MemoryStore) Events(runID string) []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Event(nil), s.events[runID]...)
}

func (t *trace) apply(input Input) error {
	if t.terminal {
		return fmt.Errorf("event %s cannot follow a terminal run event", input.Type)
	}
	if !t.created && input.Type != RunCreated {
		return fmt.Errorf("first event must be %s", RunCreated)
	}

	switch input.Type {
	case RunCreated:
		if t.created {
			return fmt.Errorf("run is already created")
		}
		t.created = true
	case TurnStarted:
		if t.openTurn != 0 {
			return fmt.Errorf("turn %d is still open", t.openTurn)
		}
		if input.Turn != t.nextTurn {
			return fmt.Errorf("turn %d is out of sequence; expected %d", input.Turn, t.nextTurn)
		}
		t.openTurn = input.Turn
		t.nextStep = 1
	case TurnCompleted:
		if input.Turn != t.openTurn || t.openTurn == 0 {
			return fmt.Errorf("turn %d does not match open turn %d", input.Turn, t.openTurn)
		}
		if t.openStep != 0 {
			return fmt.Errorf("step %d is still open", t.openStep)
		}
		t.openTurn = 0
		t.nextTurn++
	case StepStarted:
		if input.Turn != t.openTurn || t.openTurn == 0 {
			return fmt.Errorf("step must belong to open turn %d", t.openTurn)
		}
		if t.openStep != 0 {
			return fmt.Errorf("step %d is still open", t.openStep)
		}
		if input.Step != t.nextStep {
			return fmt.Errorf("step %d is out of sequence; expected %d", input.Step, t.nextStep)
		}
		t.openStep = input.Step
	case StepCompleted, StepFailed:
		if err := t.requireOpenStep(input); err != nil {
			return err
		}
		if len(t.pendingCalls) != 0 {
			return fmt.Errorf("step has %d unresolved tool calls", len(t.pendingCalls))
		}
		t.openStep = 0
		t.nextStep++
	case ModelRequested, ToolSchemaProjected, ModelCompleted, ModelFailed, ContextBuilt, ContextCompacted, MemoryRetrieved, MemoryRouted, MemoryInjected, MemorySuppressed:
		if err := t.requireOpenStep(input); err != nil {
			return err
		}
	case ToolCalled:
		if err := t.requireOpenStep(input); err != nil {
			return err
		}
		if input.CallID == "" {
			return fmt.Errorf("tool call_id is required")
		}
		if _, exists := t.pendingCalls[input.CallID]; exists {
			return fmt.Errorf("tool call_id %q is already pending", input.CallID)
		}
		t.pendingCalls[input.CallID] = struct{}{}
	case ToolApprovalRequested, ToolApprovalResolved:
		if err := t.requirePendingCall(input); err != nil {
			return err
		}
	case ToolCompleted, ToolFailed:
		if err := t.requirePendingCall(input); err != nil {
			return err
		}
		delete(t.pendingCalls, input.CallID)
	case RunCompleted, RunFailed, RunCancelled:
		if t.openTurn != 0 || t.openStep != 0 || len(t.pendingCalls) != 0 {
			return fmt.Errorf("run cannot terminate with open work")
		}
		t.terminal = true
	case WorkflowCreated, WorkflowRoutingDecided, RunAttemptCreated, WorkflowResumed, WorkflowSuspended, WorkflowCompleted, TurnCreated,
		RunClaimed, RunResumed, RunSuspended, RunCancelRequested, CheckpointCreated, EvalCompleted, IdentityCompiled, SkillActivated, ModelResolved, MemoryExtractionRequested, MemoryExtractionStarted, MemoryCandidateProposed, MemoryCreated, MemoryUpdated, MemoryDeleted, MemoryMerged, MemorySuperseded, MemoryReviewRequired, MemoryExtractionCompleted, MemoryExtractionFailed, MemoryVerified, MemoryContradicted, MemoryTeamPromotionRequested, MemoryTeamPromoted, MemorySourceRevisionObserved, DelegationRequested, DelegationCompleted, DelegationFailed, ExecutionModeSelected, PlanCreated, PlanUpdated, ProgressReviewCreated, ReviewDecisionRecorded, PlanCompletionBlocked, FinalOutputRejected, VerificationIntentCreated, VerificationSpecCompiled, VerificationSpecRevised, VerificationCompleted, VerificationFailed, VerificationLoopDetected, UserInputRequested, UserInputReceived:
		// These events do not change Turn/Step structural invariants.
	default:
		return fmt.Errorf("unknown event type %q", input.Type)
	}
	return nil
}

func (t *trace) requireOpenStep(input Input) error {
	if input.Turn != t.openTurn || input.Step != t.openStep || t.openStep == 0 {
		return fmt.Errorf("event %s must belong to open turn/step %d/%d", input.Type, t.openTurn, t.openStep)
	}
	return nil
}

func (t *trace) requirePendingCall(input Input) error {
	if err := t.requireOpenStep(input); err != nil {
		return err
	}
	if input.CallID == "" {
		return fmt.Errorf("tool call_id is required")
	}
	if _, exists := t.pendingCalls[input.CallID]; !exists {
		return fmt.Errorf("tool call_id %q is not pending", input.CallID)
	}
	return nil
}
