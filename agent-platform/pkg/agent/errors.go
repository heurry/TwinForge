package agent

import "errors"

// WorkflowCandidate is the minimum task projection needed to resolve an
// ambiguous continuation without asking the model to guess an ID.
type WorkflowCandidate struct {
	ID     string `json:"workflow_id"`
	Goal   string `json:"goal,omitempty"`
	Status string `json:"status"`
}

// WorkflowAmbiguousError is returned when auto-routing has more than one
// resumable Workflow in a Session. The API exposes candidates to the UI; no
// Run is created until the user selects one explicitly.
type WorkflowAmbiguousError struct {
	Candidates []WorkflowCandidate `json:"candidates"`
}

func (e *WorkflowAmbiguousError) Error() string {
	return "multiple resumable workflows require an explicit selection"
}

func (e *WorkflowAmbiguousError) Unwrap() error { return ErrWorkflowAmbiguous }

var (
	// ErrDefinitionConflict reports a duplicate tenant-scoped Agent key.
	ErrDefinitionConflict = errors.New("agent definition already exists")
	// ErrDefinitionNotFound reports an unknown tenant-scoped Agent.
	ErrDefinitionNotFound = errors.New("agent definition not found")
	// ErrVersionNotFound reports an unknown tenant-scoped Agent version.
	ErrVersionNotFound = errors.New("agent version not found")
	// ErrSessionNotFound reports an unknown tenant-scoped Session.
	ErrSessionNotFound = errors.New("agent session not found")
	// ErrMemoryNotFound reports an unknown or already deleted tenant memory.
	ErrMemoryNotFound = errors.New("agent memory not found")
	// ErrMemoryConflict reports an optimistic-concurrency mismatch while
	// editing a Memory.
	ErrMemoryConflict = errors.New("agent memory revision conflict")
	// ErrRunBindingInvalid reports a version/session that is not published or
	// does not belong to the requesting tenant.
	ErrRunBindingInvalid = errors.New("agent run binding is invalid")
	// ErrRunInputInvalid reports input that violates the immutable Agent schema.
	ErrRunInputInvalid = errors.New("agent run input violates input schema")
	// ErrRunNotFound reports an unknown Run identifier.
	ErrRunNotFound = errors.New("agent run not found")
	// ErrWorkflowBusy reports another live Run attempt already owns a Workflow.
	ErrWorkflowBusy = errors.New("workflow already has an active run")
	// ErrWorkflowAmbiguous reports that auto-routing cannot safely choose a
	// Workflow when a Session contains multiple resumable tasks.
	ErrWorkflowAmbiguous = errors.New("workflow routing is ambiguous")
	// ErrWorkflowRoutingInvalid reports an intent/binding combination that
	// would otherwise silently create or mutate the wrong Workflow.
	ErrWorkflowRoutingInvalid = errors.New("workflow routing intent is invalid")
	// ErrRunTerminal reports a cancellation request for a completed Run.
	ErrRunTerminal = errors.New("agent run is already terminal")
	// ErrLeaseLost reports a stale or foreign fencing token.
	ErrLeaseLost = errors.New("agent run lease lost")
	// ErrNoRunnableRun reports an empty durable queue.
	ErrNoRunnableRun = errors.New("no runnable agent run")
)
