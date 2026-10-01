package agent

import "fmt"

// RunStatus is the durable lifecycle state of an Agent Run.
type RunStatus string

const (
	RunQueued          RunStatus = "queued"
	RunRunning         RunStatus = "running"
	RunWaitingTool     RunStatus = "waiting_tool"
	RunWaitingApproval RunStatus = "waiting_approval"
	RunWaitingInput    RunStatus = "waiting_input"
	RunWaitingExternal RunStatus = "waiting_external"
	RunSuspended       RunStatus = "suspended"
	RunCompleted       RunStatus = "completed"
	RunFailed          RunStatus = "failed"
	RunCancelled       RunStatus = "cancelled"
)

var runTransitions = map[RunStatus]map[RunStatus]struct{}{
	RunQueued:          setOf(RunRunning, RunCancelled),
	RunRunning:         setOf(RunWaitingTool, RunWaitingApproval, RunWaitingInput, RunWaitingExternal, RunSuspended, RunCompleted, RunFailed, RunCancelled),
	RunWaitingTool:     setOf(RunRunning, RunFailed, RunCancelled),
	RunWaitingApproval: setOf(RunQueued, RunRunning, RunFailed, RunCancelled),
	RunWaitingInput:    setOf(RunQueued, RunRunning, RunFailed, RunCancelled),
	RunWaitingExternal: setOf(RunRunning, RunSuspended, RunFailed, RunCancelled),
	RunSuspended:       setOf(RunQueued, RunCancelled),
	RunCompleted:       {},
	RunFailed:          {},
	RunCancelled:       {},
}

// ValidateTransition verifies one optimistic state-machine update.
func ValidateTransition(current, next RunStatus) error {
	allowed, known := runTransitions[current]
	if !known {
		return fmt.Errorf("unknown current run status %q", current)
	}
	if _, known = runTransitions[next]; !known {
		return fmt.Errorf("unknown next run status %q", next)
	}
	if _, ok := allowed[next]; !ok {
		return fmt.Errorf("run status transition %q -> %q is not allowed", current, next)
	}
	return nil
}

// Terminal reports whether no further normal transition may occur.
func (s RunStatus) Terminal() bool {
	return s == RunCompleted || s == RunFailed || s == RunCancelled
}

func setOf(values ...RunStatus) map[RunStatus]struct{} {
	result := make(map[RunStatus]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}
