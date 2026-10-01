// Package workflow contains the canonical execution vocabulary for long-lived
// Agent tasks. The legacy Run/Turn/Step fields remain on the wire while new
// code uses these names to avoid conflating planning, scheduling and actions.
package workflow

import (
	"fmt"
	"time"
)

type Status string

const (
	StatusCreated         Status = "created"
	StatusPlanning        Status = "planning"
	StatusReady           Status = "ready"
	StatusRunning         Status = "running"
	StatusWaitingUser     Status = "waiting_user"
	StatusWaitingApproval Status = "waiting_approval"
	StatusWaitingTool     Status = "waiting_tool"
	StatusWaitingAgent    Status = "waiting_agent"
	StatusVerifying       Status = "verifying"
	StatusReflecting      Status = "reflecting"
	StatusReplanning      Status = "replanning"
	StatusReviewing       Status = "reviewing"
	StatusPaused          Status = "paused"
	StatusCancelRequested Status = "cancel_requested"
	StatusCancelled       Status = "cancelled"
	StatusSucceeded       Status = "succeeded"
	StatusFailed          Status = "failed"
)

var transitions = map[Status]map[Status]struct{}{
	StatusCreated:         workflowSet(StatusPlanning, StatusReady, StatusCancelled),
	StatusPlanning:        workflowSet(StatusReady, StatusRunning, StatusWaitingUser, StatusFailed, StatusCancelled),
	StatusReady:           workflowSet(StatusRunning, StatusWaitingUser, StatusWaitingApproval, StatusReplanning, StatusReviewing, StatusCancelled),
	StatusRunning:         workflowSet(StatusWaitingUser, StatusWaitingApproval, StatusWaitingTool, StatusWaitingAgent, StatusVerifying, StatusReflecting, StatusReplanning, StatusReviewing, StatusPaused, StatusCancelRequested, StatusSucceeded, StatusFailed, StatusCancelled),
	StatusWaitingUser:     workflowSet(StatusReady, StatusRunning, StatusCancelled),
	StatusWaitingApproval: workflowSet(StatusReady, StatusRunning, StatusCancelled, StatusFailed),
	StatusWaitingTool:     workflowSet(StatusReady, StatusRunning, StatusFailed, StatusCancelled),
	StatusWaitingAgent:    workflowSet(StatusReady, StatusRunning, StatusFailed, StatusCancelled),
	StatusVerifying:       workflowSet(StatusRunning, StatusReflecting, StatusReplanning, StatusReviewing, StatusSucceeded, StatusFailed, StatusCancelled),
	StatusReflecting:      workflowSet(StatusRunning, StatusReplanning, StatusReviewing, StatusWaitingUser, StatusFailed, StatusCancelled),
	StatusReplanning:      workflowSet(StatusReady, StatusRunning, StatusWaitingUser, StatusFailed, StatusCancelled),
	StatusReviewing:       workflowSet(StatusRunning, StatusReplanning, StatusWaitingApproval, StatusWaitingAgent, StatusSucceeded, StatusFailed, StatusCancelled),
	StatusPaused:          workflowSet(StatusReady, StatusCancelled),
	StatusCancelRequested: workflowSet(StatusCancelled, StatusFailed),
	StatusCancelled:       workflowSet(),
	StatusSucceeded:       workflowSet(),
	StatusFailed:          workflowSet(),
}

func workflowSet(values ...Status) map[Status]struct{} {
	result := make(map[Status]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}

// ValidateTransition is the Workflow-owned lifecycle guard. The legacy Run
// state machine remains in pkg/agent until API consumers complete migration.
func ValidateTransition(current, next Status) error {
	allowed, ok := transitions[current]
	if !ok {
		return fmt.Errorf("unknown workflow status %q", current)
	}
	if _, ok := transitions[next]; !ok {
		return fmt.Errorf("unknown workflow status %q", next)
	}
	if _, ok := allowed[next]; !ok {
		return fmt.Errorf("workflow status transition %q -> %q is not allowed", current, next)
	}
	return nil
}

func (s Status) Terminal() bool {
	return s == StatusCancelled || s == StatusSucceeded || s == StatusFailed
}

// Workflow is one durable autonomous task. A user-facing Session may contain
// multiple Workflows, but pause/resume/approval must keep the same WorkflowID.
type Workflow struct {
	ID                  string    `json:"workflow_id"`
	SessionID           string    `json:"session_id,omitempty"`
	Status              string    `json:"status"`
	Phase               Status    `json:"phase"`
	Goal                string    `json:"goal,omitempty"`
	WorkspaceID         string    `json:"workspace_id,omitempty"`
	ActiveRunID         string    `json:"active_run_id,omitempty"`
	ActivePlanID        string    `json:"active_plan_id,omitempty"`
	LatestCheckpointID  string    `json:"latest_checkpoint_id,omitempty"`
	LatestStateSeq      int64     `json:"latest_state_sequence,omitempty"`
	ExecutionGeneration int64     `json:"execution_generation"`
	WorkspaceRevision   int64     `json:"workspace_revision"`
	LatestWorkflowSeq   int64     `json:"latest_workflow_sequence"`
	CreatedAt           time.Time `json:"created_at,omitempty"`
	UpdatedAt           time.Time `json:"updated_at,omitempty"`
}

// PlanNode is a Todo in a versioned plan DAG. It is not a model call and is
// not an execution step; one node may have multiple ActionAttempts.
type PlanNode struct {
	ID           string   `json:"plan_node_id"`
	Description  string   `json:"description"`
	Status       string   `json:"status"`
	Dependencies []string `json:"dependencies,omitempty"`
}

// DecisionCycle is one model observation/decision iteration. It may produce
// zero or more ActionIntents and is intentionally separate from PlanNode.
type DecisionCycle struct {
	ID         string `json:"decision_cycle_id"`
	WorkflowID string `json:"workflow_id"`
	PlanNodeID string `json:"plan_node_id,omitempty"`
	Number     int    `json:"number"`
}

// ActionAttempt is one executable activity, such as a tool, MCP call, A2A
// request or child Workflow invocation.
type ActionAttempt struct {
	ID              string `json:"action_id"`
	WorkflowID      string `json:"workflow_id"`
	DecisionCycleID string `json:"decision_cycle_id,omitempty"`
	PlanNodeID      string `json:"plan_node_id,omitempty"`
	Kind            string `json:"kind"`
	Name            string `json:"name"`
	Attempt         int    `json:"attempt"`
	Status          string `json:"status"`
}

var actionTransitions = map[string]map[string]struct{}{
	"created":          stringSet("claimed", "cancelled"),
	"claimed":          stringSet("started", "cancelled", "failed"),
	"started":          stringSet("waiting_approval", "waiting_tool", "completed", "failed", "cancelled"),
	"waiting_approval": stringSet("claimed", "started", "cancelled", "failed"),
	"waiting_tool":     stringSet("claimed", "started", "completed", "failed", "cancelled"),
	"completed":        stringSet(),
	"failed":           stringSet(),
	"cancelled":        stringSet(),
}

func stringSet(values ...string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}

// ValidateActionTransition guards one executable activity independently from
// the plan node that requested it. This is the boundary needed for retries:
// one PlanNode can have multiple ActionAttempt records without reopening the
// whole plan.
func ValidateActionTransition(current, next string) error {
	allowed, ok := actionTransitions[current]
	if !ok {
		return fmt.Errorf("unknown action status %q", current)
	}
	if _, ok := actionTransitions[next]; !ok {
		return fmt.Errorf("unknown action status %q", next)
	}
	if _, ok := allowed[next]; !ok {
		return fmt.Errorf("action status transition %q -> %q is not allowed", current, next)
	}
	return nil
}

// Legacy aliases make migrations explicit at call sites and in API adapters.
// They can be removed after all consumers have moved to the canonical names.
const (
	LegacyRunAsWorkflow = "run_is_workflow_compat"
	LegacyTurnAsCycle   = "turn_is_decision_cycle_compat"
	LegacyStepAsAction  = "step_is_not_plan_node"
)
