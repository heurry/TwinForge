// Package harness defines the stable contract implemented by Agent loops.
package harness

import (
	"context"
	"encoding/json"

	"github.com/heurry/cloudnative-infra-platform/agent-platform/pkg/model"
)

// Request contains context already selected for a single Agent Run.
type Request struct {
	RunID string `json:"run_id"`
	// WorkspaceID scopes files across continuation Runs in one Session while
	// keeping RunID as the immutable audit/call identity.
	WorkspaceID string `json:"workspace_id,omitempty"`
	// Turn is the physical per-Run lifecycle counter. TurnID is the stable
	// Workflow-level identity used when a new Run continues an existing Turn.
	TurnID           string           `json:"turn_id,omitempty"`
	Messages         []model.Message  `json:"messages"`
	Turn             int              `json:"turn,omitempty"`
	StartStep        int              `json:"start_step,omitempty"`
	Resume           bool             `json:"resume,omitempty"`
	Usage            model.Usage      `json:"usage,omitempty"`
	PendingToolCalls []model.ToolCall `json:"pending_tool_calls,omitempty"`
	ActiveToolCallID string           `json:"active_tool_call_id,omitempty"`
	ContextManifest  json.RawMessage  `json:"context_manifest,omitempty"`
	// ContextState is durable projection metadata. It is deliberately separate
	// from Messages: Context Collapse may change the model-facing view without
	// replacing the complete resumable history.
	ContextState json.RawMessage `json:"context_state,omitempty"`
	// ExecutionLedger is a structured, durable action/observation snapshot.
	// It survives context compaction independently from conversational history.
	ExecutionLedger json.RawMessage `json:"execution_ledger,omitempty"`
}

// Checkpoint is durable, model-visible state sufficient to resume a ReAct
// loop. It deliberately excludes private chain-of-thought.
type Checkpoint struct {
	RunID string `json:"run_id"`
	// SourceRunID identifies the Run attempt that produced this snapshot. RunID
	// is rebound to the current attempt when a Workflow is continued.
	SourceRunID string `json:"source_run_id,omitempty"`
	// StateSeq is monotonic across every Run Attempt in one Workflow. It is
	// the durable state generation used to reject stale writers and resume the
	// newest graph/message projection after a worker takeover.
	StateSeq         int64            `json:"state_sequence,omitempty"`
	Turn             int              `json:"turn"`
	NextStep         int              `json:"next_step"`
	Messages         []model.Message  `json:"messages"`
	Usage            model.Usage      `json:"usage"`
	PendingToolCalls []model.ToolCall `json:"pending_tool_calls,omitempty"`
	ActiveToolCallID string           `json:"active_tool_call_id,omitempty"`
	ExecutionLedger  json.RawMessage  `json:"execution_ledger,omitempty"`
	// ContextState records the active read-time projection generation. Events
	// retain the complete audit history; Messages is a bounded takeover view
	// containing the current summary, protected anchors, and uncovered tail.
	ContextState json.RawMessage `json:"context_state,omitempty"`
	Completed    bool            `json:"completed"`
	Answer       model.Message   `json:"answer,omitempty"`
}

// CheckpointSink persists one resumable execution snapshot.
type CheckpointSink interface {
	Save(ctx context.Context, checkpoint Checkpoint) error
}

// Result is the terminal answer and the complete model-visible message list.
type Result struct {
	Answer   model.Message   `json:"answer"`
	Messages []model.Message `json:"messages"`
	Steps    int             `json:"steps"`
	Usage    model.Usage     `json:"usage"`
}

// Runner executes one Harness implementation.
type Runner interface {
	Name() string
	Run(ctx context.Context, request Request) (Result, error)
}
