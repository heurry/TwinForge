package agent

import "time"

// RunAudit is a durable, tenant-scoped usage projection for one Run. It is
// derived from Run, model-call, and tool-execution records rather than
// process-local metrics, so it remains available after restarts.
type RunAudit struct {
	RunID                  string     `json:"run_id"`
	WorkflowID             string     `json:"workflow_id"`
	Status                 RunStatus  `json:"status"`
	StartedAt              *time.Time `json:"started_at,omitempty"`
	FinishedAt             *time.Time `json:"finished_at,omitempty"`
	DurationMS             int64      `json:"duration_ms"`
	ModelCalls             int64      `json:"model_calls"`
	SuccessfulModelCalls   int64      `json:"successful_model_calls"`
	FailedModelCalls       int64      `json:"failed_model_calls"`
	InputTokens            int64      `json:"input_tokens"`
	OutputTokens           int64      `json:"output_tokens"`
	TotalTokens            int64      `json:"total_tokens"`
	ToolCalls              int64      `json:"tool_calls"`
	SuccessfulToolCalls    int64      `json:"successful_tool_calls"`
	FailedToolCalls        int64      `json:"failed_tool_calls"`
	TerminalToolCalls      int64      `json:"terminal_tool_calls"`
	ToolSuccessRatePercent float64    `json:"tool_success_rate_percent"`
}

// SessionAudit aggregates persisted execution usage for one conversation and
// includes its Run-level breakdown for drill-down and reconciliation.
type SessionAudit struct {
	SessionID              string     `json:"session_id"`
	RunCount               int64      `json:"run_count"`
	CompletedRuns          int64      `json:"completed_runs"`
	FailedRuns             int64      `json:"failed_runs"`
	ActiveRuns             int64      `json:"active_runs"`
	FirstStartedAt         *time.Time `json:"first_started_at,omitempty"`
	LastActivityAt         *time.Time `json:"last_activity_at,omitempty"`
	WallDurationMS         int64      `json:"wall_duration_ms"`
	ExecutionDurationMS    int64      `json:"execution_duration_ms"`
	ModelCalls             int64      `json:"model_calls"`
	SuccessfulModelCalls   int64      `json:"successful_model_calls"`
	FailedModelCalls       int64      `json:"failed_model_calls"`
	InputTokens            int64      `json:"input_tokens"`
	OutputTokens           int64      `json:"output_tokens"`
	TotalTokens            int64      `json:"total_tokens"`
	ToolCalls              int64      `json:"tool_calls"`
	SuccessfulToolCalls    int64      `json:"successful_tool_calls"`
	FailedToolCalls        int64      `json:"failed_tool_calls"`
	TerminalToolCalls      int64      `json:"terminal_tool_calls"`
	ToolSuccessRatePercent float64    `json:"tool_success_rate_percent"`
	Runs                   []RunAudit `json:"runs"`
}
