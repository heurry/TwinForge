package agent

// ObservabilitySummary is a tenant-scoped aggregate derived from durable Run,
// model-call and tool-execution records.
type ObservabilitySummary struct {
	TotalRuns              int64            `json:"total_runs"`
	CompletedRuns          int64            `json:"completed_runs"`
	FailedRuns             int64            `json:"failed_runs"`
	ActiveRuns             int64            `json:"active_runs"`
	ModelCalls             int64            `json:"model_calls"`
	ToolCalls              int64            `json:"tool_calls"`
	FailedToolCalls        int64            `json:"failed_tool_calls"`
	InputTokens            int64            `json:"input_tokens"`
	OutputTokens           int64            `json:"output_tokens"`
	AverageRunLatencyMS    float64          `json:"average_run_latency_ms"`
	AverageModelLatencyMS  float64          `json:"average_model_latency_ms"`
	ToolFailuresByCode     map[string]int64 `json:"tool_failures_by_code,omitempty"`
	ModelProtocolFailures  int64            `json:"model_protocol_failures"`
	VerificationFailures   int64            `json:"verification_failures"`
	PlanFailures           int64            `json:"plan_failures"`
	CompactionCount        int64            `json:"compaction_count"`
	CompactionBeforeTokens int64            `json:"compaction_before_tokens"`
	CompactionAfterTokens  int64            `json:"compaction_after_tokens"`
}
