CREATE INDEX IF NOT EXISTS idx_agent_model_calls_run
    ON agent_platform.agent_model_calls (run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_agent_tool_executions_run_status
    ON agent_platform.agent_tool_executions (run_id, status);
