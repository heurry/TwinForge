ALTER TABLE agent_platform.agent_tool_executions
    ALTER COLUMN step_id DROP NOT NULL,
    ADD COLUMN lease_owner VARCHAR(128),
    ADD COLUMN lease_token BIGINT NOT NULL DEFAULT 0;

CREATE INDEX idx_agent_tool_execution_status
    ON agent_platform.agent_tool_executions (status, started_at);
