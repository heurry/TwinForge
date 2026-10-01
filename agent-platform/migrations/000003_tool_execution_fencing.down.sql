DROP INDEX IF EXISTS agent_platform.idx_agent_tool_execution_status;
ALTER TABLE agent_platform.agent_tool_executions
    DROP COLUMN IF EXISTS lease_token,
    DROP COLUMN IF EXISTS lease_owner;
