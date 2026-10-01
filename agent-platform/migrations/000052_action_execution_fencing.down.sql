DROP INDEX IF EXISTS agent_platform.idx_agent_tool_execution_fence;

ALTER TABLE agent_platform.agent_tool_executions
    DROP CONSTRAINT IF EXISTS chk_agent_tool_execution_application_status,
    DROP CONSTRAINT IF EXISTS chk_agent_tool_execution_generations,
    DROP COLUMN IF EXISTS application_status,
    DROP COLUMN IF EXISTS finished_workspace_revision,
    DROP COLUMN IF EXISTS expected_workspace_revision,
    DROP COLUMN IF EXISTS node_revision,
    DROP COLUMN IF EXISTS plan_revision,
    DROP COLUMN IF EXISTS workflow_generation;

ALTER TABLE agent_platform.agent_plan_node_states
    DROP COLUMN IF EXISTS node_revision;
