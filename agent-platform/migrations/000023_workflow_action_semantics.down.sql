DROP INDEX IF EXISTS agent_platform.idx_agent_observations_workflow_action;
DROP INDEX IF EXISTS agent_platform.idx_agent_tool_action_trace;
ALTER TABLE agent_platform.agent_observations
    DROP COLUMN IF EXISTS parent_action_id,
    DROP COLUMN IF EXISTS action_id,
    DROP COLUMN IF EXISTS decision_cycle,
    DROP COLUMN IF EXISTS plan_node_id,
    DROP COLUMN IF EXISTS workflow_id;
ALTER TABLE agent_platform.agent_tool_executions
    DROP COLUMN IF EXISTS action_kind,
    DROP COLUMN IF EXISTS action_id,
    DROP COLUMN IF EXISTS decision_cycle,
    DROP COLUMN IF EXISTS plan_node_id,
    DROP COLUMN IF EXISTS workflow_id;
