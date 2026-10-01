DROP INDEX IF EXISTS agent_platform.idx_agent_tool_execution_plan_evidence;
ALTER TABLE agent_platform.agent_tool_executions DROP COLUMN IF EXISTS plan_step_key;
