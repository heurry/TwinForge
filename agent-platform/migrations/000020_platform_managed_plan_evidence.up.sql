ALTER TABLE agent_platform.agent_tool_executions
    ADD COLUMN plan_step_key VARCHAR(128);

CREATE INDEX idx_agent_tool_execution_plan_evidence
    ON agent_platform.agent_tool_executions(tenant_id,run_id,plan_step_key,finished_at DESC)
    WHERE status='succeeded' AND plan_step_key IS NOT NULL;
