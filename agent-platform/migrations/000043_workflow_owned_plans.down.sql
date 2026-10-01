DROP INDEX IF EXISTS agent_platform.idx_agent_task_plans_tenant_workflow;

ALTER TABLE agent_platform.agent_task_plans
    DROP CONSTRAINT IF EXISTS fk_agent_task_plans_last_modified_run,
    DROP CONSTRAINT IF EXISTS fk_agent_task_plans_created_run,
    DROP COLUMN IF EXISTS last_modified_run_id;

ALTER TABLE agent_platform.agent_task_plans
    ADD CONSTRAINT agent_task_plans_run_id_fkey
    FOREIGN KEY (run_id) REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE;
