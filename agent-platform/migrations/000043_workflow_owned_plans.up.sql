-- A Plan belongs to a Workflow. run_id is retained as the immutable creating
-- attempt for compatibility; last_modified_run_id records which Run produced
-- the current revision without transferring ownership.
ALTER TABLE agent_platform.agent_task_plans
    ADD COLUMN IF NOT EXISTS last_modified_run_id UUID;

UPDATE agent_platform.agent_task_plans
SET last_modified_run_id = run_id
WHERE last_modified_run_id IS NULL;

ALTER TABLE agent_platform.agent_task_plans
    ALTER COLUMN last_modified_run_id SET NOT NULL;

ALTER TABLE agent_platform.agent_task_plans
    DROP CONSTRAINT IF EXISTS agent_task_plans_run_id_fkey;

ALTER TABLE agent_platform.agent_task_plans
    ADD CONSTRAINT fk_agent_task_plans_created_run
    FOREIGN KEY (run_id) REFERENCES agent_platform.agent_runs(id) ON DELETE RESTRICT;

ALTER TABLE agent_platform.agent_task_plans
    ADD CONSTRAINT fk_agent_task_plans_last_modified_run
    FOREIGN KEY (last_modified_run_id) REFERENCES agent_platform.agent_runs(id) ON DELETE RESTRICT;

CREATE INDEX IF NOT EXISTS idx_agent_task_plans_tenant_workflow
    ON agent_platform.agent_task_plans(tenant_id, workflow_id);
