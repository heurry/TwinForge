-- Finish the ownership transition: plan_id is the row identity, workflow_id
-- is the unique owner, and Run foreign keys are provenance only.
ALTER TABLE agent_platform.agent_workflows
    DROP CONSTRAINT IF EXISTS fk_agent_workflows_active_plan;

ALTER TABLE agent_platform.agent_task_plans
    DROP CONSTRAINT IF EXISTS agent_task_plans_pkey,
    DROP CONSTRAINT IF EXISTS uq_agent_task_plans_plan_id;

ALTER TABLE agent_platform.agent_task_plans
    ADD CONSTRAINT agent_task_plans_pkey PRIMARY KEY (plan_id),
    ADD CONSTRAINT uq_agent_task_plans_created_run UNIQUE (run_id);

ALTER TABLE agent_platform.agent_workflows
    ADD CONSTRAINT fk_agent_workflows_active_plan
    FOREIGN KEY (active_plan_id)
    REFERENCES agent_platform.agent_task_plans(plan_id);

COMMENT ON COLUMN agent_platform.agent_task_plans.run_id IS
    'Immutable Run attempt that created the Workflow-owned Plan; provenance, not ownership.';
COMMENT ON COLUMN agent_platform.agent_task_plans.last_modified_run_id IS
    'Run attempt that produced the current Plan revision; provenance, not ownership.';
