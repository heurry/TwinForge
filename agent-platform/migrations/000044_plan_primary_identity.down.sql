ALTER TABLE agent_platform.agent_workflows
    DROP CONSTRAINT IF EXISTS fk_agent_workflows_active_plan;

ALTER TABLE agent_platform.agent_task_plans
    DROP CONSTRAINT IF EXISTS agent_task_plans_pkey,
    DROP CONSTRAINT IF EXISTS uq_agent_task_plans_created_run;

ALTER TABLE agent_platform.agent_task_plans
    ADD CONSTRAINT agent_task_plans_pkey PRIMARY KEY (run_id),
    ADD CONSTRAINT uq_agent_task_plans_plan_id UNIQUE (plan_id);

ALTER TABLE agent_platform.agent_workflows
    ADD CONSTRAINT fk_agent_workflows_active_plan
    FOREIGN KEY (active_plan_id)
    REFERENCES agent_platform.agent_task_plans(plan_id);
