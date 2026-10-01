ALTER TABLE agent_platform.agent_workflows
    DROP CONSTRAINT IF EXISTS fk_agent_workflows_active_plan;

UPDATE agent_platform.agent_workflows AS workflow
SET active_plan_id = plan.run_id
FROM agent_platform.agent_task_plans AS plan
WHERE plan.workflow_id = workflow.id;

ALTER TABLE agent_platform.agent_task_plans
    DROP CONSTRAINT IF EXISTS uq_agent_task_plans_plan_id;

ALTER TABLE agent_platform.agent_task_plans
    DROP COLUMN IF EXISTS plan_id;

ALTER TABLE agent_platform.agent_workflows
    ADD CONSTRAINT fk_agent_workflows_active_plan
    FOREIGN KEY (active_plan_id)
    REFERENCES agent_platform.agent_task_plans(run_id);
