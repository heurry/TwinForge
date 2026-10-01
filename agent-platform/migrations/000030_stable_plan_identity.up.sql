-- A Workflow plan has a stable identity independent of the Run attempt that
-- most recently updated it. The legacy run_id remains mutable for verification
-- and API compatibility, but must not be used as the Workflow projection FK.
ALTER TABLE agent_platform.agent_workflows
    DROP CONSTRAINT IF EXISTS fk_agent_workflows_active_plan;

ALTER TABLE agent_platform.agent_task_plans
    ADD COLUMN IF NOT EXISTS plan_id UUID;

UPDATE agent_platform.agent_task_plans
SET plan_id = gen_random_uuid()
WHERE plan_id IS NULL;

ALTER TABLE agent_platform.agent_task_plans
    ALTER COLUMN plan_id SET DEFAULT gen_random_uuid(),
    ALTER COLUMN plan_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_agent_task_plans_plan_id'
    ) THEN
        ALTER TABLE agent_platform.agent_task_plans
            ADD CONSTRAINT uq_agent_task_plans_plan_id UNIQUE (plan_id);
    END IF;
END $$;

UPDATE agent_platform.agent_workflows AS workflow
SET active_plan_id = plan.plan_id
FROM agent_platform.agent_task_plans AS plan
WHERE plan.workflow_id = workflow.id;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_agent_workflows_active_plan'
    ) THEN
        ALTER TABLE agent_platform.agent_workflows
            ADD CONSTRAINT fk_agent_workflows_active_plan
            FOREIGN KEY (active_plan_id)
            REFERENCES agent_platform.agent_task_plans(plan_id);
    END IF;
END $$;
