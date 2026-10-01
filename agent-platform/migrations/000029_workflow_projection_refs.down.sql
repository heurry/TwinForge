DROP INDEX IF EXISTS agent_platform.idx_agent_workflows_projection_refs;
ALTER TABLE agent_platform.agent_workflows
    DROP CONSTRAINT IF EXISTS fk_agent_workflows_latest_checkpoint,
    DROP CONSTRAINT IF EXISTS fk_agent_workflows_active_plan,
    DROP COLUMN IF EXISTS latest_checkpoint_id,
    DROP COLUMN IF EXISTS active_plan_id;
