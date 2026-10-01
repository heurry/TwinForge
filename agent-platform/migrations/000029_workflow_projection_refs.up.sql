-- Keep the two primary Workflow projections addressable without forcing API
-- consumers to guess from updated_at.  A task plan is identified by its
-- historical Run ID in the legacy schema; the Workflow column makes that
-- compatibility explicit.  latest_checkpoint_id is populated for terminal
-- snapshots; the live, non-terminal state remains in agent_run_states.
ALTER TABLE agent_platform.agent_workflows
    ADD COLUMN IF NOT EXISTS active_plan_id UUID,
    ADD COLUMN IF NOT EXISTS latest_checkpoint_id UUID;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fk_agent_workflows_active_plan') THEN
        ALTER TABLE agent_platform.agent_workflows
            ADD CONSTRAINT fk_agent_workflows_active_plan
            FOREIGN KEY (active_plan_id) REFERENCES agent_platform.agent_task_plans(run_id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fk_agent_workflows_latest_checkpoint') THEN
        ALTER TABLE agent_platform.agent_workflows
            ADD CONSTRAINT fk_agent_workflows_latest_checkpoint
            FOREIGN KEY (latest_checkpoint_id) REFERENCES agent_platform.agent_checkpoints(id);
    END IF;
END $$;

UPDATE agent_platform.agent_workflows AS workflow
SET active_plan_id = plan.run_id
FROM agent_platform.agent_task_plans AS plan
WHERE plan.workflow_id = workflow.id AND workflow.active_plan_id IS NULL;

UPDATE agent_platform.agent_workflows AS workflow
SET latest_checkpoint_id = checkpoint.id
FROM (
    SELECT DISTINCT ON (workflow_id) workflow_id, id
    FROM agent_platform.agent_checkpoints
    ORDER BY workflow_id, COALESCE(NULLIF(state_seq,0),event_seq) DESC, created_at DESC, id DESC
) AS checkpoint
WHERE checkpoint.workflow_id = workflow.id AND workflow.latest_checkpoint_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_agent_workflows_projection_refs
    ON agent_platform.agent_workflows(active_plan_id, latest_checkpoint_id);
