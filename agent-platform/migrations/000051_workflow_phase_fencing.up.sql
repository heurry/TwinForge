-- Introduce the canonical Workflow lifecycle alongside the legacy UI status.
-- Runtime code dual-writes both columns during the compatibility window. The
-- legacy status can be retired after API/UI readers have moved to phase.

ALTER TABLE agent_platform.agent_workflows
    ADD COLUMN IF NOT EXISTS phase VARCHAR(32),
    ADD COLUMN IF NOT EXISTS execution_generation BIGINT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS workspace_revision BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS latest_workflow_seq BIGINT NOT NULL DEFAULT 0;

UPDATE agent_platform.agent_workflows AS workflow
SET phase = CASE
    WHEN workflow.status = 'completed' THEN 'succeeded'
    WHEN workflow.status = 'failed' THEN 'failed'
    WHEN workflow.status = 'cancelled' THEN 'cancelled'
    WHEN workflow.status = 'archived' THEN COALESCE((
        SELECT CASE run.status
            WHEN 'completed' THEN 'succeeded'
            WHEN 'failed' THEN 'failed'
            WHEN 'cancelled' THEN 'cancelled'
            ELSE 'paused'
        END
        FROM agent_platform.agent_runs AS run
        WHERE run.id = workflow.active_run_id
    ), 'paused')
    WHEN workflow.status = 'waiting' THEN COALESCE((
        SELECT CASE run.status
            WHEN 'waiting_approval' THEN 'waiting_approval'
            WHEN 'waiting_input' THEN 'waiting_user'
            WHEN 'waiting_tool' THEN 'waiting_tool'
            WHEN 'waiting_external' THEN 'waiting_tool'
            WHEN 'suspended' THEN 'paused'
            ELSE 'paused'
        END
        FROM agent_platform.agent_runs AS run
        WHERE run.id = workflow.active_run_id
    ), 'paused')
    ELSE COALESCE((
        SELECT CASE run.status
            WHEN 'queued' THEN 'ready'
            WHEN 'running' THEN 'running'
            WHEN 'waiting_approval' THEN 'waiting_approval'
            WHEN 'waiting_input' THEN 'waiting_user'
            WHEN 'waiting_tool' THEN 'waiting_tool'
            WHEN 'waiting_external' THEN 'waiting_tool'
            WHEN 'suspended' THEN 'paused'
            WHEN 'completed' THEN 'ready'
            WHEN 'failed' THEN 'failed'
            WHEN 'cancelled' THEN 'cancelled'
            ELSE 'created'
        END
        FROM agent_platform.agent_runs AS run
        WHERE run.id = workflow.active_run_id
    ), 'created')
END
WHERE workflow.phase IS NULL;

UPDATE agent_platform.agent_workflows AS workflow
SET execution_generation = GREATEST(1, (
    SELECT count(*)
    FROM agent_platform.agent_runs AS run
    WHERE run.workflow_id = workflow.id
      AND run.parent_run_id IS NULL
));

ALTER TABLE agent_platform.agent_workflows
    ALTER COLUMN phase SET DEFAULT 'created',
    ALTER COLUMN phase SET NOT NULL;

ALTER TABLE agent_platform.agent_workflows
    ADD CONSTRAINT chk_agent_workflow_phase CHECK (phase IN (
        'created','planning','ready','running','verifying','reflecting','replanning','reviewing',
        'waiting_user','waiting_approval','waiting_tool','waiting_agent','paused',
        'cancel_requested','succeeded','failed','cancelled'
    )),
    ADD CONSTRAINT chk_agent_workflow_generation_positive CHECK (execution_generation > 0),
    ADD CONSTRAINT chk_agent_workflow_revision_nonnegative CHECK (
        workspace_revision >= 0 AND latest_workflow_seq >= 0
    );

CREATE INDEX IF NOT EXISTS idx_agent_workflows_phase
    ON agent_platform.agent_workflows(tenant_id, phase, updated_at DESC);
