-- P0: promote the long-lived task identity above an individual Run attempt.
-- Existing rows get one compatibility Workflow per Run. New continuation Runs
-- are assigned to the active Workflow by the application layer.

CREATE TABLE IF NOT EXISTS agent_platform.agent_workflows (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id         VARCHAR(128) NOT NULL,
    session_id        UUID REFERENCES agent_platform.agent_sessions(id) ON DELETE CASCADE,
    agent_version_id  UUID NOT NULL REFERENCES agent_platform.agent_versions(id),
    status            VARCHAR(32) NOT NULL DEFAULT 'active',
    goal              TEXT,
    goal_hash         VARCHAR(128),
    workspace_id      VARCHAR(256) NOT NULL,
    active_run_id     UUID,
    latest_state_seq  BIGINT NOT NULL DEFAULT 0,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_agent_workflow_status CHECK (status IN ('active','waiting','completed','failed','cancelled','archived'))
);

ALTER TABLE agent_platform.agent_runs
    ADD COLUMN IF NOT EXISTS workflow_id UUID;

-- Compatibility identity: before this migration each Run was its own durable
-- task. Keeping that identity avoids rewriting historical event ownership.
UPDATE agent_platform.agent_runs
SET workflow_id = COALESCE(workflow_id, gen_random_uuid())
WHERE workflow_id IS NULL;

INSERT INTO agent_platform.agent_workflows
    (id, tenant_id, session_id, agent_version_id, status, workspace_id, created_at, updated_at)
SELECT DISTINCT ON (run.workflow_id)
    run.workflow_id, run.tenant_id, run.session_id, run.agent_version_id,
    CASE WHEN run.status IN ('completed','cancelled','failed') THEN run.status ELSE 'active' END,
    COALESCE(run.session_id::text, run.workflow_id::text), run.created_at, run.updated_at
FROM agent_platform.agent_runs AS run
WHERE run.workflow_id IS NOT NULL
ON CONFLICT (id) DO NOTHING;

UPDATE agent_platform.agent_workflows AS workflow
SET active_run_id = latest.id,
    updated_at = GREATEST(workflow.updated_at, latest.updated_at)
FROM (
    SELECT DISTINCT ON (run.workflow_id) run.workflow_id, run.id, run.updated_at
    FROM agent_platform.agent_runs AS run
    ORDER BY run.workflow_id, run.updated_at DESC, run.id DESC
) AS latest
WHERE workflow.id = latest.workflow_id;

ALTER TABLE agent_platform.agent_runs
    ALTER COLUMN workflow_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname='fk_agent_runs_workflow'
    ) THEN
        ALTER TABLE agent_platform.agent_runs
            ADD CONSTRAINT fk_agent_runs_workflow
            FOREIGN KEY (workflow_id) REFERENCES agent_platform.agent_workflows(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_agent_workflows_session
    ON agent_platform.agent_workflows(tenant_id, session_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_workflows_active
    ON agent_platform.agent_workflows(tenant_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_workflow
    ON agent_platform.agent_runs(tenant_id, workflow_id, created_at DESC);

ALTER TABLE agent_platform.agent_task_plans
    ADD COLUMN IF NOT EXISTS workflow_id UUID;
UPDATE agent_platform.agent_task_plans AS plan
SET workflow_id = run.workflow_id
FROM agent_platform.agent_runs AS run
WHERE plan.run_id = run.id AND plan.workflow_id IS NULL;
ALTER TABLE agent_platform.agent_task_plans
    ALTER COLUMN workflow_id SET NOT NULL;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fk_agent_task_plans_workflow') THEN
        ALTER TABLE agent_platform.agent_task_plans
            ADD CONSTRAINT fk_agent_task_plans_workflow
            FOREIGN KEY (workflow_id) REFERENCES agent_platform.agent_workflows(id);
    END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_task_plans_workflow
    ON agent_platform.agent_task_plans(workflow_id);
CREATE INDEX IF NOT EXISTS idx_agent_task_plans_workflow_updated
    ON agent_platform.agent_task_plans(tenant_id, workflow_id, updated_at DESC);

ALTER TABLE agent_platform.agent_checkpoints
    ADD COLUMN IF NOT EXISTS workflow_id UUID;
UPDATE agent_platform.agent_checkpoints AS checkpoint
SET workflow_id = run.workflow_id
FROM agent_platform.agent_runs AS run
WHERE checkpoint.run_id = run.id AND checkpoint.workflow_id IS NULL;
ALTER TABLE agent_platform.agent_checkpoints
    ALTER COLUMN workflow_id SET NOT NULL;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fk_agent_checkpoints_workflow') THEN
        ALTER TABLE agent_platform.agent_checkpoints
            ADD CONSTRAINT fk_agent_checkpoints_workflow
            FOREIGN KEY (workflow_id) REFERENCES agent_platform.agent_workflows(id);
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS idx_agent_checkpoints_workflow
    ON agent_platform.agent_checkpoints(workflow_id, event_seq DESC);

ALTER TABLE agent_platform.agent_run_states
    ADD COLUMN IF NOT EXISTS workflow_id UUID;
UPDATE agent_platform.agent_run_states AS state
SET workflow_id = run.workflow_id
FROM agent_platform.agent_runs AS run
WHERE state.run_id = run.id AND state.workflow_id IS NULL;
ALTER TABLE agent_platform.agent_run_states
    ALTER COLUMN workflow_id SET NOT NULL;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='fk_agent_run_states_workflow') THEN
        ALTER TABLE agent_platform.agent_run_states
            ADD CONSTRAINT fk_agent_run_states_workflow
            FOREIGN KEY (workflow_id) REFERENCES agent_platform.agent_workflows(id);
    END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_run_states_workflow
    ON agent_platform.agent_run_states(workflow_id);
CREATE INDEX IF NOT EXISTS idx_agent_run_states_workflow_seq
    ON agent_platform.agent_run_states(workflow_id, event_seq DESC);

-- Normalized node projection. The Plan JSON remains the API snapshot, while
-- this table makes scheduling/branching/metrics queries independent of JSON
-- text and allows a node to be retried without rewriting its full graph.
CREATE TABLE IF NOT EXISTS agent_platform.agent_plan_node_states (
    workflow_id         UUID NOT NULL REFERENCES agent_platform.agent_workflows(id) ON DELETE CASCADE,
    node_id             VARCHAR(256) NOT NULL,
    plan_revision       INTEGER NOT NULL,
    status              VARCHAR(32) NOT NULL,
    output              TEXT,
    artifact_ids        JSONB NOT NULL DEFAULT '[]'::jsonb,
    tests               JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_tokens        BIGINT NOT NULL DEFAULT 0,
    output_tokens       BIGINT NOT NULL DEFAULT 0,
    total_tokens        BIGINT NOT NULL DEFAULT 0,
    cost_usd            DOUBLE PRECISION NOT NULL DEFAULT 0,
    duration_ms         BIGINT NOT NULL DEFAULT 0,
    attempts            INTEGER NOT NULL DEFAULT 0,
    last_run_id         UUID,
    last_event_sequence BIGINT NOT NULL DEFAULT 0,
    blocked_reason      TEXT,
    retry_from_node_id  VARCHAR(256),
    next_node_ids       JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workflow_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_plan_node_states_ready
    ON agent_platform.agent_plan_node_states(workflow_id, status, updated_at);
