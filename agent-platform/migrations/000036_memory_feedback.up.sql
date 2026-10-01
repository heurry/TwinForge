CREATE TABLE IF NOT EXISTS agent_platform.memory_feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   VARCHAR(128) NOT NULL,
    memory_id   UUID NOT NULL REFERENCES agent_platform.agent_memories(id) ON DELETE CASCADE,
    run_id      UUID REFERENCES agent_platform.agent_runs(id) ON DELETE SET NULL,
    turn_id     VARCHAR(128),
    action      VARCHAR(32) NOT NULL,
    actor       VARCHAR(128),
    reason      TEXT,
    evidence    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_memory_feedback_action
        CHECK (action IN ('used','ignored','contradicted','verified','updated')),
    CONSTRAINT chk_memory_feedback_evidence
        CHECK (jsonb_typeof(evidence)='object')
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_feedback_idempotency
    ON agent_platform.memory_feedback (tenant_id,memory_id,COALESCE(run_id,'00000000-0000-0000-0000-000000000000'::uuid),COALESCE(turn_id,''),action,COALESCE(actor,''));

CREATE INDEX IF NOT EXISTS idx_memory_feedback_memory
    ON agent_platform.memory_feedback (tenant_id,memory_id,created_at DESC);
