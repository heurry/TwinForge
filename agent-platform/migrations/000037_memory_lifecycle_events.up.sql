CREATE TABLE IF NOT EXISTS agent_platform.memory_lifecycle_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       VARCHAR(128) NOT NULL,
    memory_id       UUID REFERENCES agent_platform.agent_memories(id) ON DELETE CASCADE,
    source_id       UUID REFERENCES agent_platform.memory_sources(id) ON DELETE SET NULL,
    run_id          UUID REFERENCES agent_platform.agent_runs(id) ON DELETE SET NULL,
    event_type      VARCHAR(64) NOT NULL,
    actor           VARCHAR(128),
    idempotency_key VARCHAR(256),
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_memory_lifecycle_event_type CHECK (event_type IN (
        'MEMORY_EXTRACTION_REQUESTED','MEMORY_EXTRACTION_STARTED',
        'MEMORY_CANDIDATE_PROPOSED','MEMORY_CREATED','MEMORY_UPDATED','MEMORY_DELETED',
        'MEMORY_MERGED','MEMORY_SUPERSEDED','MEMORY_REVIEW_REQUIRED',
        'MEMORY_EXTRACTION_COMPLETED','MEMORY_EXTRACTION_FAILED',
        'MEMORY_ROUTED','MEMORY_INJECTED','MEMORY_SUPPRESSED',
        'MEMORY_VERIFIED','MEMORY_CONTRADICTED',
        'MEMORY_TEAM_PROMOTION_REQUESTED','MEMORY_TEAM_PROMOTED',
        'MEMORY_SOURCE_REVISION_OBSERVED'
    )),
    CONSTRAINT chk_memory_lifecycle_event_payload CHECK (jsonb_typeof(payload)='object')
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_lifecycle_event_idempotency
    ON agent_platform.memory_lifecycle_events (tenant_id,event_type,idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memory_lifecycle_events_memory
    ON agent_platform.memory_lifecycle_events (tenant_id,memory_id,created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_lifecycle_events_source
    ON agent_platform.memory_lifecycle_events (tenant_id,source_id,created_at DESC)
    WHERE source_id IS NOT NULL;
