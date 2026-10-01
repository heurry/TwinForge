CREATE TABLE IF NOT EXISTS agent_platform.memory_revisions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(128) NOT NULL,
    memory_id           UUID NOT NULL REFERENCES agent_platform.agent_memories(id) ON DELETE CASCADE,
    revision             BIGINT NOT NULL,
    title               VARCHAR(256) NOT NULL,
    description         VARCHAR(1200) NOT NULL,
    body                TEXT NOT NULL,
    structured_data     JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_message_ids  JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_run_id       UUID REFERENCES agent_platform.agent_runs(id),
    source_event_from   BIGINT,
    source_event_to     BIGINT,
    reason              VARCHAR(64) NOT NULL,
    created_by_type     VARCHAR(32) NOT NULL,
    created_by          VARCHAR(128),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_memory_revision_number UNIQUE (memory_id, revision),
    CONSTRAINT chk_memory_revision_reason CHECK (reason IN ('create','update','merge','supersede','manual','verify','consolidate')),
    CONSTRAINT chk_memory_revision_creator CHECK (created_by_type IN ('user','agent','system','import')),
    CONSTRAINT chk_memory_revision_source_messages CHECK (jsonb_typeof(source_message_ids)='array'),
    CONSTRAINT chk_memory_revision_structured_data CHECK (jsonb_typeof(structured_data)='object'),
    CONSTRAINT chk_memory_revision_body CHECK (length(btrim(body)) BETWEEN 1 AND 32000)
);

CREATE INDEX IF NOT EXISTS idx_memory_revisions_lookup
    ON agent_platform.memory_revisions (tenant_id, memory_id, revision DESC);

CREATE TABLE IF NOT EXISTS agent_platform.memory_write_jobs (
    id                              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                       VARCHAR(128) NOT NULL,
    run_id                          UUID REFERENCES agent_platform.agent_runs(id),
    session_id                      UUID REFERENCES agent_platform.agent_sessions(id),
    turn_id                         VARCHAR(128) NOT NULL,
    trigger                         VARCHAR(32) NOT NULL,
    status                          VARCHAR(32) NOT NULL DEFAULT 'pending',
    attempt                         INTEGER NOT NULL DEFAULT 0,
    lease_token                     UUID,
    available_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_event_from               BIGINT,
    source_event_to                 BIGINT,
    last_memory_write_sequence      BIGINT,
    input_hash                      VARCHAR(128) NOT NULL,
    result_summary                  JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_error                      TEXT,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at                      TIMESTAMPTZ,
    finished_at                     TIMESTAMPTZ,
    CONSTRAINT uq_memory_write_job_input UNIQUE (tenant_id, run_id, turn_id, trigger, input_hash),
    CONSTRAINT chk_memory_write_job_trigger CHECK (trigger IN ('turn_complete','collapse_barrier','manual','consolidation')),
    CONSTRAINT chk_memory_write_job_status CHECK (status IN ('pending','running','completed','failed','cancelled')),
    CONSTRAINT chk_memory_write_job_result CHECK (jsonb_typeof(result_summary)='object')
);

CREATE INDEX IF NOT EXISTS idx_memory_write_jobs_claim
    ON agent_platform.memory_write_jobs (status, available_at, created_at)
    WHERE status IN ('pending','running');
