CREATE TABLE IF NOT EXISTS agent_platform.memory_retrievals (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(128) NOT NULL,
    run_id              UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    turn_id             VARCHAR(128) NOT NULL,
    query_hash          VARCHAR(128) NOT NULL,
    candidate_ids       JSONB NOT NULL DEFAULT '[]'::jsonb,
    routed_ids          JSONB NOT NULL DEFAULT '[]'::jsonb,
    injected_ids        JSONB NOT NULL DEFAULT '[]'::jsonb,
    suppressed_ids      JSONB NOT NULL DEFAULT '[]'::jsonb,
    scores              JSONB NOT NULL DEFAULT '{}'::jsonb,
    suppression_reasons JSONB NOT NULL DEFAULT '{}'::jsonb,
    router_model        VARCHAR(255),
    manifest_tokens     INTEGER NOT NULL DEFAULT 0,
    body_tokens         INTEGER NOT NULL DEFAULT 0,
    latency_ms          BIGINT NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_memory_retrieval_candidate_ids CHECK (jsonb_typeof(candidate_ids)='array'),
    CONSTRAINT chk_memory_retrieval_routed_ids CHECK (jsonb_typeof(routed_ids)='array'),
    CONSTRAINT chk_memory_retrieval_injected_ids CHECK (jsonb_typeof(injected_ids)='array'),
    CONSTRAINT chk_memory_retrieval_suppressed_ids CHECK (jsonb_typeof(suppressed_ids)='array'),
    CONSTRAINT chk_memory_retrieval_scores CHECK (jsonb_typeof(scores)='object'),
    CONSTRAINT chk_memory_retrieval_reasons CHECK (jsonb_typeof(suppression_reasons)='object'),
    CONSTRAINT chk_memory_retrieval_tokens CHECK (manifest_tokens >= 0 AND body_tokens >= 0 AND latency_ms >= 0)
);

CREATE INDEX IF NOT EXISTS idx_memory_retrievals_run
    ON agent_platform.memory_retrievals (tenant_id, run_id, created_at DESC);
