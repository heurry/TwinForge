CREATE TABLE IF NOT EXISTS agent_platform.memory_sources (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       VARCHAR(128) NOT NULL,
    source_layer    VARCHAR(32) NOT NULL,
    agent_id        UUID REFERENCES agent_platform.agent_definitions(id),
    user_id         VARCHAR(128),
    project_key     VARCHAR(512),
    team_id         VARCHAR(128),
    uri             TEXT NOT NULL,
    display_name    VARCHAR(512) NOT NULL,
    content_hash    VARCHAR(128) NOT NULL,
    revision        BIGINT NOT NULL DEFAULT 1,
    authority       DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    writable_by     VARCHAR(32) NOT NULL DEFAULT 'owner',
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    git_commit      VARCHAR(128),
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_memory_source_layer CHECK (source_layer IN ('managed','user','project','local','auto','team')),
    CONSTRAINT chk_memory_source_authority CHECK (authority >= 0 AND authority <= 1),
    CONSTRAINT chk_memory_source_writable_by CHECK (writable_by IN ('admin','owner','agent','team','none')),
    CONSTRAINT chk_memory_source_uri CHECK (length(btrim(uri)) BETWEEN 1 AND 4096),
    CONSTRAINT chk_memory_source_display_name CHECK (length(btrim(display_name)) BETWEEN 1 AND 512)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_sources_revision
    ON agent_platform.memory_sources (tenant_id, uri, revision);
CREATE INDEX IF NOT EXISTS idx_memory_sources_lookup
    ON agent_platform.memory_sources (tenant_id, source_layer, project_key, team_id, updated_at DESC)
    WHERE enabled;

ALTER TABLE agent_platform.agent_memories
    ADD COLUMN IF NOT EXISTS source_id UUID REFERENCES agent_platform.memory_sources(id),
    ADD COLUMN IF NOT EXISTS source_layer VARCHAR(32) NOT NULL DEFAULT 'auto',
    ADD COLUMN IF NOT EXISTS semantic_type VARCHAR(32) NOT NULL DEFAULT 'project',
    ADD COLUMN IF NOT EXISTS project_key VARCHAR(512),
    ADD COLUMN IF NOT EXISTS team_id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS title VARCHAR(256) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS description VARCHAR(1200) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS body TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS structured_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    ADD COLUMN IF NOT EXISTS freshness_class VARCHAR(32) NOT NULL DEFAULT 'normal',
    ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS verification_hint TEXT,
    ADD COLUMN IF NOT EXISTS canonical_key VARCHAR(512) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS content_hash VARCHAR(128) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS supersedes_id UUID REFERENCES agent_platform.agent_memories(id);

UPDATE agent_platform.agent_memories
SET body = CASE WHEN btrim(body)='' THEN content ELSE body END,
    description = CASE WHEN btrim(description)='' THEN left(content, 1000) ELSE description END,
    semantic_type = CASE kind
        WHEN 'preference' THEN 'feedback'
        WHEN 'episodic' THEN 'project'
        ELSE 'project'
    END,
    source_layer = CASE
        WHEN COALESCE(metadata->>'source_layer','') IN ('managed','user','project','local','auto','team')
            THEN metadata->>'source_layer'
        ELSE 'auto'
    END,
    confidence = GREATEST(0, LEAST(1, importance)),
    canonical_key = CASE WHEN btrim(canonical_key)='' THEN md5(lower(btrim(content))) ELSE canonical_key END,
    content_hash = CASE WHEN btrim(content_hash)='' THEN md5(content) ELSE content_hash END
WHERE TRUE;

ALTER TABLE agent_platform.agent_memories
    ADD CONSTRAINT chk_agent_memory_source_layer
        CHECK (source_layer IN ('managed','user','project','local','auto','team')),
    ADD CONSTRAINT chk_agent_memory_semantic_type
        CHECK (semantic_type IN ('user','feedback','project','reference')),
    ADD CONSTRAINT chk_agent_memory_taxonomy_status
        CHECK (status IN ('active','review','superseded','expired','deleted')),
    ADD CONSTRAINT chk_agent_memory_confidence
        CHECK (confidence >= 0 AND confidence <= 1),
    ADD CONSTRAINT chk_agent_memory_freshness_class
        CHECK (freshness_class IN ('stable','normal','volatile')),
    ADD CONSTRAINT chk_agent_memory_structured_data
        CHECK (jsonb_typeof(structured_data)='object'),
    ADD CONSTRAINT chk_agent_memory_body
        CHECK (length(btrim(body)) BETWEEN 1 AND 32000),
    ADD CONSTRAINT chk_agent_memory_taxonomy_window
        CHECK (valid_until IS NULL OR valid_from IS NULL OR valid_until > valid_from);

CREATE INDEX IF NOT EXISTS idx_agent_memories_taxonomy
    ON agent_platform.agent_memories (tenant_id, source_layer, semantic_type, status, updated_at DESC)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_agent_memories_project
    ON agent_platform.agent_memories (tenant_id, project_key, source_layer, updated_at DESC)
    WHERE deleted_at IS NULL AND project_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_memories_manifest
    ON agent_platform.agent_memories (tenant_id, source_layer, semantic_type, pinned DESC, updated_at DESC)
    WHERE deleted_at IS NULL AND status='active';
CREATE INDEX IF NOT EXISTS idx_agent_memories_canonical_key
    ON agent_platform.agent_memories (tenant_id, source_layer, canonical_key)
    WHERE deleted_at IS NULL AND status IN ('active','review');
