CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE agent_platform.agent_memories (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     VARCHAR(128) NOT NULL,
    scope         VARCHAR(32) NOT NULL,
    agent_id      UUID REFERENCES agent_platform.agent_definitions(id),
    user_id       VARCHAR(128),
    session_id    UUID REFERENCES agent_platform.agent_sessions(id),
    kind          VARCHAR(32) NOT NULL DEFAULT 'semantic',
    content       TEXT NOT NULL,
    importance    DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    source_run_id UUID REFERENCES agent_platform.agent_runs(id),
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    expires_at    TIMESTAMPTZ,
    deleted_at    TIMESTAMPTZ,
    created_by    VARCHAR(128),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_agent_memory_scope CHECK (scope IN ('tenant','agent','user','session')),
    CONSTRAINT chk_agent_memory_kind CHECK (kind IN ('semantic','episodic','preference')),
    CONSTRAINT chk_agent_memory_importance CHECK (importance >= 0 AND importance <= 1),
    CONSTRAINT chk_agent_memory_owner CHECK (
        (scope='tenant' AND agent_id IS NULL AND user_id IS NULL AND session_id IS NULL) OR
        (scope='agent' AND agent_id IS NOT NULL AND user_id IS NULL AND session_id IS NULL) OR
        (scope='user' AND user_id IS NOT NULL AND session_id IS NULL) OR
        (scope='session' AND session_id IS NOT NULL)
    ),
    CONSTRAINT chk_agent_memory_content CHECK (length(btrim(content)) BETWEEN 1 AND 8000)
);

CREATE INDEX idx_agent_memories_tenant_scope
    ON agent_platform.agent_memories (tenant_id, scope, updated_at DESC)
    WHERE deleted_at IS NULL;
CREATE INDEX idx_agent_memories_agent
    ON agent_platform.agent_memories (tenant_id, agent_id, updated_at DESC)
    WHERE deleted_at IS NULL AND agent_id IS NOT NULL;
CREATE INDEX idx_agent_memories_user
    ON agent_platform.agent_memories (tenant_id, user_id, updated_at DESC)
    WHERE deleted_at IS NULL AND user_id IS NOT NULL;
CREATE INDEX idx_agent_memories_session
    ON agent_platform.agent_memories (tenant_id, session_id, updated_at DESC)
    WHERE deleted_at IS NULL AND session_id IS NOT NULL;
CREATE INDEX idx_agent_memories_content_trgm
    ON agent_platform.agent_memories USING gin (content gin_trgm_ops)
    WHERE deleted_at IS NULL;
