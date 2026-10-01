CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE agent_platform.agent_memories
    ADD COLUMN embedding vector(1024),
    ADD COLUMN embedding_model VARCHAR(255),
    ADD COLUMN embedding_status VARCHAR(32) NOT NULL DEFAULT 'pending',
    ADD COLUMN embedding_error TEXT,
    ADD COLUMN embedded_at TIMESTAMPTZ,
    ADD CONSTRAINT chk_agent_memory_embedding_status
        CHECK (embedding_status IN ('pending', 'ready', 'failed')),
    ADD CONSTRAINT chk_agent_memory_embedding_state
        CHECK (
            (embedding_status='ready' AND embedding IS NOT NULL AND NULLIF(btrim(embedding_model),'') IS NOT NULL AND embedding_error IS NULL AND embedded_at IS NOT NULL) OR
            (embedding_status IN ('pending','failed') AND embedding IS NULL)
        );

CREATE INDEX idx_agent_memories_embedding_hnsw
    ON agent_platform.agent_memories USING hnsw (embedding vector_cosine_ops)
    WHERE deleted_at IS NULL AND embedding_status='ready';

CREATE INDEX idx_agent_memories_embedding_reconcile
    ON agent_platform.agent_memories (embedding_status, updated_at, id)
    WHERE deleted_at IS NULL AND embedding_status<>'ready';
