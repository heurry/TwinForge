DROP INDEX IF EXISTS agent_platform.idx_agent_memories_embedding_reconcile;
DROP INDEX IF EXISTS agent_platform.idx_agent_memories_embedding_hnsw;

ALTER TABLE agent_platform.agent_memories
    DROP CONSTRAINT IF EXISTS chk_agent_memory_embedding_state,
    DROP CONSTRAINT IF EXISTS chk_agent_memory_embedding_status,
    DROP COLUMN IF EXISTS embedded_at,
    DROP COLUMN IF EXISTS embedding_error,
    DROP COLUMN IF EXISTS embedding_status,
    DROP COLUMN IF EXISTS embedding_model,
    DROP COLUMN IF EXISTS embedding;
