CREATE INDEX idx_agent_memories_embedding_model
    ON agent_platform.agent_memories (embedding_model)
    WHERE deleted_at IS NULL AND embedding_status='ready';
