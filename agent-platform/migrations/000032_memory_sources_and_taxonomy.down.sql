DROP INDEX IF EXISTS agent_platform.idx_agent_memories_canonical_key;
DROP INDEX IF EXISTS agent_platform.idx_agent_memories_manifest;
DROP INDEX IF EXISTS agent_platform.idx_agent_memories_project;
DROP INDEX IF EXISTS agent_platform.idx_agent_memories_taxonomy;

ALTER TABLE agent_platform.agent_memories
    DROP CONSTRAINT IF EXISTS chk_agent_memory_taxonomy_window,
    DROP CONSTRAINT IF EXISTS chk_agent_memory_body,
    DROP CONSTRAINT IF EXISTS chk_agent_memory_structured_data,
    DROP CONSTRAINT IF EXISTS chk_agent_memory_freshness_class,
    DROP CONSTRAINT IF EXISTS chk_agent_memory_confidence,
    DROP CONSTRAINT IF EXISTS chk_agent_memory_taxonomy_status,
    DROP CONSTRAINT IF EXISTS chk_agent_memory_semantic_type,
    DROP CONSTRAINT IF EXISTS chk_agent_memory_source_layer,
    DROP COLUMN IF EXISTS supersedes_id,
    DROP COLUMN IF EXISTS pinned,
    DROP COLUMN IF EXISTS content_hash,
    DROP COLUMN IF EXISTS canonical_key,
    DROP COLUMN IF EXISTS verification_hint,
    DROP COLUMN IF EXISTS last_verified_at,
    DROP COLUMN IF EXISTS valid_until,
    DROP COLUMN IF EXISTS valid_from,
    DROP COLUMN IF EXISTS freshness_class,
    DROP COLUMN IF EXISTS confidence,
    DROP COLUMN IF EXISTS status,
    DROP COLUMN IF EXISTS structured_data,
    DROP COLUMN IF EXISTS body,
    DROP COLUMN IF EXISTS description,
    DROP COLUMN IF EXISTS title,
    DROP COLUMN IF EXISTS team_id,
    DROP COLUMN IF EXISTS project_key,
    DROP COLUMN IF EXISTS semantic_type,
    DROP COLUMN IF EXISTS source_layer,
    DROP COLUMN IF EXISTS source_id;

DROP INDEX IF EXISTS agent_platform.idx_memory_sources_lookup;
DROP INDEX IF EXISTS agent_platform.uq_memory_sources_revision;
DROP TABLE IF EXISTS agent_platform.memory_sources;
