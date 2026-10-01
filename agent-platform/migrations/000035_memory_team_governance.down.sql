ALTER TABLE agent_platform.memory_sources
    DROP CONSTRAINT IF EXISTS chk_memory_source_team_binding;

ALTER TABLE agent_platform.agent_memories
    DROP CONSTRAINT IF EXISTS chk_agent_memory_team_binding;

DROP INDEX IF EXISTS agent_platform.idx_memory_team_memberships_user;
DROP TABLE IF EXISTS agent_platform.memory_team_memberships;
