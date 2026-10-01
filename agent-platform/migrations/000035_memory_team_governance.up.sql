-- Team memories are visible only to explicitly enrolled users. The platform
-- currently derives the acting user from agent_sessions.user_id, so this table
-- is the durable ACL boundary used by every Run-scoped memory query.
CREATE TABLE IF NOT EXISTS agent_platform.memory_team_memberships (
    tenant_id   VARCHAR(128) NOT NULL,
    team_id     VARCHAR(128) NOT NULL,
    user_id     VARCHAR(128) NOT NULL,
    role        VARCHAR(32) NOT NULL DEFAULT 'member',
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, team_id, user_id),
    CONSTRAINT chk_memory_team_membership_role
        CHECK (role IN ('member','manager','owner'))
);

CREATE INDEX IF NOT EXISTS idx_memory_team_memberships_user
    ON agent_platform.memory_team_memberships (tenant_id, user_id, team_id)
    WHERE enabled;

ALTER TABLE agent_platform.agent_memories
    DROP CONSTRAINT IF EXISTS chk_agent_memory_team_binding;

-- Older taxonomy rows may have been labeled team before team_id was
-- mandatory. Quarantine those ambiguous rows instead of making migration
-- failure depend on historical data quality.
UPDATE agent_platform.agent_memories
SET source_layer='auto', team_id=NULL, status=CASE WHEN status='active' THEN 'review' ELSE status END, updated_at=now()
WHERE source_layer='team' AND NULLIF(btrim(team_id),'') IS NULL;

ALTER TABLE agent_platform.agent_memories
    ADD CONSTRAINT chk_agent_memory_team_binding
        CHECK (source_layer <> 'team' OR NULLIF(btrim(team_id),'') IS NOT NULL);

ALTER TABLE agent_platform.memory_sources
    DROP CONSTRAINT IF EXISTS chk_memory_source_team_binding;

UPDATE agent_platform.memory_sources
SET source_layer='auto', team_id=NULL, writable_by='agent', updated_at=now()
WHERE source_layer='team' AND NULLIF(btrim(team_id),'') IS NULL;

ALTER TABLE agent_platform.memory_sources
    ADD CONSTRAINT chk_memory_source_team_binding
        CHECK (source_layer <> 'team' OR NULLIF(btrim(team_id),'') IS NOT NULL);
