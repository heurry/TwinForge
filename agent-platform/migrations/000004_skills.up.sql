CREATE TABLE agent_platform.skill_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    skill_key VARCHAR(128) NOT NULL,
    name VARCHAR(256) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_skill_definition UNIQUE (tenant_id, skill_key)
);

CREATE TABLE agent_platform.skill_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    definition_id UUID NOT NULL REFERENCES agent_platform.skill_definitions(id),
    version INTEGER NOT NULL,
    spec JSONB NOT NULL,
    spec_hash VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'published',
    created_by VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_skill_version UNIQUE (definition_id, version)
);

CREATE TABLE agent_platform.skillset_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    skillset_key VARCHAR(128) NOT NULL,
    name VARCHAR(256) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_skillset_definition UNIQUE (tenant_id, skillset_key)
);

CREATE TABLE agent_platform.skillset_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    definition_id UUID NOT NULL REFERENCES agent_platform.skillset_definitions(id),
    version INTEGER NOT NULL,
    spec JSONB NOT NULL,
    spec_hash VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'published',
    created_by VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_skillset_version UNIQUE (definition_id, version)
);
