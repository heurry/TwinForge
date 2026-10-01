CREATE TABLE agent_platform.prompt_definitions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  VARCHAR(128) NOT NULL,
    prompt_key VARCHAR(128) NOT NULL,
    name       VARCHAR(256) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_prompt_definition UNIQUE (tenant_id, prompt_key)
);

CREATE TABLE agent_platform.prompt_versions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    definition_id UUID NOT NULL REFERENCES agent_platform.prompt_definitions(id),
    version       INTEGER NOT NULL,
    content       TEXT NOT NULL,
    content_hash  VARCHAR(128) NOT NULL,
    status        VARCHAR(32) NOT NULL DEFAULT 'published',
    created_by    VARCHAR(128),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_prompt_version UNIQUE (definition_id, version)
);

CREATE TABLE agent_platform.tool_definitions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  VARCHAR(128) NOT NULL,
    tool_key   VARCHAR(128) NOT NULL,
    name       VARCHAR(256) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_tool_definition UNIQUE (tenant_id, tool_key)
);

CREATE TABLE agent_platform.tool_versions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    definition_id UUID NOT NULL REFERENCES agent_platform.tool_definitions(id),
    version       INTEGER NOT NULL,
    spec          JSONB NOT NULL,
    spec_hash     VARCHAR(128) NOT NULL,
    status        VARCHAR(32) NOT NULL DEFAULT 'published',
    created_by    VARCHAR(128),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_tool_version UNIQUE (definition_id, version)
);

CREATE TABLE agent_platform.toolset_definitions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   VARCHAR(128) NOT NULL,
    toolset_key VARCHAR(128) NOT NULL,
    name        VARCHAR(256) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_toolset_definition UNIQUE (tenant_id, toolset_key)
);

CREATE TABLE agent_platform.toolset_versions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    definition_id UUID NOT NULL REFERENCES agent_platform.toolset_definitions(id),
    version       INTEGER NOT NULL,
    spec          JSONB NOT NULL,
    spec_hash     VARCHAR(128) NOT NULL,
    status        VARCHAR(32) NOT NULL DEFAULT 'published',
    created_by    VARCHAR(128),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_toolset_version UNIQUE (definition_id, version)
);
