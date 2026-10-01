CREATE TABLE agent_platform.agent_environment_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_key VARCHAR(128) NOT NULL,
    version INTEGER NOT NULL,
    name VARCHAR(256) NOT NULL,
    runtime VARCHAR(64) NOT NULL,
    image_ref VARCHAR(512) NOT NULL,
    spec_digest VARCHAR(128) NOT NULL,
    dependencies JSONB NOT NULL DEFAULT '[]'::jsonb,
    capabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_environment_template UNIQUE(template_key, version),
    CONSTRAINT chk_agent_environment_template_status CHECK(status IN ('active','deprecated','retired')),
    CONSTRAINT chk_agent_environment_template_dependencies CHECK(jsonb_typeof(dependencies)='array'),
    CONSTRAINT chk_agent_environment_template_capabilities CHECK(jsonb_typeof(capabilities)='object')
);

INSERT INTO agent_platform.agent_environment_templates
    (id,template_key,version,name,runtime,image_ref,spec_digest,dependencies,capabilities)
VALUES
    ('80000000-0000-4000-8000-000000000001','python-game',1,'Python Game Sandbox','python3.11',
     'twinforge/agent-sandbox:python-game-v1','sha256:e62d7dba1353847c6bba32a7c8757a54febe3ef900a2b8bb4d54af81ce839c3d',
     '[{"ecosystem":"python","name":"pygame","version":"2.6.1"}]'::jsonb,
     '{"shell":false,"network":false,"workspace_scope":"run","headless_sdl":true}'::jsonb)
ON CONFLICT (template_key,version) DO NOTHING;

CREATE TABLE agent_platform.agent_dependency_installs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    run_id UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    call_id VARCHAR(256) NOT NULL,
    environment_template VARCHAR(256) NOT NULL,
    ecosystem VARCHAR(32) NOT NULL,
    packages JSONB NOT NULL,
    source VARCHAR(64) NOT NULL,
    scope VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    CONSTRAINT uq_agent_dependency_install UNIQUE(tenant_id,run_id,call_id),
    CONSTRAINT chk_agent_dependency_packages CHECK(jsonb_typeof(packages)='array'),
    CONSTRAINT chk_agent_dependency_scope CHECK(scope='run'),
    CONSTRAINT chk_agent_dependency_status CHECK(status IN ('installed','failed'))
);
CREATE INDEX idx_agent_dependency_installs_run ON agent_platform.agent_dependency_installs(tenant_id,run_id,created_at DESC);
