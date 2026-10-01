CREATE TABLE agent_platform.agent_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    run_id UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    call_id VARCHAR(256),
    kind VARCHAR(64) NOT NULL,
    name VARCHAR(512) NOT NULL,
    media_type VARCHAR(256) NOT NULL,
    content BYTEA NOT NULL,
    content_hash VARCHAR(128) NOT NULL,
    size_bytes BIGINT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_agent_artifact_size CHECK (size_bytes >= 0 AND size_bytes <= 10485760)
);
CREATE INDEX idx_agent_artifacts_run ON agent_platform.agent_artifacts(run_id, created_at);

CREATE TABLE agent_platform.agent_tool_approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    run_id UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    call_id VARCHAR(256) NOT NULL,
	turn_no INTEGER NOT NULL,
	step_no INTEGER NOT NULL,
    tool_version_id UUID NOT NULL REFERENCES agent_platform.tool_versions(id),
    tool_name VARCHAR(256) NOT NULL,
    risk VARCHAR(32) NOT NULL,
    request_hash VARCHAR(128) NOT NULL,
    request JSONB NOT NULL,
    diff_artifact_id UUID REFERENCES agent_platform.agent_artifacts(id),
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    requested_by VARCHAR(128),
    decided_by VARCHAR(128),
    decision_reason TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at TIMESTAMPTZ,
    CONSTRAINT uq_agent_tool_approval UNIQUE(tenant_id, run_id, call_id),
    CONSTRAINT chk_agent_tool_approval_status CHECK(status IN ('pending','approved','rejected','expired'))
);
CREATE INDEX idx_agent_tool_approvals_pending ON agent_platform.agent_tool_approvals(tenant_id, status, created_at);

ALTER TABLE agent_platform.agent_runs
    ADD COLUMN parent_run_id UUID REFERENCES agent_platform.agent_runs(id),
    ADD COLUMN root_run_id UUID REFERENCES agent_platform.agent_runs(id),
    ADD COLUMN delegation_id UUID,
    ADD COLUMN delegation_depth INTEGER NOT NULL DEFAULT 0;
CREATE INDEX idx_agent_runs_parent ON agent_platform.agent_runs(parent_run_id, created_at);

CREATE TABLE agent_platform.agent_delegations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    parent_run_id UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    child_run_id UUID REFERENCES agent_platform.agent_runs(id),
    source_agent_version_id UUID NOT NULL REFERENCES agent_platform.agent_versions(id),
    target_agent_version_id UUID NOT NULL REFERENCES agent_platform.agent_versions(id),
    call_id VARCHAR(256) NOT NULL,
	turn_no INTEGER NOT NULL,
	step_no INTEGER NOT NULL,
    mode VARCHAR(16) NOT NULL DEFAULT 'sync',
    request_hash VARCHAR(128) NOT NULL,
    input JSONB NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    output JSONB,
    error TEXT,
    deadline TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_delegation_call UNIQUE(tenant_id, parent_run_id, call_id),
    CONSTRAINT chk_agent_delegation_mode CHECK(mode IN ('sync','async'))
);

CREATE TABLE agent_platform.mcp_server_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    server_key VARCHAR(128) NOT NULL,
    name VARCHAR(256) NOT NULL,
    owner VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_mcp_server_definition UNIQUE(tenant_id, server_key)
);
CREATE TABLE agent_platform.mcp_server_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    definition_id UUID NOT NULL REFERENCES agent_platform.mcp_server_definitions(id),
    version INTEGER NOT NULL,
    spec JSONB NOT NULL,
    spec_hash VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'published',
    created_by VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_mcp_server_version UNIQUE(definition_id, version)
);
CREATE TABLE agent_platform.mcp_tool_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    server_version_id UUID NOT NULL REFERENCES agent_platform.mcp_server_versions(id) ON DELETE CASCADE,
    tool_name VARCHAR(256) NOT NULL,
    description TEXT,
    input_schema JSONB NOT NULL,
    schema_hash VARCHAR(128) NOT NULL,
    risk VARCHAR(32) NOT NULL DEFAULT 'READ',
    enabled BOOLEAN NOT NULL DEFAULT true,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_mcp_tool_snapshot UNIQUE(server_version_id, tool_name)
);
CREATE TABLE agent_platform.mcp_connection_health (
    server_version_id UUID PRIMARY KEY REFERENCES agent_platform.mcp_server_versions(id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL,
    protocol_version VARCHAR(32),
    latency_ms BIGINT,
    error TEXT,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE agent_platform.a2a_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    agent_id UUID NOT NULL REFERENCES agent_platform.agent_definitions(id),
    run_id UUID REFERENCES agent_platform.agent_runs(id),
    context_id UUID NOT NULL DEFAULT gen_random_uuid(),
    client_message_id VARCHAR(256) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'submitted',
    message JSONB NOT NULL,
    artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_a2a_message UNIQUE(tenant_id, agent_id, client_message_id),
    CONSTRAINT chk_a2a_task_status CHECK(status IN ('submitted','working','input-required','auth-required','completed','failed','canceled','rejected'))
);
CREATE INDEX idx_a2a_tasks_tenant ON agent_platform.a2a_tasks(tenant_id, created_at DESC);
