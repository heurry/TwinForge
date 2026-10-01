-- One replayable delivery projection per Run. Artifact rows remain immutable
-- history; the manifest points at the latest artifact for each logical name.
CREATE TABLE agent_platform.agent_run_manifests (
    run_id UUID PRIMARY KEY REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(128) NOT NULL,
    workflow_id UUID NOT NULL REFERENCES agent_platform.agent_workflows(id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL,
    canonical_artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
    required_outputs JSONB NOT NULL DEFAULT '[]'::jsonb,
    verification_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    child_runs JSONB NOT NULL DEFAULT '[]'::jsonb,
    final_artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
    final_output_hash VARCHAR(128),
    final_output_present BOOLEAN NOT NULL DEFAULT false,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_agent_run_manifests_tenant ON agent_platform.agent_run_manifests(tenant_id, updated_at DESC);
CREATE INDEX idx_agent_run_manifests_workflow ON agent_platform.agent_run_manifests(tenant_id, workflow_id, updated_at DESC);
