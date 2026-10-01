CREATE TABLE agent_platform.agent_artifact_promotions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(128) NOT NULL,
    run_id UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    artifact_id UUID NOT NULL REFERENCES agent_platform.agent_artifacts(id) ON DELETE CASCADE,
    target_path VARCHAR(1024) NOT NULL,
    source_hash VARCHAR(128) NOT NULL,
    expected_target_sha256 VARCHAR(128),
    previous_target_sha256 VARCHAR(128),
    result_target_sha256 VARCHAR(128),
    status VARCHAR(32) NOT NULL DEFAULT 'requested',
    requested_by VARCHAR(128) NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    CONSTRAINT chk_agent_artifact_promotion_status CHECK(status IN ('requested','promoted','failed'))
);
CREATE INDEX idx_agent_artifact_promotions_run ON agent_platform.agent_artifact_promotions(run_id, created_at DESC);
