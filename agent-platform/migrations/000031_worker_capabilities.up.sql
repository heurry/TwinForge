CREATE TABLE IF NOT EXISTS agent_platform.agent_worker_capabilities (
    worker_id TEXT PRIMARY KEY,
    runtime_version TEXT NOT NULL,
    tool_contract_version TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    capability_hash TEXT NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_worker_capabilities_array CHECK (jsonb_typeof(capabilities)='array')
);

CREATE INDEX IF NOT EXISTS idx_agent_worker_capabilities_seen
    ON agent_platform.agent_worker_capabilities(last_seen_at);
