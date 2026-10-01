CREATE TABLE IF NOT EXISTS agent_platform.agent_observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    run_id UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    trace_id TEXT NOT NULL,
    observation_key TEXT NOT NULL,
    parent_observation_id UUID REFERENCES agent_platform.agent_observations(id) ON DELETE SET NULL,
    root_observation_id UUID REFERENCES agent_platform.agent_observations(id) ON DELETE SET NULL,
    kind VARCHAR(32) NOT NULL,
    name TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    sequence BIGINT NOT NULL,
    last_sequence BIGINT NOT NULL,
    turn_no INT,
    step_no INT,
    call_id TEXT,
    delegation_id TEXT,
    child_run_id TEXT,
    agent_version_id UUID,
    model_resolution_id TEXT,
    model_id TEXT,
    tool_version_id TEXT,
    prompt_version_id TEXT,
    skillset_version_id TEXT,
    toolset_version_id TEXT,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    total_cost NUMERIC(20,8) NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_ms BIGINT NOT NULL DEFAULT 0,
    detail_refs JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(run_id, observation_key),
    UNIQUE(run_id, sequence),
    CHECK (input_tokens >= 0 AND output_tokens >= 0 AND total_cost >= 0 AND duration_ms >= 0)
);
CREATE INDEX IF NOT EXISTS idx_agent_observations_trace_sequence ON agent_platform.agent_observations(tenant_id, trace_id, sequence);
CREATE INDEX IF NOT EXISTS idx_agent_observations_parent ON agent_platform.agent_observations(run_id, parent_observation_id, sequence);
CREATE INDEX IF NOT EXISTS idx_agent_observations_versions ON agent_platform.agent_observations(tenant_id, agent_version_id) WHERE agent_version_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS agent_platform.agent_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    run_id UUID REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    observation_id UUID REFERENCES agent_platform.agent_observations(id) ON DELETE CASCADE,
    session_id UUID REFERENCES agent_platform.agent_sessions(id) ON DELETE CASCADE,
    dataset_run_id TEXT,
    name TEXT NOT NULL,
    score_type VARCHAR(20) NOT NULL,
    value NUMERIC(20,8),
    string_value TEXT,
    source VARCHAR(32) NOT NULL,
    evaluator_version TEXT,
    agent_version_id TEXT,
    model_resolution_id TEXT,
    prompt_version_id TEXT,
    toolset_version_id TEXT,
    skillset_version_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (score_type IN ('numeric','categorical','boolean','text')),
    CHECK (value IS NOT NULL OR string_value IS NOT NULL),
    CHECK (run_id IS NOT NULL OR observation_id IS NOT NULL OR session_id IS NOT NULL OR dataset_run_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_agent_scores_run ON agent_platform.agent_scores(tenant_id, run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_scores_observation ON agent_platform.agent_scores(tenant_id, observation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_scores_name ON agent_platform.agent_scores(tenant_id, name, created_at DESC);

CREATE TABLE IF NOT EXISTS agent_platform.agent_observation_archives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL,
    run_id UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    sequence BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    object_key TEXT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    size_bytes BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'ready',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(run_id, sequence),
    UNIQUE(object_key)
);
CREATE INDEX IF NOT EXISTS idx_agent_observation_archives_run ON agent_platform.agent_observation_archives(tenant_id, run_id, sequence);
