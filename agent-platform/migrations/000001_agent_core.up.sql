CREATE SCHEMA IF NOT EXISTS agent_platform;

CREATE TABLE agent_platform.agent_definitions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id         VARCHAR(128) NOT NULL,
    agent_key         VARCHAR(128) NOT NULL,
    name              VARCHAR(256) NOT NULL,
    description       TEXT,
    owner             VARCHAR(128),
    status            VARCHAR(32) NOT NULL DEFAULT 'active',
    active_version_id UUID,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_definition UNIQUE (tenant_id, agent_key)
);

CREATE TABLE agent_platform.agent_versions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id     UUID NOT NULL REFERENCES agent_platform.agent_definitions(id),
    version      INTEGER NOT NULL,
    spec         JSONB NOT NULL,
    spec_hash    VARCHAR(128) NOT NULL,
    status       VARCHAR(32) NOT NULL DEFAULT 'draft',
    created_by   VARCHAR(128),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ,
    CONSTRAINT uq_agent_version UNIQUE (agent_id, version)
);

ALTER TABLE agent_platform.agent_definitions
    ADD CONSTRAINT fk_agent_active_version
    FOREIGN KEY (active_version_id) REFERENCES agent_platform.agent_versions(id);

CREATE TABLE agent_platform.agent_sessions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  VARCHAR(128) NOT NULL,
    agent_id   UUID NOT NULL REFERENCES agent_platform.agent_definitions(id),
    user_id    VARCHAR(128),
    status     VARCHAR(32) NOT NULL DEFAULT 'active',
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE agent_platform.agent_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(128) NOT NULL,
    session_id          UUID REFERENCES agent_platform.agent_sessions(id),
    agent_version_id    UUID NOT NULL REFERENCES agent_platform.agent_versions(id),
    status              VARCHAR(32) NOT NULL DEFAULT 'queued',
    trigger_type        VARCHAR(32) NOT NULL DEFAULT 'api',
    input               JSONB NOT NULL,
    output              JSONB,
    binding_snapshot    JSONB NOT NULL,
    current_turn        INTEGER NOT NULL DEFAULT 0,
    current_step        INTEGER NOT NULL DEFAULT 0,
    attempt             INTEGER NOT NULL DEFAULT 0,
    lease_owner         VARCHAR(128),
    lease_token         BIGINT NOT NULL DEFAULT 0,
    lease_expires_at    TIMESTAMPTZ,
    next_wakeup_at      TIMESTAMPTZ,
    cancel_requested_at TIMESTAMPTZ,
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    error_code          VARCHAR(128),
    error_message       TEXT,
    created_by          VARCHAR(128),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_agent_run_status CHECK (status IN (
        'queued', 'running', 'waiting_tool', 'waiting_approval',
        'waiting_external', 'suspended', 'completed', 'failed', 'cancelled'
    ))
);
CREATE INDEX idx_agent_runs_claim
    ON agent_platform.agent_runs (status, next_wakeup_at, lease_expires_at, created_at);
CREATE INDEX idx_agent_runs_session
    ON agent_platform.agent_runs (session_id, created_at DESC);

CREATE TABLE agent_platform.agent_turns (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    turn_no     INTEGER NOT NULL,
    status      VARCHAR(32) NOT NULL,
    end_reason  VARCHAR(64),
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    CONSTRAINT uq_agent_turn UNIQUE (run_id, turn_no)
);

CREATE TABLE agent_platform.agent_steps (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id           UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    turn_no          INTEGER NOT NULL,
    step_no          INTEGER NOT NULL,
    status           VARCHAR(32) NOT NULL,
    model_request_id UUID,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ,
    error            JSONB,
    CONSTRAINT uq_agent_step UNIQUE (run_id, turn_no, step_no)
);

CREATE TABLE agent_platform.agent_events (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id      VARCHAR(128) NOT NULL,
    run_id         UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    seq            BIGINT NOT NULL,
    event_type     VARCHAR(64) NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    turn_no        INTEGER,
    step_no        INTEGER,
    call_id        VARCHAR(256),
    causation_id   UUID,
    correlation_id UUID,
    payload        JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_uri   TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_event_seq UNIQUE (run_id, seq)
);
CREATE INDEX idx_agent_events_run ON agent_platform.agent_events (run_id, seq);
CREATE INDEX idx_agent_events_type ON agent_platform.agent_events (event_type, created_at DESC);

CREATE TABLE agent_platform.agent_model_calls (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id                UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    turn_no               INTEGER NOT NULL,
    step_no               INTEGER NOT NULL,
    provider              VARCHAR(64) NOT NULL,
    model_id              VARCHAR(256) NOT NULL,
    model_version         VARCHAR(256),
    endpoint_id           VARCHAR(256),
    context_manifest      JSONB NOT NULL,
    request_artifact_uri  TEXT,
    response_artifact_uri TEXT,
    input_tokens          BIGINT,
    output_tokens         BIGINT,
    latency_ms            BIGINT,
    status                VARCHAR(32) NOT NULL,
    error                 JSONB,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE agent_platform.agent_tool_executions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(128) NOT NULL,
    run_id              UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    step_id             UUID NOT NULL REFERENCES agent_platform.agent_steps(id) ON DELETE CASCADE,
    call_id             VARCHAR(256) NOT NULL,
    tool_name           VARCHAR(256) NOT NULL,
    tool_version        VARCHAR(128) NOT NULL,
    provider            VARCHAR(64) NOT NULL,
    request_hash        VARCHAR(128) NOT NULL,
    idempotency_key     VARCHAR(256) NOT NULL,
    status              VARCHAR(32) NOT NULL,
    attempt             INTEGER NOT NULL DEFAULT 0,
    request             JSONB,
    result              JSONB,
    result_artifact_uri TEXT,
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    error               JSONB,
    CONSTRAINT uq_agent_tool_idempotency UNIQUE (tenant_id, idempotency_key)
);
CREATE INDEX idx_agent_tool_run ON agent_platform.agent_tool_executions (run_id, call_id);

CREATE TABLE agent_platform.agent_checkpoints (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    event_seq       BIGINT NOT NULL,
    lease_token     BIGINT NOT NULL,
    state           JSONB NOT NULL,
    context_summary TEXT,
    artifact_uri    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_checkpoint UNIQUE (run_id, event_seq)
);

CREATE TABLE agent_platform.agent_outbox (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type VARCHAR(64) NOT NULL,
    aggregate_id   UUID NOT NULL,
    event_type     VARCHAR(64) NOT NULL,
    payload        JSONB NOT NULL,
    status         VARCHAR(32) NOT NULL DEFAULT 'pending',
    available_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at   TIMESTAMPTZ,
    attempt        INTEGER NOT NULL DEFAULT 0,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_agent_outbox_pending
    ON agent_platform.agent_outbox (status, available_at, created_at);
