CREATE TABLE agent_platform.agent_verification_intents (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id          VARCHAR(128) NOT NULL,
    run_id             UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    plan_revision      INTEGER NOT NULL,
    plan_step_key      VARCHAR(128) NOT NULL,
    criterion_key      VARCHAR(128) NOT NULL,
    description        TEXT NOT NULL,
    kind               VARCHAR(128),
    enforcement        VARCHAR(32) NOT NULL,
    origin             VARCHAR(32) NOT NULL,
    parameters         JSONB NOT NULL DEFAULT '{}'::jsonb,
    status             VARCHAR(32) NOT NULL,
    diagnostic_code    VARCHAR(128),
    diagnostic_message TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_verification_intent
        UNIQUE(run_id,plan_revision,plan_step_key,criterion_key),
    CONSTRAINT chk_agent_verification_intent_enforcement CHECK (
        enforcement IN ('informational','advisory','required','release_gate')
    ),
    CONSTRAINT chk_agent_verification_intent_status CHECK (
        status IN ('pending','passed','failed','skipped','invalid','unsupported','stale')
    )
);

CREATE TABLE agent_platform.agent_verification_specs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       VARCHAR(128) NOT NULL,
    run_id          UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    intent_id       UUID NOT NULL UNIQUE REFERENCES agent_platform.agent_verification_intents(id) ON DELETE CASCADE,
    provider_key    VARCHAR(128) NOT NULL,
    provider_version VARCHAR(64) NOT NULL,
    subject         JSONB NOT NULL DEFAULT '{}'::jsonb,
    execution       JSONB NOT NULL DEFAULT '{}'::jsonb,
    assertions      JSONB NOT NULL DEFAULT '[]'::jsonb,
    spec_digest     VARCHAR(64) NOT NULL,
    status          VARCHAR(32) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_agent_verification_spec_status CHECK (
        status IN ('compiled','invalid','unsupported','retired')
    )
);

CREATE TABLE agent_platform.agent_verification_attempts (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id         VARCHAR(128) NOT NULL,
    run_id            UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    spec_id           UUID NOT NULL REFERENCES agent_platform.agent_verification_specs(id) ON DELETE CASCADE,
    tool_execution_id UUID REFERENCES agent_platform.agent_tool_executions(id) ON DELETE SET NULL,
    status            VARCHAR(32) NOT NULL,
    reason_code       VARCHAR(128),
    diagnostic        TEXT,
    result            JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    CONSTRAINT chk_agent_verification_attempt_status CHECK (
        status IN ('running','passed','failed','cancelled','stale')
    )
);
CREATE UNIQUE INDEX uq_agent_verification_attempt_tool
    ON agent_platform.agent_verification_attempts(spec_id,tool_execution_id)
    WHERE tool_execution_id IS NOT NULL;

CREATE TABLE agent_platform.agent_evidence_records (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id          VARCHAR(128) NOT NULL,
    run_id             UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    intent_id          UUID NOT NULL REFERENCES agent_platform.agent_verification_intents(id) ON DELETE CASCADE,
    spec_id            UUID NOT NULL REFERENCES agent_platform.agent_verification_specs(id) ON DELETE CASCADE,
    attempt_id         UUID NOT NULL REFERENCES agent_platform.agent_verification_attempts(id) ON DELETE CASCADE,
    tool_execution_id  UUID REFERENCES agent_platform.agent_tool_executions(id) ON DELETE SET NULL,
    verdict            VARCHAR(32) NOT NULL,
    evidence           JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_digest      VARCHAR(64),
    workspace_revision VARCHAR(128),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_agent_evidence_verdict CHECK (
        verdict IN ('passed','failed','invalid','unsupported','stale')
    )
);
CREATE UNIQUE INDEX uq_agent_evidence_attempt
    ON agent_platform.agent_evidence_records(attempt_id);

CREATE INDEX idx_agent_verification_intents_run
    ON agent_platform.agent_verification_intents(tenant_id,run_id,plan_revision,plan_step_key,criterion_key);
CREATE INDEX idx_agent_verification_attempts_run
    ON agent_platform.agent_verification_attempts(tenant_id,run_id,started_at DESC);
CREATE INDEX idx_agent_evidence_records_run
    ON agent_platform.agent_evidence_records(tenant_id,run_id,created_at DESC);
