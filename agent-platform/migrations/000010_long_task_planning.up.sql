ALTER TABLE agent_platform.agent_runs DROP CONSTRAINT chk_agent_run_status;
ALTER TABLE agent_platform.agent_runs ADD CONSTRAINT chk_agent_run_status CHECK (status IN (
    'queued', 'running', 'waiting_tool', 'waiting_approval', 'waiting_input',
    'waiting_external', 'suspended', 'completed', 'failed', 'cancelled'
));

CREATE TABLE agent_platform.agent_task_plans (
    run_id       UUID PRIMARY KEY REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    tenant_id    VARCHAR(128) NOT NULL,
    revision     INTEGER NOT NULL DEFAULT 1,
    goal         TEXT NOT NULL,
    explanation  TEXT,
    steps        JSONB NOT NULL,
    last_call_id VARCHAR(256) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_agent_task_plans_tenant
    ON agent_platform.agent_task_plans(tenant_id, updated_at DESC);

CREATE TABLE agent_platform.agent_user_questions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   VARCHAR(128) NOT NULL,
    run_id      UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    call_id     VARCHAR(256) NOT NULL,
    turn_no     INTEGER NOT NULL,
    step_no     INTEGER NOT NULL,
    question    TEXT NOT NULL,
    options     JSONB NOT NULL DEFAULT '[]'::jsonb,
    context     TEXT,
    answer      TEXT,
    status      VARCHAR(32) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','answered','cancelled')),
    answered_by VARCHAR(128),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    answered_at TIMESTAMPTZ,
    CONSTRAINT uq_agent_user_question_call UNIQUE(run_id, call_id)
);
CREATE UNIQUE INDEX uq_agent_user_question_pending_run
    ON agent_platform.agent_user_questions(run_id) WHERE status='pending';
CREATE INDEX idx_agent_user_questions_tenant
    ON agent_platform.agent_user_questions(tenant_id, run_id, created_at DESC);
