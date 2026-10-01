-- Agent storage P0: make conversation messages and the latest resumable Run
-- state explicit PostgreSQL facts instead of deriving both from growing
-- history tables.

CREATE TABLE agent_platform.agent_session_messages (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id        VARCHAR(128) NOT NULL,
    session_id       UUID NOT NULL REFERENCES agent_platform.agent_sessions(id) ON DELETE CASCADE,
    sequence         BIGINT NOT NULL,
    run_id           UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    role             VARCHAR(32) NOT NULL,
    message_kind     VARCHAR(32) NOT NULL,
    content          JSONB NOT NULL,
    content_hash     VARCHAR(128) NOT NULL,
    token_count      INTEGER,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_session_message_sequence UNIQUE(session_id, sequence),
    CONSTRAINT chk_agent_session_message_role CHECK(role IN ('user','assistant')),
    CONSTRAINT chk_agent_session_message_kind CHECK(message_kind IN ('run_input','run_output','user_input'))
);
CREATE UNIQUE INDEX uq_agent_session_run_input
    ON agent_platform.agent_session_messages(run_id) WHERE message_kind='run_input';
CREATE UNIQUE INDEX uq_agent_session_run_output
    ON agent_platform.agent_session_messages(run_id) WHERE message_kind='run_output';
CREATE INDEX idx_agent_session_messages_tenant_session
    ON agent_platform.agent_session_messages(tenant_id, session_id, sequence);

WITH historical AS (
    SELECT run.tenant_id, run.session_id, run.id AS run_id, 'user'::text AS role,
           'run_input'::text AS message_kind, run.input AS content,
           run.created_at AS message_at, run.created_at AS sort_at, 0 AS position
    FROM agent_platform.agent_runs run
    WHERE run.session_id IS NOT NULL
    UNION ALL
    SELECT run.tenant_id, run.session_id, run.id AS run_id, 'assistant'::text AS role,
           'run_output'::text AS message_kind, run.output AS content,
           COALESCE(run.finished_at, run.updated_at) AS message_at,
           run.created_at AS sort_at, 1 AS position
    FROM agent_platform.agent_runs run
    WHERE run.session_id IS NOT NULL AND run.status='completed' AND run.output IS NOT NULL
), numbered AS (
    SELECT historical.*,
           row_number() OVER (PARTITION BY session_id ORDER BY sort_at, run_id, position) AS sequence
    FROM historical
)
INSERT INTO agent_platform.agent_session_messages
    (tenant_id, session_id, sequence, run_id, role, message_kind, content, content_hash, created_at)
SELECT tenant_id, session_id, sequence, run_id, role, message_kind, content,
       encode(sha256(convert_to(content::text, 'UTF8')), 'hex'), message_at
FROM numbered;

CREATE TABLE agent_platform.agent_run_states (
    run_id           UUID PRIMARY KEY REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    event_seq        BIGINT NOT NULL,
    lease_token      BIGINT NOT NULL,
    state            JSONB NOT NULL,
    context_summary  TEXT,
    state_hash       VARCHAR(128) NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO agent_platform.agent_run_states
    (run_id, event_seq, lease_token, state, context_summary, state_hash, updated_at)
SELECT DISTINCT ON (checkpoint.run_id)
       checkpoint.run_id, checkpoint.event_seq, checkpoint.lease_token,
       checkpoint.state, checkpoint.context_summary,
       encode(sha256(convert_to(checkpoint.state::text, 'UTF8')), 'hex'), checkpoint.created_at
FROM agent_platform.agent_checkpoints checkpoint
ORDER BY checkpoint.run_id, checkpoint.event_seq DESC;
