-- event_seq is scoped to a Run. A Workflow can span several Run Attempts,
-- therefore checkpoint freshness needs its own monotonic sequence.
ALTER TABLE agent_platform.agent_run_states
    ADD COLUMN IF NOT EXISTS state_seq BIGINT NOT NULL DEFAULT 0;

ALTER TABLE agent_platform.agent_checkpoints
    ADD COLUMN IF NOT EXISTS state_seq BIGINT NOT NULL DEFAULT 0;

-- Preserve the strongest legacy ordering during the one-time transition. New
-- writers use state_seq; old rows remain readable through the event_seq
-- fallback in the loader/reconciler.
WITH ranked AS (
    SELECT id,
           row_number() OVER (
               PARTITION BY workflow_id
               ORDER BY created_at, event_seq, id
           )::bigint AS next_state_seq
    FROM agent_platform.agent_checkpoints
)
UPDATE agent_platform.agent_checkpoints checkpoint
SET state_seq=ranked.next_state_seq
FROM ranked
WHERE checkpoint.id=ranked.id AND checkpoint.state_seq=0;

UPDATE agent_platform.agent_run_states state
SET state_seq=COALESCE((
    SELECT MAX(checkpoint.state_seq)
    FROM agent_platform.agent_checkpoints checkpoint
    WHERE checkpoint.workflow_id=state.workflow_id
), state.event_seq)
WHERE state.state_seq=0;

CREATE INDEX IF NOT EXISTS idx_agent_checkpoints_workflow_state_seq
    ON agent_platform.agent_checkpoints(workflow_id, state_seq DESC, event_seq DESC);

CREATE INDEX IF NOT EXISTS idx_agent_run_states_workflow_state_seq
    ON agent_platform.agent_run_states(workflow_id, state_seq DESC);
