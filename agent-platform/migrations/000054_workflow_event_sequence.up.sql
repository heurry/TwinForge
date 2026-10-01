-- Give every event both its legacy Run-local sequence and a Workflow-wide
-- monotonic sequence. The latter is the recovery cursor across continuations,
-- delegated attempts and worker hand-offs.

ALTER TABLE agent_platform.agent_events
    ADD COLUMN IF NOT EXISTS workflow_id UUID,
    ADD COLUMN IF NOT EXISTS workflow_seq BIGINT;

UPDATE agent_platform.agent_events AS event
SET workflow_id = run.workflow_id
FROM agent_platform.agent_runs AS run
WHERE run.id = event.run_id
  AND event.workflow_id IS NULL;

WITH ranked AS (
    SELECT id,
           row_number() OVER (
               PARTITION BY workflow_id
               ORDER BY created_at,id
           ) AS workflow_seq
    FROM agent_platform.agent_events
)
UPDATE agent_platform.agent_events AS event
SET workflow_seq = ranked.workflow_seq
FROM ranked
WHERE event.id = ranked.id
  AND event.workflow_seq IS NULL;

UPDATE agent_platform.agent_workflows AS workflow
SET latest_workflow_seq = COALESCE((
    SELECT max(event.workflow_seq)
    FROM agent_platform.agent_events AS event
    WHERE event.workflow_id = workflow.id
), 0);

ALTER TABLE agent_platform.agent_events
    ALTER COLUMN workflow_id SET NOT NULL,
    ALTER COLUMN workflow_seq SET NOT NULL,
    ADD CONSTRAINT fk_agent_events_workflow
        FOREIGN KEY (workflow_id) REFERENCES agent_platform.agent_workflows(id) ON DELETE CASCADE,
    ADD CONSTRAINT chk_agent_event_workflow_seq_positive CHECK (workflow_seq > 0),
    ADD CONSTRAINT uq_agent_event_workflow_seq UNIQUE (workflow_id,workflow_seq);

CREATE INDEX idx_agent_events_workflow
    ON agent_platform.agent_events(workflow_id,workflow_seq);
