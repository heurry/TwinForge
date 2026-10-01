DROP INDEX IF EXISTS agent_platform.idx_agent_events_workflow;

ALTER TABLE agent_platform.agent_events
    DROP CONSTRAINT IF EXISTS uq_agent_event_workflow_seq,
    DROP CONSTRAINT IF EXISTS chk_agent_event_workflow_seq_positive,
    DROP CONSTRAINT IF EXISTS fk_agent_events_workflow,
    DROP COLUMN IF EXISTS workflow_seq,
    DROP COLUMN IF EXISTS workflow_id;
