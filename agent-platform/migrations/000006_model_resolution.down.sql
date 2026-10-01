DROP INDEX IF EXISTS agent_platform.idx_agent_model_calls_resolved_model;

ALTER TABLE agent_platform.agent_model_calls
    DROP COLUMN IF EXISTS selection_strategy,
    DROP COLUMN IF EXISTS artifact_digest,
    DROP COLUMN IF EXISTS service_ref;
