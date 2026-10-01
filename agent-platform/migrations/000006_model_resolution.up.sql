ALTER TABLE agent_platform.agent_model_calls
    ADD COLUMN service_ref VARCHAR(128),
    ADD COLUMN artifact_digest VARCHAR(256),
    ADD COLUMN selection_strategy VARCHAR(32);

CREATE INDEX idx_agent_model_calls_resolved_model
    ON agent_platform.agent_model_calls (model_id, model_version, service_ref);
