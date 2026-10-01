ALTER TABLE agent_platform.agent_outbox
    ADD COLUMN last_error TEXT;

CREATE INDEX idx_agent_outbox_published_retention
    ON agent_platform.agent_outbox (published_at)
    WHERE status='published';
