ALTER TABLE agent_platform.agent_outbox
    DROP CONSTRAINT IF EXISTS ck_agent_outbox_publishing_token;

ALTER TABLE agent_platform.agent_outbox
    DROP COLUMN IF EXISTS delivery_token;
