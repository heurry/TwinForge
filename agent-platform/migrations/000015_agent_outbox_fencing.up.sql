ALTER TABLE agent_platform.agent_outbox
    ADD COLUMN delivery_token UUID;

-- Preserve a valid claim for rows that were already being delivered while a
-- rolling deployment applied this migration.
UPDATE agent_platform.agent_outbox
SET delivery_token = gen_random_uuid()
WHERE status = 'publishing' AND delivery_token IS NULL;

ALTER TABLE agent_platform.agent_outbox
    ADD CONSTRAINT ck_agent_outbox_publishing_token
    CHECK ((status = 'publishing') = (delivery_token IS NOT NULL));
