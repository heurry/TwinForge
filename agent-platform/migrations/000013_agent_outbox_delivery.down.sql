DROP INDEX IF EXISTS agent_platform.idx_agent_outbox_published_retention;
ALTER TABLE agent_platform.agent_outbox DROP COLUMN IF EXISTS last_error;
