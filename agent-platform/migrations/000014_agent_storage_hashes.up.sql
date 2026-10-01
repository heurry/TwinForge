UPDATE agent_platform.agent_session_messages
SET content_hash=encode(sha256(convert_to(content::text, 'UTF8')), 'hex');

UPDATE agent_platform.agent_run_states
SET state_hash=encode(sha256(convert_to(state::text, 'UTF8')), 'hex');

ALTER TABLE agent_platform.agent_session_messages
    ADD CONSTRAINT chk_agent_session_message_hash CHECK(length(content_hash)=64);
ALTER TABLE agent_platform.agent_run_states
    ADD CONSTRAINT chk_agent_run_state_hash CHECK(length(state_hash)=64);
