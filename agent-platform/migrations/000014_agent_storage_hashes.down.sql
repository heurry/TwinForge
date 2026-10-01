ALTER TABLE agent_platform.agent_run_states
    DROP CONSTRAINT IF EXISTS chk_agent_run_state_hash;
ALTER TABLE agent_platform.agent_session_messages
    DROP CONSTRAINT IF EXISTS chk_agent_session_message_hash;
