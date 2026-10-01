DROP INDEX IF EXISTS agent_platform.idx_agent_run_states_workflow_state_seq;
DROP INDEX IF EXISTS agent_platform.idx_agent_checkpoints_workflow_state_seq;
ALTER TABLE agent_platform.agent_checkpoints DROP COLUMN IF EXISTS state_seq;
ALTER TABLE agent_platform.agent_run_states DROP COLUMN IF EXISTS state_seq;
