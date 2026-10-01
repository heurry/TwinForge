DROP TABLE IF EXISTS agent_platform.agent_user_questions;
DROP TABLE IF EXISTS agent_platform.agent_task_plans;
ALTER TABLE agent_platform.agent_runs DROP CONSTRAINT chk_agent_run_status;
ALTER TABLE agent_platform.agent_runs ADD CONSTRAINT chk_agent_run_status CHECK (status IN (
    'queued', 'running', 'waiting_tool', 'waiting_approval',
    'waiting_external', 'suspended', 'completed', 'failed', 'cancelled'
));
