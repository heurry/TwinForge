-- A downgrade cannot represent independent in-flight child state in the old
-- one-row-per-Workflow schema. Child audit events and terminal checkpoints
-- remain durable; remove only the derived current-state rows before restoring
-- the former unique index.
DELETE FROM agent_platform.agent_run_states
WHERE scope_run_id IS NOT NULL;

DROP INDEX IF EXISTS agent_platform.idx_agent_run_states_scope;
DROP INDEX IF EXISTS agent_platform.uq_agent_run_states_delegated_run;
DROP INDEX IF EXISTS agent_platform.uq_agent_run_states_root_workflow;
ALTER TABLE agent_platform.agent_run_states
    DROP CONSTRAINT IF EXISTS fk_agent_run_states_scope_run;
ALTER TABLE agent_platform.agent_run_states
    DROP COLUMN IF EXISTS scope_run_id;

CREATE UNIQUE INDEX uq_agent_run_states_workflow
    ON agent_platform.agent_run_states(workflow_id);
