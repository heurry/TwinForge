-- Root continuation Runs intentionally share one current state per Workflow.
-- Delegated Runs share the workspace/Workflow identity, but their model loop,
-- pending tools and usage are independent and must never overwrite the root
-- continuation checkpoint.
ALTER TABLE agent_platform.agent_run_states
    ADD COLUMN IF NOT EXISTS scope_run_id UUID;

UPDATE agent_platform.agent_run_states AS state
SET scope_run_id = state.run_id
FROM agent_platform.agent_runs AS run
WHERE run.id = state.run_id
  AND run.parent_run_id IS NOT NULL
  AND state.scope_run_id IS NULL;

ALTER TABLE agent_platform.agent_run_states
    DROP CONSTRAINT IF EXISTS fk_agent_run_states_scope_run;
ALTER TABLE agent_platform.agent_run_states
    ADD CONSTRAINT fk_agent_run_states_scope_run
    FOREIGN KEY (scope_run_id) REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE;

DROP INDEX IF EXISTS agent_platform.uq_agent_run_states_workflow;
CREATE UNIQUE INDEX uq_agent_run_states_root_workflow
    ON agent_platform.agent_run_states(workflow_id)
    WHERE scope_run_id IS NULL;
CREATE UNIQUE INDEX uq_agent_run_states_delegated_run
    ON agent_platform.agent_run_states(scope_run_id)
    WHERE scope_run_id IS NOT NULL;

CREATE INDEX idx_agent_run_states_scope
    ON agent_platform.agent_run_states(workflow_id, scope_run_id, state_seq DESC);
