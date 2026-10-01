-- A Workflow has one mutable top-level Run attempt. Delegated child Runs are
-- activities owned by that attempt and may execute concurrently (async) or
-- while the parent waits (sync), so they must not collide with the root-run
-- single-active fence.
UPDATE agent_platform.agent_tool_executions AS execution
SET workflow_id = run.workflow_id
FROM agent_platform.agent_runs AS run
WHERE execution.run_id = run.id
  AND execution.workflow_id IS DISTINCT FROM run.workflow_id;

UPDATE agent_platform.agent_observations AS observation
SET workflow_id = run.workflow_id
FROM agent_platform.agent_runs AS run
WHERE observation.run_id = run.id
  AND observation.workflow_id IS DISTINCT FROM run.workflow_id;

DROP INDEX IF EXISTS agent_platform.uq_agent_runs_workflow_active;
CREATE UNIQUE INDEX uq_agent_runs_workflow_active
    ON agent_platform.agent_runs(workflow_id)
    WHERE parent_run_id IS NULL
      AND status IN ('queued','running','waiting_tool','waiting_approval',
                     'waiting_input','waiting_external','suspended');
