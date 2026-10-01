-- A parent Run waiting for an external/child Agent is no longer executing;
-- its child Run may be the active attempt for the same Workflow. Keep the
-- single-active-attempt fence for all states that can mutate the Workflow
-- directly, but allow waiting_external + one child attempt to coexist.
DROP INDEX IF EXISTS agent_platform.uq_agent_runs_workflow_active;
CREATE UNIQUE INDEX uq_agent_runs_workflow_active
    ON agent_platform.agent_runs(workflow_id)
    WHERE status IN ('queued','running','waiting_tool','waiting_approval',
                     'waiting_input','suspended');
