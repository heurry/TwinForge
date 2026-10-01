DROP INDEX IF EXISTS agent_platform.uq_agent_runs_workflow_active;
CREATE UNIQUE INDEX uq_agent_runs_workflow_active
    ON agent_platform.agent_runs(workflow_id)
    WHERE status IN ('queued','running','waiting_tool','waiting_approval',
                     'waiting_input','waiting_external','suspended');
