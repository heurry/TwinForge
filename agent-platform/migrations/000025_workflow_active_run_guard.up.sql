-- A Workflow may have many historical Run attempts, but only one live attempt.
-- Terminal attempts remain resumable history without blocking a new Run.
CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_runs_workflow_active
    ON agent_platform.agent_runs(workflow_id)
    WHERE status IN ('queued','running','waiting_tool','waiting_approval',
                     'waiting_input','waiting_external','suspended');
