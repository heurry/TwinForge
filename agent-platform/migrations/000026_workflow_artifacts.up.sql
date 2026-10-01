-- Artifacts are outputs of a Workflow, not only of the latest Run attempt.
-- Keep run_id for provenance, while workflow_id makes history, resume and
-- promotion queries independent of which attempt produced the bytes.
ALTER TABLE agent_platform.agent_artifacts
    ADD COLUMN IF NOT EXISTS workflow_id UUID;

UPDATE agent_platform.agent_artifacts AS artifact
SET workflow_id = run.workflow_id
FROM agent_platform.agent_runs AS run
WHERE artifact.run_id = run.id
  AND artifact.workflow_id IS NULL;

ALTER TABLE agent_platform.agent_artifacts
    ALTER COLUMN workflow_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname='fk_agent_artifacts_workflow'
    ) THEN
        ALTER TABLE agent_platform.agent_artifacts
            ADD CONSTRAINT fk_agent_artifacts_workflow
            FOREIGN KEY (workflow_id) REFERENCES agent_platform.agent_workflows(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_agent_artifacts_workflow
    ON agent_platform.agent_artifacts(tenant_id, workflow_id, created_at, id);
