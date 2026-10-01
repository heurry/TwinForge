DROP INDEX IF EXISTS agent_platform.idx_agent_artifacts_workflow;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname='fk_agent_artifacts_workflow'
    ) THEN
        ALTER TABLE agent_platform.agent_artifacts
            DROP CONSTRAINT fk_agent_artifacts_workflow;
    END IF;
END $$;

ALTER TABLE agent_platform.agent_artifacts
    DROP COLUMN IF EXISTS workflow_id;
