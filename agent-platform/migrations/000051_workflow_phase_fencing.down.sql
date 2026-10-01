DROP INDEX IF EXISTS agent_platform.idx_agent_workflows_phase;

ALTER TABLE agent_platform.agent_workflows
    DROP CONSTRAINT IF EXISTS chk_agent_workflow_revision_nonnegative,
    DROP CONSTRAINT IF EXISTS chk_agent_workflow_generation_positive,
    DROP CONSTRAINT IF EXISTS chk_agent_workflow_phase,
    DROP COLUMN IF EXISTS latest_workflow_seq,
    DROP COLUMN IF EXISTS workspace_revision,
    DROP COLUMN IF EXISTS execution_generation,
    DROP COLUMN IF EXISTS phase;
