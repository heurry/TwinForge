CREATE TABLE agent_platform.agent_workspace_mutations (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id             VARCHAR(128) NOT NULL,
    workflow_id           UUID NOT NULL REFERENCES agent_platform.agent_workflows(id) ON DELETE CASCADE,
    run_id                UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    tool_execution_id     UUID NOT NULL REFERENCES agent_platform.agent_tool_executions(id) ON DELETE CASCADE,
    action_id             VARCHAR(256) NOT NULL,
    workspace_revision    BIGINT NOT NULL,
    operation             VARCHAR(64) NOT NULL,
    path                  TEXT,
    previous_file_sha256  VARCHAR(128),
    file_sha256           VARCHAR(128),
    metadata              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_workspace_mutation_revision UNIQUE(workflow_id,workspace_revision),
    CONSTRAINT uq_agent_workspace_mutation_execution UNIQUE(tool_execution_id),
    CONSTRAINT chk_agent_workspace_mutation_revision CHECK (workspace_revision > 0)
);

CREATE INDEX idx_agent_workspace_mutations_path
    ON agent_platform.agent_workspace_mutations(workflow_id,path,workspace_revision DESC);
