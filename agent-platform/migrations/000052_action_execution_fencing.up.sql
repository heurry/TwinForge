-- Bind every durable action to the Workflow/Plan/Node/Workspace generation it
-- observed. A late receipt is still auditable, but cannot advance newer state.

ALTER TABLE agent_platform.agent_plan_node_states
    ADD COLUMN IF NOT EXISTS node_revision BIGINT NOT NULL DEFAULT 1;

UPDATE agent_platform.agent_plan_node_states
SET node_revision = GREATEST(node_revision, plan_revision, 1);

ALTER TABLE agent_platform.agent_tool_executions
    ADD COLUMN IF NOT EXISTS workflow_generation BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS plan_revision INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS node_revision BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS expected_workspace_revision BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS finished_workspace_revision BIGINT,
    ADD COLUMN IF NOT EXISTS application_status VARCHAR(32) NOT NULL DEFAULT 'applied';

UPDATE agent_platform.agent_tool_executions AS execution
SET workflow_generation = COALESCE((
        SELECT workflow.execution_generation
        FROM agent_platform.agent_workflows AS workflow
        WHERE workflow.id = execution.workflow_id
    ), 0),
    plan_revision = COALESCE((
        SELECT plan.revision
        FROM agent_platform.agent_task_plans AS plan
        WHERE plan.workflow_id = execution.workflow_id
    ), 0),
    node_revision = COALESCE((
        SELECT node.node_revision
        FROM agent_platform.agent_plan_node_states AS node
        WHERE node.workflow_id = execution.workflow_id
          AND node.node_id = execution.plan_node_id
    ), 0),
    expected_workspace_revision = COALESCE((
        SELECT workflow.workspace_revision
        FROM agent_platform.agent_workflows AS workflow
        WHERE workflow.id = execution.workflow_id
    ), 0)
WHERE execution.workflow_generation = 0;

ALTER TABLE agent_platform.agent_tool_executions
    ADD CONSTRAINT chk_agent_tool_execution_generations CHECK (
        workflow_generation >= 0 AND plan_revision >= 0 AND node_revision >= 0
        AND expected_workspace_revision >= 0
        AND (finished_workspace_revision IS NULL OR finished_workspace_revision >= 0)
    ),
    ADD CONSTRAINT chk_agent_tool_execution_application_status CHECK (
        application_status IN ('applied','stale_ignored')
    );

CREATE INDEX IF NOT EXISTS idx_agent_tool_execution_fence
    ON agent_platform.agent_tool_executions(
        tenant_id,workflow_id,workflow_generation,plan_revision,plan_node_id,node_revision
    );
