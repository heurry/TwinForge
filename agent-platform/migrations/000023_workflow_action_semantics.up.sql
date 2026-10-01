-- Canonical Workflow/PlanNode/DecisionCycle/ActionAttempt coordinates.
-- Legacy Run/Turn/Step/Call columns remain for wire compatibility; these
-- columns make joins and observability queries explicit and avoid payload
-- parsing as the primary correlation mechanism.
ALTER TABLE agent_platform.agent_tool_executions
    ADD COLUMN IF NOT EXISTS workflow_id UUID,
    ADD COLUMN IF NOT EXISTS plan_node_id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS decision_cycle INTEGER,
    ADD COLUMN IF NOT EXISTS action_id VARCHAR(256),
    ADD COLUMN IF NOT EXISTS action_kind VARCHAR(32) NOT NULL DEFAULT 'tool';

UPDATE agent_platform.agent_tool_executions
SET workflow_id = run_id,
    plan_node_id = COALESCE(plan_node_id, plan_step_key),
    decision_cycle = COALESCE(decision_cycle, 0),
    action_id = COALESCE(action_id, call_id)
WHERE workflow_id IS NULL OR action_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_agent_tool_action_trace
    ON agent_platform.agent_tool_executions(tenant_id, workflow_id, decision_cycle, started_at);

ALTER TABLE agent_platform.agent_observations
    ADD COLUMN IF NOT EXISTS workflow_id UUID,
    ADD COLUMN IF NOT EXISTS plan_node_id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS decision_cycle INTEGER,
    ADD COLUMN IF NOT EXISTS action_id TEXT,
    ADD COLUMN IF NOT EXISTS parent_action_id TEXT;

UPDATE agent_platform.agent_observations
SET workflow_id = run_id,
    plan_node_id = COALESCE(plan_node_id, metadata->>'plan_node_id'),
    decision_cycle = COALESCE(decision_cycle, turn_no, 0),
    action_id = COALESCE(action_id, call_id)
WHERE workflow_id IS NULL OR action_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_agent_observations_workflow_action
    ON agent_platform.agent_observations(tenant_id, workflow_id, decision_cycle, sequence);
