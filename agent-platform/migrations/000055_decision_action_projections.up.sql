-- Normalize model decision cycles and executable actions. Events remain the
-- immutable source; these tables are rebuildable scheduler/audit projections.

CREATE TABLE agent_platform.agent_decision_cycles (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id             VARCHAR(128) NOT NULL,
    workflow_id           UUID NOT NULL REFERENCES agent_platform.agent_workflows(id) ON DELETE CASCADE,
    run_id                UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    cycle_no              INTEGER NOT NULL,
    plan_node_id          VARCHAR(256),
    workflow_generation   BIGINT NOT NULL,
    plan_revision         INTEGER NOT NULL DEFAULT 0,
    node_revision         BIGINT NOT NULL DEFAULT 0,
    workspace_revision    BIGINT NOT NULL DEFAULT 0,
    status                VARCHAR(32) NOT NULL,
    model_call_count      INTEGER NOT NULL DEFAULT 0,
    action_count          INTEGER NOT NULL DEFAULT 0,
    first_event_sequence  BIGINT NOT NULL,
    last_event_sequence   BIGINT NOT NULL,
    started_at            TIMESTAMPTZ NOT NULL,
    finished_at           TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_decision_cycle UNIQUE(run_id,cycle_no),
    CONSTRAINT chk_agent_decision_cycle_number CHECK (cycle_no > 0),
    CONSTRAINT chk_agent_decision_cycle_status CHECK (
        status IN ('running','completed','failed','cancelled')
    )
);

CREATE INDEX idx_agent_decision_cycles_workflow
    ON agent_platform.agent_decision_cycles(workflow_id,workflow_generation,cycle_no);

CREATE TABLE agent_platform.agent_action_attempts (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id             VARCHAR(128) NOT NULL,
    workflow_id           UUID NOT NULL REFERENCES agent_platform.agent_workflows(id) ON DELETE CASCADE,
    run_id                UUID NOT NULL REFERENCES agent_platform.agent_runs(id) ON DELETE CASCADE,
    action_id             VARCHAR(256) NOT NULL,
    decision_cycle        INTEGER,
    plan_node_id          VARCHAR(256),
    action_kind           VARCHAR(32) NOT NULL,
    name                  VARCHAR(256) NOT NULL,
    attempt               INTEGER NOT NULL DEFAULT 1,
    status                VARCHAR(32) NOT NULL,
    workflow_generation   BIGINT NOT NULL,
    plan_revision         INTEGER NOT NULL DEFAULT 0,
    node_revision         BIGINT NOT NULL DEFAULT 0,
    workspace_revision    BIGINT NOT NULL DEFAULT 0,
    request               JSONB NOT NULL DEFAULT '{}'::jsonb,
    result                JSONB,
    error                 JSONB,
    first_event_sequence  BIGINT NOT NULL,
    last_event_sequence   BIGINT NOT NULL,
    started_at            TIMESTAMPTZ NOT NULL,
    finished_at           TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_action_attempt UNIQUE(run_id,action_id),
    CONSTRAINT chk_agent_action_attempt_positive CHECK (attempt > 0),
    CONSTRAINT chk_agent_action_attempt_status CHECK (
        status IN ('started','waiting_approval','waiting_user','waiting_agent','completed','failed','cancelled')
    ),
    CONSTRAINT chk_agent_action_attempt_kind CHECK (
        action_kind IN ('tool','mcp','agent','control')
    )
);

CREATE INDEX idx_agent_action_attempts_workflow
    ON agent_platform.agent_action_attempts(workflow_id,workflow_generation,status,updated_at);
CREATE INDEX idx_agent_action_attempts_node
    ON agent_platform.agent_action_attempts(workflow_id,plan_node_id,decision_cycle);

-- Backfill the projection for pre-migration ledgers. Fence values describe the
-- migration snapshot; immutable event payloads retain the original details.
WITH cycle_events AS (
    SELECT event.*,
           CASE
               WHEN COALESCE(event.payload #>> '{_workflow,decision_cycle}','') ~ '^[1-9][0-9]*$'
                   THEN (event.payload #>> '{_workflow,decision_cycle}')::integer
               ELSE event.turn_no
           END AS cycle_no,
           NULLIF(COALESCE(event.payload #>> '{_workflow,plan_node_id}',event.payload->>'plan_node_id',''),'') AS plan_node_id
    FROM agent_platform.agent_events AS event
), cycle_rollup AS (
    SELECT tenant_id,workflow_id,run_id,cycle_no,
           (array_agg(plan_node_id ORDER BY workflow_seq DESC) FILTER (WHERE plan_node_id IS NOT NULL))[1] AS plan_node_id,
           CASE
               WHEN bool_or(event_type='RUN_CANCELLED') THEN 'cancelled'
               WHEN bool_or(event_type='RUN_FAILED') THEN 'failed'
               WHEN bool_or(event_type='TURN_COMPLETED') THEN 'completed'
               ELSE 'running'
           END AS status,
           count(*) FILTER (WHERE event_type='MODEL_REQUESTED')::integer AS model_call_count,
           count(*) FILTER (WHERE event_type IN ('TOOL_CALLED','DELEGATION_REQUESTED','USER_INPUT_REQUESTED'))::integer AS action_count,
           min(workflow_seq) AS first_event_sequence,
           max(workflow_seq) AS last_event_sequence,
           min(created_at) AS started_at,
           max(created_at) FILTER (WHERE event_type IN ('TURN_COMPLETED','RUN_FAILED','RUN_CANCELLED')) AS finished_at
    FROM cycle_events
    WHERE cycle_no > 0
    GROUP BY tenant_id,workflow_id,run_id,cycle_no
)
INSERT INTO agent_platform.agent_decision_cycles(
    tenant_id,workflow_id,run_id,cycle_no,plan_node_id,
    workflow_generation,plan_revision,node_revision,workspace_revision,
    status,model_call_count,action_count,first_event_sequence,last_event_sequence,
    started_at,finished_at)
SELECT rollup.tenant_id,rollup.workflow_id,rollup.run_id,rollup.cycle_no,rollup.plan_node_id,
       workflow.execution_generation,COALESCE(plan.revision,0),COALESCE(node.node_revision,0),workflow.workspace_revision,
       rollup.status,rollup.model_call_count,rollup.action_count,
       rollup.first_event_sequence,rollup.last_event_sequence,rollup.started_at,rollup.finished_at
FROM cycle_rollup AS rollup
JOIN agent_platform.agent_workflows AS workflow ON workflow.id=rollup.workflow_id
LEFT JOIN agent_platform.agent_task_plans AS plan ON plan.workflow_id=workflow.id
LEFT JOIN agent_platform.agent_plan_node_states AS node
  ON node.workflow_id=workflow.id AND node.node_id=rollup.plan_node_id
ON CONFLICT(run_id,cycle_no) DO NOTHING;

WITH action_events AS (
    SELECT event.*,
           COALESCE(NULLIF(event.payload #>> '{_workflow,action_id}',''),NULLIF(event.call_id,''),NULLIF(event.payload->>'delegation_id','')) AS action_id,
           CASE
               WHEN COALESCE(event.payload #>> '{_workflow,decision_cycle}','') ~ '^[1-9][0-9]*$'
                   THEN (event.payload #>> '{_workflow,decision_cycle}')::integer
               ELSE event.turn_no
           END AS decision_cycle,
           NULLIF(COALESCE(event.payload #>> '{_workflow,plan_node_id}',event.payload->>'plan_node_id',''),'') AS plan_node_id
    FROM agent_platform.agent_events AS event
    WHERE event.event_type IN (
        'TOOL_CALLED','TOOL_APPROVAL_REQUESTED','TOOL_APPROVAL_RESOLVED','TOOL_COMPLETED','TOOL_FAILED',
        'DELEGATION_REQUESTED','DELEGATION_COMPLETED','DELEGATION_FAILED',
        'USER_INPUT_REQUESTED','USER_INPUT_RECEIVED'
    )
), action_rollup AS (
    SELECT tenant_id,workflow_id,run_id,action_id,
           max(decision_cycle) AS decision_cycle,
           (array_agg(plan_node_id ORDER BY workflow_seq DESC) FILTER (WHERE plan_node_id IS NOT NULL))[1] AS plan_node_id,
           CASE
               WHEN bool_or(event_type LIKE 'DELEGATION_%') THEN 'agent'
               WHEN bool_or(event_type LIKE 'USER_INPUT_%') THEN 'control'
               WHEN bool_or(COALESCE(payload->>'action_kind','')='mcp') THEN 'mcp'
               ELSE 'tool'
           END AS action_kind,
           COALESCE((array_agg(NULLIF(COALESCE(payload->>'name',payload->>'tool_name',payload->>'target_agent_version_id',''),'') ORDER BY workflow_seq DESC)
               FILTER (WHERE NULLIF(COALESCE(payload->>'name',payload->>'tool_name',payload->>'target_agent_version_id',''),'') IS NOT NULL))[1],lower(max(event_type))) AS name,
           CASE
               WHEN bool_or(event_type IN ('TOOL_FAILED','DELEGATION_FAILED')) THEN 'failed'
               WHEN bool_or(event_type IN ('TOOL_COMPLETED','DELEGATION_COMPLETED','USER_INPUT_RECEIVED')) THEN 'completed'
               WHEN bool_or(event_type='USER_INPUT_REQUESTED') THEN 'waiting_user'
               WHEN bool_or(event_type='DELEGATION_REQUESTED') THEN 'waiting_agent'
               WHEN bool_or(event_type='TOOL_APPROVAL_REQUESTED') THEN 'waiting_approval'
               ELSE 'started'
           END AS status,
           min(workflow_seq) AS first_event_sequence,
           max(workflow_seq) AS last_event_sequence,
           min(created_at) AS started_at,
           max(created_at) FILTER (WHERE event_type IN ('TOOL_COMPLETED','TOOL_FAILED','DELEGATION_COMPLETED','DELEGATION_FAILED','USER_INPUT_RECEIVED')) AS finished_at
    FROM action_events
    WHERE action_id IS NOT NULL
    GROUP BY tenant_id,workflow_id,run_id,action_id
)
INSERT INTO agent_platform.agent_action_attempts(
    tenant_id,workflow_id,run_id,action_id,decision_cycle,plan_node_id,action_kind,name,
    status,workflow_generation,plan_revision,node_revision,workspace_revision,
    first_event_sequence,last_event_sequence,started_at,finished_at)
SELECT rollup.tenant_id,rollup.workflow_id,rollup.run_id,rollup.action_id,rollup.decision_cycle,
       rollup.plan_node_id,rollup.action_kind,rollup.name,rollup.status,
       workflow.execution_generation,COALESCE(plan.revision,0),COALESCE(node.node_revision,0),workflow.workspace_revision,
       rollup.first_event_sequence,rollup.last_event_sequence,rollup.started_at,rollup.finished_at
FROM action_rollup AS rollup
JOIN agent_platform.agent_workflows AS workflow ON workflow.id=rollup.workflow_id
LEFT JOIN agent_platform.agent_task_plans AS plan ON plan.workflow_id=workflow.id
LEFT JOIN agent_platform.agent_plan_node_states AS node
  ON node.workflow_id=workflow.id AND node.node_id=rollup.plan_node_id
ON CONFLICT(run_id,action_id) DO NOTHING;
