-- Older Plan rows predate the normalized node projection. Backfill their
-- platform-owned state from the durable JSON snapshot so future observations
-- have an existing, Plan-authorized node to update.
INSERT INTO agent_platform.agent_plan_node_states (
    workflow_id,node_id,plan_revision,status,output,artifact_ids,tests,
    input_tokens,output_tokens,total_tokens,cost_usd,duration_ms,attempts,
    last_run_id,last_event_sequence,blocked_reason,retry_from_node_id,next_node_ids
)
SELECT
    plan.workflow_id,
    step->>'id',
    plan.revision,
    COALESCE(NULLIF(step->>'status',''),NULLIF(step #>> '{state,status}',''),'pending'),
    COALESCE(NULLIF(step #>> '{state,output}',''),NULLIF(step->>'result','')),
    CASE WHEN jsonb_typeof(step #> '{state,artifact_ids}')='array'
         THEN step #> '{state,artifact_ids}' ELSE '[]'::jsonb END,
    CASE WHEN jsonb_typeof(step #> '{state,tests}')='array'
         THEN step #> '{state,tests}' ELSE '[]'::jsonb END,
    COALESCE(NULLIF(step #>> '{state,usage,input_tokens}','')::bigint,0),
    COALESCE(NULLIF(step #>> '{state,usage,output_tokens}','')::bigint,0),
    COALESCE(NULLIF(step #>> '{state,usage,total_tokens}','')::bigint,0),
    COALESCE(NULLIF(step #>> '{state,usage,cost_usd}','')::double precision,0),
    COALESCE(NULLIF(step #>> '{state,usage,duration_ms}','')::bigint,0),
    COALESCE(NULLIF(step #>> '{state,attempts}','')::integer,0),
    NULLIF(step #>> '{state,last_run_id}','')::uuid,
    COALESCE(NULLIF(step #>> '{state,last_event_sequence}','')::bigint,0),
    NULLIF(step #>> '{state,blocked_reason}',''),
    NULLIF(step #>> '{state,retry_from_node_id}',''),
    CASE WHEN jsonb_typeof(step #> '{state,next_node_ids}')='array'
         THEN step #> '{state,next_node_ids}' ELSE '[]'::jsonb END
FROM agent_platform.agent_task_plans AS plan
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(plan.steps)='array' THEN plan.steps ELSE '[]'::jsonb END
) AS step
WHERE NULLIF(step->>'id','') IS NOT NULL
ON CONFLICT (workflow_id,node_id) DO NOTHING;

-- A node projection is meaningful only while the current Workflow Plan still
-- contains that node. Historical graph revisions remain available in Events.
DELETE FROM agent_platform.agent_plan_node_states AS node
WHERE NOT EXISTS (
    SELECT 1
    FROM agent_platform.agent_task_plans AS plan
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN jsonb_typeof(plan.steps)='array' THEN plan.steps ELSE '[]'::jsonb END
    ) AS step
    WHERE plan.workflow_id=node.workflow_id
      AND step->>'id'=node.node_id
);
