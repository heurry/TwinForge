-- Enrich backfilled manifests with the immutable Plan/verification projection.
-- New Runs receive the same projection transactionally from rebuildRunManifestTx.
UPDATE agent_platform.agent_run_manifests AS manifest
SET verification_summary = jsonb_build_object(
        'status', manifest.status,
        'plan_revision', plan.revision,
        'steps', plan.steps
    ),
    required_outputs = COALESCE(required.targets, '[]'::jsonb),
    updated_at = now()
FROM agent_platform.agent_task_plans AS plan
LEFT JOIN LATERAL (
    SELECT jsonb_agg(target ORDER BY target) AS targets
    FROM (
        SELECT DISTINCT criterion->'verification'->>'target' AS target
        FROM jsonb_array_elements(plan.steps) AS step,
             jsonb_array_elements(COALESCE(step->'acceptance_criteria','[]'::jsonb)) AS criterion
        WHERE criterion->>'enforcement' IN ('required','release_gate')
          AND NULLIF(criterion->'verification'->>'target','') IS NOT NULL
    ) valueset
) required ON true
WHERE manifest.workflow_id = plan.workflow_id
  AND manifest.tenant_id = plan.tenant_id;
