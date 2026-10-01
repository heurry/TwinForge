CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Seed one manifest row for Runs that existed before the manifest projector.
-- Subsequent Artifact/Run transitions replace this bounded projection.
INSERT INTO agent_platform.agent_run_manifests
    (run_id, tenant_id, workflow_id, status, canonical_artifacts, required_outputs,
     verification_summary, child_runs, final_artifacts, final_output_hash, final_output_present)
SELECT r.id, r.tenant_id, r.workflow_id, r.status,
       COALESCE(artifacts.canonical_artifacts, '[]'::jsonb),
       '[]'::jsonb, '{}'::jsonb,
       COALESCE(children.child_runs, '[]'::jsonb),
       COALESCE(artifacts.final_artifacts, '[]'::jsonb),
       CASE WHEN r.output IS NOT NULL THEN encode(digest(r.output::text, 'sha256'), 'hex') END,
       (r.output IS NOT NULL)
FROM agent_platform.agent_runs r
LEFT JOIN LATERAL (
    SELECT
        jsonb_agg(to_jsonb(latest) ORDER BY latest.name) FILTER (WHERE latest.artifact_id IS NOT NULL) AS canonical_artifacts,
        jsonb_agg(latest.artifact_id) FILTER (WHERE latest.final) AS final_artifacts
    FROM (
        SELECT DISTINCT ON (a.name)
            a.id::text AS artifact_id, a.name, a.kind, a.media_type, a.content_hash,
            a.size_bytes, a.metadata, a.created_at, true AS canonical,
            (COALESCE(a.metadata->>'phase','')='final' OR (a.kind='workspace_file' AND r.status='completed')) AS final
        FROM agent_platform.agent_artifacts a
        WHERE a.run_id=r.id AND a.tenant_id=r.tenant_id
        ORDER BY a.name, a.created_at DESC, a.id DESC
    ) latest
) artifacts ON true
LEFT JOIN LATERAL (
    SELECT jsonb_agg(child.id::text ORDER BY child.created_at) AS child_runs
    FROM agent_platform.agent_runs child
    WHERE child.parent_run_id=r.id AND child.tenant_id=r.tenant_id
) children ON true
ON CONFLICT (run_id) DO NOTHING;
