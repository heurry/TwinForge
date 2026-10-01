-- Re-project final delivery markers for manifests created before the runtime
-- learned to treat completed workspace snapshots as final artifacts. Explicit
-- metadata phase=final always wins; completed workspace_file artifacts are the
-- default final deliverables when no explicit phase was recorded.
WITH canonical AS (
    SELECT m.run_id,
           COALESCE(jsonb_agg(
               jsonb_build_object(
                   'artifact_id', latest.id::text,
                   'name', latest.name,
                   'kind', latest.kind,
                   'media_type', latest.media_type,
                   'content_hash', latest.content_hash,
                   'size_bytes', latest.size_bytes,
                   'metadata', latest.metadata,
                   'created_at', latest.created_at,
                   'canonical', true,
                   'phase', COALESCE(latest.metadata->>'phase','intermediate'),
                   'final', (COALESCE(latest.metadata->>'phase','')='final' OR (latest.kind='workspace_file' AND r.status='completed'))
               ) ORDER BY latest.name
           ) FILTER (WHERE latest.id IS NOT NULL), '[]'::jsonb) AS artifacts
    FROM agent_platform.agent_run_manifests m
    JOIN agent_platform.agent_runs r ON r.id=m.run_id AND r.tenant_id=m.tenant_id
    LEFT JOIN LATERAL (
        SELECT DISTINCT ON (a.name) a.id, a.name, a.kind, a.media_type,
            a.content_hash, a.size_bytes, a.metadata, a.created_at
        FROM agent_platform.agent_artifacts a
        WHERE a.run_id=m.run_id AND a.tenant_id=m.tenant_id
        ORDER BY a.name, a.created_at DESC, a.id DESC
    ) latest ON true
    GROUP BY m.run_id, r.status
), finals AS (
    SELECT m.run_id,
           COALESCE(jsonb_agg(a.id::text ORDER BY a.created_at, a.id) FILTER (WHERE a.id IS NOT NULL), '[]'::jsonb) AS artifact_ids
    FROM agent_platform.agent_run_manifests m
    JOIN agent_platform.agent_runs r ON r.id=m.run_id AND r.tenant_id=m.tenant_id
    LEFT JOIN agent_platform.agent_artifacts a
      ON a.run_id=m.run_id AND a.tenant_id=m.tenant_id
     AND (COALESCE(a.metadata->>'phase','')='final' OR (a.kind='workspace_file' AND r.status='completed'))
    GROUP BY m.run_id
)
UPDATE agent_platform.agent_run_manifests m
SET canonical_artifacts=c.artifacts,
    final_artifacts=f.artifact_ids,
    updated_at=now()
FROM canonical c JOIN finals f ON f.run_id=c.run_id
WHERE m.run_id=c.run_id;
