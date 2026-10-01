DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM agent_platform.agent_artifacts WHERE storage_backend='minio') THEN
        RAISE EXCEPTION 'cannot roll back Artifact object migration while MinIO-backed rows exist; rehydrate content first';
    END IF;
END $$;

DROP INDEX IF EXISTS agent_platform.idx_agent_artifacts_minio_object;
DROP INDEX IF EXISTS agent_platform.idx_agent_artifacts_storage_reconcile;

ALTER TABLE agent_platform.agent_artifacts
    DROP CONSTRAINT IF EXISTS chk_agent_artifact_storage_location,
    DROP CONSTRAINT IF EXISTS chk_agent_artifact_storage_status,
    DROP CONSTRAINT IF EXISTS chk_agent_artifact_storage_backend,
    DROP COLUMN IF EXISTS migrated_at,
    DROP COLUMN IF EXISTS storage_status,
    DROP COLUMN IF EXISTS object_key,
    DROP COLUMN IF EXISTS storage_backend;

ALTER TABLE agent_platform.agent_artifacts ALTER COLUMN content SET NOT NULL;
