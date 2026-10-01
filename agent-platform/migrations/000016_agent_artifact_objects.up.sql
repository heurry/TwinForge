ALTER TABLE agent_platform.agent_artifacts
    ALTER COLUMN content DROP NOT NULL,
    ADD COLUMN storage_backend VARCHAR(32) NOT NULL DEFAULT 'inline',
    ADD COLUMN object_key TEXT,
    ADD COLUMN storage_status VARCHAR(32) NOT NULL DEFAULT 'ready',
    ADD COLUMN migrated_at TIMESTAMPTZ;

ALTER TABLE agent_platform.agent_artifacts
    ADD CONSTRAINT chk_agent_artifact_storage_backend
        CHECK (storage_backend IN ('inline', 'minio')),
    ADD CONSTRAINT chk_agent_artifact_storage_status
        CHECK (storage_status IN ('ready', 'migration_pending', 'failed')),
    ADD CONSTRAINT chk_agent_artifact_storage_location
        CHECK (
            (storage_backend='inline' AND content IS NOT NULL AND object_key IS NULL) OR
            (storage_backend='minio' AND content IS NULL AND NULLIF(btrim(object_key),'') IS NOT NULL)
        );

CREATE INDEX idx_agent_artifacts_storage_reconcile
    ON agent_platform.agent_artifacts (storage_backend, created_at, id)
    WHERE storage_backend='inline';

CREATE INDEX idx_agent_artifacts_minio_object
    ON agent_platform.agent_artifacts (object_key)
    WHERE storage_backend='minio';
