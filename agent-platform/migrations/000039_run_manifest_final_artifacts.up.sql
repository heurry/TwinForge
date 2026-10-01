ALTER TABLE agent_platform.agent_run_manifests
    ADD COLUMN IF NOT EXISTS final_artifacts JSONB NOT NULL DEFAULT '[]'::jsonb;
