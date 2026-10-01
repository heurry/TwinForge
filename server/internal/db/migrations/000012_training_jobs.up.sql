-- Phase F / F2：分布式训练任务台账（Kubeflow PyTorchJob 的控制面记录）。
-- 提交训练即建一行（status=pending）；runner 轮询 PyTorchJob 状态回写 status + metadata（相位/副本进度/事件）；
-- 成功后（F3）把产出的 LoRA adapter 注册进 model registry，回填 model_version_id 形成血缘。
-- 与 knowledge/chat/routing 一致：走 hand-pgx（不纳入 sqlc.yaml schema）。
CREATE TABLE training_jobs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                VARCHAR(128) NOT NULL,
    framework           VARCHAR(32)  NOT NULL DEFAULT 'pytorch',
    namespace           VARCHAR(128) NOT NULL,
    base_model          VARCHAR(256) NOT NULL,
    dataset_uri         VARCHAR(512),
    image               VARCHAR(512) NOT NULL,
    workers             INT NOT NULL DEFAULT 0,
    gpus_per_worker     INT NOT NULL DEFAULT 0,
    hyperparams         JSONB NOT NULL DEFAULT '{}'::jsonb,
    status              VARCHAR(32) NOT NULL DEFAULT 'pending',  -- pending|running|succeeded|failed|cancelled
    k8s_job_ref         VARCHAR(256),                            -- namespace/name 指向 PyTorchJob
    output_artifact_uri VARCHAR(512),                            -- 成功后的 LoRA adapter MinIO key（F3）
    model_version_id    UUID,                                    -- 成功后回写 C1 注册的版本 id（F3）
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,      -- 相位/进度/replica 状态/事件日志
    created_by          VARCHAR(128),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_training_jobs_created_at ON training_jobs (created_at DESC);
CREATE INDEX idx_training_jobs_status ON training_jobs (status);
