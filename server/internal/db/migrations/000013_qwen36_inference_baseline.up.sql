-- Register the local dual-GPU Qwen3.6 inference-optimization endpoint.
INSERT INTO service_instances (name, base_url, model_id, kind, gpu_id, routing_role, status, metadata)
VALUES (
    'qwen36-27b-fp8-vllm',
    'http://host.docker.internal:8020/v1',
    'qwen36-27b-fp8',
    'vllm',
    '0,1',
    'replica',
    'unknown',
    '{"tensor_parallel_size":2,"dataset":"DianJin/DianJin-CSC-Data","purpose":"inference_optimization"}'::jsonb
)
ON CONFLICT (name) DO UPDATE SET
    base_url = EXCLUDED.base_url,
    model_id = EXCLUDED.model_id,
    gpu_id = EXCLUDED.gpu_id,
    metadata = service_instances.metadata || EXCLUDED.metadata,
    updated_at = now();

INSERT INTO models (model_id, version, display_name, base_model, status, tags, metadata)
VALUES (
    'qwen36-27b-fp8',
    'baseline',
    'Qwen3.6 27B FP8',
    'Qwen/Qwen3.6-27B-FP8',
    'registered',
    '["customer-support","fp8","inference-optimization"]'::jsonb,
    '{"local_path":"/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-FP8","tensor_parallel_size":2,"context_length":4096}'::jsonb
)
ON CONFLICT (model_id, version) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    status = EXCLUDED.status,
    tags = EXCLUDED.tags,
    metadata = models.metadata || EXCLUDED.metadata;
