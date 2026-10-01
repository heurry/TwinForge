-- Register the formal dual-GPU Qwen3.6 AWQ W4A16 inference endpoint.
INSERT INTO service_instances (name, base_url, model_id, kind, gpu_id, routing_role, status, metadata)
VALUES (
    'qwen36-27b-awq-vllm',
    'http://host.docker.internal:8021/v1',
    'qwen36-27b-awq',
    'vllm',
    '0,1',
    'replica',
    'unknown',
    '{"tensor_parallel_size":2,"dataset":"DianJin/DianJin-CSC-Data","purpose":"inference_optimization","quantization":"awq-w4a16","max_model_len":4096}'::jsonb
)
ON CONFLICT (name) DO UPDATE SET
    base_url = EXCLUDED.base_url,
    model_id = EXCLUDED.model_id,
    gpu_id = EXCLUDED.gpu_id,
    metadata = service_instances.metadata || EXCLUDED.metadata,
    updated_at = now();

INSERT INTO models (model_id, version, display_name, base_model, status, tags, metadata)
VALUES (
    'qwen36-27b-awq',
    'w4a16',
    'Qwen3.6 27B AWQ W4A16',
    'Qwen/Qwen3.6-27B',
    'registered',
    '["customer-support","awq","w4a16","inference-optimization"]'::jsonb,
    '{"local_path":"/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-AWQ-INT4","tensor_parallel_size":2,"context_length":4096,"calibration_dataset":"DianJin/DianJin-CSC-Data","calibration_samples":128}'::jsonb
)
ON CONFLICT (model_id, version) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    status = EXCLUDED.status,
    tags = EXCLUDED.tags,
    metadata = models.metadata || EXCLUDED.metadata;
