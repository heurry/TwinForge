DELETE FROM service_instances WHERE name = 'qwen36-27b-awq-vllm';
DELETE FROM models WHERE model_id = 'qwen36-27b-awq' AND version = 'w4a16';
