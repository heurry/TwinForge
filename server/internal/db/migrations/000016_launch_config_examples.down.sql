DELETE FROM config_versions WHERE config_item_id IN (
    SELECT id FROM config_items WHERE created_by='migration' AND config_key IN ('training/qwen35-lora.yaml','inference/qwen36-vllm.yaml')
);
DELETE FROM config_items WHERE created_by='migration' AND config_key IN ('training/qwen35-lora.yaml','inference/qwen36-vllm.yaml');
