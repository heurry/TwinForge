WITH training_item AS (
    INSERT INTO config_items (env, namespace, config_key, config_type, active_version, status, created_by)
    SELECT 'dev', 'training', 'training/qwen35-lora.yaml', 'yaml', 1, 'active', 'migration'
    WHERE NOT EXISTS (SELECT 1 FROM config_items WHERE config_key='training/qwen35-lora.yaml' AND env='dev')
    RETURNING id
)
INSERT INTO config_versions (config_item_id, version, content, change_reason, operator, status)
SELECT id, 1, $yaml$
kind: training
spec:
  name: qwen35-4b-customer-lora-v1
  namespace: training
  base_model: /opt/twinforge/models/Qwen3.5-4B
  dataset_uri: /opt/twinforge/data/dianjin_csc_sft_train.jsonl
  image: local/train:qwen35-v1
  workers: 1
  gpus_per_worker: 1
  model_id: qwen35-4b-customer
  version: lora-v1
  hyperparams:
    learning_rate: 0.0002
    epochs: 3
    lora_rank: 16
    lora_alpha: 32
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    max_seq_length: 1024
    precision: bf16
    deepspeed: zero2
    gradient_checkpointing: true
$yaml$, '可复现的 LoRA 训练启动模板', 'migration', 'active' FROM training_item;

WITH inference_item AS (
    INSERT INTO config_items (env, namespace, config_key, config_type, active_version, status, created_by)
    SELECT 'dev', 'serving', 'inference/qwen36-vllm.yaml', 'yaml', 1, 'active', 'migration'
    WHERE NOT EXISTS (SELECT 1 FROM config_items WHERE config_key='inference/qwen36-vllm.yaml' AND env='dev')
    RETURNING id
)
INSERT INTO config_versions (config_item_id, version, content, change_reason, operator, status)
SELECT id, 1, $yaml$
kind: inference
spec:
  profile: scheduler
  max_num_seqs: 16
  max_num_batched_tokens: 8192
  scheduling_policy: fcfs
  max_num_partial_prefills: 1
  max_long_partial_prefills: 1
  long_prefill_token_threshold: 0
  stream_interval: 1
  prefix_caching: true
  async_scheduling: true
  scheduler_reserve_full_isl: true
  disable_custom_all_reduce: true
  gpu_memory_utilization: 0.9
  max_model_len: 4096
  kv_cache_dtype: auto
  speculative_decoding: none
$yaml$, '经压测验证的双卡 vLLM 启动模板', 'migration', 'active' FROM inference_item;
