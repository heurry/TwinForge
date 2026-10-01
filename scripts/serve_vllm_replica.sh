#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

cd "${ROOT_DIR}"
PYTHON_BIN="$(resolve_python_bin)"
CONFIG_PATH="${CONFIG_PATH:-configs/serve/qwen3_4b_vllm_replica0.yaml}"

readarray -t CFG_LINES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import yaml

with open(os.environ["CONFIG_PATH"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

for key, default in [
    ("model", ""),
    ("served_model_name", "qwen3-4b-customer"),
    ("tensor_parallel_size", 1),
    ("pipeline_parallel_size", 1),
    ("dtype", "float16"),
    ("quantization", ""),
    ("max_model_len", 4096),
    ("gpu_memory_utilization", 0.60),
    ("cpu_offload_gb", ""),
    ("max_num_seqs", 64),
    ("max_num_batched_tokens", 8192),
    ("enable_prefix_caching", False),
    ("enable_chunked_prefill", False),
    ("scheduling_policy", ""),
    ("max_num_partial_prefills", ""),
    ("max_long_partial_prefills", ""),
    ("long_prefill_token_threshold", ""),
    ("async_scheduling", ""),
    ("scheduler_reserve_full_isl", ""),
    ("stream_interval", ""),
    ("kv_cache_dtype", ""),
    ("disable_custom_all_reduce", False),
    ("enable_dbo", False),
    ("dbo_prefill_token_threshold", ""),
    ("dbo_decode_token_threshold", ""),
    ("dense_tp_overlap", False),
    ("dense_tp_overlap_min_tokens", ""),
    ("enforce_eager", False),
    ("compilation_config", ""),
    ("profiler_config", ""),
    ("kernel_config", ""),
    ("attention_backend", ""),
    ("draft_model", ""),
    ("speculative_config", ""),
    ("nccl_debug", ""),
    ("nccl_p2p_disable", ""),
    ("nccl_shm_disable", ""),
    ("nccl_algo", ""),
    ("nccl_proto", ""),
    ("nccl_min_nchannels", ""),
    ("nccl_max_nchannels", ""),
    ("nccl_buffsize", ""),
    ("nccl_nthreads", ""),
    ("language_model_only", False),
    ("reasoning_parser", ""),
    ("enable_auto_tool_choice", False),
    ("tool_call_parser", ""),
    ("generation_config", ""),
    ("trust_remote_code", False),
    ("host", "0.0.0.0"),
    ("port", 8000),
    ("api_key", "EMPTY"),
]:
    value = cfg.get(key, default)
    if key in ("compilation_config", "profiler_config", "kernel_config", "speculative_config") and isinstance(value, dict):
        value = json.dumps(value, separators=(",", ":"))
    print(f"{key}={value}")
PY
)

for line in "${CFG_LINES[@]}"; do
  key="${line%%=*}"
  value="${line#*=}"
  case "${key}" in
    model) MODEL_PATH="${value}" ;;
    served_model_name) SERVED_MODEL_NAME="${value}" ;;
    tensor_parallel_size) TP_SIZE="${value}" ;;
    pipeline_parallel_size) PP_SIZE="${value}" ;;
    dtype) DTYPE="${value}" ;;
    quantization) QUANTIZATION="${value}" ;;
    max_model_len) MAX_MODEL_LEN="${value}" ;;
    gpu_memory_utilization) GPU_MEMORY_UTILIZATION="${value}" ;;
    cpu_offload_gb) CPU_OFFLOAD_GB="${value}" ;;
    max_num_seqs) MAX_NUM_SEQS="${value}" ;;
    max_num_batched_tokens) MAX_NUM_BATCHED_TOKENS="${value}" ;;
    enable_prefix_caching) ENABLE_PREFIX_CACHING="${value}" ;;
    enable_chunked_prefill) ENABLE_CHUNKED_PREFILL="${value}" ;;
    scheduling_policy) SCHEDULING_POLICY="${value}" ;;
    max_num_partial_prefills) MAX_NUM_PARTIAL_PREFILLS="${value}" ;;
    max_long_partial_prefills) MAX_LONG_PARTIAL_PREFILLS="${value}" ;;
    long_prefill_token_threshold) LONG_PREFILL_TOKEN_THRESHOLD="${value}" ;;
    async_scheduling) ASYNC_SCHEDULING="${value}" ;;
    scheduler_reserve_full_isl) SCHEDULER_RESERVE_FULL_ISL="${value}" ;;
    stream_interval) STREAM_INTERVAL="${value}" ;;
    kv_cache_dtype) KV_CACHE_DTYPE="${value}" ;;
    disable_custom_all_reduce) DISABLE_CUSTOM_ALL_REDUCE="${value}" ;;
    enable_dbo) ENABLE_DBO="${value}" ;;
    dbo_prefill_token_threshold) DBO_PREFILL_TOKEN_THRESHOLD="${value}" ;;
    dbo_decode_token_threshold) DBO_DECODE_TOKEN_THRESHOLD="${value}" ;;
    dense_tp_overlap) DENSE_TP_OVERLAP="${value}" ;;
    dense_tp_overlap_min_tokens) DENSE_TP_OVERLAP_MIN_TOKENS="${value}" ;;
    enforce_eager) ENFORCE_EAGER="${value}" ;;
    compilation_config) COMPILATION_CONFIG="${value}" ;;
    profiler_config) PROFILER_CONFIG="${value}" ;;
    kernel_config) KERNEL_CONFIG="${value}" ;;
    attention_backend) ATTENTION_BACKEND="${value}" ;;
    draft_model) DRAFT_MODEL="${value}" ;;
    speculative_config) SPECULATIVE_CONFIG="${value}" ;;
    nccl_debug) NCCL_DEBUG_VALUE="${value}" ;;
    nccl_p2p_disable) NCCL_P2P_DISABLE_VALUE="${value}" ;;
    nccl_shm_disable) NCCL_SHM_DISABLE_VALUE="${value}" ;;
    nccl_algo) NCCL_ALGO_VALUE="${value}" ;;
    nccl_proto) NCCL_PROTO_VALUE="${value}" ;;
    nccl_min_nchannels) NCCL_MIN_NCHANNELS_VALUE="${value}" ;;
    nccl_max_nchannels) NCCL_MAX_NCHANNELS_VALUE="${value}" ;;
    nccl_buffsize) NCCL_BUFFSIZE_VALUE="${value}" ;;
    nccl_nthreads) NCCL_NTHREADS_VALUE="${value}" ;;
    language_model_only) LANGUAGE_MODEL_ONLY="${value}" ;;
    reasoning_parser) REASONING_PARSER="${value}" ;;
    enable_auto_tool_choice) ENABLE_AUTO_TOOL_CHOICE="${value}" ;;
    tool_call_parser) TOOL_CALL_PARSER="${value}" ;;
    generation_config) GENERATION_CONFIG="${value}" ;;
    trust_remote_code) TRUST_REMOTE_CODE="${value}" ;;
    host) HOST="${value}" ;;
    port) PORT="${value}" ;;
    api_key) API_KEY="${value}" ;;
  esac
done

EXTRA_ARGS=()
if [[ "${ENABLE_PREFIX_CACHING}" == "True" || "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
else
  EXTRA_ARGS+=(--no-enable-prefix-caching)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "True" || "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
  EXTRA_ARGS+=(--enable-chunked-prefill)
else
  EXTRA_ARGS+=(--no-enable-chunked-prefill)
fi
if [[ -n "${SCHEDULING_POLICY}" ]]; then
  EXTRA_ARGS+=(--scheduling-policy "${SCHEDULING_POLICY}")
fi
if [[ -n "${MAX_NUM_PARTIAL_PREFILLS}" ]]; then
  EXTRA_ARGS+=(--max-num-partial-prefills "${MAX_NUM_PARTIAL_PREFILLS}")
fi
if [[ -n "${MAX_LONG_PARTIAL_PREFILLS}" ]]; then
  EXTRA_ARGS+=(--max-long-partial-prefills "${MAX_LONG_PARTIAL_PREFILLS}")
fi
if [[ -n "${LONG_PREFILL_TOKEN_THRESHOLD}" ]]; then
  EXTRA_ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
fi
if [[ "${ASYNC_SCHEDULING}" == "True" || "${ASYNC_SCHEDULING}" == "true" ]]; then
  EXTRA_ARGS+=(--async-scheduling)
elif [[ "${ASYNC_SCHEDULING}" == "False" || "${ASYNC_SCHEDULING}" == "false" ]]; then
  EXTRA_ARGS+=(--no-async-scheduling)
fi
if [[ "${SCHEDULER_RESERVE_FULL_ISL}" == "True" || "${SCHEDULER_RESERVE_FULL_ISL}" == "true" ]]; then
  EXTRA_ARGS+=(--scheduler-reserve-full-isl)
elif [[ "${SCHEDULER_RESERVE_FULL_ISL}" == "False" || "${SCHEDULER_RESERVE_FULL_ISL}" == "false" ]]; then
  EXTRA_ARGS+=(--no-scheduler-reserve-full-isl)
fi
if [[ -n "${STREAM_INTERVAL}" ]]; then
  EXTRA_ARGS+=(--stream-interval "${STREAM_INTERVAL}")
fi
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  EXTRA_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${CPU_OFFLOAD_GB}" ]]; then
  EXTRA_ARGS+=(--cpu-offload-gb "${CPU_OFFLOAD_GB}")
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "True" || "${DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
  EXTRA_ARGS+=(--disable-custom-all-reduce)
fi
if [[ "${ENABLE_DBO}" == "True" || "${ENABLE_DBO}" == "true" ]]; then
  EXTRA_ARGS+=(--enable-dbo)
fi
if [[ -n "${DBO_PREFILL_TOKEN_THRESHOLD}" ]]; then
  EXTRA_ARGS+=(--dbo-prefill-token-threshold "${DBO_PREFILL_TOKEN_THRESHOLD}")
fi
if [[ -n "${DBO_DECODE_TOKEN_THRESHOLD}" ]]; then
  EXTRA_ARGS+=(--dbo-decode-token-threshold "${DBO_DECODE_TOKEN_THRESHOLD}")
fi
if [[ "${ENFORCE_EAGER}" == "True" || "${ENFORCE_EAGER}" == "true" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ -n "${COMPILATION_CONFIG}" ]]; then
  EXTRA_ARGS+=(--compilation-config "${COMPILATION_CONFIG}")
fi
if [[ -n "${PROFILER_CONFIG}" ]]; then
  EXTRA_ARGS+=(--profiler-config "${PROFILER_CONFIG}")
fi
if [[ -n "${KERNEL_CONFIG}" ]]; then
  EXTRA_ARGS+=(--kernel-config "${KERNEL_CONFIG}")
fi
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  EXTRA_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [[ -n "${SPECULATIVE_CONFIG}" ]]; then
  EXTRA_ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi
if [[ "${LANGUAGE_MODEL_ONLY}" == "True" || "${LANGUAGE_MODEL_ONLY}" == "true" ]]; then
  EXTRA_ARGS+=(--language-model-only)
fi
if [[ -n "${REASONING_PARSER}" && "${REASONING_PARSER}" != "None" ]]; then
  EXTRA_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "True" || "${ENABLE_AUTO_TOOL_CHOICE}" == "true" ]]; then
  if [[ -z "${TOOL_CALL_PARSER}" || "${TOOL_CALL_PARSER}" == "None" ]]; then
    echo "[ERROR] tool_call_parser is required when enable_auto_tool_choice is true" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi
if [[ -n "${GENERATION_CONFIG}" && "${GENERATION_CONFIG}" != "None" ]]; then
  EXTRA_ARGS+=(--generation-config "${GENERATION_CONFIG}")
fi
if [[ -n "${QUANTIZATION}" && "${QUANTIZATION}" != "None" ]]; then
  EXTRA_ARGS+=(--quantization "${QUANTIZATION}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "True" || "${TRUST_REMOTE_CODE}" == "true" ]]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi
if [[ -n "${API_KEY}" && "${API_KEY}" != "None" ]]; then
  EXTRA_ARGS+=(--api-key "${API_KEY}")
fi

echo "[INFO] serving ${MODEL_PATH} as ${SERVED_MODEL_NAME} on ${HOST}:${PORT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

NCCL_ENV_ARGS=()
for entry in \
  "NCCL_DEBUG=${NCCL_DEBUG_VALUE}" \
  "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE_VALUE}" \
  "NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE_VALUE}" \
  "NCCL_ALGO=${NCCL_ALGO_VALUE}" \
  "NCCL_PROTO=${NCCL_PROTO_VALUE}" \
  "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS_VALUE}" \
  "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS_VALUE}" \
  "NCCL_BUFFSIZE=${NCCL_BUFFSIZE_VALUE}" \
  "NCCL_NTHREADS=${NCCL_NTHREADS_VALUE}"; do
  name="${entry%%=*}"
  value="${entry#*=}"
  if [[ -n "${value}" ]]; then
    export "${name}=${value}"
    NCCL_ENV_ARGS+=(--env "${name}=${value}")
  fi
done

if [[ "${DENSE_TP_OVERLAP}" == "True" || "${DENSE_TP_OVERLAP}" == "true" ]]; then
  export VLLM_DENSE_TP_OVERLAP=1
  NCCL_ENV_ARGS+=(--env VLLM_DENSE_TP_OVERLAP=1)
  if [[ -n "${DENSE_TP_OVERLAP_MIN_TOKENS}" ]]; then
    export VLLM_DENSE_TP_OVERLAP_MIN_TOKENS="${DENSE_TP_OVERLAP_MIN_TOKENS}"
    NCCL_ENV_ARGS+=(--env "VLLM_DENSE_TP_OVERLAP_MIN_TOKENS=${DENSE_TP_OVERLAP_MIN_TOKENS}")
  fi
fi

if [[ -n "${VLLM_DOCKER_IMAGE:-}" ]]; then
  CONTAINER_NAME="${VLLM_CONTAINER_NAME:-twinforge-vllm}"
  CACHE_DIR="${VLLM_CACHE_DIR:-${ROOT_DIR}/.cache/vllm}"
  MODEL_VOLUME_ARGS=(-v "${MODEL_PATH}:${MODEL_PATH}:ro")
  if [[ -n "${DRAFT_MODEL}" ]]; then
    MODEL_VOLUME_ARGS+=(-v "${DRAFT_MODEL}:${DRAFT_MODEL}:ro")
  fi
  DOCKER_RUN_ARGS=(run --rm)
  if [[ "${VLLM_DOCKER_DETACH:-false}" == "true" ]]; then
    DOCKER_RUN_ARGS+=(--detach)
  fi
  mkdir -p "${CACHE_DIR}"
  exec docker "${DOCKER_RUN_ARGS[@]}" \
    --name "${CONTAINER_NAME}" \
    --gpus "${VLLM_GPUS:-all}" \
    --ipc=host \
    --network=host \
    "${NCCL_ENV_ARGS[@]}" \
    "${MODEL_VOLUME_ARGS[@]}" \
    -v "${CACHE_DIR}:/root/.cache/vllm" \
    -v "${ROOT_DIR}/logs:${ROOT_DIR}/logs" \
    --entrypoint vllm \
    "${VLLM_DOCKER_IMAGE}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${EXTRA_ARGS[@]}"
fi

if [[ -x "${ROOT_DIR}/.venv/bin/vllm" ]]; then
  exec "${ROOT_DIR}/.venv/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${EXTRA_ARGS[@]}"
fi

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size "${PP_SIZE}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "${EXTRA_ARGS[@]}"
