#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_BASE="${API_BASE:-http://127.0.0.1:8081}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8021}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/inference/awq-experiments}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-awq-baseline}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/serve/qwen36_27b_awq_vllm.yaml}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-16}"
MAX_TOKENS="${MAX_TOKENS:-256}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-1.0}"
STOP_JSON="${STOP_JSON:-[\"😊\",\"😘\",\"😀\",\"❤️\",\"❤\",\"✨\",\"⭐\",\"🌟\",\"🎉\"]}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-[1,2,4,8,16]}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-[1024,2048]}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-600}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-1800}"
RESTART_SERVICE="${RESTART_SERVICE:-false}"

mkdir -p "${OUT_DIR}"

config_json() {
  CONFIG_PATH="${CONFIG_PATH}" python3 - <<'PY'
import json
import os
import yaml

with open(os.environ["CONFIG_PATH"], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

print(json.dumps(config, ensure_ascii=True, separators=(",", ":")))
PY
}

wait_for_service() {
  local deadline=$((SECONDS + START_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -fsS "${VLLM_BASE}/health" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  return 1
}

wait_for_gpu_release() {
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    local max_used
    max_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -n 1)"
    if [[ "${max_used}" =~ ^[0-9]+$ ]] && (( max_used < 4096 )); then
      return 0
    fi
    sleep 2
  done
  echo "GPU memory was not released below 4096 MiB after stopping the prior service" >&2
  return 1
}

wait_for_run() {
  local run_id="$1"
  local deadline=$((SECONDS + RUN_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    local body status
    body="$(curl -fsS "${API_BASE}/api/benchmarks/${run_id}")"
    status="$(jq -r '.status' <<<"${body}")"
    printf '%s run=%s status=%s\n' "$(date '+%H:%M:%S')" "${run_id:0:8}" "${status}"
    if [[ "${status}" == "completed" ]]; then
      printf '%s' "${body}" >"${OUT_DIR}/${EXPERIMENT_NAME}-result.json"
      curl -fsS "${API_BASE}/api/benchmarks/${run_id}/events" >"${OUT_DIR}/${EXPERIMENT_NAME}-events.json"
      return 0
    fi
    if [[ "${status}" == "failed" || "${status}" == "cancelled" ]]; then
      printf '%s' "${body}" >"${OUT_DIR}/${EXPERIMENT_NAME}-result.json"
      return 1
    fi
    sleep 10
  done
  return 1
}

cfg="$(config_json)"
gpu_id="$(jq -r 'if .tensor_parallel_size == 1 then "0" else "0,1" end' <<<"${cfg}")"
cp "${CONFIG_PATH}" "${OUT_DIR}/${EXPERIMENT_NAME}-serve.yaml"

if [[ "${RESTART_SERVICE}" == "true" ]]; then
  docker rm -f twinforge-vllm-qwen36-awq >/dev/null 2>&1 || true
  wait_for_gpu_release
  env CONFIG_PATH="${CONFIG_PATH}" VLLM_DOCKER_DETACH=true \
    "${ROOT_DIR}/scripts/serve_qwen36_27b_awq.sh" >"${OUT_DIR}/${EXPERIMENT_NAME}-vllm.log" 2>&1
fi

wait_for_service

setsid -f env API_BASE="${API_BASE}" VLLM_BASE="${VLLM_BASE}" \
  "${ROOT_DIR}/scripts/heartbeat_qwen36_awq.sh" \
  >"${OUT_DIR}/awq-heartbeat.log" 2>&1 </dev/null

curl -fsS -X POST "${API_BASE}/api/service-instances/register" \
  -H 'Content-Type: application/json' \
  --data "$(jq -nc --argjson cfg "${cfg}" --arg gpu_id "${gpu_id}" '{name:"qwen36-27b-awq-vllm",base_url:"http://host.docker.internal:8021/v1",model_id:"qwen36-27b-awq",kind:"vllm",gpu_id:$gpu_id,routing_role:"replica",metadata:{purpose:"inference_optimization",quantization:"awq-w4a16",launch_config:$cfg},operator:"awq-experiment"}')" \
  >"${OUT_DIR}/${EXPERIMENT_NAME}-registration.json"

benchmark_body="$(jq -nc \
  --argjson cfg "${cfg}" \
  --argjson requests "${REQUESTS_PER_LEVEL}" \
  --argjson max_tokens "${MAX_TOKENS}" \
  --argjson frequency_penalty "${FREQUENCY_PENALTY}" \
  --argjson stop "${STOP_JSON}" \
  --argjson concurrency "${CONCURRENCY_LEVELS}" \
  --argjson contexts "${CONTEXT_LENGTHS}" \
  '{endpoint_id:"qwen36-27b-awq-vllm",dataset:"DianJin/DianJin-CSC-Data",workload:"customer_support_shared_prefix",routing_strategy:"direct",concurrency_levels:$concurrency,context_lengths:$contexts,requests_per_level:$requests,max_tokens:$max_tokens,vllm:{max_num_seqs:$cfg.max_num_seqs,max_num_batched_tokens:$cfg.max_num_batched_tokens,gpu_memory_utilization:$cfg.gpu_memory_utilization,max_model_len:$cfg.max_model_len,prefix_caching:$cfg.enable_prefix_caching,chunked_prefill:$cfg.enable_chunked_prefill,enable_thinking:false,quantization:"awq-w4a16",frequency_penalty:$frequency_penalty,stop:$stop,disable_custom_all_reduce:$cfg.disable_custom_all_reduce}}')"

printf '%s' "${benchmark_body}" >"${OUT_DIR}/${EXPERIMENT_NAME}-request.json"
submission="$(curl -fsS -X POST "${API_BASE}/api/benchmarks/serving" -H 'Content-Type: application/json' --data "${benchmark_body}")"
printf '%s' "${submission}" >"${OUT_DIR}/${EXPERIMENT_NAME}-submission.json"
run_id="$(jq -er '.run_id' <<<"${submission}")"

wait_for_run "${run_id}"
