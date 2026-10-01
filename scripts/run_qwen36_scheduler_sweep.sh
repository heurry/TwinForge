#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_BASE="${API_BASE:-http://localhost:8081}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/logs/inference/scheduler-sweep}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-16}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
PIPELINE_LAYER_PARTITION="${PIPELINE_LAYER_PARTITION:-}"
ENABLE_DBO="${ENABLE_DBO:-false}"
DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-32}"
DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-512}"
CONTEXT_LENGTHS_JSON="${CONTEXT_LENGTHS_JSON:-[1024,2048]}"
CONCURRENCY_LEVELS_JSON="${CONCURRENCY_LEVELS_JSON:-[8,16]}"
MAX_NUM_PARTIAL_PREFILLS="${MAX_NUM_PARTIAL_PREFILLS:-1}"
MAX_LONG_PARTIAL_PREFILLS="${MAX_LONG_PARTIAL_PREFILLS:-1}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-0}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-fcfs}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"
PRIORITY_BY_CONTEXT="${PRIORITY_BY_CONTEXT:-false}"
CONTEXT_MIX_JSON="${CONTEXT_MIX_JSON:-null}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
PROFILING="${PROFILING:-false}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-true}"
SCHEDULER_RESERVE_FULL_ISL="${SCHEDULER_RESERVE_FULL_ISL:-true}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
SPECULATIVE_DECODING="${SPECULATIVE_DECODING:-none}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-600}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-900}"
REUSE_RUNTIME="${REUSE_RUNTIME:-false}"
PROFILER_CACHE_DIR="${PROFILER_CACHE_DIR:-${ROOT_DIR}/.cache/vllm/qwen36-27b-fp8/profiler}"

mkdir -p "${OUT_DIR}"

wait_for_runtime() {
  local deadline=$((SECONDS + START_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    local body status
    body="$(curl -fsS "${API_BASE}/api/inference/runtime")"
    status="$(jq -r '.status' <<<"${body}")"
    printf '%s runtime=%s\n' "$(date '+%H:%M:%S')" "${status}" >&2
    if [[ "${status}" == "ready" ]]; then
      printf '%s' "${body}"
      return 0
    fi
    [[ "${status}" != "error" && "${status}" != "stopped" ]] || return 1
    sleep 10
  done
  return 1
}

wait_for_run() {
  local run_id="$1" result_path="$2" events_path="$3"
  local deadline=$((SECONDS + RUN_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    local body status
    body="$(curl -fsS "${API_BASE}/api/benchmarks/${run_id}")"
    status="$(jq -r '.status' <<<"${body}")"
    printf '%s run=%s status=%s\n' "$(date '+%H:%M:%S')" "${run_id:0:8}" "${status}"
    if [[ "${status}" == "completed" ]]; then
      printf '%s' "${body}" >"${result_path}"
      curl -fsS "${API_BASE}/api/benchmarks/${run_id}/events" >"${events_path}"
      return 0
    fi
    [[ "${status}" != "failed" && "${status}" != "cancelled" ]] || return 1
    sleep 10
  done
  return 1
}

run_candidate() {
  local seqs="$1"
  local name="${EXPERIMENT_NAME:-tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-seqs${seqs}-tokens${MAX_BATCHED_TOKENS}}"

  local runtime_body
  runtime_body="$(jq -nc \
    --argjson seqs "${seqs}" --argjson tokens "${MAX_BATCHED_TOKENS}" \
    --arg policy "${SCHEDULING_POLICY}" \
    --argjson partial "${MAX_NUM_PARTIAL_PREFILLS}" --argjson long_partial "${MAX_LONG_PARTIAL_PREFILLS}" \
	  --argjson threshold "${LONG_PREFILL_TOKEN_THRESHOLD}" --argjson profiling "${PROFILING}" \
	  --argjson disable_custom_ar "${DISABLE_CUSTOM_ALL_REDUCE}" --argjson reserve_isl "${SCHEDULER_RESERVE_FULL_ISL}" \
	  --argjson gpu_memory "${GPU_MEMORY_UTILIZATION}" --argjson model_len "${MAX_MODEL_LEN}" --arg kv_dtype "${KV_CACHE_DTYPE}" \
	  --argjson tensor_parallel "${TENSOR_PARALLEL_SIZE}" --argjson pipeline_parallel "${PIPELINE_PARALLEL_SIZE}" --arg layer_partition "${PIPELINE_LAYER_PARTITION}" \
	  --argjson enable_dbo "${ENABLE_DBO}" --argjson dbo_decode "${DBO_DECODE_TOKEN_THRESHOLD}" --argjson dbo_prefill "${DBO_PREFILL_TOKEN_THRESHOLD}" \
	  --arg speculation "${SPECULATIVE_DECODING}" \
	  '{profile:"scheduler",tensor_parallel_size:$tensor_parallel,pipeline_parallel_size:$pipeline_parallel,pipeline_layer_partition:$layer_partition,enable_dbo:$enable_dbo,dbo_decode_token_threshold:$dbo_decode,dbo_prefill_token_threshold:$dbo_prefill,max_num_seqs:$seqs,max_num_batched_tokens:$tokens,prefix_caching:true,scheduling_policy:$policy,max_num_partial_prefills:$partial,max_long_partial_prefills:$long_partial,long_prefill_token_threshold:$threshold,scheduler_reserve_full_isl:$reserve_isl,disable_custom_all_reduce:$disable_custom_ar,stream_interval:1,profiling:$profiling,gpu_memory_utilization:$gpu_memory,max_model_len:$model_len,kv_cache_dtype:$kv_dtype,speculative_decoding:$speculation}')"
  runtime_body="$(jq -c --argjson async "${ASYNC_SCHEDULING}" '. + {async_scheduling:$async}' <<<"${runtime_body}")"
  if [[ "${REUSE_RUNTIME}" == "true" ]]; then
    curl -fsS "${API_BASE}/api/inference/runtime" >"${OUT_DIR}/${name}-ready.json"
  else
    curl -fsS -X DELETE "${API_BASE}/api/inference/runtime" >/dev/null
    curl -fsS -X POST "${API_BASE}/api/inference/runtime" -H 'Content-Type: application/json' \
      --data "${runtime_body}" >"${OUT_DIR}/${name}-start.json"
    if ! wait_for_runtime >"${OUT_DIR}/${name}-ready.json"; then
      curl -sS "${API_BASE}/api/inference/runtime" >"${OUT_DIR}/${name}-failed-status.json" || true
      docker logs twinforge-vllm-qwen36 >"${OUT_DIR}/${name}-failed-vllm.log" 2>&1 || true
      return 1
    fi
  fi

  local benchmark_body submission run_id
  benchmark_body="$(jq -nc \
    --argjson seqs "${seqs}" --argjson tokens "${MAX_BATCHED_TOKENS}" \
    --argjson requests "${REQUESTS_PER_LEVEL}" --argjson max_tokens "${MAX_TOKENS}" \
    --argjson context_mix "${CONTEXT_MIX_JSON}" --argjson priority "${PRIORITY_BY_CONTEXT}" \
    --arg policy "${SCHEDULING_POLICY}" --argjson partial "${MAX_NUM_PARTIAL_PREFILLS}" --argjson profiling "${PROFILING}" \
	  --argjson disable_custom_ar "${DISABLE_CUSTOM_ALL_REDUCE}" --argjson reserve_isl "${SCHEDULER_RESERVE_FULL_ISL}" \
	  --argjson gpu_memory "${GPU_MEMORY_UTILIZATION}" --argjson model_len "${MAX_MODEL_LEN}" --arg kv_dtype "${KV_CACHE_DTYPE}" \
	  --argjson tensor_parallel "${TENSOR_PARALLEL_SIZE}" --argjson pipeline_parallel "${PIPELINE_PARALLEL_SIZE}" --arg layer_partition "${PIPELINE_LAYER_PARTITION}" \
	  --argjson enable_dbo "${ENABLE_DBO}" --argjson dbo_decode "${DBO_DECODE_TOKEN_THRESHOLD}" --argjson dbo_prefill "${DBO_PREFILL_TOKEN_THRESHOLD}" \
	  --argjson context_lengths "${CONTEXT_LENGTHS_JSON}" --argjson concurrency_levels "${CONCURRENCY_LEVELS_JSON}" \
	  --arg speculation "${SPECULATIVE_DECODING}" \
    --argjson long_partial "${MAX_LONG_PARTIAL_PREFILLS}" --argjson threshold "${LONG_PREFILL_TOKEN_THRESHOLD}" \
	  '{endpoint_id:"qwen36-27b-fp8-vllm",dataset:"DianJin/DianJin-CSC-Data",workload:"customer_support_shared_prefix",routing_strategy:"direct",context_lengths:(if $context_mix == null then $context_lengths else [] end),context_mix:($context_mix // []),priority_by_context:$priority,concurrency_levels:(if $context_mix == null then $concurrency_levels else [16] end),requests_per_level:$requests,max_tokens:$max_tokens,vllm:{tensor_parallel_size:$tensor_parallel,pipeline_parallel_size:$pipeline_parallel,pipeline_layer_partition:$layer_partition,enable_dbo:$enable_dbo,dbo_decode_token_threshold:$dbo_decode,dbo_prefill_token_threshold:$dbo_prefill,max_num_seqs:$seqs,max_num_batched_tokens:$tokens,gpu_memory_utilization:$gpu_memory,max_model_len:$model_len,prefix_caching:true,chunked_prefill:true,enable_thinking:false,quantization:"fp8",scheduler:$policy,max_num_partial_prefills:$partial,max_long_partial_prefills:$long_partial,long_prefill_token_threshold:$threshold,scheduler_reserve_full_isl:$reserve_isl,stream_interval:1,disable_custom_all_reduce:$disable_custom_ar,kv_cache_dtype:$kv_dtype,profiling:$profiling,speculative_decoding:$speculation}}')"
	benchmark_body="$(jq -c --argjson async "${ASYNC_SCHEDULING}" '.vllm.async_scheduling=$async' <<<"${benchmark_body}")"
	if [[ "${PROFILING}" == "true" ]]; then
	  touch "${OUT_DIR}/${name}-profiler-started.marker"
	  curl -fsS -X POST http://127.0.0.1:8020/start_profile >/dev/null
	fi
	submission="$(curl -fsS -X POST "${API_BASE}/api/benchmarks/serving" -H 'Content-Type: application/json' --data "${benchmark_body}")"
  printf '%s' "${submission}" >"${OUT_DIR}/${name}-submission.json"
  run_id="$(jq -er '.run_id' <<<"${submission}")"
  local run_succeeded=true
  wait_for_run "${run_id}" "${OUT_DIR}/${name}-result.json" "${OUT_DIR}/${name}-events.json" || run_succeeded=false
  if [[ "${PROFILING}" == "true" ]]; then
    curl -fsS -X POST http://127.0.0.1:8020/stop_profile >/dev/null || true
    curl -fsS http://127.0.0.1:8020/metrics >"${OUT_DIR}/${name}-metrics.prom" || true
    mkdir -p "${OUT_DIR}/${name}-profiler"
    if [[ -d "${PROFILER_CACHE_DIR}" ]]; then
      find "${PROFILER_CACHE_DIR}" -type f -newer "${OUT_DIR}/${name}-profiler-started.marker" \
        -exec cp -t "${OUT_DIR}/${name}-profiler" {} +
    fi
  fi
  docker logs twinforge-vllm-qwen36 >"${OUT_DIR}/${name}-vllm.log" 2>&1 || true
  [[ "${run_succeeded}" == "true" ]]
}

if (( $# == 0 )); then
  set -- 8 12 16
fi
for seqs in "$@"; do
  run_candidate "${seqs}"
done
