#!/usr/bin/env bash

# TwinForge 应用层一键编排。
#
# 默认启动：Web/Nginx、Go 控制面、Python AI Service、本地 Embedding Service、
# Agent API/Worker、Go Node Agent、PostgreSQL/pgvector、Redis、MinIO。
# Chat 推理层不可用时可降级；Agent 语义记忆默认要求本地 embedding 为 live。
# 可选可观测栈：ENABLE_OBSERVABILITY=1（Prometheus、Tempo、Grafana）。
#
# 用法：
#   bash scripts/run_full_stack.sh [up|down|restart|status|logs]
#   SKIP_BUILD=1 bash scripts/run_full_stack.sh up
#   ENABLE_OBSERVABILITY=1 bash scripts/run_full_stack.sh up
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${ROOT_DIR}"

if [[ "${EUID}" -eq 0 && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
  printf '[INFO] Do not run this stack script with sudo. Re-executing as %s.\n' "${SUDO_USER}" >&2
  exec sudo -u "${SUDO_USER}" -H bash "$0" "$@"
fi

ACTION="${1:-up}"
case "${ACTION}" in
  up|start|down|stop|restart|status|logs) ;;
  *)
    printf 'Usage: %s [up|down|restart|status|logs]\n' "$0" >&2
    exit 2
    ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/full_stack/${TS}}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

COMPOSE_FILE="${COMPOSE_FILE:-deploy/compose/docker-compose.yml}"
SERVER_PORT="${SERVER_PORT:-8081}"
AI_PORT="${AI_PORT:-8200}"
AGENT_API_PORT="${AGENT_API_PORT:-8180}"
AGENT_WORKER_PORT="${AGENT_WORKER_PORT:-8181}"
NODE_AGENT_PORT="${NODE_AGENT_PORT:-8090}"
PG_PORT="${PG_PORT:-5432}"
WEB_PORT="${WEB_PORT:-5173}"
GATEWAY_PORT="${GATEWAY_PORT:-8010}"

SKIP_BUILD="${SKIP_BUILD:-0}"
ENABLE_OBSERVABILITY="${ENABLE_OBSERVABILITY:-0}"
REGEN_KUBECONFIG="${REGEN_KUBECONFIG:-0}"
ALLOW_MINIKUBE_DOWN="${ALLOW_MINIKUBE_DOWN:-0}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
REQUIRE_LIVE_EMBEDDING="${REQUIRE_LIVE_EMBEDDING:-1}"
EMBEDDING_MODEL_HOST_PATH="${EMBEDDING_MODEL_HOST_PATH:-/mnt/nvme-data/LLM/models/Qwen3-Embedding-0.6B}"
EMBEDDING_BASE_IMAGE="${EMBEDDING_BASE_IMAGE:-local/train:qwen35-v1}"
KUBECONFIG_SECRET="${KUBECONFIG_SECRET:-deploy/compose/.secrets/kubeconfig}"
WEB_PID_FILE="${WEB_PID_FILE:-.runtime/web.pid}"

COMPOSE=(docker compose -f "${COMPOSE_FILE}")
if [[ "${ENABLE_OBSERVABILITY}" = "1" ]]; then
  COMPOSE+=(--profile observability)
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run() {
  log "+ $*"
  "$@"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing command: $1"
    exit 127
  fi
}

capture_diagnostics() {
  local status="${1:-unknown}"
  log "capturing diagnostics, status=${status}, log_dir=${LOG_DIR}"
  "${COMPOSE[@]}" ps --all > "${LOG_DIR}/compose_ps.txt" 2>&1 || true
  "${COMPOSE[@]}" logs --no-color --tail=300 > "${LOG_DIR}/compose_logs.txt" 2>&1 || true
  ss -ltnp > "${LOG_DIR}/listening_ports.txt" 2>&1 || true
  docker ps > "${LOG_DIR}/docker_ps.txt" 2>&1 || true
}

on_error() {
  local status=$?
  log "failed with exit code ${status}"
  if grep -Eq 'lookup (registry-1\.docker\.io|gcr\.io).*i/o timeout' "${LOG_FILE}" 2>/dev/null; then
    log "Docker registry DNS timed out. Check host/Docker DNS, then retry."
    log "If all images were built previously, use: SKIP_BUILD=1 ./start.sh"
    log "For a local Web-only fallback, see apps/web/Dockerfile.runtime."
  fi
  capture_diagnostics "${status}"
  exit "${status}"
}
trap on_error ERR

wait_http_contains() {
  local url="$1"
  local expected="$2"
  local timeout="${3:-180}"
  local start body
  start="$(date +%s)"
  while true; do
    body="$(curl -fsS --max-time 5 -H 'Accept: application/json' "${url}" 2>/dev/null || true)"
    if [[ "${body}" == *"${expected}"* ]]; then
      return 0
    fi
    if (( "$(date +%s)" - start > timeout )); then
      log "timeout waiting for ${url}; expected response to contain: ${expected}"
      return 1
    fi
    sleep 2
  done
}

wait_live_embedding() {
  local timeout="${1:-180}"
  local start body
  start="$(date +%s)"
  while true; do
    body="$(curl -fsS --max-time 60 \
      -H 'Content-Type: application/json' \
      --data '{"texts":["Agent semantic memory readiness probe"],"is_query":false}' \
      "http://127.0.0.1:${AI_PORT}/internal/embed" 2>/dev/null || true)"
    if [[ "${body}" == *'"mode":"live"'* && "${body}" == *'"dim":1024'* ]]; then
      return 0
    fi
    if (( "$(date +%s)" - start > timeout )); then
      log "timeout waiting for live 1024-dimensional embedding through Python AI Service"
      return 1
    fi
    sleep 2
  done
}

stop_managed_host_web() {
  # 兼容旧脚本留下的宿主机 Vite 进程，只处理项目自己的 PID 文件，不抢杀任意端口进程。
  [[ -f "${WEB_PID_FILE}" ]] || return 0
  local pid cmdline
  pid="$(tr -cd '0-9' < "${WEB_PID_FILE}")"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    cmdline="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    if [[ "${cmdline}" == *"vite"* || "${cmdline}" == *"serve_web.sh"* || "${cmdline}" == *"npm run preview"* ]]; then
      log "stopping legacy host web process pid=${pid}; Compose/Nginx will own :${WEB_PORT}"
      kill "${pid}"
      for _ in {1..20}; do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.25
      done
    fi
  fi
  rm -f "${WEB_PID_FILE}"
}

stack_down() {
  log "stopping TwinForge application stack (persistent volumes are preserved)"
  stop_managed_host_web
  run "${COMPOSE[@]}" down --remove-orphans
  log "stack stopped; PostgreSQL/MinIO/Tempo volumes were preserved"
}

preflight() {
  need_cmd docker
  need_cmd curl
  need_cmd ss
  need_cmd jq

  if ! docker compose version >/dev/null 2>&1; then
    log "Docker Compose v2 plugin is required."
    exit 127
  fi
  if ! docker ps >/dev/null 2>&1; then
    log "current user cannot access the Docker daemon"
    exit 1
  fi
  if [[ ! -f "${COMPOSE_FILE}" ]]; then
    log "compose file not found: ${COMPOSE_FILE}"
    exit 2
  fi

  if [[ ! -f "${EMBEDDING_MODEL_HOST_PATH}/config.json" || ! -f "${EMBEDDING_MODEL_HOST_PATH}/model.safetensors" ]]; then
    log "local embedding model is incomplete: ${EMBEDDING_MODEL_HOST_PATH}"
    log "set EMBEDDING_MODEL_HOST_PATH to a complete Qwen3-Embedding-0.6B directory"
    exit 1
  fi
  if ! jq -e '.hidden_size == 1024' "${EMBEDDING_MODEL_HOST_PATH}/config.json" >/dev/null; then
    log "embedding model dimension is not the required 1024: ${EMBEDDING_MODEL_HOST_PATH}/config.json"
    exit 1
  fi
  if [[ "${EMBEDDING_BASE_IMAGE}" == local/* ]] && ! docker image inspect "${EMBEDDING_BASE_IMAGE}" >/dev/null 2>&1; then
    log "local embedding base image is missing: ${EMBEDDING_BASE_IMAGE}"
    log "build the local training runtime first, or set EMBEDDING_BASE_IMAGE to an available compatible image"
    exit 1
  fi
  "${COMPOSE[@]}" config --quiet

  # Compose 需要加入 minikube 外部网络，避免应用层抢占其保留地址。
  if ! docker network ls --format '{{.Name}}' | grep -qx minikube; then
    log "external Docker network 'minikube' not found"
    log "start the inference layer first: bash scripts/run_aibrix_4b_stack.sh"
    log "or, for application-only degraded mode: docker network create minikube"
    exit 1
  fi

  if docker inspect minikube >/dev/null 2>&1; then
    local mk_status
    mk_status="$(docker inspect -f '{{.State.Status}}' minikube 2>/dev/null || echo unknown)"
    if [[ "${mk_status}" != "running" && "${ALLOW_MINIKUBE_DOWN}" != "1" ]]; then
      log "minikube container is '${mk_status}', not running"
      log "start it first, or explicitly allow degraded mode: ALLOW_MINIKUBE_DOWN=1 ./start.sh"
      exit 1
    fi
    if [[ "${mk_status}" != "running" ]]; then
      log "[WARN] minikube is ${mk_status}; continuing in explicitly requested degraded mode"
    fi
  fi

  if [[ ! -f "${KUBECONFIG_SECRET}" ]]; then
    log "[WARN] kubeconfig missing: ${KUBECONFIG_SECRET}; Kubernetes reads will be degraded"
  fi
}

regen_kubeconfig() {
  [[ "${REGEN_KUBECONFIG}" = "1" ]] || return 0
  if ! command -v minikube >/dev/null 2>&1 || ! minikube status >/dev/null 2>&1; then
    log "[WARN] REGEN_KUBECONFIG=1 but minikube is unavailable; keeping the existing file"
    return 0
  fi
  need_cmd kubectl
  mkdir -p "$(dirname "${KUBECONFIG_SECRET}")"
  log "regenerating embedded-cert kubeconfig: ${KUBECONFIG_SECRET}"
  kubectl config view --minify --flatten > "${KUBECONFIG_SECRET}"
}

compose_up() {
  local args=(up -d --remove-orphans)
  local image
  stop_managed_host_web
  log "bringing up the complete application stack via Compose"

  if [[ "${SKIP_BUILD}" != "1" ]]; then
    if run env WEB_PORT="${WEB_PORT}" "${COMPOSE[@]}" "${args[@]}" --build; then
      "${COMPOSE[@]}" ps --all
      return 0
    fi

    # Registry/DNS 临时不可用时，已有的完整镜像集仍应能启动，而不是让控制台整体离线。
    while IFS= read -r image; do
      if ! docker image inspect "${image}" >/dev/null 2>&1; then
        log "build failed and required local image is missing: ${image}"
        return 1
      fi
    done < <("${COMPOSE[@]}" config --images)
    log "[WARN] image rebuild failed; all required local images exist, falling back to them"
    log "       retry with SKIP_BUILD=0 after Docker registry/network connectivity is restored"
  fi

  run env WEB_PORT="${WEB_PORT}" "${COMPOSE[@]}" "${args[@]}"
  "${COMPOSE[@]}" ps --all
}

wait_health() {
  log "waiting for local Qwen3 Embedding Service"
  run "${COMPOSE[@]}" exec -T embedding-service python3 -c \
    "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); assert data['status']=='ok' and data['dimension']==1024"
  log "waiting for Python AI Service"
  wait_http_contains "http://127.0.0.1:${AI_PORT}/internal/health" '"status"' "${HEALTH_TIMEOUT}"
  if [[ "${REQUIRE_LIVE_EMBEDDING}" = "1" ]]; then
    log "waiting for Python AI Service → local model live embedding chain"
    wait_live_embedding "${HEALTH_TIMEOUT}"
  fi
  log "waiting for Go control plane"
  wait_http_contains "http://127.0.0.1:${SERVER_PORT}/api/health" '"status":"ok"' "${HEALTH_TIMEOUT}"
  log "waiting for Agent API"
  wait_http_contains "http://127.0.0.1:${AGENT_API_PORT}/health/ready" '"status":"ready"' "${HEALTH_TIMEOUT}"
  log "waiting for isolated Agent Sandbox"
  run "${COMPOSE[@]}" exec -T agent-sandbox /app/agent-sandbox --healthcheck
  log "waiting for approved project access broker"
  run "${COMPOSE[@]}" exec -T agent-project-broker /app/agent-project-broker --healthcheck
  log "waiting for Agent Worker metrics"
  # 冷启动且尚无 Run 时业务指标可能还没有样本；Go runtime collector 始终存在。
  wait_http_contains "http://127.0.0.1:${AGENT_WORKER_PORT}/metrics" '# HELP go_' "${HEALTH_TIMEOUT}"
  log "waiting for Web/Nginx and validating both reverse-proxy routes"
  wait_http_contains "http://127.0.0.1:${WEB_PORT}/" '<div id="root">' "${HEALTH_TIMEOUT}"
  wait_http_contains "http://127.0.0.1:${WEB_PORT}/api/health" '"status":"ok"' "${HEALTH_TIMEOUT}"
  wait_http_contains "http://127.0.0.1:${WEB_PORT}/agent-api/health/ready" '"status":"ready"' "${HEALTH_TIMEOUT}"
}

smoke_test() {
  local capability_body
  log "smoke: Web → Go control plane proxy"
  run curl -fsS "http://127.0.0.1:${WEB_PORT}/api/health" | tee "${LOG_DIR}/go-health.json"
  echo

  log "smoke: Web → Agent Platform proxy and tenant-scoped capability API"
  capability_body="$(curl -fsS -H 'X-Tenant-ID: demo' "http://127.0.0.1:${WEB_PORT}/agent-api/api/v1/capabilities")"
  [[ "${capability_body}" == *'"data"'* ]]
  printf '%s\n' "${capability_body}" > "${LOG_DIR}/agent-capabilities.json"
  log "  /agent-api/api/v1/capabilities OK"

  log "smoke: PostgreSQL-backed service registry"
  curl -fsS "http://127.0.0.1:${WEB_PORT}/api/service-instances" >/dev/null
  log "  /api/service-instances OK"
}

show_status() {
  "${COMPOSE[@]}" ps --all
  echo
  local name url expected state body
  while IFS='|' read -r name url expected; do
    body="$(curl -fsS --max-time 3 "${url}" 2>/dev/null || true)"
    state="DOWN"
    [[ "${body}" == *"${expected}"* ]] && state="OK"
    printf '%-24s %-5s %s\n' "${name}" "${state}" "${url}"
  done <<EOF
Web UI|http://127.0.0.1:${WEB_PORT}/|<div id="root">
Web → Go API|http://127.0.0.1:${WEB_PORT}/api/health|"status":"ok"
Web → Agent API|http://127.0.0.1:${WEB_PORT}/agent-api/health/ready|"status":"ready"
Python AI Service|http://127.0.0.1:${AI_PORT}/internal/health|"status"
Agent Worker metrics|http://127.0.0.1:${AGENT_WORKER_PORT}/metrics|# HELP go_
EOF

  local sandbox_id sandbox_health
  sandbox_id="$("${COMPOSE[@]}" ps -q agent-sandbox)"
  sandbox_health="$(docker inspect -f '{{.State.Health.Status}}' "${sandbox_id}" 2>/dev/null || echo unavailable)"
  printf '%-24s %-5s %s\n' "Agent Sandbox" "$( [[ "${sandbox_health}" == "healthy" ]] && echo OK || echo DOWN )" "internal-only (${sandbox_health})"

  local broker_id broker_health
  broker_id="$("${COMPOSE[@]}" ps -q agent-project-broker)"
  broker_health="$(docker inspect -f '{{.State.Health.Status}}' "${broker_id}" 2>/dev/null || echo unavailable)"
  printf '%-24s %-5s %s\n' "Project Access Broker" "$( [[ "${broker_health}" == "healthy" ]] && echo OK || echo DOWN )" "internal-only (${broker_health})"

  local embedding_id embedding_health embedding_body embedding_mode
  embedding_id="$("${COMPOSE[@]}" ps -q embedding-service)"
  embedding_health="$(docker inspect -f '{{.State.Health.Status}}' "${embedding_id}" 2>/dev/null || echo unavailable)"
  printf '%-24s %-5s %s\n' "Embedding Service" "$( [[ "${embedding_health}" == "healthy" ]] && echo OK || echo DOWN )" "internal-only (${embedding_health})"
  embedding_body="$(curl -fsS --max-time 60 -H 'Content-Type: application/json' --data '{"texts":["readiness"],"is_query":false}' "http://127.0.0.1:${AI_PORT}/internal/embed" 2>/dev/null || true)"
  embedding_mode="$(jq -r '.mode // "unavailable"' <<<"${embedding_body}" 2>/dev/null || echo unavailable)"
  printf '%-24s %-5s %s\n' "Semantic Memory (${embedding_mode})" "$( [[ "${embedding_mode}" == "live" ]] && echo OK || echo DOWN )" "AI Service → Qwen3 → pgvector"

  body="$(curl -fsS --max-time 8 "http://127.0.0.1:${SERVER_PORT}/api/inference/releases?model_id=qwen3-4b-customer" 2>/dev/null || true)"
  state="$(jq -r '.serving_status.overall // "unavailable"' <<<"${body}" 2>/dev/null || echo unavailable)"
  printf '%-24s %-5s %s\n' "Serving/LLM (${state})" "$( [[ "${state}" == "ready" ]] && echo OK || echo DOWN )" "http://minikube:30080/v1"
}

summary() {
  local serving_state="degraded/optional (no live gateway detected)"
  local serving_body serving_overall
  serving_body="$(curl -fsS --max-time 8 "http://127.0.0.1:${SERVER_PORT}/api/inference/releases?model_id=qwen3-4b-customer" 2>/dev/null || true)"
  serving_overall="$(jq -r '.serving_status.overall // "unavailable"' <<<"${serving_body}" 2>/dev/null || echo unavailable)"
  [[ "${serving_overall}" == "ready" ]] && serving_state="live (workload + AIBrix gateway + model route ready)"
  log "================ TwinForge is ready ================"
  log "Web UI / API gateway:       http://127.0.0.1:${WEB_PORT}"
  log "Go control plane:           http://127.0.0.1:${SERVER_PORT}/api/health"
  log "Agent Platform API:         http://127.0.0.1:${AGENT_API_PORT}/health/ready"
  log "Agent Worker metrics:       http://127.0.0.1:${AGENT_WORKER_PORT}/metrics"
  log "Agent Sandbox:              internal-only / healthy"
  log "Python AI Service:          http://127.0.0.1:${AI_PORT}/internal/health"
  log "Embedding Service (live):   internal-only; verified through Python AI Service"
  log "Go Node Agent:              http://127.0.0.1:${NODE_AGENT_PORT}/healthz"
  log "PostgreSQL / Redis / MinIO: :${PG_PORT} / :6379 / :9000 (:9001 console)"
  log "Serving/LLM layer:          ${serving_state}"
  if [[ "${ENABLE_OBSERVABILITY}" = "1" ]]; then
    log "Grafana / Prometheus:       http://127.0.0.1:3000 / http://127.0.0.1:9090"
  fi
  log "Run logs:                   ${LOG_DIR}"
  log "Status / logs / stop:       ./start.sh status | logs | down"
  log "====================================================="
}

main() {
  case "${ACTION}" in
    down|stop)
      need_cmd docker
      stack_down
      return
      ;;
    status)
      need_cmd docker
      need_cmd curl
      need_cmd jq
      show_status
      return
      ;;
    logs)
      need_cmd docker
      "${COMPOSE[@]}" logs -f --tail="${TAIL_LINES:-200}"
      return
      ;;
    restart)
      need_cmd docker
      stack_down
      ;;
  esac

  log "log_dir=${LOG_DIR}"
  log "compose_file=${COMPOSE_FILE}"
  preflight
  regen_kubeconfig
  compose_up
  wait_health
  smoke_test
  capture_diagnostics "success"
  summary
}

main "$@"
