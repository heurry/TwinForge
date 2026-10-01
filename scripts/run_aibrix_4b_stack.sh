#!/usr/bin/env bash

# NVIDIA_PLUGIN_IMAGE=ghcr.io/nvidia/k8s-device-plugin:e79cad2f bash scripts/run_aibrix_4b_stack.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

cd "${ROOT_DIR}"

if [[ "${EUID}" -eq 0 && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
  printf '[INFO] Do not run this stack script with sudo. Re-executing as %s so minikube uses the correct profile.\n' "${SUDO_USER}" >&2
  exec sudo -u "${SUDO_USER}" -H bash "$0" "$@"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/aibrix_4b/${TS}}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/Qwen3.5-4B}"
MODEL_REAL_PATH="$(readlink -f "${MODEL_PATH}")"
MINIKUBE_MODEL_PATH="${MINIKUBE_MODEL_PATH:-${ROOT_DIR}/model/Qwen3.5-4B}"
AIBRIX_VERSION="${AIBRIX_VERSION:-v0.6.0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-customer}"
AIBRIX_PORT="${AIBRIX_PORT:-8010}"
AIBRIX_NODE_PORT="${AIBRIX_NODE_PORT:-30080}"
# port-forward 监听地址（逗号分隔列表）：默认同时绑回环 + docker 网桥网关，不暴露到局域网。
# - 127.0.0.1：脚本自身的健康探针（wait_http / curl）与宿主访问走回环，必须保留；
# - 172.17.0.1：compose 容器经 host.docker.internal（= 网桥网关）访问网关 / cAdvisor / vLLM。
# kubectl port-forward 默认只绑 127.0.0.1，容器侧会 Connection refused，故显式补网桥地址。
# 网桥网关非 172.17.0.1 时按 `docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}'`
# 调整；想暴露所有网卡可设 PF_BIND_ADDRESS=0.0.0.0。
PF_BIND_ADDRESS="${PF_BIND_ADDRESS:-127.0.0.1,172.17.0.1}"
CADVISOR_PORT="${CADVISOR_PORT:-18081}"
CADVISOR_TIMEOUT="${CADVISOR_TIMEOUT:-120s}"
START_MINIKUBE_IF_STOPPED="${START_MINIKUBE_IF_STOPPED:-1}"
MINIKUBE_STATUS_TIMEOUT="${MINIKUBE_STATUS_TIMEOUT:-30s}"
MINIKUBE_CPUS="${MINIKUBE_CPUS:-12}"
MINIKUBE_MEMORY="${MINIKUBE_MEMORY:-24576}"
MINIKUBE_IMAGE_REPOSITORY="${MINIKUBE_IMAGE_REPOSITORY:-}"
NVIDIA_PLUGIN_VERSION="${NVIDIA_PLUGIN_VERSION:-v0.17.1}"
NVIDIA_PLUGIN_IMAGE="${NVIDIA_PLUGIN_IMAGE:-}"
GPU_ALLOCATABLE_TIMEOUT="${GPU_ALLOCATABLE_TIMEOUT:-120}"
ENABLE_CADVISOR="${ENABLE_CADVISOR:-1}"
SKIP_IMAGE_PREFETCH="${SKIP_IMAGE_PREFETCH:-0}"
SKIP_IMAGE_BUILD="${SKIP_IMAGE_BUILD:-0}"
FORCE_MINIKUBE_IMAGE_LOAD="${FORCE_MINIKUBE_IMAGE_LOAD:-0}"
SKIP_BACKEND_REDEPLOY="${SKIP_BACKEND_REDEPLOY:-0}"
SYNC_MODEL="${SYNC_MODEL:-1}"
VLLM_IMAGE="${VLLM_IMAGE:-local/vllm-openai:qwen35-v0.19.1-deepepfix}"
VLLM_DOCKERFILE="${VLLM_DOCKERFILE:-deploy/aibrix/Dockerfile.vllm-qwen35}"
STACK_TMPDIR="${STACK_TMPDIR:-/mnt/nvme-data/tmp}"

if ! mkdir -p "${STACK_TMPDIR}" 2>/dev/null || [[ ! -w "${STACK_TMPDIR}" ]]; then
  printf '[WARN] STACK_TMPDIR is not writable: %s. Falling back to %s/runs/tmp\n' "${STACK_TMPDIR}" "${ROOT_DIR}" >&2
  STACK_TMPDIR="${ROOT_DIR}/runs/tmp"
  mkdir -p "${STACK_TMPDIR}"
fi
export TMPDIR="${STACK_TMPDIR}"
export CADVISOR_URL="${CADVISOR_URL:-http://127.0.0.1:${CADVISOR_PORT}/metrics}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run() {
  log "+ $*"
  "$@"
}

capture_diagnostics() {
  local status="${1:-unknown}"
  log "capturing diagnostics, status=${status}, log_dir=${LOG_DIR}"
  kubectl get pods -A -o wide > "${LOG_DIR}/kubectl_pods.txt" 2>&1 || true
  kubectl get svc -A -o wide > "${LOG_DIR}/kubectl_services.txt" 2>&1 || true
  kubectl get gateway,httproute,gatewayclass -A > "${LOG_DIR}/kubectl_gateway_api.txt" 2>&1 || true
  kubectl get events -A --sort-by=.lastTimestamp > "${LOG_DIR}/kubectl_events.txt" 2>&1 || true
  kubectl describe deployment "${SERVED_MODEL_NAME}" -n default > "${LOG_DIR}/describe_${SERVED_MODEL_NAME}_deployment.txt" 2>&1 || true
  kubectl describe pod -n default -l "app=${SERVED_MODEL_NAME}" > "${LOG_DIR}/describe_${SERVED_MODEL_NAME}_pods.txt" 2>&1 || true
  kubectl logs -n default -l "app=${SERVED_MODEL_NAME}" --all-containers --tail=300 > "${LOG_DIR}/logs_${SERVED_MODEL_NAME}.txt" 2>&1 || true
  kubectl get pods -n default -l "app=${SERVED_MODEL_NAME}" -o yaml > "${LOG_DIR}/pods_${SERVED_MODEL_NAME}.yaml" 2>&1 || true
  kubectl logs -n aibrix-system -l app=gateway-plugins --all-containers --tail=300 > "${LOG_DIR}/logs_aibrix_gateway_plugins.txt" 2>&1 || true
  kubectl logs -n aibrix-system -l app=metadata-service --all-containers --tail=300 > "${LOG_DIR}/logs_aibrix_metadata_service.txt" 2>&1 || true
  kubectl get pods -n observability -o wide > "${LOG_DIR}/kubectl_observability_pods.txt" 2>&1 || true
  kubectl logs -n observability -l app=cadvisor --all-containers --tail=200 > "${LOG_DIR}/logs_cadvisor.txt" 2>&1 || true
  nvidia-smi > "${LOG_DIR}/nvidia_smi.txt" 2>&1 || true
  ss -ltnp > "${LOG_DIR}/listening_ports.txt" 2>&1 || true
  docker ps > "${LOG_DIR}/docker_ps.txt" 2>&1 || true
}

on_error() {
  local status=$?
  log "failed with exit code ${status}"
  capture_diagnostics "${status}"
  exit "${status}"
}
trap on_error ERR

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing command: $1"
    exit 127
  fi
}

kill_port() {
  local port="$1"
  local pids
  pids="$(ss -ltnp 2>/dev/null | sed -n "s/.*127\\.0\\.0\\.1:${port}.*pid=\\([0-9][0-9]*\\).*/\\1/p" | sort -u || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi
  for pid in ${pids}; do
    log "stopping pid ${pid} on 127.0.0.1:${port}"
    kill "${pid}" 2>/dev/null || true
  done
  sleep 2
}

wait_http() {
  local url="$1"
  local timeout="${2:-180}"
  local start
  start="$(date +%s)"
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if (( "$(date +%s)" - start > timeout )); then
      log "timeout waiting for ${url}"
      return 1
    fi
    sleep 2
  done
}

ensure_gpu_device_plugin() {
  log "checking NVIDIA device plugin"
  if kubectl get ds -n kube-system nvidia-device-plugin-daemonset >/dev/null 2>&1; then
    log "NVIDIA device plugin DaemonSet exists"
  else
    log "installing NVIDIA device plugin ${NVIDIA_PLUGIN_VERSION}"
    run kubectl apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${NVIDIA_PLUGIN_VERSION}/deployments/static/nvidia-device-plugin.yml"
  fi

  if [[ -n "${NVIDIA_PLUGIN_IMAGE}" ]]; then
    log "overriding NVIDIA device plugin image: ${NVIDIA_PLUGIN_IMAGE}"
    run kubectl -n kube-system set image ds/nvidia-device-plugin-daemonset \
      nvidia-device-plugin-ctr="${NVIDIA_PLUGIN_IMAGE}"
  fi

  if ! kubectl -n kube-system rollout status ds/nvidia-device-plugin-daemonset --timeout=180s; then
    log "NVIDIA device plugin did not become ready"
    kubectl -n kube-system describe pod -l name=nvidia-device-plugin-ds | tail -n 120 || true
    return 1
  fi

  log "waiting for nvidia.com/gpu to become allocatable"
  local gpu_allocatable=""
  local gpu_wait_start
  gpu_wait_start="$(date +%s)"
  while (( "$(date +%s)" - gpu_wait_start <= GPU_ALLOCATABLE_TIMEOUT )); do
    gpu_allocatable="$(kubectl get node minikube -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)"
    if [[ "${gpu_allocatable}" =~ ^[1-9][0-9]*$ ]]; then
      log "minikube GPU allocatable nvidia.com/gpu=${gpu_allocatable}"
      return 0
    fi
    sleep 2
  done

  log "nvidia.com/gpu is not allocatable on minikube after ${GPU_ALLOCATABLE_TIMEOUT}s"
  kubectl describe node minikube | grep -i "nvidia.com/gpu" -A4 || true
  return 1
}

ensure_minikube_running() {
  local status_output
  local status_code
  local status_file="${LOG_DIR}/minikube_status.txt"
  log "checking Docker daemon access"
  if ! docker ps >/dev/null 2>&1; then
    log "current user cannot access Docker daemon. Add user to docker group and re-login:"
    log "  sudo usermod -aG docker ${USER}"
    return 1
  fi

  log "checking minikube status"
  if timeout "${MINIKUBE_STATUS_TIMEOUT}" minikube status > "${status_file}" 2>&1; then
    status_code=0
  else
    status_code=$?
  fi
  status_output="$(cat "${status_file}" 2>/dev/null || true)"

  if [[ "${status_code}" -eq 124 ]]; then
    log "minikube status timed out after ${MINIKUBE_STATUS_TIMEOUT}"
    printf '%s\n' "${status_output}"
    return 124
  fi

  if [[ "${status_code}" -eq 0 ]]; then
    log "minikube is already running"
    printf '%s\n' "${status_output}"
    ensure_gpu_device_plugin
    return 0
  fi

  log "minikube is not running, status_code=${status_code}"
  printf '%s\n' "${status_output}"

  if [[ "${START_MINIKUBE_IF_STOPPED}" != "1" ]]; then
    log "START_MINIKUBE_IF_STOPPED=${START_MINIKUBE_IF_STOPPED}; not starting minikube automatically"
    return "${status_code}"
  fi

  if ! nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi is not available; cannot start GPU minikube"
    return 1
  fi

  local args=(
    --driver=docker
    --container-runtime=docker
    --gpus all
    --cpus "${MINIKUBE_CPUS}"
    --memory "${MINIKUBE_MEMORY}"
  )
  if [[ -n "${MINIKUBE_IMAGE_REPOSITORY}" ]]; then
    args+=(--image-repository "${MINIKUBE_IMAGE_REPOSITORY}")
  fi
  local proxy_var
  for proxy_var in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    if [[ -n "${!proxy_var:-}" ]]; then
      args+=(--docker-env "${proxy_var}=${!proxy_var}")
    fi
  done

  # 反向检查：minikube 未运行，但其保留的控制面 IP 192.168.49.2 已被非 minikube 容器占用
  # （通常是先跑了 run_full_stack.sh，go-server 抢到了它）→ `minikube start` 必撞
  # "Address already in use"。先于 start 快速失败，给出释放指引。
  if docker network inspect minikube >/dev/null 2>&1; then
    local ip_holder
    ip_holder="$(docker network inspect minikube \
      -f '{{range .Containers}}{{if eq .IPv4Address "192.168.49.2/24"}}{{.Name}} {{end}}{{end}}' 2>/dev/null || true)"
    ip_holder="${ip_holder% }"
    if [[ -n "${ip_holder}" && "${ip_holder}" != "minikube" ]]; then
      log "[FATAL] 192.168.49.2 (minikube's reserved control-plane IP) is held by: ${ip_holder}"
      log "        'minikube start' would fail with 'Address already in use'."
      log "        This usually means the Go stack came up before minikube and grabbed it."
      log "        Release it first, then re-run in the correct order:"
      log "          docker compose -f deploy/compose/docker-compose.yml down"
      log "          bash scripts/run_aibrix_4b_stack.sh"
      log "          bash scripts/run_full_stack.sh"
      return 1
    fi
  fi

  log "starting GPU minikube"
  run minikube start "${args[@]}"
  ensure_gpu_device_plugin
  run minikube status
}

ensure_host_image() {
  local image="$1"
  local mirror="${2:-}"
  if docker image inspect "${image}" >/dev/null 2>&1; then
    return 0
  fi
  if docker pull "${image}"; then
    return 0
  fi
  if [[ -n "${mirror}" ]]; then
    run docker pull "${mirror}"
    run docker tag "${mirror}" "${image}"
    return 0
  fi
  return 1
}

ensure_minikube_image() {
  local image="$1"
  local mirror="${2:-}"
  if minikube image ls | grep -F "${image}" >/dev/null 2>&1; then
    return 0
  fi
  ensure_host_image "${image}" "${mirror}"
  run minikube image load "${image}"
}

ensure_vllm_runtime_image() {
  if [[ ! -f "${VLLM_DOCKERFILE}" ]]; then
    log "missing Dockerfile: ${VLLM_DOCKERFILE}"
    exit 2
  fi

  if docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
    log "host image exists: ${VLLM_IMAGE}"
  elif [[ "${SKIP_IMAGE_BUILD}" = "1" ]]; then
    log "SKIP_IMAGE_BUILD=1 but image is missing: ${VLLM_IMAGE}"
    exit 2
  else
    log "building vLLM runtime image: ${VLLM_IMAGE}"
    log "docker build log: ${LOG_DIR}/docker_build_vllm_image.log"
    docker build -f "${VLLM_DOCKERFILE}" -t "${VLLM_IMAGE}" deploy/aibrix \
      2>&1 | tee "${LOG_DIR}/docker_build_vllm_image.log"
  fi

  if [[ "${FORCE_MINIKUBE_IMAGE_LOAD}" != "1" ]] && minikube image ls | grep -F "${VLLM_IMAGE}" >/dev/null 2>&1; then
    log "minikube image exists: ${VLLM_IMAGE}"
  else
    run minikube image load --overwrite=true "${VLLM_IMAGE}"
  fi
}

install_aibrix_if_needed() {
  if kubectl get ns aibrix-system >/dev/null 2>&1; then
    log "aibrix-system namespace exists, skip AIBrix manifest install"
  else
    log "installing AIBrix ${AIBRIX_VERSION}"
    run kubectl apply -f "https://github.com/vllm-project/aibrix/releases/download/${AIBRIX_VERSION}/aibrix-dependency-${AIBRIX_VERSION}.yaml" --server-side
    run kubectl apply -f "https://github.com/vllm-project/aibrix/releases/download/${AIBRIX_VERSION}/aibrix-core-${AIBRIX_VERSION}.yaml"
  fi

  run kubectl rollout status deployment/aibrix-controller-manager -n aibrix-system --timeout=300s
  run kubectl rollout status deployment/aibrix-gateway-plugins -n aibrix-system --timeout=300s
  run kubectl rollout status deployment/aibrix-metadata-service -n aibrix-system --timeout=300s
  run kubectl rollout status deployment/aibrix-redis-master -n aibrix-system --timeout=300s
  run kubectl rollout status deployment/envoy-gateway -n envoy-gateway-system --timeout=300s
}

sync_model_into_minikube() {
  if [[ ! -f "${MODEL_REAL_PATH}/config.json" ]]; then
    log "model path does not contain config.json: ${MODEL_REAL_PATH}"
    exit 2
  fi
  if [[ "${SYNC_MODEL}" != "1" ]]; then
    log "SYNC_MODEL=${SYNC_MODEL}, skip model sync"
    return 0
  fi
  log "syncing model into minikube"
  log "source: ${MODEL_REAL_PATH}"
  log "target: minikube:${MINIKUBE_MODEL_PATH}"
  run minikube ssh -- "sudo rm -rf '${MINIKUBE_MODEL_PATH}' && sudo mkdir -p '${MINIKUBE_MODEL_PATH}'"
  run docker cp "${MODEL_REAL_PATH}/." "minikube:${MINIKUBE_MODEL_PATH}/"
  run minikube ssh -- "test -f '${MINIKUBE_MODEL_PATH}/config.json'"
}

deploy_4b_backend() {
  if [[ "${SKIP_BACKEND_REDEPLOY}" = "1" ]]; then
    log "SKIP_BACKEND_REDEPLOY=1, verify existing 4B backend"
    run kubectl rollout status "deployment/${SERVED_MODEL_NAME}" -n default --timeout=1200s
    run kubectl get endpoints "${SERVED_MODEL_NAME}" -n default -o wide
    return 0
  fi

  log "removing any existing 4B backend before redeploy"
  kubectl delete deployment "${SERVED_MODEL_NAME}" -n default --ignore-not-found=true --wait=true
  kubectl wait --for=delete pod -n default -l "app=${SERVED_MODEL_NAME}" --timeout=300s >/dev/null 2>&1 || true

  log "applying 4B model PV and vLLM deployment"
  run kubectl apply -f deploy/aibrix/qwen35-4b-model-pv.yaml
  run kubectl apply -f deploy/aibrix/vllm-qwen35-4b-deployment.yaml
  run kubectl rollout status "deployment/${SERVED_MODEL_NAME}" -n default --timeout=1200s
  run kubectl get endpoints "${SERVED_MODEL_NAME}" -n default -o wide
}

start_aibrix_port_forward() {
  kill_port "${AIBRIX_PORT}"
  local service
  service="$(kubectl get svc -n envoy-gateway-system -o name | grep 'envoy-aibrix-system-aibrix-eg' | head -1)"
  if [[ -z "${service}" ]]; then
    log "AIBrix Envoy service not found"
    return 1
  fi
  log "starting AIBrix port-forward ${service} ${AIBRIX_PORT}:80 (--address ${PF_BIND_ADDRESS})"
  nohup kubectl port-forward --address "${PF_BIND_ADDRESS}" -n envoy-gateway-system "${service}" "${AIBRIX_PORT}:80" \
    > "${LOG_DIR}/aibrix_port_forward.log" 2>&1 &
  echo "$!" > "${LOG_DIR}/aibrix_port_forward.pid"
  sleep 3
  ss -ltnp | grep ":${AIBRIX_PORT} "
}

start_cadvisor_observability() {
  if [[ "${ENABLE_CADVISOR}" != "1" ]]; then
    log "ENABLE_CADVISOR=${ENABLE_CADVISOR}, skip cAdvisor"
    return 0
  fi
  if [[ ! -f deploy/observability/cadvisor.yaml ]]; then
    log "missing deploy/observability/cadvisor.yaml, skip cAdvisor"
    return 0
  fi

  log "starting cAdvisor observability, url=${CADVISOR_URL}"
  if ! kubectl apply -f deploy/observability/cadvisor.yaml; then
    log "cAdvisor manifest apply failed, continue without cAdvisor"
    return 0
  fi
  if ! kubectl rollout status daemonset/cadvisor -n observability --timeout="${CADVISOR_TIMEOUT}"; then
    log "cAdvisor DaemonSet is not ready, continue without cAdvisor"
    return 0
  fi

  kill_port "${CADVISOR_PORT}"
  local attempt
  for attempt in $(seq 1 6); do
    log "starting cAdvisor port-forward service/cadvisor ${CADVISOR_PORT}:8080, attempt=${attempt} (--address ${PF_BIND_ADDRESS})"
    nohup kubectl port-forward --address "${PF_BIND_ADDRESS}" -n observability service/cadvisor "${CADVISOR_PORT}:8080" \
      > "${LOG_DIR}/cadvisor_port_forward.log" 2>&1 &
    local pf_pid="$!"
    echo "${pf_pid}" > "${LOG_DIR}/cadvisor_port_forward.pid"
    local probe
    for probe in $(seq 1 15); do
      if curl -fsS "${CADVISOR_URL}" >/dev/null 2>&1; then
        log "cAdvisor metrics are reachable: ${CADVISOR_URL}"
        return 0
      fi
      if ! kill -0 "${pf_pid}" 2>/dev/null; then
        log "cAdvisor port-forward exited before metrics became ready"
        break
      fi
      sleep 2
    done
    kill_port "${CADVISOR_PORT}"
  done
  log "cAdvisor metrics are not reachable, dashboard will show unavailable"
}

start_direct_round_robin_port_forwards() {
  local pods=()
  mapfile -t pods < <(
    kubectl get pods -n default -l "app=${SERVED_MODEL_NAME}" \
      --field-selector=status.phase=Running \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' \
      | awk '$2 == "True" {print $1}' \
      | sort
  )
  local desired_replicas
  desired_replicas="$(kubectl get deployment "${SERVED_MODEL_NAME}" -n default -o jsonpath='{.spec.replicas}')"
  if [[ -z "${desired_replicas}" || "${desired_replicas}" -lt 1 ]]; then
    log "deployment ${SERVED_MODEL_NAME} has no desired replicas"
    return 1
  fi
  if [[ "${#pods[@]}" -lt "${desired_replicas}" ]]; then
    log "expected ${desired_replicas} ready backend pod(s) for ${SERVED_MODEL_NAME}, found ${#pods[@]}"
    kubectl get pods -n default -l "app=${SERVED_MODEL_NAME}" -o wide || true
    return 1
  fi

  kill_port 8000
  kill_port 8001

  local idx
  for ((idx = 0; idx < desired_replicas; idx++)); do
    local pod="${pods[$idx]}"
    local port="$((8000 + idx))"
    log "starting round-robin port-forward pod/${pod} ${port}:8000 (--address ${PF_BIND_ADDRESS})"
    nohup kubectl port-forward --address "${PF_BIND_ADDRESS}" -n default "pod/${pod}" "${port}:8000" \
      > "${LOG_DIR}/vllm_replica_${idx}.log" 2>&1 &
    echo "$!" > "${LOG_DIR}/vllm_replica_${idx}.pid"
  done

  sleep 3
  for ((idx = 0; idx < desired_replicas; idx++)); do
    wait_http "http://127.0.0.1:$((8000 + idx))/health" 120
  done
  ss -ltnp | rg ':800[0-9] '
}

main() {
  log "log_dir=${LOG_DIR}"
  log "model_path=${MODEL_PATH}"
  log "model_real_path=${MODEL_REAL_PATH}"
  log "tmpdir=${TMPDIR}"
  log "cadvisor_url=${CADVISOR_URL}"

  need_cmd kubectl
  need_cmd minikube
  need_cmd docker
  need_cmd curl
  need_cmd ss
  need_cmd nvidia-smi

  ensure_minikube_running
  run kubectl get nodes -o wide

  if [[ "${SKIP_IMAGE_PREFETCH}" != "1" ]]; then
    ensure_minikube_image "docker.io/envoyproxy/gateway:v1.2.8" "swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/envoyproxy/gateway:v1.2.8"
    ensure_vllm_runtime_image
  else
    log "SKIP_IMAGE_PREFETCH=1, skip image checks"
  fi

  install_aibrix_if_needed
  # 正式应用链路通过固定 NodePort 访问，不依赖易失的 kubectl
  # port-forward。8010 仅保留给宿主机人工调试。
  run kubectl apply -f deploy/aibrix/twinforge-aibrix-gateway-service.yaml
  sync_model_into_minikube
  deploy_4b_backend
  start_direct_round_robin_port_forwards
  start_aibrix_port_forward
  start_cadvisor_observability
  capture_diagnostics "success"

  log "AIBrix 4B serving layer is ready"
  log "AIBrix stable application endpoint: http://minikube:${AIBRIX_NODE_PORT}/v1"
  log "AIBrix Gateway: http://127.0.0.1:${AIBRIX_PORT}/v1"
  log "vLLM direct endpoint(s): http://127.0.0.1:8000 ... (one port per desired replica)"
  log "cAdvisor:       ${CADVISOR_URL}"
  log "Logs:           ${LOG_DIR}"
  log "Next: bring up the Go control plane + dashboard with scripts/run_full_stack.sh"
}

main "$@"
