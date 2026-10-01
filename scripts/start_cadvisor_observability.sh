#!/usr/bin/env bash
set -Eeuo pipefail

CADVISOR_PORT="${CADVISOR_PORT:-18081}"
CADVISOR_TIMEOUT="${CADVISOR_TIMEOUT:-300s}"
LOG_DIR="${LOG_DIR:-logs/observability}"
MANIFEST="${MANIFEST:-deploy/observability/cadvisor.yaml}"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
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

log "applying ${MANIFEST}"
kubectl apply -f "${MANIFEST}"

log "waiting for cAdvisor DaemonSet"
kubectl rollout status daemonset/cadvisor -n observability --timeout="${CADVISOR_TIMEOUT}"

for attempt in $(seq 1 6); do
  kill_port "${CADVISOR_PORT}"
  log "starting cAdvisor port-forward 127.0.0.1:${CADVISOR_PORT} -> service/cadvisor:8080, attempt=${attempt}"
  nohup kubectl port-forward -n observability service/cadvisor "${CADVISOR_PORT}:8080" \
    > "${LOG_DIR}/cadvisor_port_forward.log" 2>&1 &
  pf_pid="$!"
  echo "${pf_pid}" > "${LOG_DIR}/cadvisor_port_forward.pid"

  for probe in $(seq 1 15); do
    if curl -fsS "http://127.0.0.1:${CADVISOR_PORT}/metrics" >/dev/null 2>&1; then
      log "cAdvisor metrics: http://127.0.0.1:${CADVISOR_PORT}/metrics"
      exit 0
    fi
    if ! kill -0 "${pf_pid}" 2>/dev/null; then
      log "cAdvisor port-forward exited before metrics became ready"
      break
    fi
    sleep 2
  done
done

log "cAdvisor metrics did not become reachable"
tail -n 80 "${LOG_DIR}/cadvisor_port_forward.log" || true
exit 1
