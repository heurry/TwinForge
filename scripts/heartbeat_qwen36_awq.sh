#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:8081}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8021}"
INSTANCE_NAME="${INSTANCE_NAME:-qwen36-27b-awq-vllm}"
LOCK_FILE="${LOCK_FILE:-/tmp/twinforge-qwen36-awq-heartbeat.lock}"

exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

while docker inspect -f '{{.State.Running}}' twinforge-vllm-qwen36-awq 2>/dev/null | grep -qx true; do
  if curl -fsS "${VLLM_BASE}/health" >/dev/null; then
    curl -fsS -X POST "${API_BASE}/api/service-instances/${INSTANCE_NAME}/heartbeat" >/dev/null || true
  fi
  sleep 10
done
