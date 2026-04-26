#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

cd "${ROOT_DIR}"
export TOKENIZERS_PARALLELISM=false
STATUS_FILE="runs/cpt/optimize/run_status.tsv"

mkdir -p "$(dirname "${STATUS_FILE}")"
printf "step\tstatus\texit_code\n" > "${STATUS_FILE}"

record_status() {
  local name="$1"
  local status="$2"
  local exit_code="$3"
  printf "%s\t%s\t%s\n" "${name}" "${status}" "${exit_code}" >> "${STATUS_FILE}"
}

run_step() {
  local name="$1"
  shift
  echo "[OPTIMIZE] ${name}"
  if "$@"; then
    record_status "${name}" "completed" "0"
    return 0
  else
    local status=$?
    record_status "${name}" "failed" "${status}"
    FAILED_STEPS+=("${name}:${status}")
    echo "[WARN] ${name} failed with exit code ${status}; continuing to generate the optimization report from available artifacts."
    return 0
  fi
}

PYTHON_BIN="$(resolve_python_bin)"
FAILED_STEPS=()

run_step "native_ddp_base" bash scripts/train/optimize/native/cpt_ddp_base.sh
run_step "native_ddp_workers4" bash scripts/train/optimize/native/cpt_ddp_workers4.sh
run_step "native_ddp_no_gc" bash scripts/train/optimize/native/cpt_ddp_no_gc.sh
run_step "zero2_base" bash scripts/train/optimize/deepspeed/cpt_zero2_base.sh
run_step "zero2_workers4" bash scripts/train/optimize/deepspeed/cpt_zero2_workers4.sh
run_step "zero2_no_gc" bash scripts/train/optimize/deepspeed/cpt_zero2_no_gc.sh
run_step "report_optimization" "${PYTHON_BIN}" scripts/15_report_cpt_optimization.py --output reports/cpt_optimization.md

echo "[DONE] all optimization runs finished. report updated:"
echo "  - reports/cpt_optimization.md"

if ((${#FAILED_STEPS[@]} > 0)); then
  echo "[WARN] failed steps:"
  for item in "${FAILED_STEPS[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi
