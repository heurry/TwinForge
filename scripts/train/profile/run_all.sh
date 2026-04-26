#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

cd "${ROOT_DIR}"
export TOKENIZERS_PARALLELISM=false

run_step() {
  local name="$1"
  shift
  echo "[PROFILE] ${name}"
  if "$@"; then
    return 0
  fi

  local status=$?
  FAILED_STEPS+=("${name}:${status}")
  echo "[WARN] ${name} failed with exit code ${status}; continuing to generate reports from available artifacts."
  return 0
}

PYTHON_BIN="$(resolve_python_bin)"
FAILED_STEPS=()

run_step "native_single" bash scripts/train/profile/native/cpt_single.sh
run_step "native_ddp" bash scripts/train/profile/native/cpt_ddp.sh
run_step "zero2" bash scripts/train/profile/deepspeed/cpt_zero2.sh
run_step "zero3_offload" bash scripts/train/profile/deepspeed/cpt_zero3_offload.sh
run_step "report_profile" "${PYTHON_BIN}" scripts/13_report_cpt_profile.py --output reports/cpt_profile.md
run_step "report_comms" "${PYTHON_BIN}" scripts/14_report_cpt_comms.py --output reports/cpt_comms.md

echo "[DONE] all profile runs finished. reports updated:"
echo "  - reports/cpt_profile.md"
echo "  - reports/cpt_comms.md"

if ((${#FAILED_STEPS[@]} > 0)); then
  echo "[WARN] failed steps:"
  for item in "${FAILED_STEPS[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi
