#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[RESUME] native_ddp"
bash "${SCRIPT_DIR}/native/cpt_ddp_validate.sh"

echo "[RESUME] zero2"
bash "${SCRIPT_DIR}/deepspeed/cpt_zero2_validate.sh"

echo "[DONE] resume validation finished. report updated:"
echo "  - reports/cpt_resume_validation.md"
