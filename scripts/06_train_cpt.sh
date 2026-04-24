#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/06_train_cpt_zero2.sh" "$@"
