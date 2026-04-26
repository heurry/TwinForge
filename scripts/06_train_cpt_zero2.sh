#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/train/sanity/cpt_zero2.sh" "$@"
