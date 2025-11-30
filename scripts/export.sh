#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

CONFIG=${1:-"configs/face_swap/export.yaml"}
WORK_DIR=${2:-""}

if [[ -n "$WORK_DIR" ]]; then
  python -m interfaces.cli export --config "$CONFIG" --work-dir "$WORK_DIR"
else
  python -m interfaces.cli export --config "$CONFIG"
fi
