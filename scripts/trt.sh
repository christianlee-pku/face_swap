#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"

CONFIG=${1:-"configs/face_swap/trt.yaml"}
WORK_DIR=${2:-""}

if [[ -n "$WORK_DIR" ]]; then
  python -m interfaces.cli trt --config "$CONFIG" --work-dir "$WORK_DIR"
else
  python -m interfaces.cli trt --config "$CONFIG"
fi
