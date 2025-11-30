#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
DEFAULT_WD="work_dirs/face_swap/lfw-unet-baseline-001"
if [[ $# -gt 0 ]]; then
  python -m interfaces.cli eval --config configs/face_swap/eval.yaml --work-dir "$1"
else
  python -m interfaces.cli eval --config configs/face_swap/eval.yaml --work-dir "$DEFAULT_WD"
fi
