#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
if [[ $# -gt 0 ]]; then
  python -m interfaces.cli infer --config configs/face_swap/infer.yaml --work-dir "$1"
else
  python -m interfaces.cli infer --config configs/face_swap/infer.yaml
fi
