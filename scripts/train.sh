#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
if [[ $# -gt 0 ]]; then
  python -m interfaces.cli train --config configs/face_swap/baseline.yaml --work-dir "$1"
else
  python -m interfaces.cli train --config configs/face_swap/baseline.yaml
fi
