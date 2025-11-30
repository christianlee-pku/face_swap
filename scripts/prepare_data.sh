#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
python -m interfaces.cli prepare-data --config configs/face_swap/data_prepare.yaml "$@"
