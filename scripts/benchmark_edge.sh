#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
EXPORT_DIR=${1:-"work_dirs/exports/baseline"}
python -m interfaces.cli benchmark-edge --config configs/face_swap/export_edge.yaml --checkpoint work_dirs/.../checkpoints/best.pth --export-dir "$EXPORT_DIR" --target jetson
