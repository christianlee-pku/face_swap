#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
python -m interfaces.cli validate-manifest --manifest data/lfw/manifest.json --processed-dir data/lfw/processed "$@"
