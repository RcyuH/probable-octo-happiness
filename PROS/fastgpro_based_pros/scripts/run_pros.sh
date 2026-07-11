#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-PROS/fastgpro_based_pros/configs/pros_fastgrpo_example.json}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" PROS/fastgpro_based_pros/train_pros.py --config "$CONFIG_PATH" "$@"
