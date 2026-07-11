#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPO_ROOT"

"$PYTHON_BIN" PROS/fastgpro_based_pros/train_pros.py \
  --dry-run true \
  --train-draft false \
  --verification-capacity 256 \
  --max-draft-token-length 6 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --lora-bias none

"$PYTHON_BIN" PROS/fastgpro_based_pros/train_pros.py \
  --dry-run true \
  --generation-backend target \
  --verification-capacity 1 \
  --train-draft false \
  --use-lora false

"$PYTHON_BIN" -m unittest discover -s FastGRPO/tests -v
"$PYTHON_BIN" -m unittest discover -s PROS/fastgpro_based_pros/tests -v
