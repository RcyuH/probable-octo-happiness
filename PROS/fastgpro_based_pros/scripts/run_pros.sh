#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-PROS/fastgpro_based_pros/configs/pros_fastgrpo_example.json}"

python3 PROS/fastgpro_based_pros/train_pros.py --config "$CONFIG_PATH" "$@"
