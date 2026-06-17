#!/usr/bin/env bash
set -euo pipefail

python3 PROS/fastgpro_based_pros/train_pros.py --dry-run true --train-draft false
python3 -m unittest discover -s PROS/fastgpro_based_pros/tests -v
