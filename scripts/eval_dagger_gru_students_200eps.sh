#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
source "$repo_root/scripts/_python_env.sh"

"${PYTHON_RUNNER[@]}" src/eval_student_checkpoints.py \
  --checkpoints "imitation_runs/dagger_gru_balanced/checkpoints/student_iter_*.pt" \
  --episodes 200 \
  --start-seed 500000 \
  --output-dir imitation_data/eval/dagger_gru_balanced_200eps
