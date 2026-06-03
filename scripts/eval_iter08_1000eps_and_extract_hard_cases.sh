#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
source "$repo_root/scripts/_python_env.sh"

"${PYTHON_RUNNER[@]}" src/eval_student_checkpoints.py \
  --checkpoints "imitation_runs/dagger_gru_balanced/checkpoints/student_iter_08.pt" \
  --episodes 1000 \
  --start-seed 600000 \
  --output-dir imitation_data/eval/dagger_gru_iter08_1000eps

"${PYTHON_RUNNER[@]}" src/extract_hard_eval_seeds.py \
  --eval-summary imitation_data/eval/dagger_gru_iter08_1000eps/student_iter_08_eval_1000eps.json \
  --output imitation_data/eval/dagger_gru_iter08_1000eps/student_iter_08_hard_cases.json \
  --near-miss-distance 0.6 \
  --slow-success-steps 100
