#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
source "$repo_root/scripts/_python_env.sh"

"${PYTHON_RUNNER[@]}" src/run_gru_hard_seed_dagger.py \
  --base-config configs/imitation/dagger_gru_balanced.json \
  --failed-seeds-json imitation_data/eval/dagger_gru_balanced_200eps/student_iter_08_failed_seeds.json \
  --student-checkpoint imitation_runs/dagger_gru_balanced/checkpoints/student_iter_08.pt \
  --output-root imitation_runs/dagger_gru_hard_seeds \
  --beta 0.0 \
  --epochs 10
