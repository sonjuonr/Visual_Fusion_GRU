#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
source "$repo_root/scripts/_python_env.sh"

"${PYTHON_RUNNER[@]}" src/eval_imitation_policy.py --config configs/imitation/eval_current_best_student.json
