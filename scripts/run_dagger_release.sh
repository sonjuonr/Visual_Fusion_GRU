#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

python src/run_dagger.py --config configs/imitation/dagger_balanced.json

