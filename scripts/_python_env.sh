#!/usr/bin/env bash

if [[ -x "../python.sh" ]]; then
  PYTHON_RUNNER=("../python.sh")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_RUNNER=("python3")
elif command -v python >/dev/null 2>&1; then
  PYTHON_RUNNER=("python")
else
  echo "No Python runner found. Run from the Isaac Sim-adjacent release folder or install python3." >&2
  exit 127
fi

if [[ -z "${FISH_TANK_USD_PATH:-}" && -f "../final_project/watertank.usd" ]]; then
  export FISH_TANK_USD_PATH="$(cd .. && pwd)/final_project/watertank.usd"
fi
