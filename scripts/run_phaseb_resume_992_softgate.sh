#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

ts="$(date +%Y%m%d_%H%M%S)"
tb_run_name="fish_vla_phaseb_resume_992_softgate_${ts}"
model_out="my_projects/models/my_underwater_robot_policy_phaseb_resume_992_softgate_${ts}"
monitor_path="my_projects/monitor_logs/monitor_phaseb_resume_992_softgate_${ts}.monitor.csv"

./python.sh my_projects/train_phaseb_hybrid.py \
  --resume-from my_projects/models/my_underwater_robot_policy_phaseb_resume_990to1000_20260420_024451.zip \
  --learning-rate 5e-5 \
  --timesteps 400000 \
  --tb-run-name "$tb_run_name" \
  --model-output-path "$model_out" \
  --monitor-log-path "$monitor_path" \
  --alpha-start 0.992 \
  --alpha-max 1.0 \
  --alpha-step 0.0005 \
  --alpha-targets 0.992,0.9925,0.993,0.9935,0.994,0.9945,0.995,0.9955,0.996,0.9965,0.997,0.9975,0.998,0.9985,0.999,0.9995,1.000 \
  --success-threshold 0.90 \
  --success-window 80 \
  --min-episodes-before-update 80 \
  --cooldown-episodes 40 \
  --no-allow-alpha-decrease \
  --progress-bar \
  --semantic-update-interval-steps 1 \
  --renderer HydraStorm
