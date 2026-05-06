#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

ts="$(date +%Y%m%d_%H%M%S)"
tb_run_name="fish_vla_phaseb_resume_990to1000_${ts}"
model_out="my_projects/models/my_underwater_robot_policy_phaseb_resume_990to1000_${ts}"
monitor_path="my_projects/monitor_logs/monitor_phaseb_resume_990to1000_${ts}.monitor.csv"

./python.sh my_projects/train_phaseb_hybrid.py \
  --resume-from my_projects/models/my_underwater_robot_policy_phaseb_1269200_dense960to1000.zip \
  --learning-rate 5e-5 \
  --timesteps 400000 \
  --tb-run-name "$tb_run_name" \
  --model-output-path "$model_out" \
  --monitor-log-path "$monitor_path" \
  --alpha-start 0.99 \
  --alpha-max 1.0 \
  --alpha-step 0.001 \
  --alpha-targets 0.990,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999,1.000 \
  --success-threshold 0.97 \
  --success-window 80 \
  --min-episodes-before-update 80 \
  --cooldown-episodes 40 \
  --no-allow-alpha-decrease \
  --progress-bar \
  --semantic-update-interval-steps 2 \
  --renderer HydraStorm
