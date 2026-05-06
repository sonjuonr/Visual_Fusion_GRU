$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& (Join-Path $repoRoot "python.bat") `
    my_projects\train_phaseb_hybrid.py `
    --resume-from (Join-Path $repoRoot "my_projects\models\my_underwater_robot_policy_phaseb_dense98_ladder.zip") `
    --learning-rate 1e-4 `
    --model-output-path (Join-Path $repoRoot "my_projects\models\my_underwater_robot_policy_phaseb_slowalpha_resume") `
    --tb-run-name fish_vla_phaseb_slowalpha_resume `
    --monitor-log-path (Join-Path $repoRoot "my_projects\monitor_logs\monitor_phaseb_slowalpha_resume.monitor.csv") `
    --timesteps 500000 `
    --alpha-start 0.95 `
    --alpha-max 1.0 `
    --alpha-targets 0.95,0.952,0.954,0.956,0.958,0.96,0.962,0.964,0.966,0.968,0.97,0.972,0.974,0.976,0.978,0.98,0.982,0.984,0.986,0.988,0.99,0.992,0.994,0.996,0.998,1.0 `
    --success-threshold 0.97 `
    --success-window 80 `
    --min-episodes-before-update 80 `
    --cooldown-episodes 40 `
    --allow-alpha-decrease `
    --success-lower-threshold 0.2
