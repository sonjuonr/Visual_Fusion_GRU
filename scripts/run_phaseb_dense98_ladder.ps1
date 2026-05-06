$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& (Join-Path $repoRoot "python.bat") `
    my_projects\train_phaseb_hybrid.py `
    --resume-from (Join-Path $repoRoot "my_projects\models\my_underwater_robot_policy_phaseb_from566k_alpha80_ladder2p.zip") `
    --model-output-path (Join-Path $repoRoot "my_projects\models\my_underwater_robot_policy_phaseb_dense98_ladder") `
    --tb-run-name fish_vla_phaseb_dense98_ladder `
    --monitor-log-path (Join-Path $repoRoot "my_projects\monitor_logs\monitor_phaseb_dense98_ladder.monitor.csv") `
    --timesteps 150000 `
    --alpha-start 0.96 `
    --alpha-max 1.0 `
    --alpha-targets 0.96,0.97,0.972,0.974,0.976,0.978,0.98,0.982,0.984,0.986,0.988,0.99,0.992,0.994,0.996,0.998,1.0 `
    --success-threshold 0.97 `
    --success-window 50 `
    --min-episodes-before-update 50 `
    --cooldown-episodes 30 `
    --no-allow-alpha-decrease
