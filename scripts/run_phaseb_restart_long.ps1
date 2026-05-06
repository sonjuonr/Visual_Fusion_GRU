$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$tbRunName = "fish_vla_phaseb_restart_$ts"
$modelOut = Join-Path $repoRoot "my_projects\models\my_underwater_robot_policy_phaseb_restart_$ts"
$monitorPath = Join-Path $repoRoot "my_projects\monitor_logs\monitor_phaseb_restart_$ts.monitor.csv"
$resumeCkpt = Get-ChildItem (Join-Path $repoRoot "my_projects\checkpoints") -Filter "fish_vla_phaseb_*_steps.zip" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $resumeCkpt) {
    throw "No phaseb checkpoint found under my_projects\\checkpoints."
}

Write-Output "Resume checkpoint: $($resumeCkpt.FullName)"

& (Join-Path $repoRoot "python.bat") `
    my_projects\train_phaseb_hybrid.py `
    --resume-from $resumeCkpt.FullName `
    --learning-rate 1e-4 `
    --timesteps 1000000 `
    --tb-run-name $tbRunName `
    --model-output-path $modelOut `
    --monitor-log-path $monitorPath `
    --alpha-start 0.95 `
    --alpha-max 1.0 `
    --alpha-targets 0.95,0.952,0.954,0.956,0.958,0.96,0.962,0.964,0.966,0.968,0.97,0.972,0.974,0.976,0.978,0.98,0.982,0.984,0.986,0.988,0.99,0.992,0.994,0.996,0.998,1.0 `
    --success-threshold 0.97 `
    --success-window 80 `
    --min-episodes-before-update 80 `
    --cooldown-episodes 40 `
    --allow-alpha-decrease `
    --success-lower-threshold 0.2 `
    --no-progress-bar `
    --semantic-update-interval-steps 2 `
    --renderer HydraStorm
