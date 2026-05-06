param(
    [string]$TaskName = "IsaacSim-PhaseB-Watchdog",
    [string]$WatchdogScript = "my_projects\watchdog_phaseb_restart.ps1"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$scriptPath = if ([System.IO.Path]::IsPathRooted($WatchdogScript)) {
    $WatchdogScript
} else {
    Join-Path $repoRoot $WatchdogScript
}

if (-not (Test-Path $scriptPath)) {
    throw "Watchdog script not found: $scriptPath"
}

$escapedScriptPath = $scriptPath.Replace('"', '""')
$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$escapedScriptPath`""

Write-Output "Registering task '$TaskName'..."
schtasks /Create /SC ONLOGON /TN $TaskName /TR $taskCommand /RL LIMITED /F | Out-Host

Write-Output ""
Write-Output "Task registered. Current task info:"
schtasks /Query /TN $TaskName /V /FO LIST
