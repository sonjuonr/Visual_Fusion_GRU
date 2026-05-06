$ErrorActionPreference = "Stop"

param(
    [string]$TaskName = "IsaacSim-PhaseB-Watchdog"
)

Write-Output "Removing task '$TaskName'..."
schtasks /Delete /TN $TaskName /F | Out-Host
Write-Output "Done."
