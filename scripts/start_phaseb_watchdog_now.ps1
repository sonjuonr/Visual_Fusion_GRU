$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$watchdog = Join-Path $repoRoot "my_projects\watchdog_phaseb_restart.ps1"
if (-not (Test-Path $watchdog)) {
    throw "Watchdog script not found: $watchdog"
}

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $watchdog) `
    -WorkingDirectory $repoRoot `
    -PassThru

Write-Output "Watchdog started. PID=$($proc.Id)"
