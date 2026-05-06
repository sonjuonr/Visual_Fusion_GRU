param(
    [string]$LauncherScript = "my_projects\run_phaseb_restart_long.ps1",
    [int]$CheckIntervalSeconds = 60,
    [int]$CooldownSeconds = 30
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$stateDir = Join-Path $repoRoot "my_projects\watchdog_state"
$logFile = Join-Path $stateDir "watchdog_phaseb.log"
$pidFile = Join-Path $stateDir "phaseb_runner.pid"

New-Item -ItemType Directory -Path $stateDir -Force | Out-Null

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts [watchdog] $Message" | Tee-Object -FilePath $logFile -Append
}

function Get-TrackedPid {
    if (-not (Test-Path $pidFile)) { return $null }
    $raw = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ([int]::TryParse($raw, [ref]$null)) { return [int]$raw }
    return $null
}

function Start-Training {
    $launcherPath = if ([System.IO.Path]::IsPathRooted($LauncherScript)) {
        $LauncherScript
    } else {
        Join-Path $repoRoot $LauncherScript
    }

    if (-not (Test-Path $launcherPath)) {
        throw "Launcher script not found: $launcherPath"
    }

    $proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $launcherPath) `
        -WorkingDirectory $repoRoot `
        -PassThru

    Set-Content -Path $pidFile -Value $proc.Id -Encoding ASCII
    Write-Log "Started training launcher PID=$($proc.Id) script=$launcherPath"
}

Write-Log "Watchdog started. interval=${CheckIntervalSeconds}s cooldown=${CooldownSeconds}s"

while ($true) {
    try {
        $trackedPid = Get-TrackedPid
        if ($null -eq $trackedPid) {
            Write-Log "No tracked PID found. Launching training."
            Start-Training
            Start-Sleep -Seconds $CooldownSeconds
        } else {
            $proc = Get-Process -Id $trackedPid -ErrorAction SilentlyContinue
            if ($null -eq $proc) {
                Write-Log "Tracked PID $trackedPid is not running. Relaunching training."
                Start-Training
                Start-Sleep -Seconds $CooldownSeconds
            }
        }
    } catch {
        Write-Log "Watchdog error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $CheckIntervalSeconds
}
