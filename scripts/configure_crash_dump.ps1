$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    Write-Error "This script must be run as Administrator."
    exit 1
}

$crashControl = "HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl"

# Kernel dump + keep dump/logging enabled for post-mortem analysis.
Set-ItemProperty -Path $crashControl -Name CrashDumpEnabled -Type DWord -Value 2
Set-ItemProperty -Path $crashControl -Name LogEvent -Type DWord -Value 1
Set-ItemProperty -Path $crashControl -Name Overwrite -Type DWord -Value 1
Set-ItemProperty -Path $crashControl -Name AutoReboot -Type DWord -Value 1
Set-ItemProperty -Path $crashControl -Name AlwaysKeepMemoryDump -Type DWord -Value 1
Set-ItemProperty -Path $crashControl -Name MinidumpsCount -Type DWord -Value 20
Set-ItemProperty -Path $crashControl -Name DumpFile -Type ExpandString -Value "%SystemRoot%\MEMORY.DMP"
Set-ItemProperty -Path $crashControl -Name MinidumpDir -Type ExpandString -Value "%SystemRoot%\Minidump"

# Ensure dump directory exists.
$miniDumpPath = Join-Path $env:SystemRoot "Minidump"
New-Item -ItemType Directory -Path $miniDumpPath -Force | Out-Null

Write-Output "Crash dump settings applied."
Get-ItemProperty -Path $crashControl |
    Select-Object CrashDumpEnabled,LogEvent,Overwrite,AutoReboot,AlwaysKeepMemoryDump,MinidumpsCount,DumpFile,MinidumpDir |
    Format-List

Write-Output ""
Write-Output "Recommended next check:"
Write-Output "1) System Properties -> Advanced -> Startup and Recovery confirms 'Kernel memory dump'."
Write-Output "2) Page file is enabled on C: (required for reliable dump generation)."
