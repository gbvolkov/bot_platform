param(
    [switch]$DryRun,
    [switch]$NoPatternFallback,
    [switch]$KeepProcessFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = Split-Path -Parent $PSCommandPath
}

Set-Location -LiteralPath $ScriptDir

$ProcessFile = Join-Path $ScriptDir ".start-processes.json"
$KnownCommandPatterns = @(
    "services.sales_lead_retrieval.main:app",
    "services.sales_lead_retrieval.worker",
    "services.kb_manager.app",
    "bot_service.main:app",
    "services.task_queue.worker",
    "openai_proxy.main:app"
)

function Write-Status {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Get-AllProcessInfo {
    try {
        return @(Get-CimInstance Win32_Process -ErrorAction Stop)
    }
    catch {
        return @(Get-WmiObject Win32_Process -ErrorAction Stop)
    }
}

function Add-TargetProcess {
    param(
        [Parameter(Mandatory = $true)][hashtable]$TargetsByPid,
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][string]$Reason
    )

    if ($ProcessId -le 0 -or $ProcessId -eq $PID) {
        return
    }

    if ($TargetsByPid.ContainsKey($ProcessId)) {
        $TargetsByPid[$ProcessId] = "$($TargetsByPid[$ProcessId]); $Reason"
        return
    }

    $TargetsByPid[$ProcessId] = $Reason
}

function Add-RecordedTargets {
    param(
        [Parameter(Mandatory = $true)][hashtable]$TargetsByPid,
        [Parameter(Mandatory = $true)][object[]]$AllProcesses
    )

    if (-not (Test-Path -LiteralPath $ProcessFile -PathType Leaf)) {
        Write-Status "No process file found at $(Split-Path -Leaf $ProcessFile)"
        return
    }

    try {
        $state = Get-Content -Raw -LiteralPath $ProcessFile | ConvertFrom-Json
    }
    catch {
        Write-Status "Could not read $(Split-Path -Leaf $ProcessFile): $($_.Exception.Message)"
        return
    }

    foreach ($record in @($state.processes)) {
        if (-not $record.pid) {
            continue
        }

        $name = if ($record.name) { [string]$record.name } else { "recorded startup process" }
        $processId = [int]$record.pid
        $process = $AllProcesses | Where-Object { [int]$_.ProcessId -eq $processId } | Select-Object -First 1
        if (-not $process) {
            Write-Status "Recorded PID $processId is not running ($name)"
            continue
        }

        if (-not $record.encoded_command) {
            Write-Status "Skipping legacy recorded PID $processId without command fingerprint ($name)"
            continue
        }

        $commandLine = [string]$process.CommandLine
        if ($commandLine.IndexOf([string]$record.encoded_command, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
            Write-Status "Skipping stale or reused recorded PID $processId ($name)"
            continue
        }

        Add-TargetProcess -TargetsByPid $TargetsByPid -ProcessId ([int]$record.pid) -Reason "recorded: $name"
    }
}

function Add-PatternTargets {
    param(
        [Parameter(Mandatory = $true)][hashtable]$TargetsByPid,
        [Parameter(Mandatory = $true)][object[]]$AllProcesses
    )

    foreach ($process in $AllProcesses) {
        $commandLine = [string]$process.CommandLine
        if ([string]::IsNullOrWhiteSpace($commandLine)) {
            continue
        }

        foreach ($pattern in $KnownCommandPatterns) {
            if ($commandLine.IndexOf($pattern, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                Add-TargetProcess -TargetsByPid $TargetsByPid -ProcessId ([int]$process.ProcessId) -Reason "matched command: $pattern"
                break
            }
        }
    }
}

function Get-ProcessSummary {
    param(
        [Parameter(Mandatory = $true)][int]$ProcessId,
        [Parameter(Mandatory = $true)][object[]]$AllProcesses
    )

    $info = $AllProcesses | Where-Object { [int]$_.ProcessId -eq $ProcessId } | Select-Object -First 1
    if (-not $info) {
        return "PID $ProcessId"
    }

    $name = if ($info.Name) { [string]$info.Name } else { "process" }
    $commandLine = [string]$info.CommandLine
    if ($commandLine.Length -gt 140) {
        $commandLine = $commandLine.Substring(0, 137) + "..."
    }

    if ([string]::IsNullOrWhiteSpace($commandLine)) {
        return "PID $ProcessId ($name)"
    }

    return "PID $ProcessId ($name): $commandLine"
}

$targetsByPid = @{}
$allProcesses = Get-AllProcessInfo

Add-RecordedTargets -TargetsByPid $targetsByPid -AllProcesses $allProcesses

if (-not $NoPatternFallback) {
    Add-PatternTargets -TargetsByPid $targetsByPid -AllProcesses $allProcesses
}

if ($targetsByPid.Count -eq 0) {
    Write-Status "No matching startup processes found"
    if ((-not $KeepProcessFile) -and (Test-Path -LiteralPath $ProcessFile -PathType Leaf)) {
        Remove-Item -LiteralPath $ProcessFile -Force
    }
    exit 0
}

$allProcesses = Get-AllProcessInfo
$processIdsToStop = @()

foreach ($entry in $targetsByPid.GetEnumerator()) {
    $processId = [int]$entry.Key
    $processInfo = $allProcesses | Where-Object { [int]$_.ProcessId -eq $processId } | Select-Object -First 1
    if (-not $processInfo) {
        Write-Status "Already stopped PID $processId ($($entry.Value))"
        continue
    }

    $processIdsToStop += $processId
}

$processIdsToStop = @($processIdsToStop | Where-Object { $_ -ne $PID } | Select-Object -Unique)

if ($processIdsToStop.Count -eq 0) {
    Write-Status "No running startup processes found"
    if ((-not $KeepProcessFile) -and (Test-Path -LiteralPath $ProcessFile -PathType Leaf)) {
        Remove-Item -LiteralPath $ProcessFile -Force
    }
    exit 0
}

if ($DryRun) {
    Write-Status "Dry run: these processes would be stopped"
    foreach ($processId in $processIdsToStop) {
        Write-Status (Get-ProcessSummary -ProcessId ([int]$processId) -AllProcesses $allProcesses)
    }
    exit 0
}

foreach ($processId in $processIdsToStop) {
    Write-Status "Stopping $(Get-ProcessSummary -ProcessId ([int]$processId) -AllProcesses $allProcesses)"
    try {
        Stop-Process -Id ([int]$processId) -Force -ErrorAction Stop
    }
    catch {
        Write-Status "Could not stop PID ${processId}: $($_.Exception.Message)"
    }
}

Start-Sleep -Milliseconds 500
$remainingProcesses = Get-AllProcessInfo
$remaining = @($processIdsToStop | Where-Object {
    $processId = [int]$_
    $remainingProcesses | Where-Object { [int]$_.ProcessId -eq $processId } | Select-Object -First 1
})

if ($remaining.Count -gt 0) {
    Write-Status "Some processes are still running: $($remaining -join ', ')"
    exit 1
}

if ((-not $KeepProcessFile) -and (Test-Path -LiteralPath $ProcessFile -PathType Leaf)) {
    Remove-Item -LiteralPath $ProcessFile -Force
}

Write-Status "All startup processes have been stopped"
