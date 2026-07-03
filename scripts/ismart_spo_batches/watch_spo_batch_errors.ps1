cd C:\Projects\bot_platform
chcp 65001 | Out-Null
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

$LogDirs = @(
    "logs\ismart_spo_batches",
    "logs\ismart_spo_incomplete_rerun"
)
$ScanIntervalSeconds = 60
$script:ShowAll = $true
$script:Offsets = @{}
$script:Contexts = @{}

foreach ($logDir in $LogDirs) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

function Get-DatasetName([string]$FileName) {
    if ($FileName -match '^(rerun_)?(?<dataset>basic_spo|advanced_spo)_batch_') {
        return $Matches.dataset
    }
    return [System.IO.Path]::GetFileNameWithoutExtension($FileName)
}

function Get-LogFiles {
    $logs = @()
    foreach ($logDir in $LogDirs) {
        $logs += Get-ChildItem -LiteralPath $logDir -File -Filter "*.log" -ErrorAction SilentlyContinue
    }
    return $logs | Sort-Object FullName
}

function Get-DisplayLogName($LogFile) {
    $parent = Split-Path -Leaf $LogFile.DirectoryName
    return (Join-Path $parent $LogFile.Name)
}

function Get-FileContext($LogFile) {
    $key = $LogFile.FullName
    if (-not $script:Contexts.ContainsKey($key)) {
        $script:Contexts[$key] = [ordered]@{
            dataset = Get-DatasetName $LogFile.Name
            lesson_number = ""
            task_id = ""
        }
    }
    return $script:Contexts[$key]
}

function Set-OffsetsToEnd {
    foreach ($log in (Get-LogFiles)) {
        $script:Offsets[$log.FullName] = [int64]$log.Length
        $null = Get-FileContext $log
    }
}

function Read-LogText([string]$Path, [int64]$Offset) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        if ($Offset -gt $stream.Length) {
            $Offset = 0
        }
        $stream.Seek($Offset, [System.IO.SeekOrigin]::Begin) | Out-Null
        $reader = [System.IO.StreamReader]::new($stream, [Text.UTF8Encoding]::new($false, $true), $true, 4096, $true)
        try {
            $text = $reader.ReadToEnd()
            return [pscustomobject]@{
                Text = $text
                Offset = $stream.Position
            }
        }
        finally {
            $reader.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Try-ParseJsonLine([string]$Line) {
    $trimmed = $Line.Trim()
    if ($trimmed.StartsWith("{")) {
        try { return $trimmed | ConvertFrom-Json } catch { return $null }
    }

    $jsonStart = $trimmed.IndexOf("{")
    if ($jsonStart -lt 0) {
        return $null
    }

    $jsonCandidate = $trimmed.Substring($jsonStart)
    if ($jsonCandidate -notmatch '"event"\s*:') {
        return $null
    }

    try { return $jsonCandidate | ConvertFrom-Json } catch { return $null }
}

function Get-ErrorMessageFromLine([string]$Line, $JsonEvent) {
    if ($null -ne $JsonEvent) {
        $event = [string]$JsonEvent.event
        $status = [string]$JsonEvent.status

        if ($event -eq "task.error") {
            if ($JsonEvent.error) {
                return [string]$JsonEvent.error
            }
            return $Line.Trim()
        }

        if ($event -eq "task.done" -and $status -and $status -notin @("approved", "completed_with_skips", "skipped")) {
            return "Task finished with status=$status"
        }

        if ($event -eq "sequential.done" -and $status -and $status -notin @("approved", "completed_with_skips", "skipped", "completed")) {
            return "Sequential run finished with status=$status"
        }
    }

    $trimmed = $Line.Trim()
    if ($trimmed -match 'subagent\.structured_output\.exception\s+(?<payload>\{.*\})') {
        try {
            $payload = $Matches.payload | ConvertFrom-Json
            return ("{0} structured output failed: {1}: {2}" -f [string]$payload.agent_type, [string]$payload.error_type, [string]$payload.error)
        }
        catch {
            return $trimmed
        }
    }

    if ($trimmed -match '^(error|fatal):\s*(?<message>.+)$') {
        return $Matches.message
    }

    if ($trimmed -match '(Failed to parse iSMART generator request|iSMART generation failed|Traceback \(most recent call last\)|Unhandled exception|Internal Server Error|Failed to export span batch|LLM request failed|structured_output_failed|validation_error|failed after .*generation/validation attempts)') {
        return $trimmed
    }

    return $null
}

function Write-ErrorReport([string]$Dataset, [string]$LessonNumber, [string]$TaskId, [string]$Message, [string]$LogName, [bool]$Historical) {
    $eventName = if ($Historical) { "HISTORY_ERROR" } else { "ERROR" }
    $lesson = if ([string]::IsNullOrWhiteSpace($LessonNumber)) { "unknown" } else { $LessonNumber }
    $task = if ([string]::IsNullOrWhiteSpace($TaskId)) { "" } else { " task=$TaskId" }
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $eventName dataset=$Dataset lesson=$lesson$task log=$LogName :: $Message" -ForegroundColor Red
}

function Write-LessonLifecycleReport(
    [string]$EventName,
    [string]$Dataset,
    [string]$LessonNumber,
    [string]$TaskId,
    [string]$LessonTitle,
    [string]$Status,
    [string]$LogName
) {
    $lesson = if ([string]::IsNullOrWhiteSpace($LessonNumber)) { "unknown" } else { $LessonNumber }
    $task = if ([string]::IsNullOrWhiteSpace($TaskId)) { "" } else { " task=$TaskId" }
    $title = if ([string]::IsNullOrWhiteSpace($LessonTitle)) { "" } else { " title=""$LessonTitle""" }
    $statusPart = if ([string]::IsNullOrWhiteSpace($Status)) { "" } else { " status=$Status" }
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $EventName dataset=$Dataset lesson=$lesson$task$statusPart log=$LogName$title" -ForegroundColor Cyan
}

function Process-LogLine([string]$Line, $Context, [string]$LogName, [bool]$Historical) {
    if ([string]::IsNullOrWhiteSpace($Line)) {
        return
    }

    $jsonEvent = Try-ParseJsonLine $Line
    if ($null -ne $jsonEvent) {
        if ($jsonEvent.lesson_number) { $Context.lesson_number = [string]$jsonEvent.lesson_number }
        if ($jsonEvent.task_id) { $Context.task_id = [string]$jsonEvent.task_id }
        if ($jsonEvent.course_level) { $Context.dataset = "$($jsonEvent.course_level)_spo" }

        $event = [string]$jsonEvent.event
        if ($event -eq "task.start") {
            $eventName = if ($Historical) { "HISTORY_LESSON_START" } else { "LESSON_START" }
            Write-LessonLifecycleReport -EventName $eventName -Dataset ([string]$Context.dataset) -LessonNumber ([string]$Context.lesson_number) -TaskId ([string]$Context.task_id) -LessonTitle ([string]$jsonEvent.lesson_title) -Status "" -LogName $LogName
        }
        elseif ($event -eq "task.done") {
            $eventName = if ($Historical) { "HISTORY_LESSON_DONE" } else { "LESSON_DONE" }
            Write-LessonLifecycleReport -EventName $eventName -Dataset ([string]$Context.dataset) -LessonNumber ([string]$Context.lesson_number) -TaskId ([string]$Context.task_id) -LessonTitle ([string]$jsonEvent.lesson_title) -Status ([string]$jsonEvent.status) -LogName $LogName
        }
        elseif ($event -eq "task.error") {
            $eventName = if ($Historical) { "HISTORY_LESSON_DONE" } else { "LESSON_DONE" }
            Write-LessonLifecycleReport -EventName $eventName -Dataset ([string]$Context.dataset) -LessonNumber ([string]$Context.lesson_number) -TaskId ([string]$Context.task_id) -LessonTitle ([string]$jsonEvent.lesson_title) -Status "error" -LogName $LogName
        }
    }

    $message = Get-ErrorMessageFromLine -Line $Line -JsonEvent $jsonEvent
    if ($message) {
        Write-ErrorReport -Dataset ([string]$Context.dataset) -LessonNumber ([string]$Context.lesson_number) -TaskId ([string]$Context.task_id) -Message $message -LogName $LogName -Historical $Historical
    }
}

function Write-MonitorHeader {
    $mode = if ($script:ShowAll) { "ALL" } else { "SINCE_C" }
    Write-Host "Watching $($LogDirs -join ', ') for iSMART SPO generation." -ForegroundColor Green
    Write-Host "Interval: $ScanIntervalSeconds sec. Mode: $mode. Press C to toggle, Ctrl+C to stop." -ForegroundColor Green
    Write-Host ""
}

function Reset-MonitorScreen {
    Clear-Host
    Write-MonitorHeader
}

function Replay-All {
    Reset-MonitorScreen
    Write-Host "Historical log stream:" -ForegroundColor Yellow

    foreach ($log in (Get-LogFiles)) {
        $context = Get-FileContext $log
        $displayLogName = Get-DisplayLogName $log
        $readResult = Read-LogText -Path $log.FullName -Offset 0
        $script:Offsets[$log.FullName] = [int64]$readResult.Offset

        if ([string]::IsNullOrEmpty($readResult.Text)) {
            continue
        }

        foreach ($line in ($readResult.Text -split "`r?`n")) {
            Process-LogLine -Line $line -Context $context -LogName $displayLogName -Historical $true
        }
    }
}

function Toggle-Mode {
    $script:ShowAll = -not $script:ShowAll
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    if ($script:ShowAll) {
        Replay-All
        Write-Host "[$timestamp] Mode switched to ALL. Showing all history and live events." -ForegroundColor Yellow
    }
    else {
        Set-OffsetsToEnd
        Reset-MonitorScreen
        Write-Host "[$timestamp] Mode switched to SINCE_C. Showing only new events after this moment." -ForegroundColor Yellow
    }
}

function Handle-ConsoleInput {
    try {
        while ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            if ($key.KeyChar -eq 'c' -or $key.KeyChar -eq 'C') {
                Toggle-Mode
            }
        }
    }
    catch {
        return
    }
}

function Wait-WithConsoleInput([int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        Handle-ConsoleInput
        Start-Sleep -Milliseconds 250
    }
}

Replay-All

while ($true) {
    Handle-ConsoleInput

    foreach ($log in (Get-LogFiles)) {
        $context = Get-FileContext $log
        $displayLogName = Get-DisplayLogName $log
        $offset = if ($script:Offsets.ContainsKey($log.FullName)) { [int64]$script:Offsets[$log.FullName] } else { 0 }
        $readResult = Read-LogText -Path $log.FullName -Offset $offset
        $script:Offsets[$log.FullName] = [int64]$readResult.Offset

        if ([string]::IsNullOrEmpty($readResult.Text)) {
            continue
        }

        foreach ($line in ($readResult.Text -split "`r?`n")) {
            Process-LogLine -Line $line -Context $context -LogName $displayLogName -Historical $false
        }
    }

    Wait-WithConsoleInput -Seconds $ScanIntervalSeconds
}
