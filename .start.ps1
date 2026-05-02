param(
    [int]$PollIntervalSeconds = 1,
    [switch]$KeepLogs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = Split-Path -Parent $PSCommandPath
}

Set-Location -LiteralPath $ScriptDir

$ReadyMessage = "Application startup complete"
$BotLog = Join-Path $ScriptDir "log_bot_service.log"
$ProcessFile = Join-Path $ScriptDir ".start-processes.json"
$VenvScripts = Join-Path $ScriptDir ".venv\Scripts"
$StartupSessionStartedAt = (Get-Date).ToString("o")
$StartedProcesses = @()

function Write-Status {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function ConvertTo-QuotedLiteral {
    param([Parameter(Mandatory = $true)][string]$Value)

    return "'" + $Value.Replace("'", "''") + "'"
}

function Reset-Log {
    param([Parameter(Mandatory = $true)][string]$LogPath)

    if ($KeepLogs) {
        return
    }

    New-Item -ItemType File -Force -Path $LogPath | Out-Null
    Clear-Content -LiteralPath $LogPath
}

function Save-StartedProcesses {
    $payload = [pscustomobject]@{
        started_at = $StartupSessionStartedAt
        script_dir = $ScriptDir
        processes = $StartedProcesses
    }

    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ProcessFile -Encoding UTF8
}

function Register-StartedProcess {
    param(
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$ScriptName,
        [Parameter(Mandatory = $true)][string]$DisplayName,
        [Parameter(Mandatory = $true)][string]$LogName,
        [Parameter(Mandatory = $true)][string]$EncodedCommand
    )

    $script:StartedProcesses += [pscustomobject]@{
        name = $DisplayName
        script = $ScriptName
        log = $LogName
        pid = $Process.Id
        encoded_command = $EncodedCommand
        started_at = (Get-Date).ToString("o")
    }

    Save-StartedProcesses
}

function Get-PowerShellExe {
    $currentProcess = Get-Process -Id $PID
    if ($currentProcess.Path) {
        return $currentProcess.Path
    }

    $pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($pwsh) {
        return $pwsh.Source
    }

    return "powershell.exe"
}

function Start-StartupScript {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptName,
        [Parameter(Mandatory = $true)][string]$DisplayName,
        [Parameter(Mandatory = $true)][string]$LogName
    )

    $scriptPath = Join-Path $ScriptDir $ScriptName
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        throw "missing $ScriptName"
    }

    $logPath = Join-Path $ScriptDir $LogName
    Reset-Log -LogPath $logPath

    $scriptDirLiteral = ConvertTo-QuotedLiteral $ScriptDir
    $venvScriptsLiteral = ConvertTo-QuotedLiteral $VenvScripts
    $scriptPathLiteral = ConvertTo-QuotedLiteral $scriptPath
    $logPathLiteral = ConvertTo-QuotedLiteral $logPath

    $childCommand = @"
`$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $scriptDirLiteral
if (Test-Path -LiteralPath $venvScriptsLiteral) {
    `$env:PATH = $venvScriptsLiteral + [IO.Path]::PathSeparator + `$env:PATH
}
`$env:PYTHONUTF8 = "1"
`$env:PYTHONIOENCODING = "utf-8"
`$OutputEncoding = [Text.UTF8Encoding]::new(`$false)
try {
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new(`$false)
    [Console]::InputEncoding = [Text.UTF8Encoding]::new(`$false)
}
catch {
    # Hidden/background PowerShell hosts may not expose a real console.
}
& $scriptPathLiteral *>> $logPathLiteral
if (Get-Variable -Name LASTEXITCODE -Scope Global -ErrorAction SilentlyContinue) {
    `$exitCode = `$global:LASTEXITCODE
}
elseif (`$?) {
    `$exitCode = 0
}
else {
    `$exitCode = 1
}
"STARTUP WRAPPER EXITED WITH CODE `$exitCode" | Add-Content -LiteralPath $logPathLiteral
exit `$exitCode
"@

    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($childCommand))
    $powerShellExe = Get-PowerShellExe

    Write-Status "Starting $DisplayName via $ScriptName (log: $LogName)"
    $process = Start-Process `
        -FilePath $powerShellExe `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encodedCommand) `
        -WorkingDirectory $ScriptDir `
        -WindowStyle Hidden `
        -PassThru

    Register-StartedProcess -Process $process -ScriptName $ScriptName -DisplayName $DisplayName -LogName $LogName -EncodedCommand $encodedCommand
    return $process
}

function Wait-ForBotReady {
    param([Parameter(Mandatory = $true)]$BotProcess)

    Write-Status "Waiting for bot service readiness message in $(Split-Path -Leaf $BotLog)"

    while ($true) {
        if ((Test-Path -LiteralPath $BotLog -PathType Leaf) -and
            (Select-String -LiteralPath $BotLog -Pattern $ReadyMessage -SimpleMatch -Quiet)) {
            Write-Status "Bot service is ready"
            return
        }

        if ($BotProcess.HasExited) {
            throw "bot service exited before readiness; check $(Split-Path -Leaf $BotLog)"
        }

        Start-Sleep -Seconds $PollIntervalSeconds
    }
}

# Windows split of the Bash .start_sales step.
Save-StartedProcesses
Start-StartupScript -ScriptName ".start_ss.ps1" -DisplayName "sales lead retrieval service" -LogName "log_sales_service.log" | Out-Null
Start-StartupScript -ScriptName ".start_sw.ps1" -DisplayName "sales lead retrieval worker" -LogName "log_sales_worker.log" | Out-Null

Start-StartupScript -ScriptName ".start_gaz.ps1" -DisplayName "GAZ knowledge service" -LogName "log_gaz_service.log" | Out-Null
$botProcess = Start-StartupScript -ScriptName ".start_bs.ps1" -DisplayName "bot service" -LogName "log_bot_service.log"

Wait-ForBotReady -BotProcess $botProcess

Start-StartupScript -ScriptName ".start_qw.ps1" -DisplayName "task queue worker" -LogName "log_task_queue_worker.log" | Out-Null
Start-StartupScript -ScriptName ".start_oai.ps1" -DisplayName "OpenAI-compatible proxy" -LogName "log_openai_proxy.log" | Out-Null

Write-Status "All startup scripts have been triggered"
