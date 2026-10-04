param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetId,
    [Parameter(Mandatory = $true)]
    [string]$InputJson,
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,
    [Parameter(Mandatory = $true)]
    [string]$TargetRoot,
    [Parameter(Mandatory = $true)]
    [int[]]$Lessons,
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [string]$LogFileName
)

cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$ErrorActionPreference = "Stop"

$RunnerOutput = Join-Path $TargetRoot "_runner_manifests"
$LogDir = "logs\ismart_${DatasetId}_complete_batches"
$LogFile = Join-Path $LogDir $LogFileName

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
New-Item -ItemType Directory -Force -Path $RunnerOutput | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Remove-Item -LiteralPath $LogFile -Force -ErrorAction SilentlyContinue

foreach ($Lesson in $Lessons) {
    $SourceDir = Join-Path $SourceRoot "$Lesson-lesson-$Lesson"
    $TargetDir = Join-Path $TargetRoot "$Lesson-lesson-$Lesson"
    if (-not (Test-Path -LiteralPath $SourceDir)) {
        throw "Missing source lesson folder: $SourceDir"
    }
    if (-not (Test-Path -LiteralPath $TargetDir)) {
        Copy-Item -LiteralPath $SourceDir -Destination $TargetDir -Recurse -Force
    }
}

$LessonArgs = @()
foreach ($Lesson in $Lessons) {
    $LessonArgs += "--lesson-number"
    $LessonArgs += [string]$Lesson
}

$CommandArgs = @(
    "run",
    "--no-sync",
    "python",
    "-m",
    "agents.ismart_generator_agent.sequential_runner",
    "--input",
    $InputJson
) + $LessonArgs + @(
    "--output",
    $RunnerOutput,
    "--run-name",
    $RunName,
    "--resume-missing-from",
    $TargetRoot,
    "--verbose"
)

"start $(Get-Date -Format o) dataset=$DatasetId run=$RunName lessons=$($Lessons -join ',')" | Out-File -FilePath $LogFile -Encoding utf8
& uv @CommandArgs 2>&1 | Out-File -FilePath $LogFile -Encoding utf8 -Append
$ExitCode = $LASTEXITCODE
"finish $(Get-Date -Format o) dataset=$DatasetId run=$RunName exit_code=$ExitCode" | Out-File -FilePath $LogFile -Encoding utf8 -Append

if ($ExitCode -ne 0) {
    throw "Generation failed with exit code $ExitCode. See $LogFile"
}

Write-Host "Completed $RunName"
Write-Host "Target root: $TargetRoot"
Write-Host "Log: $LogFile"
