cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$InputJson = "data\ismart\generator\data\generation_input_8_9_from_tracker.json"
$OutputRoot = "docs\generated output_basic_8-9_full"
$RunName = "basic_8_9_lessons_002_003_full"
$RunDir = Join-Path $OutputRoot $RunName
$LogFile = "logs\ismart_basic_8_9_lessons_002_003_full.log"

Remove-Item -LiteralPath $LogFile -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input $InputJson `
    --lesson-number 2 `
    --lesson-number 3 `
    --output $OutputRoot `
    --run-name $RunName `
    --preserve-source-index `
    --verbose 2>&1 | Out-File -FilePath $LogFile -Encoding utf8

if ($LASTEXITCODE -ne 0) {
    throw "Main generation failed with exit code $LASTEXITCODE. See $LogFile"
}

$LessonDirs = @(
    (Join-Path $RunDir "002-2-lesson-2"),
    (Join-Path $RunDir "003-3-lesson-3")
)

foreach ($LessonDir in $LessonDirs) {
    if (-not (Test-Path -LiteralPath $LessonDir)) {
        throw "Expected lesson output directory not found: $LessonDir"
    }

    "practice_guidance.start lesson_output=$LessonDir" | Out-File -FilePath $LogFile -Encoding utf8 -Append
    uv run --no-sync python -m agents.ismart_generator_agent.practice_guidance_cli `
        --lesson-output $LessonDir `
        --verbose 2>&1 | Out-File -FilePath $LogFile -Encoding utf8 -Append

    if ($LASTEXITCODE -ne 0) {
        throw "Practice guidance generation failed for $LessonDir with exit code $LASTEXITCODE. See $LogFile"
    }
    "practice_guidance.done lesson_output=$LessonDir" | Out-File -FilePath $LogFile -Encoding utf8 -Append
}

"done output=$RunDir log=$LogFile" | Out-File -FilePath $LogFile -Encoding utf8 -Append
Write-Host "Generated full package: $RunDir"
Write-Host "Log: $LogFile"
