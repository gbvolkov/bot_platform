cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$InputJson = "data\ismart\generator\data\generation_input_8_9_from_tracker.json"
$ExistingSourceRoot = "docs\basic_8-9"
$OutputRoot = "docs\generated output_basic_8-9_full"
$RunName = "basic_8_9_lessons_001_003_resume_missing"
$RunDir = Join-Path $OutputRoot $RunName
$LogFile = "logs\ismart_basic_8_9_lessons_001_003_resume_missing.log"

Remove-Item -LiteralPath $LogFile -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$Workspace = [IO.Path]::GetFullPath((Get-Location).Path)
$FullRunDir = [IO.Path]::GetFullPath((Join-Path (Get-Location) $RunDir))
$AllowedRoot = [IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputRoot))
if (-not $FullRunDir.StartsWith($AllowedRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean unexpected output path: $FullRunDir"
}
if (-not $AllowedRoot.StartsWith($Workspace, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside workspace: $AllowedRoot"
}
Remove-Item -LiteralPath $RunDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

foreach ($Lesson in 1, 2, 3) {
    $SourceDir = Join-Path $ExistingSourceRoot "$Lesson-lesson-$Lesson"
    if (-not (Test-Path -LiteralPath $SourceDir)) {
        throw "Missing source lesson folder: $SourceDir"
    }
    Copy-Item -LiteralPath $SourceDir -Destination $RunDir -Recurse -Force
}

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input $InputJson `
    --lesson-number 1 `
    --lesson-number 2 `
    --lesson-number 3 `
    --output $OutputRoot `
    --run-name $RunName `
    --resume-missing-from $RunDir `
    --verbose 2>&1 | Out-File -FilePath $LogFile -Encoding utf8

if ($LASTEXITCODE -ne 0) {
    throw "Resume missing generation failed with exit code $LASTEXITCODE. See $LogFile"
}

"done output=$RunDir log=$LogFile" | Out-File -FilePath $LogFile -Encoding utf8 -Append
Write-Host "Resume-missing package: $RunDir"
Write-Host "Log: $LogFile"
