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

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input $InputJson `
    --lesson-number 12 `
    --output $OutputRoot `
    --run-name $RunName `
    --preserve-source-index `
    --verbose 2>&1 | Out-File -FilePath $LogFile -Encoding utf8

if ($LASTEXITCODE -ne 0) {
    throw "Generation failed with exit code $LASTEXITCODE. See $LogFile"
}

"done output=$RunDir log=$LogFile" | Out-File -FilePath $LogFile -Encoding utf8 -Append
Write-Host "Generated full package: $RunDir"
Write-Host "Log: $LogFile"
