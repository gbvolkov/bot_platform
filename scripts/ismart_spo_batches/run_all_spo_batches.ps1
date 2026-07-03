cd C:\Projects\bot_platform
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = "C:\Projects\bot_platform"
$scripts = @(
    "run_spo_basic_batch_01_lessons_001_010.ps1",
    "run_spo_basic_batch_02_lessons_011_020.ps1",
    "run_spo_basic_batch_03_lessons_021_030.ps1",
    "run_spo_basic_batch_04_lessons_031_040.ps1",
    "run_spo_basic_batch_05_lessons_041_050.ps1",
    "run_spo_basic_batch_06_lessons_051_060.ps1",
    "run_spo_basic_batch_07_lessons_061_074.ps1",
    "run_spo_advanced_batch_01_lessons_001_010.ps1",
    "run_spo_advanced_batch_02_lessons_011_020.ps1",
    "run_spo_advanced_batch_03_lessons_021_030.ps1",
    "run_spo_advanced_batch_04_lessons_031_040.ps1",
    "run_spo_advanced_batch_05_lessons_041_050.ps1",
    "run_spo_advanced_batch_06_lessons_051_060.ps1",
    "run_spo_advanced_batch_07_lessons_061_074.ps1"
)

foreach ($script in $scripts) {
    $scriptPath = Join-Path $scriptDir $script
    Start-Process powershell.exe -WorkingDirectory $repoRoot -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "`"$scriptPath`""
    )
}
