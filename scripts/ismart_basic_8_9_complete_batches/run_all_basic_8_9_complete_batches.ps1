cd C:\Projects\bot_platform
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = "C:\Projects\bot_platform"
$scripts = @(
    "run_basic_8_9_complete_batch_01_lessons_001_005.ps1",
    "run_basic_8_9_complete_batch_02_lessons_006_010.ps1",
    "run_basic_8_9_complete_batch_03_lessons_011_015.ps1",
    "run_basic_8_9_complete_batch_04_lessons_016_020.ps1",
    "run_basic_8_9_complete_batch_05_lessons_021_025.ps1",
    "run_basic_8_9_complete_batch_06_lessons_026_030.ps1",
    "run_basic_8_9_complete_batch_07_lessons_031_035.ps1",
    "run_basic_8_9_complete_batch_08_lessons_036_040.ps1",
    "run_basic_8_9_complete_batch_09_lessons_041_045.ps1",
    "run_basic_8_9_complete_batch_10_lessons_046_050.ps1",
    "run_basic_8_9_complete_batch_11_lessons_051_055.ps1",
    "run_basic_8_9_complete_batch_12_lessons_056_060.ps1",
    "run_basic_8_9_complete_batch_13_lessons_061_065.ps1",
    "run_basic_8_9_complete_batch_14_lessons_066_070.ps1",
    "run_basic_8_9_complete_batch_15_lessons_071_074.ps1"
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
