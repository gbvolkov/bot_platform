cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001 | Out-Null
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$ErrorActionPreference = "Continue"

New-Item -ItemType Directory -Force -Path "logs\ismart_spo_incomplete_rerun" | Out-Null

function Invoke-IsmartSpoRerun {
    param(
        [Parameter(Mandatory = $true)][string]$InputFile,
        [Parameter(Mandatory = $true)][string]$OutputDir,
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)][int[]]$Lessons,
        [Parameter(Mandatory = $true)][string]$LogName
    )

    $lessonArgs = @()
    foreach ($lesson in $Lessons) {
        $lessonArgs += @("--lesson-number", [string]$lesson)
    }

    $logFile = Join-Path "logs\ismart_spo_incomplete_rerun" $LogName
    Write-Host "Rerun $RunName lessons: $($Lessons -join ', ')"
    Write-Host "Log: $logFile"

    & uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
        --input $InputFile `
        @lessonArgs `
        --output $OutputDir `
        --run-name $RunName `
        --preserve-source-index `
        --verbose 2>&1 | Out-File -FilePath $logFile -Encoding utf8
}

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_basic_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_basic_batches" `
#    -RunName "basic_spo_batch_01_lessons_001_010" `
#    -Lessons @(9, 10) `
#    -LogName "rerun_basic_spo_batch_01_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_basic_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_basic_batches" `
#    -RunName "basic_spo_batch_03_lessons_021_030" `
#    -Lessons @(28, 29, 30) `
#    -LogName "rerun_basic_spo_batch_03_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_basic_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_basic_batches" `
#    -RunName "basic_spo_batch_05_lessons_041_050" `
#    -Lessons @(50) `
#    -LogName "rerun_basic_spo_batch_05_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_basic_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_basic_batches" `
#    -RunName "basic_spo_batch_06_lessons_051_060" `
#    -Lessons @(60) `
#    -LogName "rerun_basic_spo_batch_06_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_basic_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_basic_batches" `
#    -RunName "basic_spo_batch_07_lessons_061_074" `
#    -Lessons @(72) `
#    -LogName "rerun_basic_spo_batch_07_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_advanced_batches" `
#    -RunName "advanced_spo_batch_01_lessons_001_010" `
#    -Lessons @(8, 9, 10) `
#    -LogName "rerun_advanced_spo_batch_01_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_advanced_batches" `
#    -RunName "advanced_spo_batch_02_lessons_011_020" `
#    -Lessons @(18, 19, 20) `
#    -LogName "rerun_advanced_spo_batch_02_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_advanced_batches" `
#    -RunName "advanced_spo_batch_03_lessons_021_030" `
#    -Lessons @(28, 29, 30) `
#    -LogName "rerun_advanced_spo_batch_03_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_advanced_batches" `
#    -RunName "advanced_spo_batch_04_lessons_031_040" `
#    -Lessons @(38, 39, 40) `
#    -LogName "rerun_advanced_spo_batch_04_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_advanced_batches" `
#    -RunName "advanced_spo_batch_05_lessons_041_050" `
#    -Lessons @(48, 49, 50) `
#    -LogName "rerun_advanced_spo_batch_05_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_advanced_batches" `
#    -RunName "advanced_spo_batch_06_lessons_051_060" `
#    -Lessons @(59, 60) `
#    -LogName "rerun_advanced_spo_batch_06_incomplete.log"

#Invoke-IsmartSpoRerun `
#    -InputFile "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
#    -OutputDir "docs\generated output_spo_advanced_batches" `
#    -RunName "advanced_spo_batch_07_lessons_061_074" `
#    -Lessons @(67, 68, 69, 70, 71, 72, 73, 74) `
#    -LogName "rerun_advanced_spo_batch_07_incomplete.log"

Invoke-IsmartSpoRerun `
    -InputFile "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
    -OutputDir "docs\generated output_spo_advanced_batches" `
    -RunName "advanced_spo_batch_07_lessons_061_074" `
    -Lessons @(71, 72, 73, 74) `
    -LogName "rerun_advanced_spo_batch_07_incomplete.log"
