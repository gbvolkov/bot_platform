cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
New-Item -ItemType Directory -Force -Path "logs\ismart_spo_batches" | Out-Null

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
    --from-lesson 31 `
    --to-lesson 40 `
    --output "docs\generated output_spo_advanced_batches" `
    --run-name "advanced_spo_batch_04_lessons_031_040" `
    --verbose 2>&1 | Out-File -FilePath "logs\ismart_spo_batches\advanced_spo_batch_04_lessons_031_040.log" -Encoding utf8
