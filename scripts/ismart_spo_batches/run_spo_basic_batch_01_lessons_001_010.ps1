cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
New-Item -ItemType Directory -Force -Path "logs\ismart_spo_batches" | Out-Null

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input "data\ismart\generator\data\generation_input_basic_spo_from_tracker.json" `
    --from-lesson 1 `
    --to-lesson 10 `
    --output "docs\generated output_spo_basic_batches" `
    --run-name "basic_spo_batch_01_lessons_001_010" `
    --verbose 2>&1 | Out-File -FilePath "logs\ismart_spo_batches\basic_spo_batch_01_lessons_001_010.log" -Encoding utf8
