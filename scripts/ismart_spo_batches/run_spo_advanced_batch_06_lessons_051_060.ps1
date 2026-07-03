cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
New-Item -ItemType Directory -Force -Path "logs\ismart_spo_batches" | Out-Null

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input "data\ismart\generator\data\generation_input_advanced_spo_from_tracker.json" `
    --from-lesson 51 `
    --to-lesson 60 `
    --output "docs\generated output_spo_advanced_batches" `
    --run-name "advanced_spo_batch_06_lessons_051_060" `
    --verbose 2>&1 | Out-File -FilePath "logs\ismart_spo_batches\advanced_spo_batch_06_lessons_051_060.log" -Encoding utf8
