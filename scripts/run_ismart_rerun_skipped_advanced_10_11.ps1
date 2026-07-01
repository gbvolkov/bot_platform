cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input "data\ismart\generator\data\generation_input_advanced_10_11_from_tracker.json" `
    --lesson-number 31 `
    --lesson-number 68 `
    --lesson-number 69 `
    --lesson-number 70 `
    --lesson-number 73 `
    --lesson-number 74 `
    --output "docs\generated output_adv_10-11_rerun_skipped" `
    --verbose 2>&1 | Out-File -FilePath "logs\ismart_rerun_skipped_advanced_10-11.log" -Encoding utf8

