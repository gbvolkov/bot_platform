cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input "data\ismart\generator\data\generation_input_advanced_10_11_from_tracker.json" `
    --lesson-number 34 `
    --output "docs\generated output_adv_10-11_rerun_skipped" `
    --verbose 2>&1 | Out-File -FilePath "logs\ismart_rerun_skipped_advanced_10-11.log" -Encoding utf8

