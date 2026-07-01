cd C:\Projects\bot_platform
.\.venv\Scripts\activate
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

uv run --no-sync python -m agents.ismart_generator_agent.sequential_runner `
    --input "data\ismart\generator\data\generation_input_basic_10_11_from_tracker.json" `
    --lesson-number 12 `
    --lesson-number 13 `
    --lesson-number 14 `
    --lesson-number 30 `
    --lesson-number 31 `
    --lesson-number 32 `
    --lesson-number 48 `
    --lesson-number 49 `
    --lesson-number 50 `
    --lesson-number 57 `
    --lesson-number 60 `
    --lesson-number 62 `
    --lesson-number 63 `
    --lesson-number 64 `
    --lesson-number 65 `
    --lesson-number 66 `
    --lesson-number 67 `
    --lesson-number 68 `
    --lesson-number 69 `
    --lesson-number 70 `
    --lesson-number 73 `
    --lesson-number 74 `
    --output "docs\generated output_basic_10-11_rerun_skipped" `
    --verbose 2>&1 | Out-File -FilePath "logs\ismart_rerun_skipped_basic_10-11.log" -Encoding utf8

