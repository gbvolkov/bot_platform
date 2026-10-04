cd C:\Projects\bot_platform
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = "C:\Projects\bot_platform"
$scripts = Get-ChildItem -LiteralPath $scriptDir -File -Filter "run_*_batch_*.ps1" | Sort-Object Name

foreach ($script in $scripts) {
    Start-Process powershell.exe -WorkingDirectory $repoRoot -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "`"$($script.FullName)`""
    )
}
