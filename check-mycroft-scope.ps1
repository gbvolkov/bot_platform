Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$allowedExact = @(
    "mycroft_agent_cli.py",
    "check-mycroft-scope.ps1"
)

$allowedPrefixes = @(
    "agents/mycroft_agent/",
    "tests/unit/test_mycroft_",
    "tests/functional/test_mycroft_"
)

function Normalize-RepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return $Path.Trim().Replace("\", "/")
}

function Test-AllowedPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ($allowedExact -contains $Path) {
        return $true
    }

    foreach ($prefix in $allowedPrefixes) {
        if ($Path.StartsWith($prefix)) {
            return $true
        }
    }

    return $false
}

Get-Command git | Out-Null

$repoRoot = (& git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "Unable to resolve git repository root."
}

Set-Location $repoRoot

$trackedChanged = @(& git diff --name-only HEAD)
$untrackedChanged = @(& git ls-files --others --exclude-standard)

$changedFiles = @(
    $trackedChanged
    $untrackedChanged
) | Where-Object { $_ -and $_.Trim() } |
    ForEach-Object { Normalize-RepoPath -Path $_ } |
    Sort-Object -Unique

if ($changedFiles.Count -eq 0) {
    Write-Host "Scope check passed. No changed files."
    exit 0
}

$disallowedFiles = @(
    foreach ($file in $changedFiles) {
        if (-not (Test-AllowedPath -Path $file)) {
            $file
        }
    }
)

if ($disallowedFiles.Count -gt 0) {
    Write-Host "Out-of-scope changes detected:"
    foreach ($file in $disallowedFiles) {
        Write-Host "  $file"
    }
    exit 1
}

Write-Host "Scope check passed."
Write-Host "Changed files:"
foreach ($file in $changedFiles) {
    Write-Host "  $file"
}
