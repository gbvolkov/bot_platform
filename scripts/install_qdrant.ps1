param(
    [ValidateSet('Docker', 'Windows')][string]$Mode = 'Docker',
    [string]$DataDirectory = (Join-Path $PSScriptRoot '../data/services/qdrant'),
    [ValidateRange(1024, 65535)][int]$Port = 6333,
    [switch]$InstallOnly
)
$ErrorActionPreference = 'Stop'
$version = '1.19.2'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($Mode -eq 'Docker') {
    $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
    $docker = if ($dockerCommand) { $dockerCommand.Source } else { 'C:\Program Files\Docker\Docker\resources\bin\docker.exe' }
    $env:PLATFORM_QDRANT_HTTP_PORT = [string]$Port
    & $docker compose -f (Join-Path $repoRoot 'deploy/qdrant.compose.yml') pull qdrant
    if ($LASTEXITCODE -ne 0) { throw 'Qdrant image installation failed; check the Docker daemon.' }
    if ($InstallOnly) { return }
    & $docker compose -f (Join-Path $repoRoot 'deploy/qdrant.compose.yml') up -d --wait qdrant
    if ($LASTEXITCODE -ne 0) { throw 'Qdrant did not become ready.' }
} else {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'The pinned package requires Windows x64.' }
    $serviceRoot = [IO.Path]::GetFullPath($DataDirectory)
    $binaryRoot = Join-Path $serviceRoot "bin/$version"
    New-Item -ItemType Directory -Path $binaryRoot -Force | Out-Null
    $archive = Join-Path $binaryRoot 'qdrant.zip'
    $executable = Join-Path $binaryRoot 'qdrant.exe'
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -Uri "https://github.com/qdrant/qdrant/releases/download/v$version/qdrant-x86_64-pc-windows-msvc.zip" -OutFile $archive
    }
    $expected = '7d86596f16c6e85d45a50312f5e16ccb51e059b62e3d7308110cb65b4c799a4d'
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw 'Official Qdrant package SHA256 does not match the pinned release.'
    }
    if (Test-Path -LiteralPath $executable) {
        $zip = [IO.Compression.ZipFile]::OpenRead($archive)
        try {
            $entry = $zip.GetEntry('qdrant.exe')
            if (-not $entry) { throw 'The official archive has no expected executable.' }
            $stream = $entry.Open()
            $algorithm = [Security.Cryptography.SHA256]::Create()
            try { $binaryHash = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
            finally { $stream.Dispose(); $algorithm.Dispose() }
        } finally { $zip.Dispose() }
        if ((Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant() -ne $binaryHash) {
            throw 'Installed executable differs from the verified archive; review it before reinstalling.'
        }
    } else {
        Expand-Archive -LiteralPath $archive -DestinationPath $binaryRoot -Force
    }
    if (-not (Test-Path -LiteralPath $executable)) { throw 'Qdrant executable is missing from the release archive.' }
    $configuration = Join-Path $serviceRoot 'local.yaml'
    $storage = (Join-Path $serviceRoot 'storage').Replace('\', '/').Replace("'", "''")
    $snapshots = (Join-Path $serviceRoot 'snapshots').Replace('\', '/').Replace("'", "''")
    # A dedicated config keeps local ports and persistent data independent of cwd.
    $yaml = "storage:`n  storage_path: '$storage'`n  snapshots_path: '$snapshots'`nservice:`n  host: 127.0.0.1`n  http_port: $Port`n  grpc_port: $($Port + 1)`ntelemetry_disabled: true`n"
    if ((Test-Path -LiteralPath $configuration) -and (Get-Content -Raw -LiteralPath $configuration) -ne $yaml) {
        throw "Existing configuration differs. Review $configuration before changing this installation."
    }
    [IO.File]::WriteAllText($configuration, $yaml, [Text.UTF8Encoding]::new($false))
    if ($InstallOnly) { Write-Output "Installed Qdrant $version in $binaryRoot"; return }
    $existing = $null
    try { $existing = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/" -TimeoutSec 2 } catch { }
    if ($existing) {
        if ($existing.title -ne 'qdrant - vector search engine' -or $existing.version -ne $version) {
            throw "Port $Port belongs to a different service/version; nothing was stopped."
        }
    } else {
        $process = Start-Process -FilePath $executable -ArgumentList @('--config-path', ('"' + $configuration + '"')) -WorkingDirectory $serviceRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $serviceRoot 'stdout.log') -RedirectStandardError (Join-Path $serviceRoot 'stderr.log') -PassThru
        [IO.File]::WriteAllText((Join-Path $serviceRoot 'qdrant.pid'), [string]$process.Id)
    }
}
$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/readyz" -TimeoutSec 2
        $ready = $true
        break
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) { throw 'Qdrant readiness check failed. Inspect the service logs.' }
$server = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/"
Write-Output "Qdrant $($server.version) is ready at http://127.0.0.1:$Port; persistent data is enabled."
