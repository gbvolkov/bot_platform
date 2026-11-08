# Standalone Qdrant installation

The optional installer provides Qdrant Server **1.19.2** independently of the platform services.

## Docker

```powershell
./scripts/install_qdrant.ps1 -Mode Docker
```

Equivalent command:

```text
docker compose -f deploy/qdrant.compose.yml up -d --wait qdrant
```

The HTTP port is bound to loopback. Data and snapshots are stored in persistent named volumes.

## Windows

```powershell
./scripts/install_qdrant.ps1 -Mode Windows
```

The installer verifies the release checksum, starts a hidden process, checks `/readyz`, and stores data, logs, and the process ID under `data/services/qdrant`. Run it again after a reboot; it does not register a system service.

Keep the data and snapshots when restarting or upgrading. See the [Qdrant release](https://github.com/qdrant/qdrant/releases/tag/v1.19.2) and [installation documentation](https://qdrant.tech/documentation/operations/installation/).
