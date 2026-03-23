#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ ! -d "data/gaz-docs" ]]; then
  echo "[run_gaz_documents_service.sh] ERROR: missing data/gaz-docs" >&2
  exit 1
fi

if [[ ! -d "data/gaz_index" ]]; then
  echo "[run_gaz_documents_service.sh] ERROR: missing data/gaz_index" >&2
  exit 1
fi

echo "[run_gaz_documents_service.sh] Starting GAZ documents service from $REPO_DIR"
exec python -m services.kb_manager.app
