#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SOURCE_DIR="${1:-}"
REPO_DIR="$(pwd)"

usage() {
  cat <<'EOF'
Install the minimal standalone GAZ documents service bundle.

Usage:
  scripts/install_gaz_service_bundle.sh /path/to/bundle

Required source layout:
  <bundle>/data/gaz-docs/
  <bundle>/data/gaz_index/

Copy targets:
  <repo>/data/gaz-docs/
  <repo>/data/gaz_index/

Notes:
  - This is for services/kb_manager only.
  - load.json, .env, and bot_service config are not used here.
  - gv.env is not copied because the standalone service does not need it
    when a ready-made data/gaz_index bundle is provided.
EOF
}

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

copy_dir() {
  local src="$1"
  local dest="$2"

  mkdir -p "$dest"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src"/ "$dest"/
  else
    cp -a "$src"/. "$dest"/
  fi
  log "Copied $src -> $dest"
}

if [[ -z "$SOURCE_DIR" || "$SOURCE_DIR" == "-h" || "$SOURCE_DIR" == "--help" ]]; then
  usage
  [[ -n "$SOURCE_DIR" ]] && exit 0
  exit 1
fi

[[ -d "$SOURCE_DIR" ]] || die "Source directory does not exist: $SOURCE_DIR"
[[ -f "$REPO_DIR/services/kb_manager/app.py" ]] || die "Current directory is not the bot_platform repo root: $REPO_DIR"
[[ -d "$SOURCE_DIR/data/gaz-docs" ]] || die "Missing source directory: $SOURCE_DIR/data/gaz-docs"
[[ -d "$SOURCE_DIR/data/gaz_index" ]] || die "Missing source directory: $SOURCE_DIR/data/gaz_index"

log "Installing standalone GAZ service data bundle"
log "Source: $SOURCE_DIR"
log "Repo: $REPO_DIR"

copy_dir "$SOURCE_DIR/data/gaz-docs" "$REPO_DIR/data/gaz-docs"
copy_dir "$SOURCE_DIR/data/gaz_index" "$REPO_DIR/data/gaz_index"

log "Installation complete"
