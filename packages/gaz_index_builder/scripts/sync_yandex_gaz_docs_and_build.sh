#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Sync GAZ source documents from Yandex Object Storage and rebuild the local index.

Usage:
  scripts/sync_yandex_gaz_docs_and_build.sh [options]

Options:
  --env-file PATH       Env file to load. Default: ./.env when present.
  --bucket NAME         Yandex Object Storage bucket. Env: YANDEX_OBJECT_STORAGE_BUCKET or YANDEX_S3_BUCKET.
  --prefix PREFIX       Object prefix inside bucket. Env: YANDEX_OBJECT_STORAGE_PREFIX or YANDEX_S3_PREFIX.
  --endpoint URL        S3 endpoint. Default: https://storage.yandexcloud.net.
  --docs-root PATH      Local docs directory. Default: ./data/gaz-docs.
  --cache-root PATH     Local index/cache directory. Default: ./data/gaz-index.
  --collection-id ID    Collection id. Default: gaz.
  --delete              Delete local files not present in cloud prefix during sync.
  --no-force            Do not force index rebuild when existing artifacts are available.
  --dry-run             Print commands without syncing or rebuilding.
  -h, --help            Show this help.

Required auth:
  Configure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or AWS_PROFILE, for an
  account with read access to the Yandex Object Storage bucket.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  echo "[gaz-index-sync] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-.env}"
ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      [[ $# -ge 2 ]] || die "--env-file requires a value"
      ENV_FILE="$2"
      shift 2
      ;;
    --env-file=*)
      ENV_FILE="${1#*=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

set -- "${ORIGINAL_ARGS[@]}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

AWS_BIN="${AWS_BIN:-aws}"
BUCKET="${YANDEX_OBJECT_STORAGE_BUCKET:-${YANDEX_S3_BUCKET:-}}"
PREFIX="${YANDEX_OBJECT_STORAGE_PREFIX:-${YANDEX_S3_PREFIX:-gaz-docs}}"
ENDPOINT="${YANDEX_OBJECT_STORAGE_ENDPOINT:-${YANDEX_S3_ENDPOINT:-https://storage.yandexcloud.net}}"
DOCS_ROOT="${GAZ_DOCS_ROOT:-./data/gaz-docs}"
CACHE_ROOT="${GAZ_INDEX_ROOT:-./data/gaz-index}"
COLLECTION_ID="${GAZ_COLLECTION_ID:-gaz}"
SYNC_DELETE="${GAZ_SYNC_DELETE:-0}"
FORCE_REBUILD="${GAZ_INDEX_FORCE:-1}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      [[ $# -ge 2 ]] || die "--bucket requires a value"
      BUCKET="$2"
      shift 2
      ;;
    --bucket=*)
      BUCKET="${1#*=}"
      shift
      ;;
    --prefix)
      [[ $# -ge 2 ]] || die "--prefix requires a value"
      PREFIX="$2"
      shift 2
      ;;
    --prefix=*)
      PREFIX="${1#*=}"
      shift
      ;;
    --endpoint)
      [[ $# -ge 2 ]] || die "--endpoint requires a value"
      ENDPOINT="$2"
      shift 2
      ;;
    --endpoint=*)
      ENDPOINT="${1#*=}"
      shift
      ;;
    --docs-root)
      [[ $# -ge 2 ]] || die "--docs-root requires a value"
      DOCS_ROOT="$2"
      shift 2
      ;;
    --docs-root=*)
      DOCS_ROOT="${1#*=}"
      shift
      ;;
    --cache-root)
      [[ $# -ge 2 ]] || die "--cache-root requires a value"
      CACHE_ROOT="$2"
      shift 2
      ;;
    --cache-root=*)
      CACHE_ROOT="${1#*=}"
      shift
      ;;
    --collection-id)
      [[ $# -ge 2 ]] || die "--collection-id requires a value"
      COLLECTION_ID="$2"
      shift 2
      ;;
    --collection-id=*)
      COLLECTION_ID="${1#*=}"
      shift
      ;;
    --delete)
      SYNC_DELETE="1"
      shift
      ;;
    --no-force)
      FORCE_REBUILD="0"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --env-file|--env-file=*)
      # Already handled before sourcing the env file.
      if [[ "$1" == "--env-file" ]]; then
        shift 2
      else
        shift
      fi
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$BUCKET" ]] || die "Set --bucket or YANDEX_OBJECT_STORAGE_BUCKET."

if [[ -n "$PREFIX" ]]; then
  SOURCE_URI="s3://${BUCKET%/}/${PREFIX#/}"
else
  SOURCE_URI="s3://${BUCKET%/}"
fi

SYNC_ARGS=(
  "$AWS_BIN" s3 sync "$SOURCE_URI" "$DOCS_ROOT"
  --endpoint-url "$ENDPOINT"
  --only-show-errors
)
if [[ "$SYNC_DELETE" == "1" || "$SYNC_DELETE" == "true" ]]; then
  SYNC_ARGS+=(--delete)
fi

BUILD_ARGS=(
  gaz-index-build
  --collection-id "$COLLECTION_ID"
  --docs-root "$DOCS_ROOT"
  --cache-root "$CACHE_ROOT"
)
if [[ -f "$ENV_FILE" ]]; then
  BUILD_ARGS+=(--env-file "$ENV_FILE")
fi
if [[ "$FORCE_REBUILD" == "1" || "$FORCE_REBUILD" == "true" ]]; then
  BUILD_ARGS+=(--force)
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  log "dry run"
  printf 'sync command:'
  printf ' %q' "${SYNC_ARGS[@]}"
  printf '\n'
  printf 'build command:'
  if command -v uv >/dev/null 2>&1 && [[ -f "$PACKAGE_DIR/pyproject.toml" ]]; then
    printf ' %q' uv --project "$PACKAGE_DIR" run
  fi
  printf ' %q' "${BUILD_ARGS[@]}"
  printf '\n'
  exit 0
fi

command -v "$AWS_BIN" >/dev/null 2>&1 || die "Command not found: $AWS_BIN. Install AWS CLI v2 or set AWS_BIN."

mkdir -p "$DOCS_ROOT" "$CACHE_ROOT"

log "syncing $SOURCE_URI -> $DOCS_ROOT"
"${SYNC_ARGS[@]}"

log "building index in $CACHE_ROOT"
if command -v uv >/dev/null 2>&1 && [[ -f "$PACKAGE_DIR/pyproject.toml" ]]; then
  uv --project "$PACKAGE_DIR" run "${BUILD_ARGS[@]}"
else
  "${BUILD_ARGS[@]}"
fi
