# GAZ Index Builder

Standalone package for building the GAZ document manifest, segment store, and
Chroma vector index.

This package intentionally contains only the index-build path. It does not
include `bot_service`, `openai_proxy`, FastAPI service startup, task queues, or
agent runtime code.

## Contents

- `gaz_index_builder.gaz_runtime.GazRuntimeService`
  - copied from `services/kb_manager/gaz_runtime.py`
  - build/search/read procedures are preserved
- `gaz_index_builder.build`
  - CLI wrapper adapted from `scripts/build_gaz_index.py`
  - import route changed to the package-local runtime
- `models.toml`
  - package-local model alias registry used by `rag_lib.llm.factory`

## Install

From this directory:

```powershell
uv sync
```

Or install into an existing Python 3.13 environment:

```powershell
pip install .
```

The dependency set is intentionally narrower than the full platform. It includes
`rag-lib`, Chroma, LangChain integration packages, document parsers, and embedding
support only.

## Environment

The builder reads a local `.env` from the directory where you run the command.
It does not need the platform-level `gv.env`.

Choose one template:

```powershell
Copy-Item .\env.openai.example .\.env
```

or:

```powershell
Copy-Item .\env.local-embeddings.example .\.env
```

Then edit `.env` and fill `OPENAI_API_KEY`.

Minimal variables:

```text
OPENAI_API_KEY=
LLM_PROVIDER=openai
LLM_MODEL=mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

If you run from another directory, pass the env file explicitly:

```powershell
gaz-index-build --env-file C:\path\to\.env --docs-root C:\path\to\gaz-docs --cache-root C:\path\to\gaz_index --force
```

Notes:

- Image files require `tesseract` on `PATH`, because the reused image loader uses OCR.
- Table summaries and image summaries use the configured LLM provider.
- Relative `--docs-root` and `--cache-root` are resolved against the directory where the command is started.
- Legacy `EMBEDDING_MODEL` is accepted as an alias for `EMBEDDING_MODEL_NAME` when the latter is absent.

## Build

```powershell
gaz-index-build --docs-root C:\data\gaz-docs --cache-root C:\data\gaz_index --force
```

Equivalent module form:

```powershell
python -m gaz_index_builder --docs-root C:\data\gaz-docs --cache-root C:\data\gaz_index --force
```

Default values match the platform script:

```text
collection-id = gaz
docs-root     = data/gaz-docs
cache-root    = data/gaz_index
```

The output is a JSON payload containing `rebuild` and `status`.

Expected ready status:

```json
{
  "available": true,
  "manifest_built": true,
  "segment_store_built": true,
  "rag_index_built": true,
  "document_count": 227
}
```

## Rebuild Safety

`--force` rebuilds the collection in place. Back up the existing collection
directory before rebuilding production artifacts:

```powershell
Copy-Item C:\data\gaz_index\gaz C:\data\gaz_index\gaz_backup -Recurse
gaz-index-build --docs-root C:\data\gaz-docs --cache-root C:\data\gaz_index --force
```

## Sync From Yandex Object Storage

The package includes a Bash helper that copies source documents from Yandex
Object Storage into `./data/gaz-docs`, then builds the index into
`./data/gaz-index`.

Prerequisites:

- AWS CLI v2 available as `aws`
- Yandex Object Storage static access key configured through `AWS_ACCESS_KEY_ID`
  and `AWS_SECRET_ACCESS_KEY`, or through `AWS_PROFILE`
- local `.env` copied from one of the templates above

Set these values in `.env`:

```env
YANDEX_OBJECT_STORAGE_BUCKET=gbv-gazsales
YANDEX_OBJECT_STORAGE_PREFIX=gaz-docs
YANDEX_OBJECT_STORAGE_ENDPOINT=https://storage.yandexcloud.net
YANDEX_CLOUD_FOLDER_ID=b1gt23jcnbib316dc05q
YANDEX_OBJECT_STORAGE_RESOURCE_ID=e3e3b2vb681dc13mlm7i
YANDEX_OBJECT_STORAGE_CLASS=STANDARD
YANDEX_OBJECT_STORAGE_VERSIONING=disabled
YANDEX_OBJECT_STORAGE_CREATED_AT=2026-05-22T13:02:12.984088Z

GAZ_DOCS_ROOT=./data/gaz-docs
GAZ_INDEX_ROOT=./data/gaz-index
GAZ_COLLECTION_ID=gaz
```

The sync command uses only bucket, prefix, endpoint, and AWS-compatible
credentials. Folder ID, resource ID, storage class, versioning, and creation
date are retained in `.env` as Yandex Cloud bucket metadata for audit/logging.

Run from the package directory:

```bash
bash scripts/sync_yandex_gaz_docs_and_build.sh
```

Useful variants:

```bash
bash scripts/sync_yandex_gaz_docs_and_build.sh --bucket my-bucket --prefix path/to/gaz-docs
bash scripts/sync_yandex_gaz_docs_and_build.sh --dry-run
bash scripts/sync_yandex_gaz_docs_and_build.sh --delete
```

`--delete` mirrors the cloud prefix exactly by removing local files that no
longer exist in Object Storage. Leave it off when you only want additive sync.
