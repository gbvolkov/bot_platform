# GAZ Documents Service

Back to the [documentation index](../index.md).

## Purpose

The GAZ documents service is the GAZ-specific runtime surface inside `services/kb_manager`.

It owns:

- loading source sales materials from `data/gaz-docs`
- building and serving the GAZ manifest, material artifacts, and vector index under `data/gaz_index`
- exposing HTTP endpoints used by `gaz_agent` for broad search, branch-focused retrieval, targeted reads, and collection health checks

Unlike `bot_service`, this service does not manage conversations, agent registration, or message persistence.

Default bind address:

- `0.0.0.0:8081`

Primary implementation files:

- `services/kb_manager/app.py`
- `services/kb_manager/gaz_runtime.py`
- `services/kb_manager/utils/loader.py`
- `agents/gaz_agent/documents.py`

## Runtime Data Model

The service is instantiated with:

- `docs_root=Path("data/gaz-docs")`
- `cache_root=Path("data/gaz_index")`

For collection `gaz`, the service reads and writes:

- `data/gaz-docs/`
- `data/gaz_index/gaz/manifest.json`
- `data/gaz_index/gaz/material_artifacts.json`
- `data/gaz_index/gaz/status.json`
- `data/gaz_index/gaz/vector_store/`

`manifest.json`

- one record per source document
- includes document classification metadata such as `doc_kind`, `branches`, `product_families`, `competitor_tags`, and `body_tags`

`material_artifacts.json`

- cached extracted text split into chunks per candidate document
- used by the runtime search and excerpt-read endpoints

`status.json`

- cached build summary for the collection

`vector_store/`

- persisted Chroma-backed RAG index used during rebuild

## Settings

The standalone GAZ service has very few effective settings.

Directly relevant settings:

- `EMBEDDING_MODEL`
  - read from `config.py`
  - used only when rebuilding the vector index
  - default is `/models/multilingual-e5-large`
- `gv.env`
  - loaded by `config.py` from `./gv.env` first, then `~/.env/gv.env`
  - only matters here if you need to override `EMBEDDING_MODEL` or other shared runtime environment

Fixed service behavior in code:

- HTTP port is fixed to `8081` in `services/kb_manager/app.py`
- GAZ docs path is fixed to `data/gaz-docs`
- GAZ cache path is fixed to `data/gaz_index`
- default collection id is `gaz`

Not used by the standalone GAZ documents service:

- `data/config/bot_service/load.json`
- `BOT_SERVICE_*`
- `GAZ_DOCUMENTS_SERVICE_URL`

`GAZ_DOCUMENTS_SERVICE_URL` is used by `gaz_agent` as the client-side base URL for this service, not by the service itself.

## Run Procedure

### Minimal prerequisites

- repo checkout with the current code
- Python environment with project dependencies installed
- populated `data/gaz-docs`
- either:
  - populated `data/gaz_index`, or
  - ability to rebuild the collection locally

### Rebuild manifest, artifacts, and vector index

Run from repo root:

```bash
python scripts/build_gaz_index.py
```

Useful variants:

```bash
python scripts/build_gaz_index.py --force
CUDA_VISIBLE_DEVICES="" FORCE_CPU=1 NO_CUDA=True python scripts/build_gaz_index.py
```

This rebuilds:

- `manifest.json`
- `material_artifacts.json`
- `status.json`
- `vector_store/`

### Start the service

Run from repo root:

```bash
python -m services.kb_manager.app
```

Or use the repo helper on Ubuntu:

```bash
./scripts/start_gaz.sh
```

### Verify readiness

```bash
curl http://127.0.0.1:8081/gaz/runtime/collections/gaz/status
```

Expected readiness signals:

- `available=true`
- `manifest_built=true`
- `material_artifacts_built=true`
- `rag_index_built=true`

## HTTP Contract

Base URL example:

- `http://127.0.0.1:8081`

All GAZ runtime endpoints are rooted at:

- `/gaz/runtime/collections/{collection_id}`

### `GET /gaz/runtime/collections/{collection_id}/status`

Purpose:

- report whether the collection is ready for runtime use

Request body:

- none

Response shape:

```json
{
  "collection_id": "gaz",
  "available": true,
  "doc_count": 227,
  "docs_root": "/abs/path/data/gaz-docs",
  "manifest_path": "/abs/path/data/gaz_index/gaz/manifest.json",
  "material_artifacts_path": "/abs/path/data/gaz_index/gaz/material_artifacts.json",
  "vector_store_path": "/abs/path/data/gaz_index/gaz/vector_store",
  "manifest_built": true,
  "material_artifacts_built": true,
  "rag_index_built": true
}
```

### `POST /gaz/runtime/collections/{collection_id}/rebuild`

Purpose:

- rebuild the collection manifest, material artifacts, and vector index through HTTP

Request body:

```json
{
  "force": false
}
```

Response shape:

```json
{
  "collection_id": "gaz",
  "available": true,
  "doc_count": 227,
  "manifest_built": true,
  "material_artifacts_built": true,
  "rag_index_built": true,
  "manifest_path": "...",
  "material_artifacts_path": "...",
  "vector_store_path": "...",
  "force": false
}
```

### `POST /gaz/runtime/collections/{collection_id}/materials/search`

Purpose:

- broad candidate search over cached GAZ sales materials

Request body:

```json
{
  "query": "city bus comparison",
  "intent": "compare",
  "families": ["vector next", "citymax"],
  "competitor": "paz",
  "top_k": 5
}
```

Response shape:

```json
{
  "collection_id": "gaz",
  "intent": "compare",
  "query": "city bus comparison",
  "candidate_count": 2,
  "candidates": [
    {
      "candidate_id": "cand_123",
      "title": "Vector Next_Comparison",
      "doc_kind": "comparison",
      "rationale": "supports compare answer; matches competitor context",
      "branch_relevance": null,
      "preview_snippet": "....",
      "metadata": {
        "relative_path": "Vector Next_Comparison.xlsx",
        "score": 14,
        "branches": ["comparison", "passenger_route"],
        "product_families": ["vector_next"],
        "competitor_tags": ["paz"],
        "body_tags": []
      }
    }
  ]
}
```

### `POST /gaz/runtime/collections/{collection_id}/materials/estimate`

Purpose:

- estimate whether deeper research should require a wait/confirmation step

Request body:

```json
{
  "query": "need detailed financing comparison",
  "intended_depth": "deep_research",
  "intent": "financing",
  "families": ["gazelle nn"],
  "competitor": ""
}
```

Response shape:

```json
{
  "collection_id": "gaz",
  "query": "need detailed financing comparison",
  "intent": "financing",
  "intended_depth": "deep_research",
  "estimated_remaining_cost": "high",
  "positive_match_count": 9,
  "max_match_score": 18,
  "requires_hitl_wait_confirmation": true,
  "rationale": "deep_search_needed"
}
```

### `POST /gaz/runtime/collections/{collection_id}/packs/{branch}`

Purpose:

- fetch a branch-focused candidate pack using the current sales slots and problem summary

Request body:

```json
{
  "slots": {
    "competitor": "paz",
    "transport_type": "passenger"
  },
  "problem_summary": "Need a route bus with lower TCO",
  "top_k": 4
}
```

Response shape:

```json
{
  "collection_id": "gaz",
  "branch": "comparison",
  "candidates": [
    {
      "candidate_id": "cand_123",
      "title": "Vector Next_Comparison",
      "doc_kind": "comparison",
      "rationale": "supports comparison reasoning",
      "branch_relevance": "comparison",
      "preview_snippet": "....",
      "metadata": {
        "relative_path": "Vector Next_Comparison.xlsx",
        "score": 17,
        "branches": ["comparison", "passenger_route"],
        "product_families": ["vector_next"],
        "competitor_tags": ["paz"],
        "body_tags": []
      }
    }
  ]
}
```

### `POST /gaz/runtime/collections/{collection_id}/materials/{candidate_id}/read`

Purpose:

- return the most relevant excerpts for a previously surfaced candidate

Request body:

```json
{
  "focus": "payload and diesel engine",
  "max_segments": 3
}
```

Response shape:

```json
{
  "candidate_id": "cand_123",
  "title": "Gazelle NN_Base and options",
  "focus": "payload and diesel engine",
  "excerpts": [
    {
      "excerpt": "....",
      "relevance_reason": "focus match score 6",
      "metadata": {
        "score": 6,
        "chunk_index": 0
      }
    }
  ],
  "metadata": {
    "relative_path": "Gazelle NN_Base and options.xlsx",
    "doc_kind": "configuration",
    "product_families": ["gazelle_nn"],
    "artifact_chunk_count": 8
  }
}
```

### `POST /gaz/runtime/collections/{collection_id}/candidates/{candidate_id}/read`

Purpose:

- legacy alias for the same targeted read operation

Request body and response:

- same as `/materials/{candidate_id}/read`

## Error Behavior

- `404`
  - unknown `candidate_id` on read
- `503`
  - collection runtime assets are missing
  - collection is unavailable
  - read/search/pack execution failed because runtime prerequisites are not ready

The Python client raises `GazDocumentsClientError` for both HTTP and transport failures.

## How `gaz_agent` Uses This Service

The integration point is `GazDocumentsClient` in `agents/gaz_agent/documents.py`.

`gaz_agent` creates the client in `initialize_agent(...)`:

- base URL comes from `GAZ_DOCUMENTS_SERVICE_URL`
- default base URL is `http://127.0.0.1:8081`
- default collection id is `gaz`

The current usage pattern inside `gaz_agent` is:

### Startup availability check

In the `init` node, `gaz_agent` calls:

- `GET /gaz/runtime/collections/gaz/status`

It stores:

- whether the service is reachable
- whether the collection is available
- how many documents are present

If this check fails, the agent records runtime warnings and continues with degraded behavior.

### Planning-time research estimate

In the answer-planning node, `gaz_agent` may call:

- `POST /gaz/runtime/collections/gaz/materials/estimate`

This is used to decide whether a deeper research path should require a wait/confirmation step.

### Tool calls during response generation

During the sales response loop, the tool layer calls:

- `search_sales_materials`
  - maps to `/materials/search`
  - establishes the current allowed candidate ids for the turn
- `read_material`
  - maps to `/materials/{candidate_id}/read`
  - reads targeted excerpts only from previously surfaced candidates
- `get_branch_pack`
  - maps to `/packs/{branch}`
  - narrows candidate documents by branch and current slot state

Composite tools such as sales landscape, comparison digests, and product snapshot generation are built on top of the same client and retrieval primitives.

### Guardrails on top of the contract

`gaz_agent` adds its own orchestration rules above the raw service contract:

- duplicate search prevention per turn
- read budgets per candidate
- explicit allow-list of candidate ids before reads
- branch validation before branch-pack retrieval

These rules live in `agents/gaz_agent/tools.py`, not in the service.
