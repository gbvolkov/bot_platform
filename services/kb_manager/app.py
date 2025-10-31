"""HTTP entrypoint for the knowledge base manager service."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import AnyHttpUrl, BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install 'fastapi' and 'pydantic' to run the KB manager service API.") from exc

import config

from .enums import ChunkingStrategy, ChunkSizeUnit, EmbeddingBackend, KnowledgeBasePreparationMethod
from .models import ChunkingConfig, WebhookRegistration
from .notifications import get_broadcaster
from .service import KnowledgeBaseManagerService

logger = logging.getLogger(__name__)

app = FastAPI(title="Knowledge Base Manager", version="0.1.0")


class ChunkingConfigPayload(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE_LENGTH
    size: int = Field(512, ge=1)
    overlap: int = Field(64, ge=0)
    size_unit: ChunkSizeUnit = ChunkSizeUnit.TOKENS
    respect_sentence_boundaries: bool = False
    respect_table_rows: bool = False
    join_on_retrieval: bool = False

    def to_config(self) -> ChunkingConfig:
        return ChunkingConfig(
            strategy=self.strategy,
            size=self.size,
            overlap=self.overlap,
            size_unit=self.size_unit,
            respect_sentence_boundaries=self.respect_sentence_boundaries,
            respect_table_rows=self.respect_table_rows,
            join_on_retrieval=self.join_on_retrieval,
        )


class DocumentPayload(BaseModel):
    content: str
    document_id: Optional[str] = None
    metadata: Dict[str, Any] | None = None
    auto_chunk: bool = True
    auto_index: bool = False
    initiated_by: Optional[str] = None


class IndexingPayload(BaseModel):
    document_ids: Sequence[str] = ()
    initiated_by: Optional[str] = None
    notify_agents: bool = True
    index_path: Optional[str] = None
    overwrite: bool = True


class ReloadPayload(BaseModel):
    reason: str = "manual reload"
    document_ids: Sequence[str] = ()
    initiated_by: Optional[str] = None


class DirectoryIngestPayload(BaseModel):
    directory: str
    product: Optional[str] = None
    extensions: Optional[Sequence[str]] = None
    auto_chunk: bool = True
    chunking: Optional[ChunkingConfigPayload] = None
    initiated_by: Optional[str] = None


class WebhookPayload(BaseModel):
    listener_id: str = Field(..., min_length=1)
    url: AnyHttpUrl
    secret: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)

    def to_registration(self) -> WebhookRegistration:
        return WebhookRegistration(
            listener_id=self.listener_id,
            url=str(self.url),
            secret=self.secret,
            headers=dict(self.headers),
        )


class IndexPathPayload(BaseModel):
    path: str


kb_service = KnowledgeBaseManagerService(
    reload_broadcaster=get_broadcaster(),
    default_index_path=Path(config.PRODUCT_INDEX_FOLDER),
)


@app.post("/documents", response_model=Dict[str, str])
def add_document(payload: DocumentPayload) -> Dict[str, str]:
    doc_id = kb_service.add_document(
        payload.content,
        document_id=payload.document_id,
        metadata=payload.metadata or {},
        auto_chunk=payload.auto_chunk,
        auto_index=payload.auto_index,
        initiated_by=payload.initiated_by,
    )
    return {"document_id": doc_id}


@app.post("/ingest/directory", response_model=Dict[str, Any])
def ingest_directory(payload: DirectoryIngestPayload) -> Dict[str, Any]:
    chunk_config = payload.chunking.to_config() if payload.chunking else None
    document_ids = kb_service.ingest_directory(
        payload.directory,
        product=payload.product,
        allowed_extensions=list(payload.extensions) if payload.extensions else None,
        chunk_on_ingest=payload.auto_chunk,
        chunking=chunk_config,
        initiated_by=payload.initiated_by,
    )
    return {"document_ids": document_ids, "count": len(document_ids)}


@app.post("/documents/{document_id}/chunk", response_model=Dict[str, Any])
def chunk_document(document_id: str, payload: ChunkingConfigPayload | None = None) -> Dict[str, Any]:
    try:
        chunks = kb_service.chunk_document(
            document_id,
            chunking=payload.to_config() if payload else None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"document_id": document_id, "chunk_count": len(chunks)}


@app.get("/documents/{document_id}/chunks", response_model=Dict[str, Any])
def list_chunks(document_id: str) -> Dict[str, Any]:
    try:
        chunks = kb_service.show_chunked_document(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "document_id": document_id,
        "chunks": [chunk.__dict__ for chunk in chunks],
    }


@app.post("/documents/index", response_model=Dict[str, Any])
def index_documents(payload: IndexingPayload) -> Dict[str, Any]:
    try:
        event = kb_service.index_document(
            payload.document_ids,
            initiated_by=payload.initiated_by,
            notify_agents=payload.notify_agents,
            vector_store_path=Path(payload.index_path).expanduser() if payload.index_path else None,
            overwrite=payload.overwrite,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "event_type": event.event_type,
        "document_ids": list(event.document_ids),
        "payload": event.payload,
    }


@app.post("/webhooks", response_model=Dict[str, str])
def register_webhook(payload: WebhookPayload) -> Dict[str, str]:
    kb_service.register_webhook(payload.to_registration())
    return {"listener_id": payload.listener_id}


@app.get("/webhooks", response_model=Dict[str, Any])
def list_webhooks() -> Dict[str, Any]:
    registrations = [
        {
            "listener_id": registration.listener_id,
            "url": registration.url,
            "has_secret": bool(registration.secret),
            "header_keys": list(registration.headers.keys()),
        }
        for registration in kb_service.list_webhooks()
    ]
    return {"webhooks": registrations}


@app.delete("/webhooks/{listener_id}", response_model=Dict[str, str])
def remove_webhook(listener_id: str) -> Dict[str, str]:
    kb_service.remove_webhook(listener_id)
    return {"listener_id": listener_id}


@app.post("/reload", response_model=Dict[str, Any])
def trigger_reload(payload: ReloadPayload) -> JSONResponse:
    results = kb_service.trigger_retriever_reload(
        reason=payload.reason,
        document_ids=payload.document_ids,
        initiated_by=payload.initiated_by,
    )
    return JSONResponse({"results": results})


@app.post("/config/index-path", response_model=Dict[str, str])
def configure_index_path(payload: IndexPathPayload) -> Dict[str, str]:
    resolved = Path(payload.path).expanduser()
    kb_service.set_default_index_path(resolved)
    return {"path": str(resolved)}


@app.post("/config/embedding", response_model=Dict[str, str])
def set_embedding(payload: Dict[str, str]) -> Dict[str, str]:
    try:
        backend = EmbeddingBackend(payload["backend"])
    except KeyError as exc:
        raise HTTPException(status_code=400, detail="Missing 'backend' field") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    kb_service.set_embedding_backend(backend)
    return {"backend": backend.value}


@app.post("/config/preparation", response_model=Dict[str, str])
def set_preparation(payload: Dict[str, str]) -> Dict[str, str]:
    try:
        method = KnowledgeBasePreparationMethod(payload["method"])
    except KeyError as exc:
        raise HTTPException(status_code=400, detail="Missing 'method' field") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    kb_service.set_preparation_method(method)
    return {"method": method.value}


@app.post("/config/chunking", response_model=Dict[str, Any])
def configure_chunking(payload: ChunkingConfigPayload) -> Dict[str, Any]:
    config = kb_service.configure_chunking(
        strategy=payload.strategy,
        size=payload.size,
        overlap=payload.overlap,
        size_unit=payload.size_unit,
        respect_sentences=payload.respect_sentence_boundaries,
        respect_table_rows=payload.respect_table_rows,
        join_on_retrieval=payload.join_on_retrieval,
    )
    return {"chunking": config.__dict__}


def create_app() -> FastAPI:
    """Factory that returns the FastAPI application instance."""
    return app


def main() -> None:  # pragma: no cover - executable entrypoint
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Install 'uvicorn' to run the KB manager service.") from exc
    uvicorn.run("services.kb_manager.app:app", host="0.0.0.0", port=8081, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
