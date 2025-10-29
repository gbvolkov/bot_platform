"""Data models for the knowledge base manager service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from .enums import ChunkingStrategy, ChunkSizeUnit, EmbeddingBackend, KnowledgeBasePreparationMethod


Metadata = Dict[str, Any]


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration describing how a document should be chunked."""

    strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE_LENGTH
    size: int = 512
    overlap: int = 64
    size_unit: ChunkSizeUnit = ChunkSizeUnit.TOKENS
    respect_sentence_boundaries: bool = False
    respect_table_rows: bool = False
    join_on_retrieval: bool = False


@dataclass(slots=True)
class DocumentChunk:
    """Smallest retrievable unit produced from a source document."""

    chunk_id: str
    document_id: str
    content: str
    metadata: Metadata = field(default_factory=dict)
    graph_links: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class DocumentRecord:
    """In-memory representation of a stored document."""

    document_id: str
    source_path: Optional[Path] = None
    content: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    chunks: MutableMapping[str, DocumentChunk] = field(default_factory=dict)

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = datetime.now(timezone.utc)


@dataclass(slots=True)
class KnowledgeBaseSubscriber:
    """Callback holder used to notify interested parties about KB changes."""

    name: str
    callback: Any
    interested_events: Optional[Sequence[str]] = None


@dataclass(slots=True)
class IndexingRequest:
    """Input payload describing an indexing intention."""

    document_ids: Sequence[str]
    embedding_backend: EmbeddingBackend
    preparation_method: KnowledgeBasePreparationMethod
    chunking: ChunkingConfig
    initiated_by: Optional[str] = None


@dataclass(slots=True)
class WebhookRegistration:
    """Definition of an external webhook subscriber."""

    listener_id: str
    url: str
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
