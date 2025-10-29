"""Event primitives for knowledge base lifecycle notifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Sequence

from .enums import KnowledgeBasePreparationMethod


class KnowledgeBaseEventType(str):
    """String constants describing knowledge base events."""

    DOCUMENT_ADDED = "document_added"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENT_REMOVED = "document_removed"
    DOCUMENT_CHUNKED = "document_chunked"
    DOCUMENT_INDEXED = "document_indexed"
    CHUNK_GRAPH_UPDATED = "chunk_graph_updated"
    RETRIEVER_RELOAD_REQUESTED = "retriever_reload_requested"
    RETRIEVER_RELOAD_COMPLETED = "retriever_reload_completed"


@dataclass(slots=True, frozen=True)
class KnowledgeBaseEvent:
    """A structured description of a knowledge base change."""

    event_type: str
    source: str
    document_ids: Sequence[str] = field(default_factory=tuple)
    payload: Dict[str, Any] = field(default_factory=dict)
    initiated_by: Optional[str] = None
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def with_payload(self, **extra: Any) -> "KnowledgeBaseEvent":
        """Return a cloned event with additional payload values."""
        merged = {**self.payload, **extra}
        return KnowledgeBaseEvent(
            event_type=self.event_type,
            source=self.source,
            document_ids=self.document_ids,
            payload=merged,
            initiated_by=self.initiated_by,
            triggered_at=self.triggered_at,
        )

    def affects(self, document_id: str) -> bool:
        """Check whether the event refers to the given document."""
        return document_id in self.document_ids


def make_index_event(
    document_ids: Iterable[str],
    *,
    source: str,
    initiated_by: Optional[str],
    preparation: KnowledgeBasePreparationMethod,
    embedding_backend: str,
) -> KnowledgeBaseEvent:
    """Convenience helper to build a document indexed event."""
    return KnowledgeBaseEvent(
        event_type=KnowledgeBaseEventType.DOCUMENT_INDEXED,
        source=source,
        document_ids=tuple(document_ids),
        payload={
            "preparation_method": preparation.value,
            "embedding_backend": embedding_backend,
        },
        initiated_by=initiated_by,
    )
