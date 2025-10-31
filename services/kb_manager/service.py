"""Skeleton implementation of the standalone knowledge base manager service."""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TYPE_CHECKING
import pickle

from langchain_classic.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from filelock import FileLock, Timeout as FileLockTimeout  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    FileLock = None  # type: ignore
    FileLockTimeout = RuntimeError  # type: ignore

if TYPE_CHECKING:
    from filelock import FileLock as FileLockType
else:
    FileLockType = Any

from .enums import (
    ChunkingStrategy,
    ChunkSizeUnit,
    EmbeddingBackend,
    KnowledgeBasePreparationMethod,
)
from .events import KnowledgeBaseEvent, KnowledgeBaseEventType, make_index_event
from .models import (
    ChunkingConfig,
    DocumentChunk,
    DocumentRecord,
    KnowledgeBaseSubscriber,
    WebhookRegistration,
)
from .notifications import (
    KBReloadContext,
    ReloadBroadcaster,
    WebhookRegistry,
    get_broadcaster,
    get_webhook_registry,
)
from .utils import load_documents

logger = logging.getLogger(__name__)


class KnowledgeBaseManagerService:
    """Facade that orchestrates document ingestion, chunking, indexing, and notifications."""

    def __init__(
        self,
        *,
        lock_path: Optional[Path] = None,
        default_chunking: Optional[ChunkingConfig] = None,
        default_embedding: EmbeddingBackend = EmbeddingBackend.DEFAULT,
        default_preparation: KnowledgeBasePreparationMethod = KnowledgeBasePreparationMethod.CLEAR_CHUNKING,
        reload_broadcaster: Optional[ReloadBroadcaster] = None,
        webhook_registry: Optional[WebhookRegistry] = None,
        default_index_path: Optional[Path] = None,
    ) -> None:
        self._documents: MutableMapping[str, DocumentRecord] = {}
        self._subscribers: Dict[str, KnowledgeBaseSubscriber] = {}
        self._chunking_config = default_chunking or ChunkingConfig()
        self._embedding_backend = default_embedding
        self._preparation_method = default_preparation
        self._mutex = RLock()
        self._lock_path = lock_path
        self._file_lock: Optional[FileLockType] = None
        if lock_path and FileLock:
            self._file_lock = FileLock(str(lock_path))
        self._source_name = "kb_manager_service"
        self._reload_broadcaster = reload_broadcaster or get_broadcaster()
        self._webhook_registry = webhook_registry or get_webhook_registry()
        self._default_index_path = Path(default_index_path) if default_index_path else None
        logger.debug(
            "KnowledgeBaseManagerService initialised (embedding=%s, preparation=%s, lock=%s)",
            self._embedding_backend.value,
            self._preparation_method.value,
            lock_path,
        )

    # ----------------------------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------------------------

    @property
    def embedding_backend(self) -> EmbeddingBackend:
        return self._embedding_backend

    @property
    def preparation_method(self) -> KnowledgeBasePreparationMethod:
        return self._preparation_method

    @property
    def chunking_config(self) -> ChunkingConfig:
        return self._chunking_config

    def set_embedding_backend(self, backend: EmbeddingBackend) -> None:
        logger.info("Switching embedding backend to %s", backend.value)
        self._embedding_backend = backend

    def set_preparation_method(self, method: KnowledgeBasePreparationMethod) -> None:
        logger.info("Switching preparation method to %s", method.value)
        self._preparation_method = method

    def configure_chunking(
        self,
        *,
        strategy: Optional[ChunkingStrategy] = None,
        size: Optional[int] = None,
        overlap: Optional[int] = None,
        size_unit: Optional[ChunkSizeUnit] = None,
        respect_sentences: Optional[bool] = None,
        respect_table_rows: Optional[bool] = None,
        join_on_retrieval: Optional[bool] = None,
    ) -> ChunkingConfig:
        """Update the default chunking configuration."""
        config = ChunkingConfig(
            strategy=strategy or self._chunking_config.strategy,
            size=size or self._chunking_config.size,
            overlap=overlap if overlap is not None else self._chunking_config.overlap,
            size_unit=size_unit or self._chunking_config.size_unit,
            respect_sentence_boundaries=(
                respect_sentences if respect_sentences is not None else self._chunking_config.respect_sentence_boundaries
            ),
            respect_table_rows=(
                respect_table_rows if respect_table_rows is not None else self._chunking_config.respect_table_rows
            ),
            join_on_retrieval=(
                join_on_retrieval if join_on_retrieval is not None else self._chunking_config.join_on_retrieval
            ),
        )
        logger.info("Updated chunking config: %s", config)
        self._chunking_config = config
        return config

    def set_default_index_path(self, path: Path) -> None:
        """Update the default location for persisted vector stores."""
        resolved = Path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        self._default_index_path = resolved
        logger.info("Default index path set to %s", resolved)

    # ----------------------------------------------------------------------------------
    # Subscription management
    # ----------------------------------------------------------------------------------

    def register_subscriber(self, subscriber: KnowledgeBaseSubscriber) -> None:
        """Register a callback that will receive KB events."""
        self._subscribers[subscriber.name] = subscriber
        logger.debug("Registered KB subscriber '%s'", subscriber.name)

    def remove_subscriber(self, name: str) -> None:
        """Remove a previously registered subscriber."""
        if name in self._subscribers:
            del self._subscribers[name]
            logger.debug("Removed KB subscriber '%s'", name)

    def _dispatch_event(self, event: KnowledgeBaseEvent) -> None:
        """Send an event to all interested subscribers."""
        for subscriber in list(self._subscribers.values()):
            if subscriber.interested_events and event.event_type not in subscriber.interested_events:
                continue
            try:
                subscriber.callback(event)
            except Exception:  # pragma: no cover - external callbacks
                logger.exception("Subscriber '%s' failed to handle event '%s'", subscriber.name, event.event_type)

    def register_webhook(self, registration: WebhookRegistration) -> None:
        """Register an external webhook endpoint for reload notifications."""
        self._webhook_registry.register(registration)

    def remove_webhook(self, listener_id: str) -> None:
        """Remove a webhook endpoint by identifier."""
        self._webhook_registry.unregister(listener_id)

    def list_webhooks(self) -> Tuple[WebhookRegistration, ...]:
        """Return the registered webhook endpoints."""
        return self._webhook_registry.list()

    # ----------------------------------------------------------------------------------
    # Document lifecycle
    # ----------------------------------------------------------------------------------

    def ingest_directory(
        self,
        directory_path: str,
        *,
        product: Optional[str] = None,
        allowed_extensions: Optional[Sequence[str]] = None,
        chunk_on_ingest: bool = True,
        chunking: Optional[ChunkingConfig] = None,
        initiated_by: Optional[str] = None,
    ) -> List[str]:
        """Load documents from a filesystem directory and register them in the KB."""
        base_path = Path(directory_path).expanduser().resolve()
        if not base_path.exists() or not base_path.is_dir():
            raise FileNotFoundError(f"Directory '{directory_path}' does not exist or is not a directory.")

        extension_filter = None
        if allowed_extensions:
            extension_filter = [
                ext if ext.startswith(".") else f".{ext}"
                for ext in (item.lower() for item in allowed_extensions)
            ]
        documents = load_documents(str(base_path), extentions=extension_filter)
        if not documents:
            logger.warning("No documents discovered under %s", base_path)
            return []

        registered_ids: List[str] = []
        for doc in documents:
            metadata = dict(doc.metadata or {})
            if product and "product" not in metadata:
                metadata["product"] = product
            relative_path = metadata.get("relative_path")
            source_path = base_path / relative_path if isinstance(relative_path, str) else None
            doc_id = self.add_document(
                doc.page_content,
                metadata=metadata,
                source_path=source_path,
                auto_chunk=False,
                auto_index=False,
                initiated_by=initiated_by,
            )
            registered_ids.append(doc_id)

        if chunk_on_ingest:
            for doc_id in registered_ids:
                self.chunk_document(doc_id, chunking=chunking, initiated_by=initiated_by)

        logger.info("Ingested %s documents from %s", len(registered_ids), base_path)
        return registered_ids

    def add_document(
        self,
        content: str,
        *,
        document_id: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
        source_path: Optional[Path] = None,
        auto_chunk: bool = True,
        auto_index: bool = False,
        initiated_by: Optional[str] = None,
    ) -> str:
        """Store a new document and optionally trigger chunking/indexing."""
        doc_id = document_id or uuid.uuid4().hex
        resolved_source = Path(source_path) if isinstance(source_path, (str, Path)) else None
        record_metadata = dict(metadata or {})
        with self._writable_storage():
            record = DocumentRecord(
                document_id=doc_id,
                source_path=resolved_source,
                content=content,
                metadata=record_metadata,
            )
            self._documents[doc_id] = record
            record.touch()
        logger.info("Document '%s' added to knowledge base", doc_id)
        self._dispatch_event(
            KnowledgeBaseEvent(
                event_type=KnowledgeBaseEventType.DOCUMENT_ADDED,
                source=self._source_name,
                document_ids=(doc_id,),
                payload={"metadata": record_metadata},
                initiated_by=initiated_by,
            )
        )
        if auto_chunk:
            self.chunk_document(doc_id, initiated_by=initiated_by)
        if auto_index:
            if self._default_index_path is None:
                raise ValueError("auto_index=True but no default index path configured.")
            self.index_document(
                (doc_id,),
                initiated_by=initiated_by,
                notify_agents=True,
                vector_store_path=self._default_index_path,
            )
        return doc_id

    def chunk_document(
        self,
        document_id: str,
        *,
        chunking: Optional[ChunkingConfig] = None,
        initiated_by: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Chunk a document according to the provided configuration."""
        record = self._documents.get(document_id)
        if record is None:
            raise KeyError(f"Document '{document_id}' is not present in the knowledge base.")

        chunk_config = chunking or self._chunking_config
        text = self._ensure_document_content(record)
        if not text:
            logger.warning("Document '%s' contains no text to chunk.", document_id)
            record.chunks.clear()
            return []

        splitter = self._build_text_splitter(chunk_config)
        segments = splitter.split_text(text)
        logger.debug(
            "Chunking document '%s' into %s segments using %s",
            document_id,
            len(segments),
            chunk_config.strategy.value,
        )

        record.chunks.clear()
        for index, segment in enumerate(segments):
            chunk_id = f"{document_id}:{index}"
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "chunk_strategy": chunk_config.strategy.value,
                    "chunk_size": len(segment),
                    "chunk_overlap": chunk_config.overlap,
                }
            )
            record.chunks[chunk_id] = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content=segment,
                metadata=metadata,
            )

        record.touch()
        self._dispatch_event(
            KnowledgeBaseEvent(
                event_type=KnowledgeBaseEventType.DOCUMENT_CHUNKED,
                source=self._source_name,
                document_ids=(document_id,),
                payload={
                    "chunking": {
                        "strategy": chunk_config.strategy.value,
                        "size": chunk_config.size,
                        "overlap": chunk_config.overlap,
                        "size_unit": chunk_config.size_unit.value,
                        "respect_sentences": chunk_config.respect_sentence_boundaries,
                        "respect_table_rows": chunk_config.respect_table_rows,
                        "join_on_retrieval": chunk_config.join_on_retrieval,
                    },
                    "chunk_count": len(record.chunks),
                },
                initiated_by=initiated_by,
            )
        )
        return list(record.chunks.values())

    def _build_text_splitter(self, config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
        """Construct a text splitter based on the chunking configuration."""
        strategy = config.strategy
        if strategy not in (
            ChunkingStrategy.SIMPLE_LENGTH,
            ChunkingStrategy.SENTENCE,
        ):
            logger.warning("Chunking strategy '%s' is not fully supported; falling back to simple length.", strategy.value)

        if config.size_unit == ChunkSizeUnit.TOKENS:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=config.size,
                chunk_overlap=config.overlap,
                encoding_name="cl100k_base",
            )
        else:
            separators = ["\n\n", "\n", " ", ""]
            if strategy == ChunkingStrategy.SENTENCE:
                separators = ["\n\n", ". ", "! ", "? ", "\n", " ", ""]
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.size,
                chunk_overlap=config.overlap,
                separators=separators,
            )
        return splitter

    def _ensure_document_content(self, record: DocumentRecord) -> str:
        """Ensure the document record has textual content loaded."""
        if isinstance(record.content, str) and record.content:
            return record.content
        if record.source_path and record.source_path.exists():
            try:
                text = record.source_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = record.source_path.read_text(encoding="utf-8", errors="ignore")
            record.content = text
            return text
        return ""

    def _collect_chunk_documents(self, document_ids: Sequence[str]) -> List[Document]:
        """Ensure chunks exist for the supplied documents and materialise them as LangChain Documents."""
        chunk_docs: List[Document] = []
        for doc_id in document_ids:
            record = self._documents[doc_id]
            if not record.chunks:
                self.chunk_document(doc_id)
            for chunk in record.chunks.values():
                if not chunk.content:
                    continue
                metadata = dict(chunk.metadata)
                metadata.setdefault("document_id", doc_id)
                chunk_docs.append(Document(page_content=chunk.content, metadata=metadata))
        return chunk_docs

    def _prepare_documents_for_index(self, chunk_docs: List[Document]) -> List[Document]:
        """Apply the configured preparation strategy to generate vectorisable documents."""
        if self._preparation_method == KnowledgeBasePreparationMethod.CLEAR_CHUNKING:
            return chunk_docs
        if self._preparation_method == KnowledgeBasePreparationMethod.RAPTOR:
            return self._augment_with_raptor(chunk_docs)
        if self._preparation_method == KnowledgeBasePreparationMethod.QA_TABLE:
            raise NotImplementedError("QA table preparation is not implemented yet.")
        raise NotImplementedError(f"Preparation method '{self._preparation_method.value}' is not supported.")

    def _augment_with_raptor(self, chunk_docs: List[Document]) -> List[Document]:
        """Augment chunk documents with hierarchical RAPTOR summaries."""
        try:
            from .utils.raptor.tree_builder import recursive_embed_cluster_summarize  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RAPTOR preparation requires optional dependencies (umap-learn, scikit-learn, langchain-openai)."
            ) from exc

        leaf_texts = [doc.page_content for doc in chunk_docs if isinstance(doc.page_content, str) and doc.page_content.strip()]
        if not leaf_texts:
            return chunk_docs

        tree_results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
        augmented = list(chunk_docs)
        summary_counter = 0
        for level, (_, summary_df) in sorted(tree_results.items()):
            if "summaries" not in summary_df:
                continue
            for summary in summary_df["summaries"].tolist():
                if not isinstance(summary, str) or not summary.strip():
                    continue
                metadata = {
                    "type": "raptor_summary",
                    "raptor_level": level,
                    "summary_index": summary_counter,
                }
                augmented.append(Document(page_content=summary, metadata=metadata))
                summary_counter += 1
        return augmented

    def _get_embedding_model(self):
        """Resolve the embedding model for index construction."""
        if self._embedding_backend == EmbeddingBackend.DEFAULT:
            from agents.retrievers.utils.models_builder import getEmbeddingModel

            return getEmbeddingModel()
        raise NotImplementedError(f"Embedding backend '{self._embedding_backend.value}' is not supported.")

    def _persist_docstore(self, destination: Path, document_ids: Sequence[str]) -> None:
        """Persist original documents for metadata lookups alongside the FAISS index."""
        docstore_documents: List[Document] = []
        for doc_id in document_ids:
            record = self._documents[doc_id]
            text = self._ensure_document_content(record)
            docstore_documents.append(Document(page_content=text, metadata=record.metadata))
        with open(destination / "docstore.pkl", "wb") as file:
            pickle.dump(docstore_documents, file)

    def show_chunked_document(self, document_id: str) -> List[DocumentChunk]:
        """Return the currently registered chunks for a document."""
        record = self._documents.get(document_id)
        if record is None:
            raise KeyError(f"Document '{document_id}' not found.")
        return list(record.chunks.values())

    def remove_chunk(self, document_id: str, chunk_id: str) -> None:
        """Remove a specific chunk from a document."""
        record = self._documents.get(document_id)
        if record is None:
            raise KeyError(f"Document '{document_id}' not found.")
        if chunk_id in record.chunks:
            del record.chunks[chunk_id]
            record.touch()
            self._dispatch_event(
                KnowledgeBaseEvent(
                    event_type=KnowledgeBaseEventType.DOCUMENT_UPDATED,
                    source=self._source_name,
                    document_ids=(document_id,),
                    payload={"removed_chunk": chunk_id},
                )
            )

    def update_chunk(
        self,
        document_id: str,
        chunk_id: str,
        *,
        content: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
        graph_links: Optional[Mapping[str, object]] = None,
    ) -> DocumentChunk:
        """Update an existing chunk with new content or metadata."""
        record = self._documents.get(document_id)
        if record is None:
            raise KeyError(f"Document '{document_id}' not found.")
        chunk = record.chunks.get(chunk_id)
        if chunk is None:
            raise KeyError(f"Chunk '{chunk_id}' not found inside document '{document_id}'.")
        if content is not None:
            chunk.content = content
        if metadata is not None:
            chunk.metadata = dict(metadata)
        if graph_links is not None:
            chunk.graph_links = dict(graph_links)
        record.touch()
        self._dispatch_event(
            KnowledgeBaseEvent(
                event_type=KnowledgeBaseEventType.CHUNK_GRAPH_UPDATED,
                source=self._source_name,
                document_ids=(document_id,),
                payload={"chunk_id": chunk_id},
            )
        )
        return chunk

    def show_chunks_graph(self, document_id: str) -> Dict[str, object]:
        """Return a lightweight graph representation for a document."""
        record = self._documents.get(document_id)
        if record is None:
            raise KeyError(f"Document '{document_id}' not found.")
        nodes = []
        edges = []
        for chunk in record.chunks.values():
            nodes.append({"id": chunk.chunk_id, "metadata": chunk.metadata})
            for target, relation in chunk.graph_links.items():
                edges.append({"source": chunk.chunk_id, "target": target, "relation": relation})
        return {"document_id": document_id, "nodes": nodes, "edges": edges}

    # ----------------------------------------------------------------------------------
    # Indexing & reload commands
    # ----------------------------------------------------------------------------------

    def index_document(
        self,
        document_ids: Sequence[str],
        *,
        initiated_by: Optional[str] = None,
        notify_agents: bool = True,
        vector_store_path: Optional[Path] = None,
        overwrite: bool = True,
    ) -> KnowledgeBaseEvent:
        """Trigger indexing flow for a set of documents."""
        target_ids: Tuple[str, ...] = tuple(document_ids) if document_ids else tuple(self._documents.keys())
        if not target_ids:
            raise ValueError("No documents were provided for indexing.")

        missing = [doc_id for doc_id in target_ids if doc_id not in self._documents]
        if missing:
            raise KeyError(f"Documents not found in KB: {', '.join(missing)}")

        destination_input = vector_store_path or self._default_index_path
        if destination_input is None:
            raise ValueError("Index destination is not configured. Provide 'vector_store_path' or set a default index path.")
        destination = Path(destination_input)
        destination.mkdir(parents=True, exist_ok=True)

        if overwrite:
            for artefact in ("index.faiss", "index.pkl"):
                artefact_path = destination / artefact
                if artefact_path.exists():
                    artefact_path.unlink()

        logger.info(
            "Indexing documents: %s [embedding=%s, preparation=%s] -> %s",
            ", ".join(target_ids),
            self._embedding_backend.value,
            self._preparation_method.value,
            destination,
        )

        chunk_docs = self._collect_chunk_documents(target_ids)
        prepared_docs = self._prepare_documents_for_index(chunk_docs)
        if not prepared_docs:
            raise ValueError("No content available to build the vector index.")

        embedding_model = self._get_embedding_model()
        vector_store = FAISS.from_documents(prepared_docs, embedding_model)
        vector_store.save_local(str(destination))
        self._persist_docstore(destination, target_ids)

        event = make_index_event(
            target_ids,
            source=self._source_name,
            initiated_by=initiated_by,
            preparation=self._preparation_method,
            embedding_backend=self._embedding_backend.value,
        ).with_payload(index_path=str(destination))

        self._dispatch_event(event)

        if notify_agents:
            self.trigger_retriever_reload(
                reason=f"Indexed documents: {', '.join(target_ids)}",
                document_ids=target_ids,
                initiated_by=initiated_by,
            )
        return event

    def trigger_retriever_reload(
        self,
        *,
        reason: str,
        document_ids: Sequence[str] = (),
        initiated_by: Optional[str] = None,
    ) -> Dict[str, Dict[str, bool]]:
        """Notify all registered listeners that the KB has changed."""
        context = KBReloadContext(
            reason=reason,
            source=self._source_name,
            document_ids=tuple(document_ids),
            initiated_by=initiated_by,
        )
        self._dispatch_event(
            KnowledgeBaseEvent(
                event_type=KnowledgeBaseEventType.RETRIEVER_RELOAD_REQUESTED,
                source=self._source_name,
                document_ids=tuple(document_ids),
                payload={"reason": reason},
                initiated_by=initiated_by,
            )
        )
        if self._reload_broadcaster is None:
            logger.warning("No reload broadcaster configured; skipping in-process notifications.")
            listener_results: Dict[str, bool] = {}
        else:
            listener_results = self._reload_broadcaster.broadcast(context)

        if self._webhook_registry is None:
            webhook_results: Dict[str, bool] = {}
        else:
            webhook_results = self._webhook_registry.dispatch(context)

        combined_results: Dict[str, Dict[str, bool]] = {
            "listeners": listener_results,
            "webhooks": webhook_results,
        }
        self._dispatch_event(
            KnowledgeBaseEvent(
                event_type=KnowledgeBaseEventType.RETRIEVER_RELOAD_COMPLETED,
                source=self._source_name,
                document_ids=tuple(document_ids),
                payload={"results": combined_results, "reason": reason},
                initiated_by=initiated_by,
            )
        )
        return combined_results

    # ----------------------------------------------------------------------------------
    # Storage helpers
    # ----------------------------------------------------------------------------------

    @contextmanager
    def _writable_storage(self) -> Iterator[None]:
        """Ensure exclusive write access to the storage backend."""
        if self._file_lock is not None:
            try:
                self._file_lock.acquire(timeout=10)
            except FileLockTimeout as exc:  # pragma: no cover - external dependency
                raise RuntimeError("Knowledge base storage is currently in use.") from exc
            try:
                with self._mutex:
                    yield
            finally:
                self._file_lock.release()
        else:
            with self._mutex:
                yield

    def iter_documents(self) -> Iterable[DocumentRecord]:
        """Iterate through all known documents."""
        return list(self._documents.values())

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Fetch a document by id."""
        return self._documents.get(document_id)
