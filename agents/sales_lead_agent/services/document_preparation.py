from __future__ import annotations

import contextlib
import hashlib
import json
import mimetypes
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_lib.core.domain import Segment
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.vectors.factory import create_vector_store

from services.kb_manager.utils.loader import load_single_document

from ..schemas import (
    DocSearchMatch,
    DocSearchResponse,
    PreparedDocument,
    PreparedDocumentEntities,
    SourceKind,
)
from ..settings import SalesLeadAgentSettings


_INN_RE = re.compile(r"\b\d{10,12}\b")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{8,}\d")
_DATE_RE = re.compile(r"\b\d{2}[./-]\d{2}[./-]\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")
_AMOUNT_RE = re.compile(r"\b\d[\d\s.,]{2,}\s?(?:руб\.?|₽|RUB)\b", re.IGNORECASE)
_COMPANY_RE = re.compile(
    r"\b(?:ООО|АО|ПАО|ИП|ФГБУ|ФГУП|ГБУ|МУП|ОАО)\s+\"?[A-Za-zА-Яа-яЁё0-9 .,-]+\"?"
)


@dataclass(frozen=True)
class RunWorkspace:
    run_id: str
    index_id: str
    root_dir: Path
    downloads_dir: Path
    web_dir: Path
    index_dir: Path
    artifacts_dir: Path
    transcripts_dir: Path


class RunWorkspaceManager:
    def __init__(self, settings: SalesLeadAgentSettings) -> None:
        self._settings = settings

    def _index_registry_dir(self) -> Path:
        path = self._settings.work_root / "_index_registry"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _metadata_path(self, root: Path) -> Path:
        return root / "workspace.json"

    def _write_metadata(self, *, root: Path, run_id: str, index_id: str) -> None:
        self._metadata_path(root).write_text(
            json.dumps({"run_id": run_id, "index_id": index_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._index_registry_dir() / f"{index_id}.txt").write_text(run_id, encoding="utf-8")

    def _read_metadata(self, root: Path) -> dict[str, str]:
        payload = json.loads(self._metadata_path(root).read_text(encoding="utf-8"))
        run_id = str(payload["run_id"])
        index_id = str(payload["index_id"])
        return {"run_id": run_id, "index_id": index_id}

    def create_run(self) -> RunWorkspace:
        run_id = f"run_{uuid.uuid4().hex}"
        index_id = f"index_{uuid.uuid4().hex}"
        root = self._settings.work_root / run_id
        downloads = root / "downloads"
        web = root / "web"
        index = root / "index"
        artifacts = root / "artifacts"
        transcripts = root / "transcripts"
        for path in (downloads, web, index, artifacts, transcripts):
            path.mkdir(parents=True, exist_ok=True)
        self._write_metadata(root=root, run_id=run_id, index_id=index_id)
        return RunWorkspace(
            run_id=run_id,
            index_id=index_id,
            root_dir=root,
            downloads_dir=downloads,
            web_dir=web,
            index_dir=index,
            artifacts_dir=artifacts,
            transcripts_dir=transcripts,
        )

    def get(self, run_id: str) -> RunWorkspace:
        root = self._settings.work_root / run_id
        try:
            metadata = self._read_metadata(root)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Workspace metadata is missing or invalid for run_id={run_id}.") from exc
        if metadata["run_id"] != run_id:
            raise ValueError(
                f"Workspace metadata mismatch: requested run_id={run_id}, metadata points to {metadata['run_id']}."
            )
        return RunWorkspace(
            run_id=metadata["run_id"],
            index_id=metadata["index_id"],
            root_dir=root,
            downloads_dir=root / "downloads",
            web_dir=root / "web",
            index_dir=root / "index",
            artifacts_dir=root / "artifacts",
            transcripts_dir=root / "transcripts",
        )

    def get_by_index(self, index_id: str) -> RunWorkspace:
        registry_path = self._index_registry_dir() / f"{index_id}.txt"
        try:
            run_id = registry_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise ValueError(f"Index id {index_id} is not registered.") from exc
        if not run_id:
            raise ValueError(f"Run id not found for index_id={index_id}.")
        workspace = self.get(run_id)
        if workspace.index_id != index_id:
            raise ValueError(
                f"Index registry mismatch: requested {index_id}, workspace {run_id} points to {workspace.index_id}."
            )
        return workspace


class DocumentPreparationService:
    def __init__(self, settings: SalesLeadAgentSettings) -> None:
        self._settings = settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    def _create_embeddings(self):
        provider = self._settings.embedding_provider.strip().lower()
        model_name = self._settings.embedding_model.strip()
        if not provider:
            raise RuntimeError("Sales lead agent embedding provider is not configured.")
        if not model_name:
            raise RuntimeError("Sales lead agent embedding model is not configured.")
        if provider == "openai" and not (config.OPENAI_API_KEY or "").strip():
            raise RuntimeError(
                "OPENAI_API_KEY is required when SALES_LEAD_AGENT_EMBEDDING_PROVIDER=openai."
            )
        return create_embeddings_model(provider=provider, model_name=model_name)

    @contextmanager
    def _vector_env(self, workspace: RunWorkspace):
        workspace.index_dir.mkdir(parents=True, exist_ok=True)
        previous = os.environ.get("VECTOR_PATH")
        os.environ["VECTOR_PATH"] = str(workspace.index_dir)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("VECTOR_PATH", None)
            else:
                os.environ["VECTOR_PATH"] = previous

    def _extract_entities(self, text: str) -> PreparedDocumentEntities:
        company_names = []
        seen_names: set[str] = set()
        for match in _COMPANY_RE.finditer(text):
            value = match.group(0).strip()
            if value not in seen_names:
                seen_names.add(value)
                company_names.append(value)
        return PreparedDocumentEntities(
            inn=list(dict.fromkeys(_INN_RE.findall(text)))[:20],
            company_names=company_names[:20],
            emails=list(dict.fromkeys(_EMAIL_RE.findall(text)))[:20],
            phones=list(dict.fromkeys(_PHONE_RE.findall(text)))[:20],
            dates=list(dict.fromkeys(_DATE_RE.findall(text)))[:20],
            amounts=list(dict.fromkeys(_AMOUNT_RE.findall(text)))[:20],
        )

    def _detect_file_type(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix == ".docx":
            return "docx"
        if suffix in {".xlsx", ".xls"}:
            return "xlsx"
        if suffix in {".html", ".htm"}:
            return "html"
        return "other"

    def _guess_content_type(self, file_path: Path, explicit: str | None = None) -> str | None:
        if explicit:
            return explicit
        guessed, _ = mimetypes.guess_type(file_path.name)
        return guessed

    def _load_docs(self, file_path: Path) -> list[Document]:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".json"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=text, metadata={"source": str(file_path)})]
        return load_single_document(str(file_path))

    def _doc_page(self, metadata: dict[str, Any]) -> int | None:
        for key in ("page", "page_number", "page_num"):
            value = metadata.get(key)
            if value is None or value == "":
                continue
            try:
                return int(value)
            except Exception:
                continue
        return None

    def _doc_locator(self, metadata: dict[str, Any], *, chunk_index: int) -> str | None:
        for key in ("locator", "anchor", "xpath", "section", "heading", "relative_path"):
            value = metadata.get(key)
            if value:
                return str(value)
        return f"chunk:{chunk_index}"

    def _segment_id(self, document_id: str, chunk_index: int) -> str:
        return f"{document_id}:{chunk_index}"

    def _index_documents(
        self,
        *,
        workspace: RunWorkspace,
        segments: list[Segment],
    ) -> None:
        if not segments:
            return
        embeddings = self._create_embeddings()
        with self._vector_env(workspace):
            vector_store = create_vector_store(
                provider="chroma",
                embeddings=embeddings,
                collection_name=f"sales_lead_{workspace.index_id}",
                cleanup=False,
            )
            Indexer(vector_store=vector_store, embeddings=embeddings).index(segments, batch_size=32)

    def _build_segments(
        self,
        *,
        document_id: str,
        origin: SourceKind,
        bundle_id: str,
        registry_number: str | None,
        source_url: str | None,
        docs: list[Document],
    ) -> list[Segment]:
        segments: list[Segment] = []
        chunk_counter = 0
        for doc in docs:
            metadata = dict(doc.metadata or {})
            raw_text = doc.page_content or ""
            if not raw_text.strip():
                continue
            chunks = self._splitter.split_text(raw_text)
            page = self._doc_page(metadata)
            for text in chunks:
                segment_metadata = dict(metadata)
                segment_metadata.update(
                    {
                        "source_kind": origin,
                        "source_url": source_url or metadata.get("source"),
                        "bundle_id": bundle_id,
                        "document_id": document_id,
                        "registry_number": registry_number or "",
                        "page": page if page is not None else "",
                        "locator": self._doc_locator(metadata, chunk_index=chunk_counter),
                    }
                )
                segments.append(
                    Segment(
                        content=text,
                        metadata=segment_metadata,
                        segment_id=self._segment_id(document_id, chunk_counter),
                        original_format=metadata.get("source") or "text",
                    )
                )
                chunk_counter += 1
        return segments

    def prepare_files(
        self,
        *,
        workspace: RunWorkspace,
        origin: SourceKind,
        bundle_id: str,
        registry_number: str | None,
        source_url: str | None,
        file_paths: list[str],
        provenance_by_path: dict[str, dict[str, Any]] | None = None,
    ) -> list[PreparedDocument]:
        prepared: list[PreparedDocument] = []
        all_segments: list[Segment] = []
        for raw_path in file_paths:
            file_path = Path(raw_path)
            document_id = f"doc_{hashlib.sha1(str(file_path).encode('utf-8')).hexdigest()[:16]}"
            provenance = dict((provenance_by_path or {}).get(str(file_path), {}))
            original_source_url = provenance.get("original_source_url") or source_url
            original_file_name = provenance.get("original_file_name") or file_path.name
            original_content_type = self._guess_content_type(
                Path(str(original_file_name)),
                provenance.get("original_content_type"),
            )
            derived_artifact_path = provenance.get("derived_artifact_path") or str(file_path)
            try:
                docs = self._load_docs(file_path)
                combined_text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
                entities = self._extract_entities(combined_text)
                segments = self._build_segments(
                    document_id=document_id,
                    origin=origin,
                    bundle_id=bundle_id,
                    registry_number=registry_number,
                    source_url=source_url,
                    docs=docs,
                )
                all_segments.extend(segments)
                prepared.append(
                    PreparedDocument(
                        document_id=document_id,
                        origin=origin,
                        bundle_id=bundle_id,
                        registry_number=registry_number,
                        source_url=source_url,
                        original_source_url=original_source_url,
                        original_file_name=str(original_file_name),
                        original_content_type=original_content_type,
                        derived_artifact_path=str(derived_artifact_path),
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_type=self._detect_file_type(file_path),
                        parse_status="success" if segments else "partial",
                        index_status="ready" if segments else "failed",
                        text_excerpt=combined_text[:400],
                        entities=entities,
                        chunks_count=len(segments),
                        error=None if segments else "No indexable content extracted.",
                    )
                )
            except Exception as exc:
                prepared.append(
                    PreparedDocument(
                        document_id=document_id,
                        origin=origin,
                        bundle_id=bundle_id,
                        registry_number=registry_number,
                        source_url=source_url,
                        original_source_url=original_source_url,
                        original_file_name=str(original_file_name),
                        original_content_type=original_content_type,
                        derived_artifact_path=str(derived_artifact_path),
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_type=self._detect_file_type(file_path),
                        parse_status="failed",
                        index_status="failed",
                        text_excerpt="",
                        entities=PreparedDocumentEntities(),
                        chunks_count=0,
                        error=str(exc),
                    )
                )
        self._index_documents(workspace=workspace, segments=all_segments)
        return prepared

    def save_text_artifact(
        self,
        *,
        workspace: RunWorkspace,
        relative_dir: str,
        file_name: str,
        content: str,
    ) -> str:
        target_dir = workspace.artifacts_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_name
        target_path.write_text(content, encoding="utf-8")
        return str(target_path)

    def search(
        self,
        *,
        workspace: RunWorkspace,
        query: str,
        top_k: int = 5,
        source_kind: SourceKind | None = None,
        bundle_id: str | None = None,
    ) -> DocSearchResponse:
        embeddings = self._create_embeddings()
        metadata_filter: dict[str, Any] = {}
        if source_kind:
            metadata_filter["source_kind"] = source_kind
        if bundle_id:
            metadata_filter["bundle_id"] = bundle_id
        with self._vector_env(workspace):
            vector_store = create_vector_store(
                provider="chroma",
                embeddings=embeddings,
                collection_name=f"sales_lead_{workspace.index_id}",
                cleanup=False,
            )
            kwargs: dict[str, Any] = {"k": top_k}
            if metadata_filter:
                kwargs["filter"] = metadata_filter
            results = vector_store.similarity_search_with_relevance_scores(query, **kwargs)
        matches: list[DocSearchMatch] = []
        for doc, score in results:
            metadata = dict(doc.metadata or {})
            matches.append(
                DocSearchMatch(
                    document_id=str(metadata.get("document_id") or ""),
                    bundle_id=str(metadata.get("bundle_id") or ""),
                    file_path=str(metadata.get("source") or metadata.get("file_path") or ""),
                    page=int(metadata["page"]) if str(metadata.get("page") or "").isdigit() else None,
                    locator=str(metadata.get("locator") or "") or None,
                    snippet=str(metadata.get("content") or doc.page_content or "")[:500],
                    score=float(score),
                    source_kind=str(metadata.get("source_kind") or "purchase"),  # type: ignore[arg-type]
                    source_url=str(metadata.get("source_url") or "") or None,
                )
            )
        return DocSearchResponse(index_id=workspace.index_id, matches=matches)

    def cleanup_run(self, workspace: RunWorkspace) -> None:
        with contextlib.suppress(Exception):
            shutil.rmtree(workspace.root_dir)
