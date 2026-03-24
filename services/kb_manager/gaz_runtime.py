"""GAZ sales-material runtime built entirely on top of rag_lib."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence

from langchain_core.documents import Document

from rag_lib.config import Settings
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.indexer import Indexer
from rag_lib.core.store import LocalPickleStore
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.llm.factory import create_llm
from rag_lib.loaders.csv_excel import CSVLoader, ExcelLoader
from rag_lib.loaders.data_loaders import JsonLoader, TextLoader
from rag_lib.loaders.docx import DocXLoader
from rag_lib.loaders.html import HTMLLoader
from rag_lib.loaders.image import ImageLoader
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.loaders.pptx import PPTXLoader
from rag_lib.chunkers.csv_table import CSVTableSplitter
from rag_lib.chunkers.html import HTMLSplitter
from rag_lib.chunkers.json import JsonSplitter
from rag_lib.chunkers.markdown_table import MarkdownTableSplitter
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.chunkers.sentence import SentenceSplitter
from rag_lib.retrieval.composition import create_scored_dual_storage_retriever
from rag_lib.retrieval.scored_retriever import HydrationMode, SearchType
from rag_lib.summarizers.table_llm import LLMTableSummarizer
from rag_lib.vectors.factory import create_vector_store

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".html",
    ".json",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
_TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"} | _IMAGE_EXTENSIONS
_MAX_RUNTIME_CHUNK_LENGTH = 1600
_PREVIEW_LENGTH = 240
_DEFAULT_SEARCH_THRESHOLD = 0.12
_DEFAULT_READ_THRESHOLD = 0.0
_SENTENCE_PASS1_CHUNK_SIZE = 2400
_SENTENCE_PASS1_OVERLAP = 240
_SENTENCE_PASS2_CHUNK_SIZE = 1200
_SENTENCE_PASS2_OVERLAP = 120

_BRANCH_HINTS = {
    "tco": "total cost ownership pricing leasing financing monthly payment economics",
    "configuration": "configuration options base package body version trim",
    "comparison": "comparison versus competitor differences advantages drawbacks",
    "service_risk": "service maintenance warranty downtime reliability spare parts",
    "internal_approval": "arguments approval procurement finance director internal decision",
    "passenger_route": "passenger route saloon seating capacity bus urban route",
    "special_body": "special body refrigerated isothermal tow vacuum tank tipper crane",
    "special_conditions": "4x4 harsh severe municipal offroad special conditions",
    "unknown_selection": "selection recommendation product direction",
}
_INTENT_HINTS = {
    "overview": "portfolio overview model families use cases",
    "compare": "comparison differences alternatives",
    "specs": "specifications engine payload dimensions power fuel",
    "financing": "finance leasing credit ownership cost economics",
    "objection": "counter objection competitor differentiation",
    "recommendation": "recommended direction fit use case",
    "materials": "documents presentation materials source proof",
    "next_step": "next step proposal pack shortlist",
}
_DOC_KIND_BRANCHES = {
    "comparison": ["comparison"],
    "tco": ["tco"],
    "configuration": ["configuration"],
    "service_book": ["service_risk"],
    "operations_manual": ["service_risk"],
    "service_evidence": ["service_risk"],
    "sales_argument": ["internal_approval"],
}
_CANONICAL_FAMILY_LABELS = {
    "gazelle_next": "Газель NEXT",
    "gazelle_nn": "Газель NN",
    "gazelle_business": "Газель Бизнес",
    "gazelle_city": "Газель City",
    "sobol_nn": "Соболь NN",
    "sobol_business": "Соболь Бизнес",
    "gazon_next": "Газон NEXT",
    "valdai": "Валдай",
    "sadko": "Садко",
    "vector_next": "Вектор NEXT",
    "citymax": "Citymax",
    "paz": "ПАЗ",
    "sat": "SAT",
}
_FAMILY_KEYWORDS = {
    "gazelle_next": ("газель next", "gazelle next"),
    "gazelle_nn": ("газель nn", "газель нн", "gazelle nn"),
    "gazelle_business": ("газель бизнес", "gazelle business"),
    "gazelle_city": ("газель city", "газель сити", "gazelle city"),
    "sobol_nn": ("соболь nn", "соболь нн", "sobol nn"),
    "sobol_business": ("соболь бизнес", "sobol business"),
    "gazon_next": ("газон next", "gazon next"),
    "valdai": ("валдай", "valdai"),
    "sadko": ("садко", "sadko"),
    "vector_next": ("вектор next", "vector next"),
    "citymax": ("citymax", "city max"),
    "paz": ("паз", "paz"),
    "sat": ("sat",),
}
_FAMILY_ALIASES: Dict[str, str] = {}
for _family_id, _variants in _FAMILY_KEYWORDS.items():
    _FAMILY_ALIASES[_family_id] = _family_id
    for _variant in _variants:
        _FAMILY_ALIASES[_variant] = _family_id


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_text(value: Any) -> str:
    text = _clean_text(value).lower().replace("ё", "е")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _slug_token(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_")


def _normalize_family(value: Any) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    compact = normalized.replace(" ", "_")
    return _FAMILY_ALIASES.get(normalized) or _FAMILY_ALIASES.get(compact) or compact


def _truncate(value: Any, limit: int = _PREVIEW_LENGTH) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _score_from_metadata(document: Document) -> float:
    try:
        return float(document.metadata.get("similarity_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


class GazRuntimeService:
    """Runtime service that builds and serves GAZ sales materials through rag_lib."""

    def __init__(self, *, docs_root: Path, cache_root: Path) -> None:
        self.docs_root = Path(docs_root)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._vector_path_lock = RLock()
        self._embeddings = None
        self._table_summarizer = None
        self._rag_settings = Settings()

    def _collection_dir(self, collection_id: str) -> Path:
        return self.cache_root / collection_id

    def _manifest_path(self, collection_id: str) -> Path:
        return self._collection_dir(collection_id) / "manifest.json"

    def _segment_store_path(self, collection_id: str) -> Path:
        return self._collection_dir(collection_id) / "segment_store.pkl"

    def _vector_store_path(self, collection_id: str) -> Path:
        return self._collection_dir(collection_id) / "vector_store"

    def _load_manifest(self, collection_id: str) -> List[Dict[str, Any]]:
        return list(_json_load(self._manifest_path(collection_id), []))

    def _save_manifest(self, collection_id: str, manifest: Sequence[Mapping[str, Any]]) -> None:
        _json_dump(list(manifest), self._manifest_path(collection_id))

    def collection_status(self, collection_id: str) -> Dict[str, Any]:
        manifest_path = self._manifest_path(collection_id)
        segment_store_path = self._segment_store_path(collection_id)
        vector_store_path = self._vector_store_path(collection_id)
        manifest = self._load_manifest(collection_id) if manifest_path.exists() else []
        segment_store_built = segment_store_path.exists()
        rag_index_built = vector_store_path.exists() and any(vector_store_path.iterdir())
        available = bool(manifest_path.exists() and segment_store_built and rag_index_built)
        return {
            "collection_id": collection_id,
            "docs_root": str(self.docs_root),
            "cache_root": str(self.cache_root),
            "manifest_path": str(manifest_path),
            "segment_store_path": str(segment_store_path),
            "vector_store_path": str(vector_store_path),
            "manifest_built": manifest_path.exists(),
            "segment_store_built": segment_store_built,
            "rag_index_built": rag_index_built,
            "material_artifacts_path": str(segment_store_path),
            "material_artifacts_built": segment_store_built,
            "document_count": len(manifest),
            "available": available,
        }

    def rebuild_collection(self, collection_id: str = "gaz", force: bool = False) -> Dict[str, Any]:
        current = self.collection_status(collection_id)
        if current["available"] and not force:
            return current

        manifest = self._build_manifest()
        segments: List[Segment] = []
        for entry in manifest:
            logger.info(f"...processing {entry.get('relative_path', "")}")
            documents = self._load_documents_for_entry(entry)
            split_segments = self._split_documents_for_entry(entry, documents)
            finalized_segments = self._finalize_segments(entry, split_segments)
            segments.extend(finalized_segments)

        self._save_manifest(collection_id, manifest)
        self._build_rag_assets(collection_id, segments)
        logger.info(
            "GAZ collection rebuilt: collection=%s documents=%s segments=%s",
            collection_id,
            len(manifest),
            len(segments),
        )
        return self.collection_status(collection_id)

    def _build_manifest(self) -> List[Dict[str, Any]]:
        manifest: List[Dict[str, Any]] = []
        for file_path in self._iter_source_files():
            manifest.append(self._build_manifest_entry(file_path))
        manifest.sort(key=lambda item: item["relative_path"])
        return manifest

    def _iter_source_files(self) -> Iterator[Path]:
        if not self.docs_root.exists():
            return iter(())
        files = [path for path in self.docs_root.rglob("*") if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS]
        files.sort(key=lambda item: item.relative_to(self.docs_root).as_posix().lower())
        return iter(files)

    def _build_manifest_entry(self, file_path: Path) -> Dict[str, Any]:
        relative_path = file_path.relative_to(self.docs_root).as_posix()
        extension = file_path.suffix.lower()
        title = file_path.stem
        candidate_id = f"cand_{hashlib.md5(relative_path.encode('utf-8')).hexdigest()[:16]}"
        family_ids = self._detect_product_families(title)
        doc_kind = self._detect_doc_kind(title)
        transport_type = self._detect_transport_type(title, family_ids)
        competitor_tags = self._detect_competitor_tags(title)
        body_tags = self._detect_body_tags(title)
        special_conditions = self._detect_special_conditions(title)
        branches = self._derive_branches(
            doc_kind=doc_kind,
            transport_type=transport_type,
            body_tags=body_tags,
            special_conditions=special_conditions,
        )
        return {
            "candidate_id": candidate_id,
            "title": title,
            "relative_path": relative_path,
            "source_path": str(file_path.resolve()),
            "extension": extension,
            "doc_kind": doc_kind,
            "branches": branches,
            "product_families": family_ids,
            "competitor_tags": competitor_tags,
            "body_tags": body_tags,
            "transport_type": transport_type,
            "special_conditions": special_conditions,
            "segment_count": 0,
            "preview_snippet": "",
        }

    def _detect_product_families(self, title: str) -> List[str]:
        normalized = _normalize_text(title)
        families: List[str] = []
        for family_id, variants in _FAMILY_KEYWORDS.items():
            if any(variant in normalized for variant in variants):
                families.append(family_id)
        if not families:
            fallback = _slug_token(title)
            if fallback:
                families.append(fallback)
        return families

    def _detect_doc_kind(self, title: str) -> str:
        normalized = _normalize_text(title)
        if "сервис" in normalized or "service book" in normalized:
            return "service_book"
        if "руководство" in normalized or "manual" in normalized:
            return "operations_manual"
        if "дефект" in normalized:
            return "service_evidence"
        if "сравнение" in normalized or "versus" in normalized or "compare" in normalized:
            return "comparison"
        if (
            "стоимость владения" in normalized
            or "ценовое позиционирование" in normalized
            or "приведенная цена" in normalized
            or "pricing" in normalized
            or "ownership cost" in normalized
            or "tco" in normalized
        ):
            return "tco"
        if "база" in normalized or "опци" in normalized or "конструктив" in normalized or "base" in normalized:
            return "configuration"
        if "аргумент" in normalized or "резюме" in normalized:
            return "sales_argument"
        return "general_sales"

    def _detect_transport_type(self, title: str, families: Sequence[str]) -> str:
        normalized = _normalize_text(title)
        if any(family in {"vector_next", "citymax", "paz", "gazelle_city"} for family in families):
            return "passenger"
        if "автобус" in normalized or "пассаж" in normalized or "route" in normalized or "city" in normalized:
            return "passenger"
        return "cargo"

    def _detect_competitor_tags(self, title: str) -> List[str]:
        normalized = _normalize_text(title)
        tags: List[str] = []
        rules = {
            "sollers": ("sollers", "соллерс"),
            "atlant": ("atlant", "атлант"),
            "tr80": ("tr80",),
            "tr120": ("tr120",),
            "tr180": ("tr180",),
            "paz": ("паз", "paz"),
        }
        for tag, variants in rules.items():
            if any(variant in normalized for variant in variants):
                tags.append(tag)
        return tags

    def _detect_body_tags(self, title: str) -> List[str]:
        normalized = _normalize_text(title)
        rules = {
            "bus": ("автобус", "bus"),
            "tourist_bus": ("туристичес",),
            "van": ("фургон", "van"),
            "chassis": ("шасси", "chassis"),
            "flatbed": ("борт", "flatbed"),
            "refrigerated": ("рефриж", "refriger", "isotherm", "изотерм"),
            "tow": ("эваку", "tow"),
            "vacuum": ("вакуум", "vacuum"),
            "tank": ("цистерн", "tank"),
            "tipper": ("самосвал", "tipper"),
            "kmu": ("кму", "crane"),
        }
        body_tags: List[str] = []
        for tag, variants in rules.items():
            if any(variant in normalized for variant in variants):
                body_tags.append(tag)
        return body_tags

    def _detect_special_conditions(self, title: str) -> List[str]:
        normalized = _normalize_text(title)
        rules = {
            "cng": ("cng",),
            "lpg": ("lpg",),
            "diesel": ("дизел", "diesel"),
            "gasoline": ("бензин", "gasoline"),
            "4x4": ("4x4",),
            "municipal": ("муницип",),
            "harsh": ("severe", "harsh", "тяжел"),
        }
        conditions: List[str] = []
        for tag, variants in rules.items():
            if any(variant in normalized for variant in variants):
                conditions.append(tag)
        return conditions

    def _derive_branches(
        self,
        *,
        doc_kind: str,
        transport_type: str,
        body_tags: Sequence[str],
        special_conditions: Sequence[str],
    ) -> List[str]:
        branches = list(_DOC_KIND_BRANCHES.get(doc_kind, []))
        if transport_type == "passenger" and "passenger_route" not in branches:
            branches.append("passenger_route")
        if body_tags and "special_body" not in branches:
            branches.append("special_body")
        if special_conditions and "special_conditions" not in branches:
            branches.append("special_conditions")
        if not branches:
            branches.append("unknown_selection")
        return branches

    def _document_metadata(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        product_families = list(entry.get("product_families") or [])
        branches = list(entry.get("branches") or [])
        competitor_tags = list(entry.get("competitor_tags") or [])
        body_tags = list(entry.get("body_tags") or [])
        special_conditions = list(entry.get("special_conditions") or [])
        metadata: Dict[str, Any] = {
            "candidate_id": entry.get("candidate_id"),
            "title": entry.get("title"),
            "relative_path": entry.get("relative_path"),
            "extension": entry.get("extension"),
            "doc_kind": entry.get("doc_kind"),
            "transport_type": entry.get("transport_type"),
            "product_families": product_families,
            "branches": branches,
            "competitor_tags": competitor_tags,
            "body_tags": body_tags,
            "special_conditions": special_conditions,
            "branches_csv": ", ".join(branches),
            "product_families_csv": ", ".join(product_families),
            "competitor_tags_csv": ", ".join(competitor_tags),
            "body_tags_csv": ", ".join(body_tags),
            "special_conditions_csv": ", ".join(special_conditions),
        }
        for branch in branches:
            metadata[f"branch__{_slug_token(branch)}"] = True
        for family in product_families:
            metadata[f"family__{_slug_token(family)}"] = True
        for competitor in competitor_tags:
            metadata[f"competitor__{_slug_token(competitor)}"] = True
        for body in body_tags:
            metadata[f"body__{_slug_token(body)}"] = True
        for condition in special_conditions:
            metadata[f"condition__{_slug_token(condition)}"] = True
        return metadata

    def _load_documents_for_entry(self, entry: Mapping[str, Any]) -> List[Document]:
        loader = self._build_loader(str(entry["source_path"]), str(entry["extension"]))
        documents = list(loader.load())
        base_metadata = self._document_metadata(entry)
        enriched: List[Document] = []
        for index, document in enumerate(documents):
            if not _clean_text(document.page_content):
                continue
            metadata = dict(document.metadata or {})
            metadata.update(base_metadata)
            metadata.setdefault("source", str(entry["source_path"]))
            metadata.setdefault("document_index", index)
            metadata.setdefault("source_type", "image" if entry["extension"] in _IMAGE_EXTENSIONS else "file")
            metadata.setdefault("output_format", "text")
            enriched.append(Document(id=f"{entry['candidate_id']}:doc:{index}", page_content=document.page_content, metadata=metadata))
        return enriched

    def _build_loader(self, file_path: str, extension: str) -> Any:
        if extension == ".csv":
            return CSVLoader(file_path, output_format="csv")
        if extension in {".xlsx", ".xls"}:
            return ExcelLoader(file_path, output_format="markdown")
        if extension == ".pdf":
            return PDFLoader(file_path, parse_mode="text")
        if extension == ".docx":
            return DocXLoader(file_path)
        if extension in {".txt", ".md"}:
            return TextLoader(file_path)
        if extension == ".html":
            return HTMLLoader(file_path, output_format="html")
        if extension == ".json":
            return JsonLoader(file_path, output_format="json", schema=".", ensure_ascii=False)
        if extension == ".pptx":
            return PPTXLoader(file_path)
        if extension in _IMAGE_EXTENSIONS:
            return ImageLoader(file_path, ocr_lang="rus+eng")
        raise ValueError(f"Unsupported extension for GAZ runtime: {extension}")

    def _get_embeddings(self) -> Any:
        if self._embeddings is None:
            self._embeddings = create_embeddings_model()
        return self._embeddings

    def _get_table_summarizer(self) -> LLMTableSummarizer:
        if self._table_summarizer is None:
            self._table_summarizer = LLMTableSummarizer(create_llm(model_name="mini", streaming=False))
        return self._table_summarizer

    def _split_documents_for_entry(self, entry: Mapping[str, Any], documents: Sequence[Document]) -> List[Segment]:
        extension = str(entry["extension"])
        if not documents:
            return []
        if extension == ".csv":
            return self._ensure_segments(
                CSVTableSplitter(
                    max_rows_per_chunk=self._rag_settings.ingestion.chunk_size,
                    summarizer=self._get_table_summarizer(),
                    summarize_table=True,
                    summarize_chunks=True,
                    inject_summaries_into_content=True,
                ).split_documents(documents),
                documents,
            )
        if extension in {".xlsx", ".xls"}:
            return self._ensure_segments(
                MarkdownTableSplitter(
                    split_table_rows=True,
                    max_rows_per_chunk=self._rag_settings.ingestion.chunk_size,
                    summarizer=self._get_table_summarizer(),
                    summarize_table=True,
                    summarize_chunks=True,
                    inject_summaries_into_content=True,
                ).split_documents(documents),
                documents,
            )
        if extension in _TEXT_EXTENSIONS:
            pass1 = SentenceSplitter(
                chunk_size=_SENTENCE_PASS1_CHUNK_SIZE,
                chunk_overlap=_SENTENCE_PASS1_OVERLAP,
                language="auto",
            )
            pass2 = SentenceSplitter(
                chunk_size=_SENTENCE_PASS2_CHUNK_SIZE,
                chunk_overlap=_SENTENCE_PASS2_OVERLAP,
                language="auto",
            )
            first_pass = self._ensure_segments(pass1.split_documents(documents), documents)
            second_docs = self._segments_to_documents(first_pass)
            return self._ensure_segments(pass2.split_documents(second_docs), second_docs)
        if extension == ".html":
            html_segments = self._ensure_segments(
                HTMLSplitter(
                    output_format="markdown",
                    split_table_rows=True,
                    summarizer=self._get_table_summarizer(),
                    summarize_table=True,
                    summarize_chunks=True,
                    inject_summaries_into_content=True,
                ).split_documents(documents),
                documents,
            )
            sentence_splitter = SentenceSplitter(
                chunk_size=_SENTENCE_PASS2_CHUNK_SIZE,
                chunk_overlap=_SENTENCE_PASS2_OVERLAP,
                language="auto",
            )
            final_segments: List[Segment] = []
            for segment in html_segments:
                if segment.type != SegmentType.TEXT:
                    final_segments.append(segment)
                    continue
                split_text = sentence_splitter.split_documents([segment.to_langchain()])
                final_segments.extend(self._ensure_segments(split_text, [segment.to_langchain()]))
            return final_segments
        if extension == ".json":
            return self._ensure_segments(
                JsonSplitter(schema=".", ensure_ascii=False, metadata_value_max_len=256).split_documents(documents),
                documents,
            )
        if extension == ".pptx":
            parent_segments = self._ensure_segments(
                RegexSplitter(
                    pattern=r"(?m)(?=^# Slide \d+: .+$)",
                    chunk_size=4000,
                    chunk_overlap=0,
                ).split_documents(documents),
                documents,
            )
            recursive = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            final_segments: List[Segment] = []
            for segment in parent_segments:
                split_children = recursive.split_documents([segment.to_langchain()])
                final_segments.extend(self._ensure_segments(split_children, [segment.to_langchain()]))
            return final_segments
        raise ValueError(f"Unsupported splitter strategy for extension: {extension}")

    def _ensure_segments(self, segments: Sequence[Segment], fallback_documents: Sequence[Document]) -> List[Segment]:
        if segments:
            return list(segments)
        fallback: List[Segment] = []
        for document in fallback_documents:
            text = _clean_text(document.page_content)
            if not text:
                continue
            fallback.append(
                Segment(
                    content=text,
                    metadata=dict(document.metadata or {}),
                    segment_id=document.id,
                    type=SegmentType.TEXT,
                    original_format=str((document.metadata or {}).get("output_format") or "text"),
                )
            )
        return fallback

    def _segments_to_documents(self, segments: Sequence[Segment]) -> List[Document]:
        return [segment.to_langchain() for segment in segments]

    def _finalize_segments(self, entry: MutableMapping[str, Any], segments: Sequence[Segment]) -> List[Segment]:
        finalized: List[Segment] = []
        candidate_id = str(entry["candidate_id"])
        total = len([segment for segment in segments if _clean_text(segment.content)])
        chunk_index = 0
        for raw in segments:
            text = _clean_text(raw.content)
            if not text:
                continue
            metadata = dict(raw.metadata or {})
            metadata.update(self._document_metadata(entry))
            metadata["chunk_index"] = chunk_index
            metadata["chunk_total"] = total
            segment = Segment(
                content=text,
                metadata=metadata,
                segment_id=f"{candidate_id}:{chunk_index}",
                parent_id=None,
                level=getattr(raw, "level", 0) or 0,
                path=list(getattr(raw, "path", []) or []),
                type=getattr(raw, "type", SegmentType.TEXT) or SegmentType.TEXT,
                original_format=getattr(raw, "original_format", "") or str(metadata.get("output_format") or "text"),
            )
            finalized.append(segment)
            chunk_index += 1
        entry["segment_count"] = len(finalized)
        entry["preview_snippet"] = _truncate(finalized[0].content if finalized else "", _PREVIEW_LENGTH)
        return finalized

    @contextmanager
    def _vector_path_context(self, collection_id: str) -> Iterator[None]:
        vector_path = self._vector_store_path(collection_id)
        vector_path.mkdir(parents=True, exist_ok=True)
        previous = os.environ.get("VECTOR_PATH")
        with self._vector_path_lock:
            os.environ["VECTOR_PATH"] = str(vector_path)
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("VECTOR_PATH", None)
                else:
                    os.environ["VECTOR_PATH"] = previous

    def _build_rag_assets(self, collection_id: str, segments: Sequence[Segment]) -> None:
        collection_dir = self._collection_dir(collection_id)
        collection_dir.mkdir(parents=True, exist_ok=True)
        segment_store_path = self._segment_store_path(collection_id)
        if segment_store_path.exists():
            segment_store_path.unlink()
        vector_store_path = self._vector_store_path(collection_id)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        doc_store = LocalPickleStore(str(segment_store_path))
        with self._vector_path_context(collection_id):
            vector_store = create_vector_store(
                provider="chroma",
                embeddings=self._get_embeddings(),
                collection_name=f"gaz_{collection_id}",
                cleanup=True,
            )
            Indexer(vector_store=vector_store, embeddings=self._get_embeddings(), doc_store=doc_store).index(list(segments), batch_size=32)

    def _require_runtime_assets(self, collection_id: str) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Any, LocalPickleStore]:
        status = self.collection_status(collection_id)
        if not status["available"]:
            raise RuntimeError(
                f"GAZ collection '{collection_id}' is not ready. "
                f"manifest={status['manifest_built']} segment_store={status['segment_store_built']} rag_index={status['rag_index_built']}"
            )
        manifest = self._load_manifest(collection_id)
        manifest_by_id = {str(item["candidate_id"]): dict(item) for item in manifest}
        doc_store = LocalPickleStore(str(self._segment_store_path(collection_id)))
        with self._vector_path_context(collection_id):
            vector_store = create_vector_store(
                provider="chroma",
                embeddings=self._get_embeddings(),
                collection_name=f"gaz_{collection_id}",
                cleanup=False,
            )
        return manifest, manifest_by_id, vector_store, doc_store

    def _build_search_query(self, query: str, intent: str, families: Sequence[str] | None, competitor: str) -> str:
        parts = [_clean_text(query), _INTENT_HINTS.get(_clean_text(intent), _clean_text(intent))]
        normalized_families = [self._family_query_label(family) for family in self._normalize_requested_families(families)]
        if normalized_families:
            parts.append(" ".join(normalized_families))
        if _clean_text(competitor):
            parts.append(_clean_text(competitor))
        return " ".join(part for part in parts if part).strip()

    def _build_branch_query(self, branch: str, slots: Mapping[str, Any] | None, problem_summary: str) -> str:
        slot_values: List[str] = []
        data = dict(slots or {})
        for key in (
            "customer_goal",
            "transport_type",
            "route_type",
            "route_mode",
            "body_type",
            "capacity_or_payload",
            "competitor",
            "decision_criterion",
            "decision_role",
        ):
            value = _clean_text(data.get(key))
            if value:
                slot_values.append(value)
        for value in data.get("special_conditions") or []:
            cleaned = _clean_text(value)
            if cleaned:
                slot_values.append(cleaned)
        branch_hint = _BRANCH_HINTS.get(branch, branch)
        return " ".join(part for part in [branch_hint, _clean_text(problem_summary), *slot_values] if part).strip()

    def _family_query_label(self, family_id: str) -> str:
        return _CANONICAL_FAMILY_LABELS.get(family_id, family_id.replace("_", " "))

    def _normalize_requested_families(self, families: Sequence[str] | None) -> List[str]:
        normalized: List[str] = []
        for family in families or []:
            canonical = _normalize_family(family)
            if canonical and canonical not in normalized:
                normalized.append(canonical)
        return normalized

    def _retrieve_hits(
        self,
        *,
        query: str,
        vector_store: Any,
        doc_store: LocalPickleStore,
        families: Sequence[str] | None = None,
        candidate_id: str = "",
        top_k: int = 8,
        score_threshold: float = _DEFAULT_SEARCH_THRESHOLD,
    ) -> List[Document]:
        search_filters: List[Optional[Dict[str, Any]]] = []
        if candidate_id:
            search_filters.append({"candidate_id": candidate_id})
        else:
            search_filters.append(None)
            for family in self._normalize_requested_families(families):
                search_filters.append({f"family__{_slug_token(family)}": True})

        merged: Dict[str, Document] = {}
        for filter_payload in search_filters:
            kwargs: Dict[str, Any] = {"k": top_k}
            if filter_payload:
                kwargs["filter"] = filter_payload
            retriever = create_scored_dual_storage_retriever(
                vector_store=vector_store,
                doc_store=doc_store,
                id_key="segment_id",
                search_kwargs=kwargs,
                search_type=SearchType.similarity_score_threshold,
                score_threshold=score_threshold,
                hydration_mode=HydrationMode.parents_replace,
            )
            for document in retriever.invoke(query):
                segment_id = str(document.metadata.get("segment_id") or document.id or "")
                if not segment_id:
                    continue
                current = merged.get(segment_id)
                if current is None or _score_from_metadata(document) > _score_from_metadata(current):
                    merged[segment_id] = document
        return sorted(merged.values(), key=_score_from_metadata, reverse=True)

    def _response_metadata(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "relative_path": entry.get("relative_path"),
            "extension": entry.get("extension"),
            "doc_kind": entry.get("doc_kind"),
            "branches": list(entry.get("branches") or []),
            "product_families": list(entry.get("product_families") or []),
            "competitor_tags": list(entry.get("competitor_tags") or []),
            "body_tags": list(entry.get("body_tags") or []),
            "transport_type": entry.get("transport_type"),
            "special_conditions": list(entry.get("special_conditions") or []),
            "segment_count": int(entry.get("segment_count") or 0),
            "preview_snippet": entry.get("preview_snippet") or "",
        }

    def _candidate_rationale(self, entry: Mapping[str, Any], document: Document) -> str:
        score = _score_from_metadata(document)
        doc_kind = _clean_text(entry.get("doc_kind")).replace("_", " ")
        families = ", ".join(entry.get("product_families") or [])
        if families:
            return f"Matched {doc_kind} evidence for {families} with similarity {score:.2f}."
        return f"Matched {doc_kind} evidence with similarity {score:.2f}."

    def _aggregate_hits_by_candidate(
        self,
        hits: Sequence[Document],
        manifest_by_id: Mapping[str, Mapping[str, Any]],
        *,
        families: Sequence[str] | None = None,
        branch: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        allowed_families = set(self._normalize_requested_families(families))
        grouped: MutableMapping[str, List[Document]] = {}
        for hit in hits:
            candidate_id = _clean_text(hit.metadata.get("candidate_id"))
            if not candidate_id or candidate_id not in manifest_by_id:
                continue
            entry = manifest_by_id[candidate_id]
            if allowed_families and not allowed_families.intersection(set(entry.get("product_families") or [])):
                continue
            if branch and branch not in set(entry.get("branches") or []):
                continue
            grouped.setdefault(candidate_id, []).append(hit)

        candidates: List[Dict[str, Any]] = []
        for candidate_id, documents in grouped.items():
            documents = sorted(documents, key=_score_from_metadata, reverse=True)
            entry = manifest_by_id[candidate_id]
            best = documents[0]
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "title": entry.get("title"),
                    "doc_kind": entry.get("doc_kind"),
                    "rationale": self._candidate_rationale(entry, best),
                    "preview_snippet": _truncate(best.page_content, _PREVIEW_LENGTH),
                    "branch_relevance": branch or (entry.get("branches") or [None])[0],
                    "metadata": self._response_metadata(entry),
                    "_score": _score_from_metadata(best),
                }
            )
        candidates.sort(key=lambda item: item["_score"], reverse=True)
        for candidate in candidates:
            candidate.pop("_score", None)
        return candidates[:top_k]

    def search_sales_materials(
        self,
        *,
        query: str,
        intent: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
        top_k: int = 5,
        collection_id: str = "gaz",
    ) -> Dict[str, Any]:
        _, manifest_by_id, vector_store, doc_store = self._require_runtime_assets(collection_id)
        query_text = self._build_search_query(query, intent, families, competitor)
        hits = self._retrieve_hits(
            query=query_text,
            vector_store=vector_store,
            doc_store=doc_store,
            families=families,
            top_k=max(8, top_k * 4),
            score_threshold=_DEFAULT_SEARCH_THRESHOLD,
        )
        candidates = self._aggregate_hits_by_candidate(hits, manifest_by_id, families=families, top_k=top_k)
        return {
            "collection_id": collection_id,
            "intent": _clean_text(intent),
            "query": _clean_text(query),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    def estimate_research_cost(
        self,
        *,
        query: str,
        intended_depth: str,
        intent: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
        collection_id: str = "gaz",
    ) -> Dict[str, Any]:
        _, manifest_by_id, vector_store, doc_store = self._require_runtime_assets(collection_id)
        query_text = self._build_search_query(query, intent, families, competitor)
        hits = self._retrieve_hits(
            query=query_text,
            vector_store=vector_store,
            doc_store=doc_store,
            families=families,
            top_k=12,
            score_threshold=_DEFAULT_SEARCH_THRESHOLD,
        )
        candidate_ids = [
            candidate_id
            for candidate_id in {
                _clean_text(hit.metadata.get("candidate_id"))
                for hit in hits
                if _clean_text(hit.metadata.get("candidate_id")) in manifest_by_id
            }
            if candidate_id
        ]
        max_score = max((_score_from_metadata(hit) for hit in hits), default=0.0)
        depth = _clean_text(intended_depth) or "bounded"
        base_cost = {
            "broad": 0.25,
            "bounded": 0.75,
            "justified": 1.5,
            "deep_research": 3.0,
        }.get(depth, 1.0)
        estimated_cost = round(base_cost + max(0, len(candidate_ids) - 1) * 0.5, 2)
        requires_hitl = depth == "deep_research" or (depth == "justified" and len(candidate_ids) >= 4)
        rationale = (
            f"Retrieval surfaced {len(candidate_ids)} unique candidates with top score {max_score:.2f}; "
            f"{'a delayed deep pass may be justified' if requires_hitl else 'a bounded answer should be possible now'}."
        )
        return {
            "query": _clean_text(query),
            "intent": _clean_text(intent),
            "intended_depth": depth,
            "positive_match_count": len(candidate_ids),
            "max_match_score": round(max_score, 4),
            "estimated_remaining_cost": estimated_cost,
            "requires_hitl_wait_confirmation": requires_hitl,
            "rationale": rationale,
        }

    def get_branch_pack(
        self,
        *,
        branch: str,
        slots: Mapping[str, Any] | None = None,
        problem_summary: str = "",
        top_k: int = 5,
        collection_id: str = "gaz",
    ) -> Dict[str, Any]:
        _, manifest_by_id, vector_store, doc_store = self._require_runtime_assets(collection_id)
        query_text = self._build_branch_query(branch, slots, problem_summary)
        hits = self._retrieve_hits(
            query=query_text,
            vector_store=vector_store,
            doc_store=doc_store,
            top_k=max(8, top_k * 4),
            score_threshold=_DEFAULT_SEARCH_THRESHOLD,
        )
        candidates = self._aggregate_hits_by_candidate(hits, manifest_by_id, branch=branch, top_k=top_k)
        return {
            "collection_id": collection_id,
            "branch": branch,
            "problem_summary": _clean_text(problem_summary),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    def read_material(
        self,
        *,
        candidate_id: str,
        focus: str,
        max_segments: int = 3,
        collection_id: str = "gaz",
    ) -> Dict[str, Any]:
        _, manifest_by_id, vector_store, doc_store = self._require_runtime_assets(collection_id)
        resolved_id = _clean_text(candidate_id)
        if resolved_id not in manifest_by_id:
            raise KeyError(f"Candidate '{candidate_id}' not found in collection '{collection_id}'")
        entry = manifest_by_id[resolved_id]
        focus_text = _clean_text(focus)
        hits = self._retrieve_hits(
            query=focus_text,
            vector_store=vector_store,
            doc_store=doc_store,
            candidate_id=resolved_id,
            top_k=max_segments,
            score_threshold=_DEFAULT_READ_THRESHOLD,
        )
        excerpts: List[Dict[str, Any]] = []
        for document in hits[:max_segments]:
            excerpts.append(
                {
                    "excerpt": _truncate(document.page_content, _MAX_RUNTIME_CHUNK_LENGTH),
                    "relevance_reason": f"Matched focus with similarity {_score_from_metadata(document):.2f}.",
                    "metadata": {
                        "segment_id": document.metadata.get("segment_id") or document.id,
                        "chunk_index": document.metadata.get("chunk_index"),
                        "chunk_total": document.metadata.get("chunk_total"),
                        "doc_kind": entry.get("doc_kind"),
                        "relative_path": entry.get("relative_path"),
                    },
                }
            )
        metadata = self._response_metadata(entry)
        metadata["artifact_chunk_count"] = metadata["segment_count"]
        return {
            "candidate_id": resolved_id,
            "title": entry.get("title"),
            "focus": focus_text,
            "excerpts": excerpts,
            "metadata": metadata,
        }


__all__ = ["GazRuntimeService"]
