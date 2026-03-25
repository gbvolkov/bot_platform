from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import uuid
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
import config
from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field
from rag_lib.chunkers.csv_table import CSVTableSplitter
from rag_lib.chunkers.html import HTMLSplitter
from rag_lib.chunkers.json import JsonSplitter
from rag_lib.chunkers.markdown_table import MarkdownTableSplitter
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.core.domain import Document, Segment, SegmentType
from rag_lib.core.indexer import Indexer
from rag_lib.config import Settings
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.llm.factory import create_llm
from rag_lib.loaders.csv_excel import CSVLoader, ExcelLoader
from rag_lib.loaders.data_loaders import JsonLoader, TextLoader
from rag_lib.loaders.docx import DocXLoader
from rag_lib.loaders.html import HTMLLoader
from rag_lib.loaders.image import ImageLoader
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.loaders.pptx import PPTXLoader
from rag_lib.loaders.web_async import AsyncWebLoader
from rag_lib.summarizers.table_llm import LLMTableSummarizer
from rag_lib.vectors.factory import create_vector_store


ToolStatus = Literal["success"]
LawType = Literal["44-FZ", "223-FZ"]
PreparedOrigin = Literal["purchase", "open_source"]
PreparedFileType = Literal["pdf", "docx", "xlsx", "html", "txt", "json", "other"]
SourceKind = Literal["purchase", "open_source"]

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
_TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"} | _IMAGE_EXTENSIONS
_SENTENCE_PASS1_CHUNK_SIZE = 2400
_SENTENCE_PASS1_OVERLAP = 240
_SENTENCE_PASS2_CHUNK_SIZE = 1200
_SENTENCE_PASS2_OVERLAP = 120
_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_INN_RE = re.compile(r"\b\d{10,12}\b")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{8,}\d")
_DATE_RE = re.compile(r"\b\d{2}[./-]\d{2}[./-]\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")
_AMOUNT_RE = re.compile(r"\b\d[\d\s.,]{2,}\s?(?:руб\.?|₽|RUB)\b", re.IGNORECASE)
_COMPANY_RE = re.compile(
    r'\b(?:ООО|АО|ПАО|ИП|ФГБУ|ФГУП|ГБУ|МУП|ОАО)\s+"?[A-Za-zА-Яа-яЁё0-9 .,-]+"?'
)


class ToolUserCorrectableError(Exception):
    """An explicit tool error that the LLM can fix by changing the next tool call."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        suggestion: str | None = None,
        input_field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.suggestion = suggestion
        self.input_field = input_field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreparedDocumentEntities(StrictBaseModel):
    inn: list[str] = Field(default_factory=list)
    company_names: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)


class PreparedDocument(StrictBaseModel):
    document_id: str
    origin: PreparedOrigin
    bundle_id: str
    purchase_id: str | None = None
    source_id: str | None = None
    parsed_at_utc: str | None = None
    registry_number: str | None = None
    source_url: str | None = None
    original_source_url: str | None = None
    original_file_name: str | None = None
    original_content_type: str | None = None
    derived_artifact_path: str | None = None
    file_path: str
    file_name: str
    file_type: PreparedFileType
    parse_status: ToolStatus = "success"
    index_status: Literal["ready"] = "ready"
    text_excerpt: str
    entities: PreparedDocumentEntities = Field(default_factory=PreparedDocumentEntities)
    chunks_count: int


class PurchaseSearchItem(StrictBaseModel):
    bundle_id: str
    registry_number: str
    law: LawType | None = None
    purchase_title: str
    customer_name: str
    price_text: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    submission_deadline: str | None = None
    detail_url: str
    common_info_url: str | None = None
    documents_url: str | None = None
    document_urls: list[str] = Field(default_factory=list)
    downloaded_files: list[str] = Field(default_factory=list)
    prepared_document_ids: list[str] = Field(default_factory=list)
    documents_json: str | None = None
    common_info_json: str | None = None
    lots_json: str | None = None
    crawl_status: str
    crawl_error: str | None = None
    crawl_ts_utc: str | None = None


class PurchaseSearchRequest(StrictBaseModel):
    run_id: str | None = None
    search_url: str | None = None
    query_texts: list[str] | None = None
    max_pages: int | None = None
    headless: bool | None = None


class PurchaseSearchResponse(StrictBaseModel):
    source: Literal["purchase_adapter"] = "purchase_adapter"
    run_id: str
    index_id: str
    status: ToolStatus = "success"
    search_urls: list[str] = Field(default_factory=list)
    items: list[PurchaseSearchItem] = Field(default_factory=list)
    prepared_documents: list[PreparedDocument] = Field(default_factory=list)


class OpenSourceFetchRequest(StrictBaseModel):
    run_id: str | None = None
    url: str
    depth: int | None = None
    follow_download_links: bool | None = None
    max_concurrency: int | None = None


class OpenSourcePage(StrictBaseModel):
    bundle_id: str
    url: str
    title: str | None = None
    text: str
    attachments: list[str] = Field(default_factory=list)
    prepared_document_ids: list[str] = Field(default_factory=list)


class OpenSourceFetchResponse(StrictBaseModel):
    source: Literal["rag_lib"] = "rag_lib"
    run_id: str
    index_id: str
    status: ToolStatus = "success"
    pages: list[OpenSourcePage] = Field(default_factory=list)
    prepared_documents: list[PreparedDocument] = Field(default_factory=list)


class DocSearchRequest(StrictBaseModel):
    index_id: str
    query: str
    top_k: int | None = None
    source_kind: SourceKind | None = None
    bundle_id: str | None = None
    purchase_id: str | None = None
    source_id: str | None = None


class DocSearchMatch(StrictBaseModel):
    document_id: str
    bundle_id: str
    purchase_id: str | None = None
    source_id: str | None = None
    parsed_at_utc: str | None = None
    file_path: str
    page: int | None = None
    locator: str | None = None
    snippet: str
    score: float
    source_kind: SourceKind
    source_url: str | None = None


class DocSearchResponse(StrictBaseModel):
    index_id: str
    matches: list[DocSearchMatch] = Field(default_factory=list)


class TopFactor(StrictBaseModel):
    name: str
    value: float | None = None
    nwoe: float | None = None


class ScorePayload(StrictBaseModel):
    risk_value: float | None = None
    risk_zone: str | None = None
    score_value: float | None = None
    score_zone: str | None = None
    reliability_value: float | None = None
    reliability_zone: str | None = None
    top_factors: list[TopFactor] = Field(default_factory=list)


class Fincoef(StrictBaseModel):
    name: str
    value: float | None = None
    norm: float | None = None
    comparison: str | None = None


class CounterpartyScoringRequest(StrictBaseModel):
    inn: str
    model: str | None = None
    include_fincoefs: bool | None = None


class CounterpartyScoringResponse(StrictBaseModel):
    source: Literal["damia_scoring"] = "damia_scoring"
    status: ToolStatus = "success"
    inn: str
    score: ScorePayload
    fincoefs: list[Fincoef] = Field(default_factory=list)


class FSSPGroupedRecord(StrictBaseModel):
    year: int
    status: str
    subject: str
    amount: float | None = None
    count: int
    proceeding_ids: list[str] = Field(default_factory=list)


class CounterpartyFSSPRequest(StrictBaseModel):
    inn: str
    from_date: str | None = None
    to_date: str | None = None
    format: Literal[1, 2] | None = None


class CounterpartyFSSPResponse(StrictBaseModel):
    source: Literal["damia_fssp"] = "damia_fssp"
    status: ToolStatus = "success"
    inn: str
    grouped: list[FSSPGroupedRecord] = Field(default_factory=list)
    raw_format: Literal[1, 2] = 1


@dataclass(frozen=True)
class SalesLeadAgentSettings:
    work_root: Path
    permanent_index_root: Path
    shared_index_id: str
    procurement_search_template: str
    purchase_headless: bool
    open_source_max_concurrency: int
    embedding_provider: str
    embedding_model: str
    damia_api_key: str
    scoring_base_url: str
    fssp_base_url: str


@lru_cache(maxsize=1)
def get_settings() -> SalesLeadAgentSettings:
    work_root = Path(
        os.environ.get("SALES_LEAD_AGENT_WORK_ROOT", "./data/sales_lead_agent/runs")
    ).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    permanent_index_root = Path(
        os.environ.get(
            "SALES_LEAD_AGENT_PERMANENT_INDEX_ROOT",
            "./data/sales_lead_agent/permanent_index",
        )
    ).resolve()
    permanent_index_root.mkdir(parents=True, exist_ok=True)
    shared_index_id = os.environ.get(
        "SALES_LEAD_AGENT_SHARED_INDEX_ID",
        "sales_lead_permanent",
    ).strip()
    if not shared_index_id:
        raise RuntimeError("SALES_LEAD_AGENT_SHARED_INDEX_ID must not be blank.")

    embedding_provider = (
        os.environ.get("SALES_LEAD_AGENT_EMBEDDING_PROVIDER", "openai").strip().lower()
    )
    embedding_model = (
        os.environ.get("SALES_LEAD_AGENT_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    )
    if not embedding_provider:
        raise RuntimeError("SALES_LEAD_AGENT_EMBEDDING_PROVIDER must not be blank.")
    if not embedding_model:
        raise RuntimeError("SALES_LEAD_AGENT_EMBEDDING_MODEL must not be blank.")
    if embedding_provider == "openai" and not (config.OPENAI_API_KEY or "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY must be configured when SALES_LEAD_AGENT_EMBEDDING_PROVIDER=openai."
        )

    template = os.environ.get(
        "SALES_LEAD_AGENT_PROCUREMENT_TEMPLATE",
        "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
        "?searchString=страхован"
        "&morphology=on"
        "&search-filter=Дате+размещения"
        "&pageNumber=1"
        "&sortDirection=false"
        "&recordsPerPage=_2"
        "&showLotsInfoHidden=false"
        "&sortBy=UPDATE_DATE"
        "&fz44=on"
        "&fz223=on"
        "&af=on"
        "&currencyIdGeneral=-1"
        "&gws=Выберите+тип+закупки",
    )

    return SalesLeadAgentSettings(
        work_root=work_root,
        permanent_index_root=permanent_index_root,
        shared_index_id=shared_index_id,
        procurement_search_template=template,
        purchase_headless=os.environ.get("SALES_LEAD_AGENT_PURCHASE_HEADLESS", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        open_source_max_concurrency=int(
            os.environ.get("SALES_LEAD_AGENT_OPEN_SOURCE_MAX_CONCURRENCY", "4")
        ),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        damia_api_key=os.environ.get("SALES_LEAD_AGENT_DAMIA_API_KEY", "").strip(),
        scoring_base_url=os.environ.get("SALES_LEAD_AGENT_SCORING_BASE_URL", "").strip().rstrip("/"),
        fssp_base_url=os.environ.get("SALES_LEAD_AGENT_FSSP_BASE_URL", "").strip().rstrip("/"),
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


class RunWorkspaceManager:
    """Manage persistent run workspaces and the run_id/index_id registry."""

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
        return {"run_id": str(payload["run_id"]), "index_id": str(payload["index_id"])}

    def create_run(self) -> RunWorkspace:
        run_id = f"run_{uuid.uuid4().hex}"
        index_id = self._settings.shared_index_id
        root = self._settings.work_root / run_id
        downloads = root / "downloads"
        web = root / "web"
        index = root / "index"
        artifacts = root / "artifacts"
        for path in (downloads, web, index, artifacts):
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
        )

    def get(self, run_id: str) -> RunWorkspace:
        root = self._settings.work_root / run_id
        metadata = self._read_metadata(root)
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
        )

    def get_by_index(self, index_id: str) -> RunWorkspace:
        registry_path = self._index_registry_dir() / f"{index_id}.txt"
        run_id = registry_path.read_text(encoding="utf-8").strip()
        if not run_id:
            raise ValueError(f"Run id not found for index_id={index_id}.")
        workspace = self.get(run_id)
        if workspace.index_id != index_id:
            raise ValueError(
                f"Index registry mismatch: requested {index_id}, workspace {run_id} points to {workspace.index_id}."
            )
        return workspace


class ProcurementQueryBuilder:
    """Build procurement URLs by replacing only `searchString` in the fixed template.

    Search guidance for `query_texts`:
    - zakupki already applies morphology, so do not enumerate inflectional forms inside one query;
    - zakupki matches all words in one `searchString` using AND semantics, not OR;
    - to simulate OR, pass multiple alternative search strings and let the tool execute each of them;
    - use short procurement-oriented phrases or stems, not long bags of synonyms;
    - for example, `страхование`, `страхованию`, `страхования` should first be queried as
      `страхован`; if that returns no results, retry with a weaker form such as `страхов`.
    """

    def __init__(self, template: str) -> None:
        self._template = template

    def build_url(self, query_text: str) -> str:
        normalized = query_text.strip()
        if not normalized:
            raise ToolUserCorrectableError(
                code="INVALID_QUERY_TEXT",
                message="purchase_search_tool requires non-empty search strings when search_url is absent.",
                suggestion="Provide a non-empty procurement-style search string and call the tool again.",
                input_field="query_texts",
            )
        parsed = urlparse(self._template)
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        params["searchString"] = normalized
        new_query = urlencode(params, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    def build_urls(self, query_texts: list[str]) -> list[str]:
        normalized = [query_text.strip() for query_text in query_texts if query_text.strip()]
        if not normalized:
            raise ToolUserCorrectableError(
                code="INVALID_QUERY_TEXTS",
                message="purchase_search_tool requires at least one non-empty search string when search_url is absent.",
                suggestion="Provide one or more short procurement-style search strings and call the tool again.",
                input_field="query_texts",
            )
        return [self.build_url(query_text) for query_text in normalized]


class PurchaseAdapter:
    """Thin wrapper around `zakupki_crawler` with no fallback behavior."""

    def __init__(self, settings: SalesLeadAgentSettings, query_builder: ProcurementQueryBuilder) -> None:
        self._settings = settings
        self._query_builder = query_builder

    def _import_scraper(self):
        from zakupki_crawler.api import scrape_purchases

        return scrape_purchases

    @property
    def downloads_root(self) -> Path:
        path = self._settings.permanent_index_root / "purchase_downloads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_search_urls(
        self,
        *,
        search_url: str | None,
        query_texts: list[str] | None,
    ) -> list[str]:
        if search_url:
            return [search_url]
        if query_texts is None:
            raise ToolUserCorrectableError(
                code="MISSING_SEARCH_INPUT",
                message="purchase_search_tool requires either search_url or query_texts.",
                suggestion="Provide a direct search_url or one or more procurement-style search strings and call the tool again.",
                input_field="query_texts",
            )
        return self._query_builder.build_urls(query_texts)

    def search(
        self,
        *,
        search_url: str | None,
        query_texts: list[str] | None,
        downloads_dir: str,
        max_pages: int | None,
        headless: bool | None,
    ) -> tuple[list[str], list[PurchaseSearchItem]]:
        resolved_search_urls = self.resolve_search_urls(search_url=search_url, query_texts=query_texts)
        scrape_purchases = self._import_scraper()
        items: list[PurchaseSearchItem] = []
        seen_registry_numbers: set[str] = set()
        resolved_downloads_dir = str(self.downloads_root)
        for resolved_search_url in resolved_search_urls:
            raw_items = self._run_scraper(
                scrape_purchases,
                resolved_search_url=resolved_search_url,
                downloads_dir=resolved_downloads_dir,
                max_pages=max_pages,
                headless=self._settings.purchase_headless if headless is None else headless,
            )

            for raw in raw_items:
                registry_number = str(raw.registry_number)
                if registry_number in seen_registry_numbers:
                    continue
                seen_registry_numbers.add(registry_number)
                items.append(
                    PurchaseSearchItem(
                        bundle_id=registry_number,
                        registry_number=registry_number,
                        law=raw.law if raw.law in {"44-FZ", "223-FZ"} else None,
                        purchase_title=str(raw.purchase_title or ""),
                        customer_name=str(raw.customer_name or ""),
                        price_text=str(raw.price_text) if raw.price_text is not None else None,
                        published_at=str(raw.published_at) if raw.published_at else None,
                        updated_at=str(raw.updated_at) if raw.updated_at else None,
                        submission_deadline=str(raw.submission_deadline)
                        if raw.submission_deadline
                        else None,
                        detail_url=str(raw.detail_url),
                        common_info_url=str(raw.common_info_url) if raw.common_info_url else None,
                        documents_url=str(raw.documents_url) if raw.documents_url else None,
                        document_urls=_normalize_list_field(raw.document_urls),
                        downloaded_files=_normalize_list_field(raw.downloaded_files),
                        documents_json=str(raw.documents_json) if raw.documents_json else None,
                        common_info_json=str(raw.common_info_json) if raw.common_info_json else None,
                        lots_json=str(raw.lots_json) if raw.lots_json else None,
                        crawl_status=str(raw.crawl_status),
                        crawl_error=str(raw.crawl_error) if raw.crawl_error else None,
                        crawl_ts_utc=str(raw.crawl_ts_utc) if raw.crawl_ts_utc else None,
                    )
                )
        return resolved_search_urls, items

    def _run_scraper(
        self,
        scrape_purchases,
        *,
        resolved_search_url: str,
        downloads_dir: str,
        max_pages: int | None,
        headless: bool,
    ):
        kwargs = {
            "downloads_dir": downloads_dir,
            "max_pages": max_pages,
            "headless": headless,
        }
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return scrape_purchases(resolved_search_url, **kwargs)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(scrape_purchases, resolved_search_url, **kwargs)
            return future.result()


class DocumentPreparationService:
    """Parse files, build embeddings, and search the shared permanent index."""

    def __init__(self, settings: SalesLeadAgentSettings) -> None:
        self._settings = settings
        self._table_summarizer = None
        self._rag_settings = Settings()

    def _create_embeddings(self):
        if self._settings.embedding_provider == "openai" and not (config.OPENAI_API_KEY or "").strip():
            raise RuntimeError(
                "OPENAI_API_KEY must be configured when SALES_LEAD_AGENT_EMBEDDING_PROVIDER=openai."
            )
        return create_embeddings_model(
            provider=self._settings.embedding_provider,
            model_name=self._settings.embedding_model,
        )

    @property
    def shared_index_id(self) -> str:
        return self._settings.shared_index_id

    def _validate_index_id(self, index_id: str) -> None:
        if index_id != self.shared_index_id:
            raise ToolUserCorrectableError(
                code="INVALID_INDEX_ID",
                message=(
                    f"Unknown index_id={index_id}. Use the shared index_id returned by "
                    "purchase_search_tool or open_source_fetch_tool."
                ),
                suggestion="Reuse the shared index_id returned by a previous acquisition tool call.",
                input_field="index_id",
            )

    def _vector_env(self):
        previous = os.environ.get("VECTOR_PATH")
        os.environ["VECTOR_PATH"] = str(self._settings.permanent_index_root)

        class _Context:
            def __enter__(self_nonlocal):
                return None

            def __exit__(self_nonlocal, exc_type, exc, tb):
                if previous is None:
                    os.environ.pop("VECTOR_PATH", None)
                else:
                    os.environ["VECTOR_PATH"] = previous
                return False

        return _Context()

    def _vector_store(self):
        embeddings = self._create_embeddings()
        with self._vector_env():
            vector_store = create_vector_store(
                provider="chroma",
                embeddings=embeddings,
                collection_name=self.shared_index_id,
                cleanup=False,
            )
        return embeddings, vector_store

    def _collection_get(self, *, where: dict[str, Any], limit: int = 1) -> dict[str, Any]:
        _embeddings, vector_store = self._vector_store()
        collection = vector_store._collection
        return collection.get(where=where, limit=limit)

    def purchase_exists(self, purchase_id: str) -> bool:
        result = self._collection_get(where={"purchase_id": purchase_id}, limit=1)
        return bool(result.get("ids"))

    def source_exists(self, source_id: str) -> bool:
        result = self._collection_get(where={"source_id": source_id}, limit=1)
        return bool(result.get("ids"))

    def _extract_entities(self, text: str) -> PreparedDocumentEntities:
        company_names: list[str] = []
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

    def _detect_file_type(self, file_path: Path) -> PreparedFileType:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix == ".docx":
            return "docx"
        if suffix in {".xlsx", ".xls"}:
            return "xlsx"
        if suffix in {".html", ".htm"}:
            return "html"
        if suffix == ".txt":
            return "txt"
        if suffix == ".json":
            return "json"
        return "other"

    def _guess_content_type(self, file_path: Path, explicit: str | None) -> str | None:
        if explicit:
            return explicit
        guessed, _ = mimetypes.guess_type(file_path.name)
        return guessed

    def _convert_loaded_documents(self, docs: list[Any]) -> list[Document]:
        return [
            Document(
                page_content=str(getattr(doc, "page_content", "") or ""),
                metadata=dict(getattr(doc, "metadata", {}) or {}),
            )
            for doc in docs
        ]

    def _build_loader(self, file_path: Path) -> Any:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return CSVLoader(str(file_path), output_format="csv")
        if suffix in {".xlsx", ".xls"}:
            return ExcelLoader(str(file_path), output_format="markdown")
        if suffix == ".pdf":
            return PDFLoader(str(file_path), parse_mode="text")
        if suffix == ".docx":
            return DocXLoader(str(file_path))
        if suffix in {".txt", ".md"}:
            return TextLoader(str(file_path))
        if suffix == ".html":
            return HTMLLoader(str(file_path), output_format="html")
        if suffix == ".json":
            return JsonLoader(
                str(file_path),
                output_format="json",
                schema=".",
                ensure_ascii=False,
            )
        if suffix == ".pptx":
            return PPTXLoader(str(file_path))
        if suffix in _IMAGE_EXTENSIONS:
            return ImageLoader(str(file_path), ocr_lang="rus+eng")
        raise ValueError(f"Unsupported file type for preparation: {file_path.suffix}")

    def _load_docs(self, file_path: Path) -> list[Document]:
        loader = self._build_loader(file_path)
        docs = self._convert_loaded_documents(list(loader.load()))
        suffix = file_path.suffix.lower()
        enriched: list[Document] = []
        file_key = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:16]
        for index, doc in enumerate(docs):
            if not _clean_text(doc.page_content):
                continue
            metadata = dict(doc.metadata or {})
            metadata.setdefault("source", str(file_path))
            metadata.setdefault("document_index", index)
            metadata.setdefault("source_type", "image" if suffix in _IMAGE_EXTENSIONS else "file")
            metadata.setdefault("output_format", "text")
            enriched.append(
                Document(
                    id=f"{file_key}:doc:{index}",
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )
        return enriched

    def _extract_archive(
        self,
        *,
        workspace: RunWorkspace,
        bundle_id: str,
        archive_path: Path,
    ) -> list[Path]:
        target_dir = workspace.artifacts_dir / bundle_id / f"archive_{archive_path.stem}"
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            target_root = target_dir.resolve()
            for member in archive.infolist():
                member_path = Path(member.filename)
                destination = (target_dir / member_path).resolve()
                if target_root not in destination.parents and destination != target_root:
                    raise ValueError(
                        f"Archive member escapes target directory: {member.filename}"
                    )
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, destination.open("wb") as target_file:
                    target_file.write(source.read())
        extracted_files = [path for path in target_dir.rglob("*") if path.is_file()]
        if not extracted_files:
            raise ValueError(f"Archive {archive_path} did not contain any files.")
        return extracted_files

    def _doc_page(self, metadata: dict[str, Any]) -> int | None:
        for key in ("page", "page_number", "page_num"):
            value = metadata.get(key)
            if value in (None, ""):
                continue
            return int(value)
        return None

    def _doc_locator(self, metadata: dict[str, Any], *, chunk_index: int) -> str:
        for key in (
            "locator",
            "anchor",
            "xpath",
            "section",
            "heading",
            "title",
            "sheet_name",
            "relative_path",
        ):
            value = metadata.get(key)
            if value:
                return str(value)
        if metadata.get("json_index") not in (None, ""):
            return f"json:{metadata['json_index']}"
        if metadata.get("table_chunk_index") not in (None, ""):
            return f"table:{metadata['table_chunk_index']}"
        return f"chunk:{chunk_index}"

    def _segment_id(self, document_id: str, chunk_index: int) -> str:
        return f"{document_id}:{chunk_index}"

    def _get_table_summarizer(self) -> LLMTableSummarizer:
        if self._table_summarizer is None:
            self._table_summarizer = LLMTableSummarizer(
                create_llm(model_name="mini", streaming=False)
            )
        return self._table_summarizer

    def _build_sentence_splitter(self, *, chunk_size: int, chunk_overlap: int) -> Any:
        from rag_lib.chunkers.sentence import SentenceSplitter

        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language="auto",
        )

    def _build_semantic_splitter(self) -> Any:
        return SemanticChunker(
            embeddings=self._create_embeddings(),
            language="auto",
            threshold=0.8,
            window_size=4,
        )

    def _ensure_segments(
        self,
        segments: list[Segment],
        fallback_documents: list[Document],
    ) -> list[Segment]:
        if segments:
            return list(segments)
        fallback: list[Segment] = []
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

    def _segments_to_documents(self, segments: list[Segment]) -> list[Document]:
        return [segment.to_langchain() for segment in segments]

    def _split_docs(self, *, file_path: Path, docs: list[Document]) -> list[Segment]:
        suffix = file_path.suffix.lower()
        if not docs:
            return []
        if suffix == ".csv":
            return self._ensure_segments(
                CSVTableSplitter(
                    max_rows_per_chunk=self._rag_settings.ingestion.chunk_size,
                    summarizer=self._get_table_summarizer(),
                    summarize_table=True,
                    summarize_chunks=False,
                    inject_summaries_into_content=True,
                ).split_documents(docs),
                docs,
            )
        if suffix in {".xlsx", ".xls"}:
            return self._ensure_segments(
                MarkdownTableSplitter(
                    split_table_rows=True,
                    max_rows_per_chunk=self._rag_settings.ingestion.chunk_size,
                    summarizer=self._get_table_summarizer(),
                    summarize_table=True,
                    summarize_chunks=False,
                    inject_summaries_into_content=True,
                ).split_documents(docs),
                docs,
            )
        if suffix in _TEXT_EXTENSIONS:
            semantic = self._build_semantic_splitter()
            sentence = self._build_sentence_splitter(
                chunk_size=_SENTENCE_PASS2_CHUNK_SIZE,
                chunk_overlap=_SENTENCE_PASS2_OVERLAP,
            )
            semantic_segments = self._ensure_segments(semantic.split_documents(docs), docs)
            sentence_docs = self._segments_to_documents(semantic_segments)
            return self._ensure_segments(sentence.split_documents(sentence_docs), sentence_docs)
        if suffix == ".html":
            html_segments = self._ensure_segments(
                HTMLSplitter(
                    output_format="markdown",
                    split_table_rows=True,
                    summarizer=self._get_table_summarizer(),
                    summarize_table=True,
                    summarize_chunks=False,
                    inject_summaries_into_content=True,
                ).split_documents(docs),
                docs,
            )
            sentence_splitter = self._build_sentence_splitter(
                chunk_size=_SENTENCE_PASS2_CHUNK_SIZE,
                chunk_overlap=_SENTENCE_PASS2_OVERLAP,
            )
            final_segments: list[Segment] = []
            for segment in html_segments:
                if segment.type != SegmentType.TEXT:
                    final_segments.append(segment)
                    continue
                split_text = sentence_splitter.split_documents([segment.to_langchain()])
                final_segments.extend(
                    self._ensure_segments(split_text, [segment.to_langchain()])
                )
            return final_segments
        if suffix == ".json":
            return self._ensure_segments(
                JsonSplitter(
                    schema=".",
                    ensure_ascii=False,
                    metadata_value_max_len=256,
                ).split_documents(docs),
                docs,
            )
        if suffix == ".pptx":
            parent_segments = self._ensure_segments(
                RegexSplitter(
                    pattern=r"(?m)(?=^# Slide \d+: .+$)",
                    chunk_size=4000,
                    chunk_overlap=0,
                ).split_documents(docs),
                docs,
            )
            recursive = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100,
            )
            final_segments: list[Segment] = []
            for segment in parent_segments:
                split_children = recursive.split_documents([segment.to_langchain()])
                final_segments.extend(
                    self._ensure_segments(split_children, [segment.to_langchain()])
                )
            return final_segments
        raise ValueError(f"Unsupported splitter strategy for extension: {suffix}")

    def _finalize_segments(
        self,
        *,
        document_id: str,
        origin: SourceKind,
        bundle_id: str,
        registry_number: str | None,
        source_url: str | None,
        source_id: str | None,
        file_path: str,
        parsed_at_utc: str,
        segments: list[Segment],
    ) -> list[Segment]:
        final_segments: list[Segment] = []
        raw_to_final_id: dict[str, str] = {}
        purchase_id = registry_number or ""
        non_empty_segments = [
            segment for segment in segments if str(segment.content or "").strip()
        ]
        for chunk_index, raw_segment in enumerate(non_empty_segments):
            metadata = dict(raw_segment.metadata or {})
            page = self._doc_page(metadata)
            final_id = self._segment_id(document_id, chunk_index)
            if raw_segment.segment_id:
                raw_to_final_id[raw_segment.segment_id] = final_id
            metadata.update(
                {
                    "source_kind": origin,
                    "source_url": source_url or metadata.get("source"),
                    "bundle_id": bundle_id,
                    "document_id": document_id,
                    "purchase_id": purchase_id,
                    "source_id": source_id or "",
                    "parsed_at_utc": parsed_at_utc,
                    "registry_number": registry_number or "",
                    "file_path": file_path,
                    "page": page if page is not None else "",
                    "locator": self._doc_locator(metadata, chunk_index=chunk_index),
                    "chunk_index": chunk_index,
                    "chunk_total": len(non_empty_segments),
                }
            )
            final_segments.append(
                Segment(
                    content=str(raw_segment.content).strip(),
                    metadata=metadata,
                    segment_id=final_id,
                    parent_id=raw_segment.parent_id,
                    level=raw_segment.level,
                    path=list(raw_segment.path or []),
                    type=raw_segment.type or SegmentType.TEXT,
                    original_format=raw_segment.original_format
                    or str(metadata.get("output_format") or "text"),
                )
            )
        for segment in final_segments:
            if segment.parent_id:
                segment.parent_id = raw_to_final_id.get(segment.parent_id)
        return final_segments

    def _index_documents(self, *, segments: list[Segment]) -> None:
        if not segments:
            return
        embeddings, vector_store = self._vector_store()
        Indexer(vector_store=vector_store, embeddings=embeddings).index(segments, batch_size=32)

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
        expanded_file_paths: list[str] = []
        expanded_provenance: dict[str, dict[str, Any]] = {}
        for raw_path in file_paths:
            file_path = Path(raw_path)
            provenance = dict((provenance_by_path or {}).get(str(file_path), {}))
            if file_path.suffix.lower() == ".zip":
                extracted_files = self._extract_archive(
                    workspace=workspace,
                    bundle_id=bundle_id,
                    archive_path=file_path,
                )
                for extracted_file in extracted_files:
                    extracted_path = str(extracted_file)
                    expanded_file_paths.append(extracted_path)
                    expanded_provenance[extracted_path] = {
                        **provenance,
                        "original_file_name": extracted_file.name,
                        "derived_artifact_path": extracted_path,
                    }
                continue
            expanded_file_paths.append(str(file_path))
            expanded_provenance[str(file_path)] = provenance

        for raw_path in expanded_file_paths:
            file_path = Path(raw_path)
            logger.info(f"...processing {file_path}")
            docs = self._load_docs(file_path)
            if not docs:
                logger.error(f"No documents extracted from {file_path}.")
                continue
            document_id = f"doc_{hashlib.sha1(str(file_path).encode('utf-8')).hexdigest()[:16]}"
            parsed_at_utc = _utc_now_iso()
            source_id = None
            if origin == "open_source":
                source_id = _source_id_from_url(source_url)
            split_segments = self._split_docs(file_path=file_path, docs=docs)
            segments = self._finalize_segments(
                document_id=document_id,
                origin=origin,
                bundle_id=bundle_id,
                registry_number=registry_number,
                source_url=source_url,
                source_id=source_id,
                file_path=str(file_path),
                parsed_at_utc=parsed_at_utc,
                segments=split_segments,
            )
            if not segments:
                logger.error(f"No indexable content extracted from {file_path}.")
                continue
                #raise ValueError(f"No indexable content extracted from {file_path}.")
            combined_text = "\n\n".join(segment.content for segment in segments)
            provenance = dict(expanded_provenance.get(str(file_path), {}))
            prepared.append(
                PreparedDocument(
                    document_id=document_id,
                    origin=origin,
                    bundle_id=bundle_id,
                    purchase_id=registry_number,
                    source_id=source_id,
                    parsed_at_utc=parsed_at_utc,
                    registry_number=registry_number,
                    source_url=source_url,
                    original_source_url=provenance.get("original_source_url") or source_url,
                    original_file_name=provenance.get("original_file_name") or file_path.name,
                    original_content_type=self._guess_content_type(
                        Path(str(provenance.get("original_file_name") or file_path.name)),
                        provenance.get("original_content_type"),
                    ),
                    derived_artifact_path=provenance.get("derived_artifact_path") or str(file_path),
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_type=self._detect_file_type(file_path),
                    text_excerpt=combined_text[:400],
                    entities=self._extract_entities(combined_text),
                    chunks_count=len(segments),
                )
            )
            all_segments.extend(segments)
        self._index_documents(segments=all_segments)
        return prepared

    def search(
        self,
        *,
        index_id: str,
        query: str,
        top_k: int = 5,
        source_kind: SourceKind | None = None,
        bundle_id: str | None = None,
        purchase_id: str | None = None,
        source_id: str | None = None,
    ) -> DocSearchResponse:
        self._validate_index_id(index_id)
        metadata_clauses: list[dict[str, Any]] = []
        if source_kind:
            metadata_clauses.append({"source_kind": source_kind})
        if bundle_id:
            metadata_clauses.append({"bundle_id": bundle_id})
        if purchase_id:
            metadata_clauses.append({"purchase_id": purchase_id})
        if source_id:
            metadata_clauses.append({"source_id": source_id})
        _embeddings, vector_store = self._vector_store()
        kwargs: dict[str, Any] = {"k": top_k}
        if len(metadata_clauses) == 1:
            kwargs["filter"] = metadata_clauses[0]
        elif metadata_clauses:
            kwargs["filter"] = {"$and": metadata_clauses}
        results = vector_store.similarity_search_with_relevance_scores(query, **kwargs)
        matches: list[DocSearchMatch] = []
        for doc, score in results:
            metadata = dict(doc.metadata or {})
            page_value = metadata.get("page")
            page = None
            if page_value not in (None, ""):
                page = int(page_value)
            matches.append(
                DocSearchMatch(
                    document_id=str(metadata.get("document_id") or ""),
                    bundle_id=str(metadata.get("bundle_id") or ""),
                    purchase_id=_optional_string(metadata.get("purchase_id")),
                    source_id=_optional_string(metadata.get("source_id")),
                    parsed_at_utc=_optional_string(metadata.get("parsed_at_utc")),
                    file_path=str(metadata.get("file_path") or metadata.get("source") or ""),
                    page=page,
                    locator=str(metadata.get("locator") or "") or None,
                    snippet=str(doc.page_content or "")[:500],
                    score=float(score),
                    source_kind=str(metadata.get("source_kind") or "purchase"),
                    source_url=str(metadata.get("source_url") or "") or None,
                )
            )
        return DocSearchResponse(index_id=self.shared_index_id, matches=matches)


class CounterpartyClients:
    """HTTP clients for scoring and FSSP with explicit error propagation."""

    def __init__(self, settings: SalesLeadAgentSettings) -> None:
        self._settings = settings

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._settings.damia_api_key:
            headers["Authorization"] = f"Bearer {self._settings.damia_api_key}"
        return headers

    def _client(self) -> httpx.Client:
        return httpx.Client(headers=self._headers(), timeout=20.0)

    def scoring(self, *, inn: str, model: str | None, include_fincoefs: bool) -> CounterpartyScoringResponse:
        if not self._settings.scoring_base_url:
            raise RuntimeError("SALES_LEAD_AGENT_SCORING_BASE_URL is not configured.")
        with self._client() as client:
            score_response = client.get(
                f"{self._settings.scoring_base_url}/scoring/score",
                params={"inn": inn, "model": model} if model else {"inn": inn},
            )
            score_response.raise_for_status()
            score_payload = score_response.json() if score_response.content else {}
            if not isinstance(score_payload, dict):
                raise TypeError("Scoring API returned a non-object payload.")
            fincoefs_payload: list[dict[str, Any]] = []
            if include_fincoefs:
                fincoefs_response = client.get(
                    f"{self._settings.scoring_base_url}/scoring/fincoefs",
                    params={"inn": inn},
                )
                fincoefs_response.raise_for_status()
                raw_fincoefs = fincoefs_response.json() if fincoefs_response.content else []
                if not isinstance(raw_fincoefs, list):
                    raise TypeError("Fincoefs API returned a non-list payload.")
                fincoefs_payload = raw_fincoefs
        return CounterpartyScoringResponse(
            inn=inn,
            score=ScorePayload(
                risk_value=_optional_float(score_payload.get("risk_value")),
                risk_zone=_optional_string(score_payload.get("risk_zone")),
                score_value=_optional_float(score_payload.get("score_value")),
                score_zone=_optional_string(score_payload.get("score_zone")),
                reliability_value=_optional_float(score_payload.get("reliability_value")),
                reliability_zone=_optional_string(score_payload.get("reliability_zone")),
                top_factors=[
                    TopFactor(
                        name=str(item["name"]),
                        value=_optional_float(item.get("value")),
                        nwoe=_optional_float(item.get("nwoe")),
                    )
                    for item in score_payload.get("top_factors") or []
                    if isinstance(item, dict)
                ],
            ),
            fincoefs=[
                Fincoef(
                    name=str(item["name"]),
                    value=_optional_float(item.get("value")),
                    norm=_optional_float(item.get("norm")),
                    comparison=_optional_string(item.get("comparison")),
                )
                for item in fincoefs_payload
                if isinstance(item, dict)
            ],
        )

    def fssp(
        self,
        *,
        inn: str,
        from_date: str | None,
        to_date: str | None,
        response_format: int,
    ) -> CounterpartyFSSPResponse:
        if not self._settings.fssp_base_url:
            raise RuntimeError("SALES_LEAD_AGENT_FSSP_BASE_URL is not configured.")
        params: dict[str, Any] = {"inn": inn, "format": response_format}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        with self._client() as client:
            response = client.get(f"{self._settings.fssp_base_url}/fssp/isps", params=params)
            response.raise_for_status()
            payload = response.json() if response.content else []
            if not isinstance(payload, list):
                raise TypeError("FSSP API returned a non-list payload.")
        return CounterpartyFSSPResponse(
            inn=inn,
            grouped=[
                FSSPGroupedRecord(
                    year=int(item["year"]),
                    status=str(item["status"]),
                    subject=str(item["subject"]),
                    amount=_optional_float(item.get("amount")),
                    count=int(item["count"]),
                    proceeding_ids=[str(value) for value in item.get("proceeding_ids") or []],
                )
                for item in payload
                if isinstance(item, dict)
            ],
            raw_format=response_format,
        )


@dataclass(frozen=True)
class SalesLeadAgentDependencies:
    workspace_manager: RunWorkspaceManager
    document_service: DocumentPreparationService
    purchase_adapter: PurchaseAdapter
    counterparty_clients: CounterpartyClients
    open_source_max_concurrency: int

    @classmethod
    def from_settings(cls, settings: SalesLeadAgentSettings) -> "SalesLeadAgentDependencies":
        return cls(
            workspace_manager=RunWorkspaceManager(settings),
            document_service=DocumentPreparationService(settings),
            purchase_adapter=PurchaseAdapter(
                settings=settings,
                query_builder=ProcurementQueryBuilder(settings.procurement_search_template),
            ),
            counterparty_clients=CounterpartyClients(settings),
            open_source_max_concurrency=settings.open_source_max_concurrency,
        )


def build_sales_lead_tools(
    dependencies: SalesLeadAgentDependencies | None = None,
) -> list[Any]:
    deps = dependencies or SalesLeadAgentDependencies.from_settings(get_settings())

    @tool(
        "purchase_search_tool",
        args_schema=PurchaseSearchRequest,
        description=(
            "Search EIS procurements, download procurement artifacts, prepare searchable documents, "
            "and return run_id/index_id for follow-up tool calls. Zakupki already applies morphology "
            "and matches all words in one query using AND semantics. To simulate OR, pass multiple "
            "alternative search strings in query_texts. Example fallback sequence: страхован -> страхов."
        ),
    )
    def purchase_search_tool(
        *,
        run_id: str | None = None,
        search_url: str | None = None,
        query_texts: list[str] | None = None,
        max_pages: int | None = None,
        headless: bool | None = None,
    ) -> dict[str, Any]:
        """Search procurements and prepare procurement artifacts for later document search.

        Args:
            run_id: Optional existing run identifier. If omitted, a new run is created.
            search_url: Optional direct EIS extended-search URL. If present, it is used as-is.
            query_texts: Optional list of contextual search phrases for the `searchString` URL
                parameter. Zakupki already applies morphology and matches all words in one query
                with AND semantics. To simulate OR, pass multiple alternative search strings.
                Recommended strategy: use short procurement-style phrases or stems, not long bags
                of synonyms. For example, `страхование`, `страхованию`, `страхования` should first
                be searched as `страхован`; if that returns no results, call the tool again with a
                weaker query such as `страхов`.
            max_pages: Optional crawler page limit.
            headless: Optional crawler headless-mode override.

        Returns:
            A dictionary with the resolved `run_id`, `index_id`, `search_urls`, raw procurement `items`,
            and `prepared_documents` ready for `doc_search_tool`.
        """
        workspace = deps.workspace_manager.get(run_id) if run_id else deps.workspace_manager.create_run()
        resolved_urls, items = deps.purchase_adapter.search(
            search_url=search_url,
            query_texts=query_texts,
            downloads_dir=str(workspace.downloads_dir),
            max_pages=max_pages,
            headless=headless,
        )
        prepared_documents: list[PreparedDocument] = []
        final_items: list[PurchaseSearchItem] = []
        for item in items:
            if deps.document_service.purchase_exists(item.registry_number):
                logger.info("purchase skipped: already indexed purchase_id=%s", item.registry_number)
                final_items.append(item)
                continue
            artifact_paths = _ensure_purchase_artifacts(workspace=workspace, item=item.model_dump())
            if artifact_paths:
                provenance_by_path = {
                    path: {
                        "original_source_url": item.detail_url,
                        "original_file_name": Path(path).name,
                        "derived_artifact_path": path,
                    }
                    for path in artifact_paths
                }
                prepared = deps.document_service.prepare_files(
                    workspace=workspace,
                    origin="purchase",
                    bundle_id=item.bundle_id,
                    registry_number=item.registry_number,
                    source_url=item.detail_url,
                    file_paths=artifact_paths,
                    provenance_by_path=provenance_by_path,
                )
                item.prepared_document_ids = [doc.document_id for doc in prepared]
                prepared_documents.extend(prepared)
            final_items.append(item)
        response = PurchaseSearchResponse(
            run_id=workspace.run_id,
            index_id=deps.document_service.shared_index_id,
            search_urls=resolved_urls,
            items=final_items,
            prepared_documents=prepared_documents,
        )
        return response.model_dump()

    @tool(
        "open_source_fetch_tool",
        args_schema=OpenSourceFetchRequest,
        description="Fetch public web pages and downloadable attachments, prepare searchable documents, and return run_id/index_id for follow-up tool calls.",
    )
    async def open_source_fetch_tool(
        *,
        run_id: str | None = None,
        url: str,
        depth: int | None = None,
        follow_download_links: bool | None = None,
        max_concurrency: int | None = None,
    ) -> dict[str, Any]:
        """Fetch open-source pages and attachments into a searchable run.

        Args:
            run_id: Optional existing run identifier. If omitted, a new run is created.
            url: Start URL for web retrieval.
            depth: Optional recursive crawl depth.
            follow_download_links: Whether download links should also be fetched.
            max_concurrency: Optional crawler concurrency override.

        Returns:
            A dictionary with `run_id`, `index_id`, fetched `pages`, and `prepared_documents`
            that can be searched with `doc_search_tool`.
        """
        workspace = deps.workspace_manager.get(run_id) if run_id else deps.workspace_manager.create_run()
        loader = AsyncWebLoader(
            url=url,
            depth=0 if depth is None else depth,
            fetch_mode="playwright",
            follow_download_links=bool(follow_download_links),
            max_concurrency=deps.open_source_max_concurrency
            if max_concurrency is None
            else max_concurrency,
            continue_on_error=False,
        )
        docs = await loader.load()
        last_errors = getattr(loader, "last_errors", None) or []
        if last_errors:
            raise ToolUserCorrectableError(
                code="FETCH_FAILED",
                message="; ".join(str(item.get("error") or item) for item in last_errors),
                suggestion="Retry the call with a different URL or a smaller crawl scope.",
                input_field="url",
            )
        if not docs:
            raise ToolUserCorrectableError(
                code="NO_CONTENT_FETCHED",
                message=f"No content fetched from {url}.",
                suggestion="Check the URL and retry with a different page or a less restrictive source.",
                input_field="url",
            )

        pages: list[OpenSourcePage] = []
        prepared_documents: list[PreparedDocument] = []
        attachments_by_parent: dict[str, list[str]] = defaultdict(list)

        for doc in docs:
            metadata = dict(doc.metadata or {})
            source_url = str(metadata.get("source") or metadata.get("url") or "").strip()
            if not source_url:
                raise ToolUserCorrectableError(
                    code="MISSING_SOURCE_URL",
                    message="Open-source loader returned content without a source URL.",
                    suggestion="Retry the call with a different URL.",
                    input_field="url",
                )
            parent_url = str(metadata.get("parent_url") or source_url).strip()
            content = str(doc.page_content or "")
            if not content.strip():
                raise ToolUserCorrectableError(
                    code="EMPTY_FETCHED_CONTENT",
                    message=f"Fetched empty content for {source_url}.",
                    suggestion="Retry the call with a different URL or a smaller crawl scope.",
                    input_field="url",
                )
            source_id = _source_id_from_url(source_url)
            if metadata.get("source_type") == "web_download" or metadata.get("download_filename"):
                bundle_id = f"download_{hashlib.sha1(source_url.encode('utf-8')).hexdigest()[:12]}"
                attachments_by_parent[parent_url].append(source_url)
                if source_id and deps.document_service.source_exists(source_id):
                    logger.info("source skipped: already indexed source_id=%s", source_id)
                    continue
                original_name = str(
                    metadata.get("download_filename")
                    or Path(urlparse(source_url).path).name
                    or "download.txt"
                )
                artifact_path = deps.document_service.save_text_artifact(
                    workspace=workspace,
                    relative_dir=bundle_id,
                    file_name=_text_artifact_name(original_name, default_stem="download"),
                    content=content,
                )
                prepared = deps.document_service.prepare_files(
                    workspace=workspace,
                    origin="open_source",
                    bundle_id=bundle_id,
                    registry_number=None,
                    source_url=source_url,
                    file_paths=[artifact_path],
                    provenance_by_path={
                        artifact_path: {
                            "original_source_url": source_url,
                            "original_file_name": original_name,
                            "original_content_type": metadata.get("content_type")
                            or metadata.get("mime_type"),
                            "derived_artifact_path": artifact_path,
                        }
                    },
                )
                prepared_documents.extend(prepared)
                continue

            bundle_id = f"page_{hashlib.sha1(source_url.encode('utf-8')).hexdigest()[:12]}"
            if source_id and deps.document_service.source_exists(source_id):
                logger.info("source skipped: already indexed source_id=%s", source_id)
                pages.append(
                    OpenSourcePage(
                        bundle_id=bundle_id,
                        url=source_url,
                        title=str(metadata.get("title")) if metadata.get("title") else None,
                        text=content,
                        prepared_document_ids=[],
                    )
                )
                continue
            original_name = Path(urlparse(source_url).path).name or "page.txt"
            artifact_path = deps.document_service.save_text_artifact(
                workspace=workspace,
                relative_dir=bundle_id,
                file_name=_text_artifact_name(original_name, default_stem="page"),
                content=content,
            )
            prepared = deps.document_service.prepare_files(
                workspace=workspace,
                origin="open_source",
                bundle_id=bundle_id,
                registry_number=None,
                source_url=source_url,
                file_paths=[artifact_path],
                provenance_by_path={
                    artifact_path: {
                        "original_source_url": source_url,
                        "original_file_name": original_name,
                        "original_content_type": metadata.get("content_type")
                        or metadata.get("mime_type"),
                        "derived_artifact_path": artifact_path,
                    }
                },
            )
            prepared_documents.extend(prepared)
            pages.append(
                OpenSourcePage(
                    bundle_id=bundle_id,
                    url=source_url,
                    title=str(metadata.get("title")) if metadata.get("title") else None,
                    text=content,
                    prepared_document_ids=[doc_item.document_id for doc_item in prepared],
                )
            )

        if not pages and not prepared_documents:
            raise ToolUserCorrectableError(
                code="NO_SEARCHABLE_ARTIFACTS",
                message=f"No searchable artifacts were prepared from {url}.",
                suggestion="Retry with a different URL or enable follow_download_links when attachments are expected.",
                input_field="url",
            )
        for page in pages:
            page.attachments = attachments_by_parent.get(page.url, [])
        response = OpenSourceFetchResponse(
            run_id=workspace.run_id,
            index_id=deps.document_service.shared_index_id,
            pages=pages,
            prepared_documents=prepared_documents,
        )
        return response.model_dump()

    @tool(
        "doc_search_tool",
        args_schema=DocSearchRequest,
        description="Search a prepared document index by explicit index_id and return exact snippets with provenance.",
    )
    def doc_search_tool(
        *,
        index_id: str,
        query: str,
        top_k: int | None = None,
        source_kind: str | None = None,
        bundle_id: str | None = None,
        purchase_id: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Search a prepared document index.

        Args:
            index_id: Explicit shared index identifier returned by a previous acquisition tool.
            query: Semantic document question.
            top_k: Optional match count limit.
            source_kind: Optional narrowing to `purchase` or `open_source`.
            bundle_id: Optional narrowing to a specific bundle.
            purchase_id: Optional narrowing to a specific procurement registry number.
            source_id: Optional narrowing to a specific open-source identifier.

        Returns:
            A dictionary with `index_id` and exact `matches`, including file path, page/locator,
            score, source kind, and source URL.
        """
        response = deps.document_service.search(
            index_id=index_id,
            query=query,
            top_k=5 if top_k is None else top_k,
            source_kind=source_kind,  # type: ignore[arg-type]
            bundle_id=bundle_id,
            purchase_id=purchase_id,
            source_id=source_id,
        )
        return response.model_dump()

    @tool(
        "counterparty_scoring_tool",
        args_schema=CounterpartyScoringRequest,
        description="Fetch scoring data and optional financial coefficients for a supplied INN.",
    )
    def counterparty_scoring_tool(
        *,
        inn: str,
        model: str | None = None,
        include_fincoefs: bool | None = None,
    ) -> dict[str, Any]:
        """Fetch counterparty scoring data.

        Args:
            inn: Company INN to analyze.
            model: Optional external scoring model identifier.
            include_fincoefs: Whether financial coefficients should also be requested.

        Returns:
            A dictionary with the normalized scoring payload for the supplied INN.
        """
        return deps.counterparty_clients.scoring(
            inn=inn,
            model=model,
            include_fincoefs=bool(include_fincoefs),
        ).model_dump()

    @tool(
        "counterparty_fssp_tool",
        args_schema=CounterpartyFSSPRequest,
        description="Fetch grouped enforcement proceedings from FSSP for a supplied INN.",
    )
    def counterparty_fssp_tool(
        *,
        inn: str,
        from_date: str | None = None,
        to_date: str | None = None,
        format: int | None = None,
    ) -> dict[str, Any]:
        """Fetch grouped FSSP enforcement data.

        Args:
            inn: Company INN to analyze.
            from_date: Optional lower bound for retrieval.
            to_date: Optional upper bound for retrieval.
            format: Optional external response format.

        Returns:
            A dictionary with grouped FSSP proceedings for the supplied INN.
        """
        return deps.counterparty_clients.fssp(
            inn=inn,
            from_date=from_date,
            to_date=to_date,
            response_format=1 if format is None else format,
        ).model_dump()

    return [
        purchase_search_tool,
        open_source_fetch_tool,
        doc_search_tool,
        counterparty_scoring_tool,
        counterparty_fssp_tool,
    ]


def _normalize_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [line for line in str(value).splitlines() if line.strip()]


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_url_for_source_id(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    query_pairs = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    normalized_query = urlencode(query_pairs, doseq=True)
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            normalized_query,
            "",
        )
    )


def _source_id_from_url(raw_url: str | None) -> str | None:
    if not raw_url:
        return None
    normalized = _normalize_url_for_source_id(raw_url)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _safe_artifact_name(name: str, *, default_stem: str) -> str:
    base = _SAFE_FILE_RE.sub("_", name.strip()) or default_stem
    if "." not in base:
        base = f"{base}.txt"
    return base


def _text_artifact_name(name: str, *, default_stem: str) -> str:
    base_name = Path(name).stem or default_stem
    base = _SAFE_FILE_RE.sub("_", base_name.strip()) or default_stem
    return f"{base}.txt"


def _ensure_purchase_artifacts(*, workspace: RunWorkspace, item: dict[str, Any]) -> list[str]:
    artifact_paths: list[str] = list(item.get("downloaded_files") or [])
    bundle_dir = workspace.artifacts_dir / str(item["bundle_id"])
    bundle_dir.mkdir(parents=True, exist_ok=True)
    json_fields = {
        "common_info_json": item.get("common_info_json"),
        "documents_json": item.get("documents_json"),
        "lots_json": item.get("lots_json"),
    }
    for name, payload in json_fields.items():
        if not payload:
            continue
        path = bundle_dir / f"{name}.json"
        path.write_text(str(payload), encoding="utf-8")
        artifact_paths.append(str(path))
    return artifact_paths
