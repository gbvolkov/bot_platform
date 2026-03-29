from __future__ import annotations

import asyncio
import contextlib
from difflib import SequenceMatcher
import hashlib
import html
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable
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
from agents.tools.yandex_search import YandexSearchTool
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command
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
from rag_lib.loaders.legacy_doc import LegacyDocLoader
from rag_lib.loaders.html import HTMLLoader
from rag_lib.loaders.image import ImageLoader
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.loaders.pptx import PPTXLoader
from rag_lib.loaders.web_async import AsyncWebLoader
from rag_lib.summarizers.table_llm import LLMTableSummarizer
from rag_lib.vectors.factory import create_vector_store
from services.sales_lead_retrieval.client import (
    RetrievalServiceConflictError,
    RetrievalServiceUserInputError,
    get_retrieval_service_client,
)


ToolStatus = Literal["success"]
LawType = Literal["44-FZ", "223-FZ"]
PreparedOrigin = Literal["purchase", "open_source"]
PreparedFileType = Literal["pdf", "docx", "xlsx", "html", "txt", "json", "other"]
SourceKind = Literal["purchase", "open_source"]

logger = logging.getLogger(__name__)

ProgressCallback = Callable[..., None]


def _safe_stream_writer() -> Callable[[Any], None]:
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None


def _emit_progress(
    writer: Callable[[Any], None] | None,
    *,
    tool_name: str,
    stage: str,
    message: str,
    **data: Any,
) -> None:
    if writer is None:
        return
    payload = {
        "type": "progress",
        "tool": tool_name,
        "stage": stage,
        "message": message,
    }
    payload.update({key: value for key, value in data.items() if value is not None})
    writer(payload)


def _tool_message(content: Any, runtime: ToolRuntime | None) -> list[ToolMessage]:
    tool_call_id = runtime.tool_call_id if runtime else None
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return [ToolMessage(content=content, tool_call_id=tool_call_id)]


def _thread_id_from_runtime(runtime: ToolRuntime | None) -> str:
    config = runtime.config if runtime else None
    configurable = config.get("configurable") if isinstance(config, dict) else None
    thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else None
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise RuntimeError(
            "purchase_search_tool requires a configured thread_id in the agent runtime."
    )
    return thread_id


def _runtime_state(runtime: ToolRuntime | None) -> dict[str, Any]:
    state = runtime.state if runtime else None
    return state if isinstance(state, dict) else {}


def _requested_run_id_from_runtime(runtime: ToolRuntime | None) -> str | None:
    state = _runtime_state(runtime)
    for field_name in ("active_retrieval_run_id", "active_run_id"):
        raw_value = state.get(field_name)
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()
    return None


def _requested_index_id_from_runtime(runtime: ToolRuntime | None) -> str | None:
    state = _runtime_state(runtime)
    raw_value = state.get("index_id")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return None


def _raise_open_source_loader_error(exc: Exception) -> None:
    message = str(exc).strip() or "Open-source fetch failed."
    message_lower = message.lower()
    if "authentication required" in message_lower or "login_processor" in message_lower:
        raise ToolUserCorrectableError(
            code="FETCH_AUTH_REQUIRED",
            message=message,
            suggestion=(
                "Retry with a publicly accessible page or provide a source that does not require "
                "authentication."
            ),
            input_field="url",
        ) from None
    raise ToolUserCorrectableError(
        code="FETCH_FAILED",
        message=message,
        suggestion="Retry the call with a different URL or a smaller crawl scope.",
        input_field="url",
    ) from None


def _has_purchase_search_inputs(
    *,
    search_url: str | None,
    query_texts: list[str] | None,
) -> bool:
    if isinstance(search_url, str) and search_url.strip():
        return True
    return any(isinstance(item, str) and item.strip() for item in (query_texts or []))


async def _resolve_existing_purchase_snapshot(
    *,
    retrieval_client: Any,
    conversation_id: str,
    runtime: ToolRuntime | None,
) -> Any | None:
    state = _runtime_state(runtime)
    active_retrieval_id = state.get("active_retrieval_id")
    get_retrieval = getattr(retrieval_client, "get_retrieval", None)
    if (
        isinstance(active_retrieval_id, str)
        and active_retrieval_id.strip()
        and callable(get_retrieval)
    ):
        snapshot = await get_retrieval(
            retrieval_id=active_retrieval_id,
            include_payloads=True,
        )
        if snapshot is not None:
            return snapshot
    get_latest_for_conversation = getattr(
        retrieval_client,
        "get_latest_for_conversation",
        None,
    )
    if not callable(get_latest_for_conversation):
        return None
    return await get_latest_for_conversation(
        conversation_id=conversation_id,
        include_payloads=True,
    )


def _search_string_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    value = str(params.get("searchString") or "").strip()
    return value or None


def _require_web_search_string(search_string: str) -> str:
    normalized = str(search_string or "").strip()
    if not normalized:
        raise ToolUserCorrectableError(
            code="INVALID_SEARCH_STRING",
            message="web_search_tool requires a non-empty search_string.",
            suggestion="Provide a concise company name, INN, or other public-web search string.",
            input_field="search_string",
        )
    return normalized


def _normalize_html_text(fragment: str) -> str:
    stripped = _HTML_TAG_RE.sub(" ", fragment)
    return _WHITESPACE_RE.sub(" ", html.unescape(stripped)).strip()


def _resolve_web_search_result_url(raw_url: str) -> str | None:
    normalized = html.unescape(str(raw_url or "").strip())
    if not normalized:
        return None
    if normalized.startswith("//"):
        normalized = f"https:{normalized}"
    parsed = urlparse(normalized)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        target_url = str(params.get("uddg") or "").strip()
        return target_url or None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return normalized


def _extract_web_search_results(html_text: str, *, max_results: int = 5) -> list[dict[str, str | None]]:
    link_matches = list(_WEB_SEARCH_LINK_RE.finditer(html_text))
    snippet_matches = list(_WEB_SEARCH_SNIPPET_RE.finditer(html_text))
    display_url_matches = list(_WEB_SEARCH_DISPLAY_URL_RE.finditer(html_text))
    results: list[dict[str, str | None]] = []
    seen_urls: set[str] = set()

    for index, match in enumerate(link_matches):
        resolved_url = _resolve_web_search_result_url(match.group(1))
        if not resolved_url or resolved_url in seen_urls:
            continue
        title = _normalize_html_text(match.group(2))
        if not title:
            continue
        snippet = None
        if index < len(snippet_matches):
            snippet = _normalize_html_text(snippet_matches[index].group(1)) or None
        display_url = None
        if index < len(display_url_matches):
            display_url = _normalize_html_text(display_url_matches[index].group(1)) or None
        results.append(
            {
                "title": title,
                "url": resolved_url,
                "snippet": snippet,
                "display_url": display_url,
            }
        )
        seen_urls.add(resolved_url)
        if len(results) >= max_results:
            break
    return results


def _validate_retrieve_page_url(url: str) -> str:
    normalized = str(url or "").strip()
    if not normalized:
        raise ToolUserCorrectableError(
            code="INVALID_URL",
            message="retrieve_page_tool requires a non-empty URL.",
            suggestion="Pass an exact page URL returned by web_search or provided by the user.",
            input_field="url",
        )
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolUserCorrectableError(
            code="INVALID_URL",
            message=f"Unsupported URL for retrieve_page_tool: {normalized}",
            suggestion="Pass a full http(s) page URL.",
            input_field="url",
        )
    return normalized


def _require_cached_document_id(document_id: str) -> str:
    normalized = str(document_id or "").strip()
    if not normalized:
        raise ToolUserCorrectableError(
            code="INVALID_DOCUMENT_ID",
            message="read_cached_document_tool requires a non-empty document_id.",
            suggestion="Pass the exact document_id returned by doc_search_tool or retrieve_page_tool.",
            input_field="document_id",
        )
    return normalized


def _normalize_cached_document_window(
    *,
    offset: int | None,
    max_chars: int | None,
) -> tuple[int, int]:
    normalized_offset = 0 if offset is None else int(offset)
    if normalized_offset < 0:
        raise ToolUserCorrectableError(
            code="INVALID_OFFSET",
            message="read_cached_document_tool requires offset >= 0.",
            suggestion="Retry with offset set to 0 or a later positive character position.",
            input_field="offset",
        )
    normalized_max_chars = _READ_CACHED_DOCUMENT_DEFAULT_MAX_CHARS if max_chars is None else int(max_chars)
    if normalized_max_chars <= 0 or normalized_max_chars > _READ_CACHED_DOCUMENT_MAX_CHARS:
        raise ToolUserCorrectableError(
            code="INVALID_MAX_CHARS",
            message=(
                "read_cached_document_tool requires max_chars to be between 1 and "
                f"{_READ_CACHED_DOCUMENT_MAX_CHARS}."
            ),
            suggestion="Retry with a smaller positive max_chars window.",
            input_field="max_chars",
        )
    return normalized_offset, normalized_max_chars


def _normalize_purchase_record_from(record_from: int | None) -> int:
    normalized = 0 if record_from is None else int(record_from)
    if normalized < 0:
        raise ToolUserCorrectableError(
            code="INVALID_RECORD_FROM",
            message="purchase_search_tool requires record_from >= 0.",
            suggestion="Retry with record_from set to 0 or a later positive record offset.",
            input_field="record_from",
        )
    return normalized


def _normalize_optional_selector(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _selector_basename(value: str) -> str:
    stripped = str(value or "").strip().strip("\"'")
    if not stripped:
        return ""
    normalized = stripped.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1]


def _normalize_file_selector_text(value: str) -> str:
    basename = _selector_basename(value) or str(value or "").strip()
    chars: list[str] = []
    for char in basename.casefold().replace("ё", "е"):
        chars.append(char if char.isalnum() else " ")
    return " ".join("".join(chars).split())


def _file_selector_token_keys(value: str) -> list[str]:
    keys: list[str] = []
    for token in _normalize_file_selector_text(value).split():
        if len(token) < 4:
            continue
        key = token[:10]
        if key not in keys:
            keys.append(key)
    return keys


def _validate_cached_document_selector(
    *,
    document_id: str | None,
    bundle_id: str | None,
    file_name: str | None,
) -> tuple[str | None, str | None, str | None]:
    normalized_document_id = _normalize_optional_selector(document_id)
    normalized_bundle_id = _normalize_optional_selector(bundle_id)
    normalized_file_name = _normalize_optional_selector(file_name)
    if normalized_document_id:
        return normalized_document_id, normalized_bundle_id, normalized_file_name
    if normalized_file_name and not normalized_bundle_id:
        raise ToolUserCorrectableError(
            code="MISSING_BUNDLE_ID",
            message="read_cached_document_tool requires bundle_id when file_name is used.",
            suggestion="Pass the bundle_id from purchase_search_tool together with the file name or downloaded file path.",
            input_field="bundle_id",
        )
    if normalized_bundle_id and not normalized_file_name:
        raise ToolUserCorrectableError(
            code="MISSING_FILE_NAME",
            message="read_cached_document_tool requires file_name when bundle_id is used.",
            suggestion="Pass the file name, downloaded file path, or a unique document hint together with the bundle_id.",
            input_field="file_name",
        )
    if normalized_bundle_id and normalized_file_name:
        return None, normalized_bundle_id, normalized_file_name
    raise ToolUserCorrectableError(
        code="MISSING_DOCUMENT_SELECTOR",
        message="read_cached_document_tool requires either document_id or bundle_id + file_name.",
        suggestion="Use a document_id from doc_search_tool, or use bundle_id together with a file name or downloaded file path from purchase_search_tool.",
        input_field="document_id",
    )

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
_ARCHIVE_EXTENSIONS = {".zip", ".rar", ".arj"}
_TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".doc"} | _IMAGE_EXTENSIONS
_SENTENCE_PASS1_CHUNK_SIZE = 2400
_SENTENCE_PASS1_OVERLAP = 240
_SENTENCE_PASS2_CHUNK_SIZE = 1200
_SENTENCE_PASS2_OVERLAP = 120
_PURCHASE_SEARCH_PAGE_SIZE = 5
_READ_CACHED_DOCUMENT_DEFAULT_MAX_CHARS = 12000
_READ_CACHED_DOCUMENT_MAX_CHARS = 30000
_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_INN_RE = re.compile(r"\b\d{10,12}\b")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{8,}\d")
_DATE_RE = re.compile(r"\b\d{2}[./-]\d{2}[./-]\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")
_AMOUNT_RE = re.compile(r"\b\d[\d\s.,]{2,}\s?(?:руб\.?|₽|RUB)\b", re.IGNORECASE)
_COMPANY_RE = re.compile(
    r'\b(?:ООО|АО|ПАО|ИП|ФГБУ|ФГУП|ГБУ|МУП|ОАО)\s+"?[A-Za-zА-Яа-яЁё0-9 .,-]+"?'
)
_QUERY_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_WEB_SEARCH_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_WEB_SEARCH_SNIPPET_RE = re.compile(
    r'<(?:a|div)[^>]+class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
    re.IGNORECASE | re.DOTALL,
)
_WEB_SEARCH_DISPLAY_URL_RE = re.compile(
    r'<a[^>]+class="result__url"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_DADATA_FIND_PARTY_URL = "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party"


def _find_7z_executable() -> str:
    for candidate in ("7z", "7zz"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        "7-Zip CLI is required to extract .zip, .rar, and .arj archives. "
        "Install 7-Zip and ensure `7z` or `7zz` is available on PATH."
    )


def _run_7z(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
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


def _raise_unexpected_tool_failure(
    *,
    tool_name: str,
    code: str,
    message: str,
    suggestion: str,
    exc: Exception,
    input_field: str | None = None,
) -> None:
    logger.exception("%s failed", tool_name)
    raise ToolUserCorrectableError(
        code=code,
        message=f"{message}: {exc}",
        suggestion=suggestion,
        input_field=input_field,
    ) from None


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
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    search_url: str | None = None
    query_texts: list[str] | None = None
    max_pages: int | None = None
    record_from: int | None = None
    runtime: ToolRuntime


class RetrievalProgress(StrictBaseModel):
    total_queries: int = 0
    completed_queries: int = 0
    total_purchases: int = 0
    processed_purchases: int = 0
    total_files: int = 0
    processed_files: int = 0
    prepared_documents: int = 0
    indexed_segments: int = 0


class PurchaseSearchResponse(StrictBaseModel):
    source: Literal["purchase_adapter"] = "purchase_adapter"
    run_id: str
    index_id: str
    status: ToolStatus = "success"
    retrieval_status: Literal["queued", "in_progress", "completed", "failed"]
    retrieval_stage: str
    message: str
    progress: RetrievalProgress = Field(default_factory=RetrievalProgress)
    search_urls: list[str] = Field(default_factory=list)
    record_from: int = 0
    returned_records: int = 0
    total_ready_records: int = 0
    next_record_from: int | None = None
    items: list[PurchaseSearchItem] = Field(default_factory=list)


class WebSearchRequest(StrictBaseModel):
    search_string: str


class WebSearchResult(StrictBaseModel):
    title: str
    url: str
    snippet: str | None = None
    display_url: str | None = None


class WebSearchResponse(StrictBaseModel):
    source: Literal["duckduckgo_html"] = "duckduckgo_html"
    status: ToolStatus = "success"
    search_string: str
    results: list[WebSearchResult] = Field(default_factory=list)


class RetrievePageRequest(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    url: str
    runtime: ToolRuntime


class RetrievedPage(StrictBaseModel):
    bundle_id: str
    url: str
    title: str | None = None
    text_excerpt: str
    attachments: list[str] = Field(default_factory=list)
    prepared_document_ids: list[str] = Field(default_factory=list)


class RetrievePageResponse(StrictBaseModel):
    source: Literal["rag_lib"] = "rag_lib"
    run_id: str
    index_id: str
    status: ToolStatus = "success"
    pages: list[RetrievedPage] = Field(default_factory=list)
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


class ReadCachedDocumentRequest(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    index_id: str | None = None
    document_id: str | None = None
    bundle_id: str | None = None
    file_name: str | None = None
    offset: int | None = None
    max_chars: int | None = None
    runtime: ToolRuntime | None = None


class ReadCachedDocumentResponse(StrictBaseModel):
    source: Literal["prepared_document_cache"] = "prepared_document_cache"
    index_id: str
    document_id: str
    bundle_id: str
    purchase_id: str | None = None
    source_id: str | None = None
    parsed_at_utc: str | None = None
    file_path: str
    file_name: str
    source_kind: SourceKind
    source_url: str | None = None
    content_source: Literal["local_file", "indexed_chunks"]
    total_chars: int
    offset: int
    returned_chars: int
    next_offset: int | None = None
    truncated: bool = False
    content: str


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


class CounterpartyLookupRequest(StrictBaseModel):
    inn: str
    include_branches: bool | None = None


class CounterpartyLookupResponse(StrictBaseModel):
    source: Literal["dadata_party"] = "dadata_party"
    status: ToolStatus = "success"
    inn: str
    found: bool
    name: str | None = None
    full_name: str | None = None
    address: str | None = None
    kpp: str | None = None
    ogrn: str | None = None
    okved: str | None = None
    state_status: str | None = None
    management_name: str | None = None
    management_post: str | None = None
    message: str | None = None


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


class CounterpartyFSSPResponse(StrictBaseModel):
    source: Literal["damia_fssp"] = "damia_fssp"
    status: ToolStatus = "success"
    inn: str
    grouped: list[FSSPGroupedRecord] = Field(default_factory=list)
    raw_format: Literal[1] = 1
    message: str | None = None


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
    scoring_base_url: str
    fssp_base_url: str
    damia_scoring_api_key: str
    damia_fssp_api_key: str
    dadata_api_key: str = ""
    scoring_default_model: str = "_problemCredit"


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
        scoring_base_url=os.environ.get("SALES_LEAD_AGENT_SCORING_BASE_URL", "").strip().rstrip("/"),
        fssp_base_url=os.environ.get("SALES_LEAD_AGENT_FSSP_BASE_URL", "").strip().rstrip("/"),
        damia_scoring_api_key=os.environ.get(
            "SALES_LEAD_AGENT_DAMIA_SCORING_API_KEY",
            "",
        ).strip(),
        damia_fssp_api_key=os.environ.get(
            "SALES_LEAD_AGENT_DAMIA_FSSP_API_KEY",
            "",
        ).strip(),
        dadata_api_key=os.environ.get("DADATA_API_KEY", "").strip(),
        scoring_default_model=(
            os.environ.get("SALES_LEAD_AGENT_SCORING_DEFAULT_MODEL", "_problemCredit").strip()
            or "_problemCredit"
        ),
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


@dataclass(frozen=True)
class PurchaseRequestSpec:
    request_hash: str
    search_urls: list[str]
    request_payload: dict[str, Any]


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
    - duplicate and overlapping query variants are canonicalized before request submission;
    - use short procurement-oriented phrases or stems, not long bags of synonyms;
    - for example, `страхование`, `страхованию`, `страхования` should first be queried as
      `страхован`; if that returns no results, retry with a weaker form such as `страхов`.
    """

    def __init__(self, template: str) -> None:
        self._template = template

    @staticmethod
    def _tokenize_query(query_text: str) -> list[str]:
        return [token.lower() for token in _QUERY_TOKEN_RE.findall(query_text)]

    @classmethod
    def _build_token_replacements(cls, query_texts: list[str]) -> dict[str, str]:
        unique_tokens = sorted(
            {
                token
                for query_text in query_texts
                for token in cls._tokenize_query(query_text)
            },
            key=lambda item: (len(item), item),
        )
        representatives: list[str] = []
        replacements: dict[str, str] = {}
        for token in unique_tokens:
            replacement = next(
                (
                    existing
                    for existing in representatives
                    if token.startswith(existing) or existing.startswith(token)
                ),
                None,
            )
            if replacement is None:
                representatives.append(token)
                replacements[token] = token
            else:
                replacements[token] = replacement
        return replacements

    @classmethod
    def _normalize_query_text(
        cls,
        query_text: str,
        replacements: dict[str, str] | None = None,
    ) -> str:
        resolved_replacements = (
            cls._build_token_replacements([query_text]) if replacements is None else replacements
        )
        tokens: list[str] = []
        seen: set[str] = set()
        for token in cls._tokenize_query(query_text):
            canonical = resolved_replacements.get(token, token)
            if canonical in seen:
                continue
            seen.add(canonical)
            tokens.append(canonical)
        return " ".join(tokens)

    @staticmethod
    def _query_key(normalized_query: str) -> tuple[str, ...]:
        return tuple(sorted(_QUERY_TOKEN_RE.findall(normalized_query.lower())))

    def build_url(self, query_text: str) -> str:
        normalized = self._normalize_query_text(query_text)
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
        replacements = self._build_token_replacements(query_texts)
        normalized: list[str] = []
        seen: set[tuple[str, ...]] = set()
        for query_text in query_texts:
            canonical = self._normalize_query_text(query_text, replacements=replacements)
            key = self._query_key(canonical)
            if not canonical or key in seen:
                continue
            seen.add(key)
            normalized.append(canonical)
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

    def build_request_spec(
        self,
        *,
        search_url: str | None,
        query_texts: list[str] | None,
        max_pages: int | None,
    ) -> PurchaseRequestSpec:
        resolved_search_urls = sorted(
            self.resolve_search_urls(search_url=search_url, query_texts=query_texts)
        )
        request_payload = {
            "search_url": search_url,
            "query_texts": list(query_texts or []),
            "search_urls": resolved_search_urls,
            "max_pages": max_pages,
        }
        request_hash = hashlib.sha256(
            json.dumps(
                {
                    "search_urls": resolved_search_urls,
                    "max_pages": max_pages,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return PurchaseRequestSpec(
            request_hash=request_hash,
            search_urls=resolved_search_urls,
            request_payload=request_payload,
        )

    def search(
        self,
        *,
        search_url: str | None,
        query_texts: list[str] | None,
        downloads_dir: str,
        max_pages: int | None,
        progress_callback: ProgressCallback | None = None,
        item_callback: Callable[[PurchaseSearchItem], None] | None = None,
    ) -> tuple[list[str], list[PurchaseSearchItem]]:
        resolved_search_urls = self.resolve_search_urls(search_url=search_url, query_texts=query_texts)
        scrape_purchases = self._import_scraper()
        items: list[PurchaseSearchItem] = []
        seen_registry_numbers: set[str] = set()
        resolved_downloads_dir = str(self.downloads_root)
        total_urls = len(resolved_search_urls)
        failed_queries = 0
        if progress_callback is not None:
            progress_callback(
                stage="crawler_plan",
                message=f"Resolved {total_urls} procurement search target(s) on zakupki.gov.ru.",
                total_queries=total_urls,
                search_urls=resolved_search_urls,
            )
        for url_index, resolved_search_url in enumerate(resolved_search_urls, start=1):
            query_text = _search_string_from_url(resolved_search_url)
            if progress_callback is not None:
                progress_callback(
                    stage="crawler_search",
                    message=(
                        f"Looking zakupki.gov.ru [{url_index}/{total_urls}] "
                        f"with search string: {query_text}"
                    )
                    if query_text
                    else (
                        f"Looking zakupki.gov.ru [{url_index}/{total_urls}] "
                        f"with search URL: {resolved_search_url}"
                    ),
                    current=url_index,
                    total=total_urls,
                    search_url=resolved_search_url,
                    query_text=query_text,
                )
            try:
                raw_items = self._run_scraper(
                    scrape_purchases,
                    resolved_search_url=resolved_search_url,
                    downloads_dir=resolved_downloads_dir,
                    max_pages=max_pages,
                )
            except Exception as exc:
                failed_queries += 1
                logger.error(
                    "Procurement crawler failed for search_url=%s reason=%s",
                    resolved_search_url,
                    exc,
                    exc_info=True,
                )
                if progress_callback is not None:
                    progress_callback(
                        stage="crawler_search_failed",
                        message=(
                            f"Failed zakupki.gov.ru search [{url_index}/{total_urls}] "
                            f"for {query_text or resolved_search_url}: {exc}"
                        ),
                        current=url_index,
                        total=total_urls,
                        search_url=resolved_search_url,
                        query_text=query_text,
                        error=str(exc),
                        error_type=exc.__class__.__name__,
                    )
                continue
            if progress_callback is not None:
                progress_callback(
                    stage="crawler_search_done",
                    message=(
                        f"Finished zakupki.gov.ru search [{url_index}/{total_urls}]. "
                        f"Parsed {len(raw_items)} procurement record(s) before deduplication."
                    ),
                    current=url_index,
                    total=total_urls,
                    search_url=resolved_search_url,
                    query_text=query_text,
                    raw_procurements=len(raw_items),
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
                if item_callback is not None:
                    item_callback(items[-1].model_copy(deep=True))
        if progress_callback is not None:
            progress_callback(
                stage="crawler_complete",
                message=(
                    f"Collected {len(items)} unique procurement record(s) "
                    f"from {total_urls} search target(s)."
                    + (f" {failed_queries} search target(s) failed." if failed_queries else "")
                ),
                unique_procurements=len(items),
                total_queries=total_urls,
                failed_queries=failed_queries,
            )
        return resolved_search_urls, items

    def _run_scraper(
        self,
        scrape_purchases,
        *,
        resolved_search_url: str,
        downloads_dir: str,
        max_pages: int | None,
    ):
        kwargs = {
            "downloads_dir": downloads_dir,
            "max_pages": max_pages,
            "headless": True,
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
                    "purchase_search_tool or retrieve_page_tool."
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

    def _collection_get(
        self,
        *,
        where: dict[str, Any],
        limit: int | None = 1,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        _embeddings, vector_store = self._vector_store()
        collection = vector_store._collection
        kwargs: dict[str, Any] = {"where": where}
        if limit is not None:
            kwargs["limit"] = limit
        if include is not None:
            kwargs["include"] = include
        return collection.get(**kwargs)

    def _collection_delete(self, *, where: dict[str, Any]) -> None:
        _embeddings, vector_store = self._vector_store()
        collection = vector_store._collection
        collection.delete(where=where)

    def purchase_exists(self, purchase_id: str) -> bool:
        result = self._collection_get(where={"purchase_id": purchase_id}, limit=1)
        return bool(result.get("ids"))

    def source_exists(self, source_id: str) -> bool:
        result = self._collection_get(where={"source_id": source_id}, limit=1)
        return bool(result.get("ids"))

    def _read_indexed_document_segments(
        self,
        *,
        document_id: str,
    ) -> list[tuple[int, str, dict[str, Any]]]:
        raw = self._collection_get(
            where={"document_id": document_id},
            limit=None,
            include=["documents", "metadatas"],
        )
        raw_ids = list(raw.get("ids") or [])
        raw_documents = list(raw.get("documents") or [])
        raw_metadatas = list(raw.get("metadatas") or [])
        rows: list[tuple[int, str, dict[str, Any]]] = []
        for row_index, raw_id in enumerate(raw_ids):
            metadata = dict(raw_metadatas[row_index] or {}) if row_index < len(raw_metadatas) else {}
            content = str(raw_documents[row_index] or "") if row_index < len(raw_documents) else ""
            chunk_index_raw = metadata.get("chunk_index")
            try:
                chunk_index = int(chunk_index_raw)
            except (TypeError, ValueError):
                try:
                    chunk_index = int(str(raw_id).rsplit(":", 1)[-1])
                except (TypeError, ValueError):
                    chunk_index = row_index
            rows.append((chunk_index, content, metadata))
        rows.sort(key=lambda item: item[0])
        return rows

    def _read_document_content_from_local_file(self, file_path: str) -> str | None:
        normalized = str(file_path or "").strip()
        if not normalized:
            return None
        path = Path(normalized)
        if not path.exists() or not path.is_file():
            return None
        docs = self._load_docs(path)
        content_parts = [_clean_text(doc.page_content) for doc in docs]
        content_parts = [part for part in content_parts if part]
        if not content_parts:
            return None
        return "\n\n".join(content_parts)

    def _slice_cached_document_content(
        self,
        *,
        content: str,
        offset: int,
        max_chars: int,
    ) -> tuple[int, str, int | None]:
        total_chars = len(content)
        safe_offset = min(offset, total_chars)
        window = content[safe_offset : safe_offset + max_chars]
        next_offset = safe_offset + len(window)
        if next_offset >= total_chars:
            next_offset = None
        return safe_offset, window, next_offset

    def _build_cached_document_response(
        self,
        *,
        index_id: str,
        document_id: str,
        bundle_id: str,
        purchase_id: str | None,
        source_id: str | None,
        parsed_at_utc: str | None,
        file_path: str,
        source_kind: SourceKind,
        source_url: str | None,
        content_source: Literal["local_file", "indexed_chunks"],
        content: str,
        offset: int,
        max_chars: int,
        ) -> ReadCachedDocumentResponse:
        safe_offset, window, next_offset = self._slice_cached_document_content(
            content=content,
            offset=offset,
            max_chars=max_chars,
        )
        return ReadCachedDocumentResponse(
            index_id=index_id,
            document_id=document_id,
            bundle_id=bundle_id,
            purchase_id=purchase_id,
            source_id=source_id,
            parsed_at_utc=parsed_at_utc,
            file_path=file_path,
            file_name=Path(file_path).name if file_path else "",
            source_kind=source_kind,
            source_url=source_url,
            content_source=content_source,
            total_chars=len(content),
            offset=safe_offset,
            returned_chars=len(window),
            next_offset=next_offset,
            truncated=next_offset is not None,
            content=window,
        )

    def _find_cached_purchase_artifact(
        self,
        *,
        workspace: RunWorkspace,
        bundle_id: str,
        file_name: str,
    ) -> Path:
        artifact_bundle_dir = workspace.artifacts_dir / bundle_id
        download_bundle_dir = self._settings.permanent_index_root / "purchase_downloads" / bundle_id
        candidate_dirs = [path for path in (artifact_bundle_dir, download_bundle_dir) if path.exists() and path.is_dir()]
        if not candidate_dirs:
            raise ToolUserCorrectableError(
                code="BUNDLE_NOT_FOUND",
                message=f"No cached procurement bundle was found for bundle_id={bundle_id}.",
                suggestion="Reuse the current procurement bundle_id from purchase_search_tool.",
                input_field="bundle_id",
            )
        candidate_files = [path for bundle_dir in candidate_dirs for path in bundle_dir.rglob("*") if path.is_file()]
        selector = str(file_name or "").strip()
        selector_basename = _selector_basename(selector)
        selector_path = selector.replace("\\", "/").rstrip("/").casefold()
        basename_path = selector_basename.replace("\\", "/").rstrip("/").casefold()

        exact_matches = [
            path
            for path in candidate_files
            if path.name.casefold() in {selector.casefold(), selector_basename.casefold()}
            or str(path).replace("\\", "/").casefold() == selector_path
            or any(
                str(path.relative_to(base_dir)).replace("\\", "/").casefold() == selector_path
                for base_dir in candidate_dirs
                if path.is_relative_to(base_dir)
            )
            or str(path).replace("\\", "/").casefold() == basename_path
            or any(
                str(path.relative_to(base_dir)).replace("\\", "/").casefold() == basename_path
                for base_dir in candidate_dirs
                if path.is_relative_to(base_dir)
            )
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise ToolUserCorrectableError(
                code="AMBIGUOUS_FILE_NAME",
                message=f"Multiple cached files matched selector {file_name!r} in bundle_id={bundle_id}.",
                suggestion="Use document_id from doc_search_tool when multiple files share the same or matching names.",
                input_field="file_name",
            )

        selector_keys = _file_selector_token_keys(selector)
        leading_key = selector_keys[0] if selector_keys else ""
        leading_matches = []
        if leading_key:
            leading_matches = [
                path
                for path in candidate_files
                if (normalized_name := _normalize_file_selector_text(path.name))
                and normalized_name.split()[0].startswith(leading_key)
            ]
            if len(leading_matches) == 1:
                return leading_matches[0]

        token_matches = []
        if selector_keys:
            token_matches = [
                path
                for path in candidate_files
                if any(
                    any(candidate_token.startswith(selector_key) for candidate_token in _normalize_file_selector_text(path.name).split())
                    for selector_key in selector_keys
                )
            ]
            if len(token_matches) == 1:
                return token_matches[0]
            if len(token_matches) > 1:
                selector_text = _normalize_file_selector_text(selector)
                scored_matches = sorted(
                    (
                        SequenceMatcher(None, selector_text, _normalize_file_selector_text(path.name)).ratio(),
                        path,
                    )
                    for path in token_matches
                )
                scored_matches.reverse()
                best_score, best_path = scored_matches[0]
                second_score = scored_matches[1][0] if len(scored_matches) > 1 else 0.0
                if best_score >= 0.45 and best_score - second_score >= 0.08:
                    return best_path

        if not exact_matches and not leading_matches and not token_matches:
            raise ToolUserCorrectableError(
                code="FILE_NOT_FOUND",
                message=f"No cached file matching selector {file_name!r} was found in bundle_id={bundle_id}.",
                suggestion="Use the file name, downloaded file path, or a clearer document hint from purchase_search_tool for that procurement.",
                input_field="file_name",
            )
        raise ToolUserCorrectableError(
            code="AMBIGUOUS_FILE_NAME",
            message=f"Multiple cached files matched selector {file_name!r} in bundle_id={bundle_id}.",
            suggestion="Use a more specific file name or document_id from doc_search_tool.",
            input_field="file_name",
        )

    def read_cached_document(
        self,
        *,
        index_id: str | None = None,
        document_id: str | None = None,
        workspace: RunWorkspace | None = None,
        bundle_id: str | None = None,
        file_name: str | None = None,
        offset: int | None = None,
        max_chars: int | None = None,
    ) -> ReadCachedDocumentResponse:
        resolved_index_id = _normalize_optional_selector(index_id) or self.shared_index_id
        self._validate_index_id(resolved_index_id)
        normalized_document_id, normalized_bundle_id, normalized_file_name = _validate_cached_document_selector(
            document_id=document_id,
            bundle_id=bundle_id,
            file_name=file_name,
        )
        normalized_offset, normalized_max_chars = _normalize_cached_document_window(
            offset=offset,
            max_chars=max_chars,
        )
        if normalized_document_id is None:
            if workspace is None:
                raise ToolUserCorrectableError(
                    code="MISSING_RUN_CONTEXT",
                    message="read_cached_document_tool needs active run context for bundle_id + file_name lookups.",
                    suggestion="Retry in the same procurement conversation after purchase_search_tool has run.",
                    input_field="bundle_id",
                )
            resolved_path = self._find_cached_purchase_artifact(
                workspace=workspace,
                bundle_id=str(normalized_bundle_id),
                file_name=str(normalized_file_name),
            )
            content = self._read_document_content_from_local_file(str(resolved_path))
            if not content:
                raise ToolUserCorrectableError(
                    code="DOCUMENT_CONTENT_UNAVAILABLE",
                    message=f"Cached content is unavailable for file {normalized_file_name!r} in bundle_id={normalized_bundle_id}.",
                    suggestion="Retry after the file finishes parsing, or choose another downloaded file.",
                    input_field="file_name",
                )
            resolved_absolute_path = resolved_path.resolve()
            artifacts_root = workspace.artifacts_dir.resolve()
            downloads_root = (self._settings.permanent_index_root / "purchase_downloads").resolve()
            if resolved_absolute_path.is_relative_to(artifacts_root):
                relative_hint = str(resolved_absolute_path.relative_to(artifacts_root)).replace("\\", "/")
            elif resolved_absolute_path.is_relative_to(downloads_root):
                relative_hint = str(resolved_absolute_path.relative_to(downloads_root)).replace("\\", "/")
            else:
                relative_hint = resolved_path.name
            source_id = self._purchase_source_id(
                workspace=workspace,
                registry_number=str(normalized_bundle_id),
                file_path=resolved_path,
                provenance={
                    "artifact_relpath": relative_hint,
                    "original_file_name": resolved_path.name,
                },
            )
            derived_document_id = f"doc_{hashlib.sha1((source_id or str(resolved_path)).encode('utf-8')).hexdigest()[:16]}"
            return self._build_cached_document_response(
                index_id=resolved_index_id,
                document_id=derived_document_id,
                bundle_id=str(normalized_bundle_id),
                purchase_id=str(normalized_bundle_id),
                source_id=source_id,
                parsed_at_utc=None,
                file_path=str(resolved_path),
                source_kind="purchase",
                source_url=None,
                content_source="local_file",
                content=content,
                offset=normalized_offset,
                max_chars=normalized_max_chars,
            )

        indexed_segments = self._read_indexed_document_segments(document_id=normalized_document_id)
        if not indexed_segments:
            raise ToolUserCorrectableError(
                code="DOCUMENT_NOT_FOUND",
                message=f"No cached document was found for document_id={normalized_document_id}.",
                suggestion="Use doc_search_tool first to get a valid document_id from the current index.",
                input_field="document_id",
            )

        first_metadata = dict(indexed_segments[0][2] or {})
        file_path = str(first_metadata.get("file_path") or "")
        content: str | None = None
        content_source: Literal["local_file", "indexed_chunks"] = "local_file"
        if file_path:
            try:
                content = self._read_document_content_from_local_file(file_path)
            except Exception as exc:
                logger.warning(
                    "Failed to reread cached local file for document_id=%s path=%s reason=%s",
                    normalized_document_id,
                    file_path,
                    exc,
                )
        if not content:
            content_source = "indexed_chunks"
            seen_chunks: set[int] = set()
            chunk_texts: list[str] = []
            for chunk_index, chunk_content, _metadata in indexed_segments:
                if chunk_index in seen_chunks:
                    continue
                seen_chunks.add(chunk_index)
                normalized_chunk = _clean_text(chunk_content)
                if normalized_chunk:
                    chunk_texts.append(normalized_chunk)
            content = "\n\n".join(chunk_texts)

        if not content:
            raise ToolUserCorrectableError(
                code="DOCUMENT_CONTENT_UNAVAILABLE",
                message=f"Cached content is unavailable for document_id={normalized_document_id}.",
                suggestion="Re-run acquisition for this source or search for a different cached document.",
                input_field="document_id",
            )

        return self._build_cached_document_response(
            index_id=resolved_index_id,
            document_id=normalized_document_id,
            bundle_id=str(first_metadata.get("bundle_id") or ""),
            purchase_id=_optional_string(first_metadata.get("purchase_id")),
            source_id=_optional_string(first_metadata.get("source_id")),
            parsed_at_utc=_optional_string(first_metadata.get("parsed_at_utc")),
            file_path=file_path,
            source_kind=str(first_metadata.get("source_kind") or "purchase"),
            source_url=_optional_string(first_metadata.get("source_url")),
            content_source=content_source,
            content=content,
            offset=normalized_offset,
            max_chars=normalized_max_chars,
        )

    def _delete_cached_document(self, *, source_id: str | None, document_id: str) -> None:
        try:
            if source_id:
                self._collection_delete(where={"source_id": source_id})
                return
            self._collection_delete(where={"document_id": document_id})
        except Exception as exc:
            logger.warning(
                "Failed to delete partially indexed document source_id=%s document_id=%s reason=%s",
                source_id,
                document_id,
                exc,
            )

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

    def _purchase_source_id(
        self,
        *,
        workspace: RunWorkspace,
        registry_number: str | None,
        file_path: Path,
        provenance: dict[str, Any],
    ) -> str | None:
        if not registry_number:
            return None
        relative_hint = str(provenance.get("artifact_relpath") or "").strip()
        if not relative_hint:
            with contextlib.suppress(ValueError):
                relative_hint = str(
                    file_path.resolve().relative_to(workspace.artifacts_dir.resolve())
                ).replace("\\", "/")
        if not relative_hint:
            relative_hint = str(provenance.get("original_file_name") or file_path.name).strip()
        if not relative_hint:
            return None
        normalized = relative_hint.replace("\\", "/").lower()
        payload = f"{registry_number}:{normalized}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

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
        if suffix == ".doc":
            return LegacyDocLoader(str(file_path))
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

    def _list_archive_members(self, archive_path: Path) -> list[dict[str, str]]:
        seven_zip = _find_7z_executable()
        result = _run_7z([seven_zip, "l", "-slt", str(archive_path)])
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"7-Zip failed to list archive {archive_path.name}: {details or f'exit code {result.returncode}'}"
            )
        records: list[dict[str, str]] = []
        current: dict[str, str] = {}
        seen_separator = False
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not seen_separator:
                if line.startswith("----------"):
                    seen_separator = True
                continue
            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue
            if " = " not in line:
                continue
            key, value = line.split(" = ", 1)
            current[key.strip()] = value
        if current:
            records.append(current)
        members = [
            record
            for record in records
            if record.get("Path") and ("Folder" in record or "Size" in record or "Packed Size" in record)
        ]
        if not members:
            raise ValueError(f"Archive {archive_path} did not contain any readable members.")
        return members

    @staticmethod
    def _validate_archive_members(target_dir: Path, members: list[dict[str, str]]) -> None:
        target_root = target_dir.resolve()
        for member in members:
            member_path = Path(member["Path"])
            destination = (target_dir / member_path).resolve()
            if target_root not in destination.parents and destination != target_root:
                raise ValueError(
                    f"Archive member escapes target directory: {member['Path']}"
                )

    def _extract_archive(
        self,
        *,
        workspace: RunWorkspace,
        bundle_id: str,
        archive_path: Path,
    ) -> list[Path]:
        target_dir = workspace.artifacts_dir / bundle_id / f"archive_{archive_path.stem}"
        target_dir.mkdir(parents=True, exist_ok=True)
        members = self._list_archive_members(archive_path)
        self._validate_archive_members(target_dir, members)
        seven_zip = _find_7z_executable()
        result = _run_7z([seven_zip, "x", "-y", f"-o{target_dir}", str(archive_path)])
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"7-Zip failed to extract archive {archive_path.name}: {details or f'exit code {result.returncode}'}"
            )
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
        progress_callback: ProgressCallback | None = None,
    ) -> list[PreparedDocument]:
        prepared: list[PreparedDocument] = []
        expanded_file_paths: list[str] = []
        expanded_provenance: dict[str, dict[str, Any]] = {}
        indexed_documents = 0
        indexed_segments = 0
        for raw_path in file_paths:
            file_path = Path(raw_path)
            provenance = dict((provenance_by_path or {}).get(str(file_path), {}))
            if file_path.suffix.lower() in _ARCHIVE_EXTENSIONS:
                extracted_files = self._extract_archive(
                    workspace=workspace,
                    bundle_id=bundle_id,
                    archive_path=file_path,
                )
                if progress_callback is not None:
                    progress_callback(
                        stage="archive_extracted",
                        message=f"Extracted {len(extracted_files)} file(s) from archive {file_path.name}.",
                        archive_file=file_path.name,
                        extracted_files=len(extracted_files),
                    )
                for extracted_file in extracted_files:
                    extracted_path = str(extracted_file)
                    expanded_file_paths.append(extracted_path)
                    artifact_relpath = None
                    with contextlib.suppress(ValueError):
                        artifact_relpath = str(
                            extracted_file.resolve().relative_to(workspace.artifacts_dir.resolve())
                        ).replace("\\", "/")
                    expanded_provenance[extracted_path] = {
                        **provenance,
                        "original_file_name": extracted_file.name,
                        "derived_artifact_path": extracted_path,
                        "artifact_relpath": artifact_relpath,
                    }
                continue
            expanded_file_paths.append(str(file_path))
            artifact_relpath = None
            with contextlib.suppress(ValueError):
                artifact_relpath = str(
                    file_path.resolve().relative_to(workspace.artifacts_dir.resolve())
                ).replace("\\", "/")
            expanded_provenance[str(file_path)] = {
                **provenance,
                "artifact_relpath": artifact_relpath,
            }

        total_files = len(expanded_file_paths)
        if progress_callback is not None and total_files:
            progress_callback(
                stage="parsing_plan",
                message=f"Parsing downloaded files [0/{total_files}].",
                total_files=total_files,
            )

        for file_index, raw_path in enumerate(expanded_file_paths, start=1):
            file_path = Path(raw_path)
            provenance = dict(expanded_provenance.get(str(file_path), {}))
            source_id = None
            if origin == "open_source":
                source_id = _source_id_from_url(source_url)
            elif origin == "purchase":
                source_id = self._purchase_source_id(
                    workspace=workspace,
                    registry_number=registry_number,
                    file_path=file_path,
                    provenance=provenance,
                )
            logger.info(f"...processing {file_path}")
            if progress_callback is not None:
                progress_callback(
                    stage="parsing_file",
                    message=f"Parsing downloaded files [{file_index}/{total_files}]: {file_path.name}",
                    current=file_index,
                    total=total_files,
                    file_name=file_path.name,
                    file_path=str(file_path),
                )
            if source_id and self.source_exists(source_id):
                logger.info("Skipping already indexed file source_id=%s path=%s", source_id, file_path)
                if progress_callback is not None:
                    progress_callback(
                        stage="file_skipped",
                        message=f"Skipping already indexed file {file_path.name}.",
                        current=file_index,
                        total=total_files,
                        file_name=file_path.name,
                        file_path=str(file_path),
                        source_id=source_id,
                    )
                continue
            try:
                docs = self._load_docs(file_path)
            except ValueError as exc:
                if not str(exc).startswith("Unsupported file type for preparation:"):
                    logger.error("Failed to parse file during preparation path=%s reason=%s", file_path, exc)
                    if progress_callback is not None:
                        progress_callback(
                            stage="file_failed",
                            message=f"Failed to parse file {file_path.name}: {exc}",
                            current=file_index,
                            total=total_files,
                            file_name=file_path.name,
                            file_path=str(file_path),
                            source_id=source_id,
                            error=str(exc),
                            error_type=exc.__class__.__name__,
                        )
                    continue
                logger.warning("Skipping unsupported file during preparation path=%s reason=%s", file_path, exc)
                if progress_callback is not None:
                    progress_callback(
                        stage="unsupported_file",
                        message=f"Skipping unsupported file {file_path.name}.",
                        current=file_index,
                        total=total_files,
                        file_name=file_path.name,
                        file_path=str(file_path),
                    )
                continue
            except Exception as exc:
                logger.error("Failed to parse file during preparation path=%s reason=%s", file_path, exc)
                if progress_callback is not None:
                    progress_callback(
                        stage="file_failed",
                        message=f"Failed to parse file {file_path.name}: {exc}",
                        current=file_index,
                        total=total_files,
                        file_name=file_path.name,
                        file_path=str(file_path),
                        source_id=source_id,
                        error=str(exc),
                        error_type=exc.__class__.__name__,
                    )
                continue
            if not docs:
                logger.error(f"No documents extracted from {file_path}.")
                if progress_callback is not None:
                    progress_callback(
                        stage="parsing_empty",
                        message=f"No documents extracted from {file_path.name}.",
                        current=file_index,
                        total=total_files,
                        file_name=file_path.name,
                        file_path=str(file_path),
                        source_id=source_id,
                    )
                continue
            document_id = f"doc_{hashlib.sha1((source_id or str(file_path)).encode('utf-8')).hexdigest()[:16]}"
            parsed_at_utc = _utc_now_iso()
            try:
                split_segments = self._split_docs(file_path=file_path, docs=docs)
            except Exception as exc:
                logger.error("Failed to split/index file during preparation path=%s reason=%s", file_path, exc)
                if progress_callback is not None:
                    progress_callback(
                        stage="file_failed",
                        message=f"Failed to prepare file {file_path.name}: {exc}",
                        current=file_index,
                        total=total_files,
                        file_name=file_path.name,
                        file_path=str(file_path),
                        source_id=source_id,
                        error=str(exc),
                        error_type=exc.__class__.__name__,
                    )
                continue
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
                if progress_callback is not None:
                    progress_callback(
                        stage="parsing_empty",
                        message=f"No indexable content extracted from {file_path.name}.",
                        current=file_index,
                        total=total_files,
                        file_name=file_path.name,
                        file_path=str(file_path),
                    )
                continue
                #raise ValueError(f"No indexable content extracted from {file_path}.")
            combined_text = "\n\n".join(segment.content for segment in segments)
            if progress_callback is not None:
                progress_callback(
                    stage="index_building",
                    message=f"Building index from {len(segments)} segment(s) for {file_path.name}.",
                    current=file_index,
                    total=total_files,
                    file_name=file_path.name,
                    file_path=str(file_path),
                    source_id=source_id,
                    total_segments=indexed_segments + len(segments),
                    total_documents=indexed_documents + 1,
                )
            try:
                self._index_documents(segments=segments)
            except Exception as exc:
                logger.error("Failed to index file during preparation path=%s reason=%s", file_path, exc)
                self._delete_cached_document(source_id=source_id, document_id=document_id)
                if progress_callback is not None:
                    progress_callback(
                        stage="index_failed",
                        message=f"Failed to index file {file_path.name}: {exc}",
                        current=file_index,
                        total=total_files,
                        file_name=file_path.name,
                        file_path=str(file_path),
                        source_id=source_id,
                        error=str(exc),
                        error_type=exc.__class__.__name__,
                    )
                continue
            if progress_callback is not None:
                progress_callback(
                    stage="index_ready",
                    message=f"Index updated with {len(segments)} segment(s) from {file_path.name}.",
                    current=file_index,
                    total=total_files,
                    file_name=file_path.name,
                    file_path=str(file_path),
                    source_id=source_id,
                    total_segments=indexed_segments + len(segments),
                    total_documents=indexed_documents + 1,
                )
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
            indexed_documents += 1
            indexed_segments += len(segments)
            if progress_callback is not None:
                progress_callback(
                    stage="file_ready",
                    message=f"Prepared {len(segments)} chunk(s) from {file_path.name}.",
                    current=file_index,
                    total=total_files,
                    file_name=file_path.name,
                    file_path=str(file_path),
                    chunks=len(segments),
                )
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
        return {"Accept": "application/json"}

    def _client(self) -> httpx.Client:
        return httpx.Client(headers=self._headers(), timeout=20.0)

    @staticmethod
    def _require_setting(value: str, env_name: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise RuntimeError(f"{env_name} is not configured.")
        return normalized

    def _damia_scoring_api_key(self) -> str:
        return self._require_setting(
            self._settings.damia_scoring_api_key,
            "SALES_LEAD_AGENT_DAMIA_SCORING_API_KEY",
        )

    def _damia_fssp_api_key(self) -> str:
        return self._require_setting(
            self._settings.damia_fssp_api_key,
            "SALES_LEAD_AGENT_DAMIA_FSSP_API_KEY",
        )

    def _dadata_api_key(self) -> str:
        return self._require_setting(
            self._settings.dadata_api_key,
            "DADATA_API_KEY",
        )

    def _scoring_base_url(self) -> str:
        return self._require_setting(
            self._settings.scoring_base_url,
            "SALES_LEAD_AGENT_SCORING_BASE_URL",
        ).rstrip("/")

    def _fssp_base_url(self) -> str:
        return self._require_setting(
            self._settings.fssp_base_url,
            "SALES_LEAD_AGENT_FSSP_BASE_URL",
        ).rstrip("/")

    def _damia_scoring_params(self, **params: Any) -> dict[str, Any]:
        prepared = {
            key: value
            for key, value in params.items()
            if value is not None and value != ""
        }
        prepared["key"] = self._damia_scoring_api_key()
        return prepared

    def _damia_fssp_params(self, **params: Any) -> dict[str, Any]:
        prepared = {
            key: value
            for key, value in params.items()
            if value is not None and value != ""
        }
        prepared["key"] = self._damia_fssp_api_key()
        return prepared

    @staticmethod
    def _first_present(payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        return None

    @staticmethod
    def _looks_like_score_metrics(payload: dict[str, Any]) -> bool:
        metric_keys = {
            "risk_value",
            "risk_zone",
            "score_value",
            "score_zone",
            "reliability_value",
            "reliability_zone",
            "РискЗнач",
            "РискЗона",
            "БаллЗнач",
            "БаллЗона",
            "НадежностьЗнач",
            "НадежностьЗона",
            "top_factors",
            "Показатели",
        }
        return any(key in payload for key in metric_keys)

    @classmethod
    def _is_empty_nested_payload(cls, payload: Any) -> bool:
        if payload is None:
            return True
        if isinstance(payload, str):
            return not payload.strip()
        if isinstance(payload, list):
            return not payload or all(cls._is_empty_nested_payload(item) for item in payload)
        if isinstance(payload, dict):
            return not payload or all(cls._is_empty_nested_payload(value) for value in payload.values())
        return False

    @staticmethod
    def _looks_like_fincoef_metrics(payload: dict[str, Any]) -> bool:
        metric_keys = {
            "value",
            "norm",
            "comparison",
            "Знач",
            "Норма",
            "НормаСравн",
            "Балл",
            "НормаНижн",
            "НормаВерхн",
        }
        return any(key in payload for key in metric_keys)

    @classmethod
    def _looks_like_fincoefs_container(cls, payload: dict[str, Any]) -> bool:
        nested_dicts = [value for value in payload.values() if isinstance(value, dict)]
        if not nested_dicts or len(nested_dicts) != len(payload):
            return False
        return any(
            cls._looks_like_fincoef_metrics(value)
            or any(
                isinstance(year_key, str) and year_key.isdigit() and isinstance(year_value, dict)
                for year_key, year_value in value.items()
            )
            for value in nested_dicts
        )

    @classmethod
    def _unwrap_object_payload(cls, payload: Any, label: str) -> dict[str, Any]:
        if isinstance(payload, str):
            error_text = payload.strip()
            if error_text.lower().startswith("ошибка"):
                raise RuntimeError(f"{label} error: {error_text}")
            try:
                return cls._unwrap_object_payload(json.loads(payload), label)
            except json.JSONDecodeError as exc:
                raise TypeError(f"{label} returned a non-object payload.") from exc
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    return item
            raise TypeError(f"{label} returned a non-object payload.")
        if isinstance(payload, dict):
            error_text = _optional_string(cls._first_present(payload, "error", "Ошибка"))
            if error_text:
                raise RuntimeError(f"{label} error: {error_text}")
            for key in ("result", "data", "item"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    return nested
            return payload
        raise TypeError(f"{label} returned a non-object payload.")

    @classmethod
    def _unwrap_list_payload(cls, payload: Any, label: str) -> list[dict[str, Any]]:
        if isinstance(payload, str):
            error_text = payload.strip()
            if error_text.lower().startswith("ошибка"):
                raise RuntimeError(f"{label} error: {error_text}")
            try:
                return cls._unwrap_list_payload(json.loads(payload), label)
            except json.JSONDecodeError as exc:
                raise TypeError(f"{label} returned a non-list payload.") from exc
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            error_text = _optional_string(cls._first_present(payload, "error", "Ошибка"))
            if error_text:
                raise RuntimeError(f"{label} error: {error_text}")
            for key in ("result", "items", "data"):
                nested = payload.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
        raise TypeError(f"{label} returned a non-list payload.")

    @staticmethod
    def _optional_float_relaxed(value: Any) -> float | None:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace("\xa0", "").replace(" ", "")
        if not text:
            return None
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        return float(text)

    @classmethod
    def _optional_int_relaxed(cls, value: Any) -> int | None:
        if value in (None, ""):
            return None
        return int(cls._optional_float_relaxed(value))

    @classmethod
    def _normalize_top_factors(cls, payload: dict[str, Any]) -> list[TopFactor]:
        raw_factors = cls._first_present(payload, "top_factors", "Показатели")
        if isinstance(raw_factors, dict):
            raw_factors = raw_factors.get("items") or raw_factors.get("result") or []
        if not isinstance(raw_factors, list):
            return []
        factors: list[TopFactor] = []
        for item in raw_factors:
            if not isinstance(item, dict):
                continue
            name = _optional_string(cls._first_present(item, "name", "Наименование"))
            if not name:
                continue
            factors.append(
                TopFactor(
                    name=name,
                    value=cls._optional_float_relaxed(
                        cls._first_present(item, "value", "Значение")
                    ),
                    nwoe=cls._optional_float_relaxed(
                        cls._first_present(item, "nwoe", "nWoE")
                    ),
                )
            )
        return factors

    @classmethod
    def _extract_score_metrics_container(cls, payload: Any) -> dict[str, Any]:
        if cls._is_empty_nested_payload(payload):
            return {}
        try:
            raw = cls._unwrap_object_payload(payload, "Scoring API")
        except TypeError:
            if cls._is_empty_nested_payload(payload):
                return {}
            raise
        if cls._looks_like_score_metrics(raw):
            return raw

        current = raw
        while isinstance(current, dict):
            if not current:
                return {}
            if cls._looks_like_score_metrics(current):
                return current

            year_candidates = [
                (key, value)
                for key, value in current.items()
                if isinstance(key, str) and key.isdigit() and isinstance(value, dict)
            ]
            if year_candidates:
                latest_year_key, latest_year_value = max(year_candidates, key=lambda item: int(item[0]))
                current = latest_year_value
                if cls._looks_like_score_metrics(current):
                    return current
                continue

            nested_dicts = [value for value in current.values() if isinstance(value, dict)]
            if len(nested_dicts) == 1:
                current = nested_dicts[0]
                if not current:
                    return {}
                continue
            break

        raise TypeError("Scoring API returned an unsupported payload shape.")

    @classmethod
    def _normalize_score_payload(cls, payload: Any) -> ScorePayload:
        raw = cls._extract_score_metrics_container(payload)
        return ScorePayload(
            risk_value=cls._optional_float_relaxed(
                cls._first_present(raw, "risk_value", "РискЗнач")
            ),
            risk_zone=_optional_string(cls._first_present(raw, "risk_zone", "РискЗона")),
            score_value=cls._optional_float_relaxed(
                cls._first_present(raw, "score_value", "БаллЗнач")
            ),
            score_zone=_optional_string(cls._first_present(raw, "score_zone", "БаллЗона")),
            reliability_value=cls._optional_float_relaxed(
                cls._first_present(raw, "reliability_value", "НадежностьЗнач")
            ),
            reliability_zone=_optional_string(
                cls._first_present(raw, "reliability_zone", "НадежностьЗона")
            ),
            top_factors=cls._normalize_top_factors(raw),
        )

    @classmethod
    def _extract_fincoefs_container(cls, payload: Any) -> list[dict[str, Any]] | dict[str, Any]:
        try:
            raw_items = cls._unwrap_list_payload(payload, "Fincoefs API")
        except TypeError:
            raw = cls._unwrap_object_payload(payload, "Fincoefs API")
        else:
            return raw_items

        current = raw
        while isinstance(current, dict):
            if cls._looks_like_fincoefs_container(current):
                return current
            nested_dicts = [value for value in current.values() if isinstance(value, dict)]
            if len(nested_dicts) == 1:
                current = nested_dicts[0]
                continue
            break
        raise TypeError("Fincoefs API returned an unsupported payload shape.")

    @classmethod
    def _normalize_fincoefs_payload(cls, payload: Any) -> list[Fincoef]:
        raw_container = cls._extract_fincoefs_container(payload)
        if isinstance(raw_container, list):
            raw_items = raw_container
        else:
            raw_items = []
            for coef_name, coef_payload in raw_container.items():
                if not isinstance(coef_payload, dict):
                    continue
                metrics = coef_payload if cls._looks_like_fincoef_metrics(coef_payload) else None
                if metrics is None:
                    year_candidates = [
                        (int(year_key), year_value)
                        for year_key, year_value in coef_payload.items()
                        if isinstance(year_key, str)
                        and year_key.isdigit()
                        and isinstance(year_value, dict)
                    ]
                    if year_candidates:
                        metrics = max(year_candidates, key=lambda item: item[0])[1]
                if metrics is None:
                    continue
                raw_items.append({"name": coef_name, **metrics})

        fincoefs: list[Fincoef] = []
        for item in raw_items:
            name = _optional_string(
                cls._first_present(item, "name", "Фин. коэффициент", "ФинКоэффициент")
            )
            if not name:
                continue
            fincoefs.append(
                Fincoef(
                    name=name,
                    value=cls._optional_float_relaxed(
                        cls._first_present(item, "value", "Знач")
                    ),
                    norm=cls._optional_float_relaxed(
                        cls._first_present(item, "norm", "Норма")
                    ),
                    comparison=_optional_string(
                        cls._first_present(item, "comparison", "НормаСравн")
                    ),
                )
            )
        return fincoefs

    @classmethod
    def _normalize_fssp_grouped_payload(cls, payload: Any) -> list[FSSPGroupedRecord]:
        raw_items = cls._unwrap_list_payload(payload, "FSSP API")
        records: list[FSSPGroupedRecord] = []
        for item in raw_items:
            year = cls._optional_int_relaxed(cls._first_present(item, "year", "Год"))
            status = _optional_string(cls._first_present(item, "status", "Статус"))
            subject = _optional_string(cls._first_present(item, "subject", "Предмет"))
            count = cls._optional_int_relaxed(cls._first_present(item, "count", "Количество"))
            if year is None or not status or not subject or count is None:
                continue
            raw_ids = cls._first_present(item, "proceeding_ids", "ИП")
            if isinstance(raw_ids, list):
                proceeding_ids = [str(value) for value in raw_ids if str(value).strip()]
            elif raw_ids in (None, ""):
                proceeding_ids = []
            else:
                proceeding_ids = [str(raw_ids)]
            records.append(
                FSSPGroupedRecord(
                    year=year,
                    status=status,
                    subject=subject,
                    amount=cls._optional_float_relaxed(
                        cls._first_present(item, "amount", "Сумма")
                    ),
                    count=count,
                    proceeding_ids=proceeding_ids,
                )
            )
        return records

    @classmethod
    def _normalize_dadata_party_payload(cls, *, inn: str, payload: Any) -> CounterpartyLookupResponse:
        raw = cls._unwrap_object_payload(payload, "DaData party lookup")
        suggestions = raw.get("suggestions")
        if not isinstance(suggestions, list):
            suggestions = []
        candidates = [item for item in suggestions if isinstance(item, dict)]
        if not candidates:
            return CounterpartyLookupResponse(
                inn=inn,
                found=False,
                message="Контрагент не найден",
            )

        def _candidate_priority(item: dict[str, Any]) -> tuple[int, int]:
            data = item.get("data") if isinstance(item.get("data"), dict) else {}
            state = data.get("state") if isinstance(data.get("state"), dict) else {}
            active = 0 if str(state.get("status") or "").strip().upper() == "ACTIVE" else 1
            exact = 0 if str(data.get("inn") or "").strip() == inn else 1
            return (active, exact)

        best = sorted(candidates, key=_candidate_priority)[0]
        data = best.get("data") if isinstance(best.get("data"), dict) else {}
        name_payload = data.get("name") if isinstance(data.get("name"), dict) else {}
        address_payload = data.get("address") if isinstance(data.get("address"), dict) else {}
        state_payload = data.get("state") if isinstance(data.get("state"), dict) else {}
        management_payload = data.get("management") if isinstance(data.get("management"), dict) else {}
        resolved_name = _optional_string(
            cls._first_present(
                name_payload,
                "short_with_opf",
                "full_with_opf",
                "short",
                "full",
            )
        ) or _optional_string(best.get("value"))
        return CounterpartyLookupResponse(
            inn=inn,
            found=True,
            name=resolved_name,
            full_name=_optional_string(
                cls._first_present(name_payload, "full_with_opf", "full")
            ),
            address=_optional_string(
                cls._first_present(address_payload, "unrestricted_value", "value")
            ),
            kpp=_optional_string(data.get("kpp")),
            ogrn=_optional_string(data.get("ogrn")),
            okved=_optional_string(data.get("okved")),
            state_status=_optional_string(state_payload.get("status")),
            management_name=_optional_string(management_payload.get("name")),
            management_post=_optional_string(management_payload.get("post")),
        )

    def scoring(self, *, inn: str, model: str | None, include_fincoefs: bool) -> CounterpartyScoringResponse:
        scoring_base_url = self._scoring_base_url()
        resolved_model = _optional_string(model) or self._settings.scoring_default_model
        with self._client() as client:
            score_response = client.get(
                f"{scoring_base_url}/scoring/score",
                params=self._damia_scoring_params(inn=inn, model=resolved_model),
            )
            score_response.raise_for_status()
            score_payload = score_response.json() if score_response.content else {}
            fincoefs_payload: Any = []
            if include_fincoefs:
                fincoefs_response = client.get(
                    f"{scoring_base_url}/scoring/fincoefs",
                    params=self._damia_scoring_params(inn=inn),
                )
                fincoefs_response.raise_for_status()
                fincoefs_payload = fincoefs_response.json() if fincoefs_response.content else []
        return CounterpartyScoringResponse(
            inn=inn,
            score=self._normalize_score_payload(score_payload),
            fincoefs=self._normalize_fincoefs_payload(fincoefs_payload) if include_fincoefs else [],
        )

    def fssp(
        self,
        *,
        inn: str,
        from_date: str | None,
        to_date: str | None,
    ) -> CounterpartyFSSPResponse:
        fssp_base_url = self._fssp_base_url()
        with self._client() as client:
            response = client.get(
                f"{fssp_base_url}/fssp/isps",
                params=self._damia_fssp_params(
                    inn=inn,
                    from_date=from_date,
                    to_date=to_date,
                    format=1,
                ),
            )
            response.raise_for_status()
            payload = response.json() if response.content else []
        return CounterpartyFSSPResponse(
            inn=inn,
            grouped=self._normalize_fssp_grouped_payload(payload),
            raw_format=1,
        )

    def lookup_party(self, *, inn: str, include_branches: bool) -> CounterpartyLookupResponse:
        api_key = self._dadata_api_key()
        payload = {"query": inn}
        if not include_branches:
            payload["branch_type"] = "MAIN"
        with httpx.Client(
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Token {api_key}",
            },
            timeout=20.0,
        ) as client:
            response = client.post(
                _DADATA_FIND_PARTY_URL,
                json=payload,
            )
            response.raise_for_status()
            payload = response.json() if response.content else {}
        return self._normalize_dadata_party_payload(inn=inn, payload=payload)


@dataclass(frozen=True)
class SalesLeadAgentDependencies:
    workspace_manager: RunWorkspaceManager
    document_service: DocumentPreparationService
    purchase_adapter: PurchaseAdapter
    counterparty_clients: CounterpartyClients
    open_source_max_concurrency: int
    retrieval_client: Any | None = None

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
            retrieval_client=None,
        )

def _retrieval_progress_from_snapshot(snapshot: Any) -> RetrievalProgress:
    raw_progress = getattr(snapshot, "progress", None)
    if isinstance(raw_progress, dict):
        progress = raw_progress
    elif hasattr(raw_progress, "model_dump"):
        progress = raw_progress.model_dump()
    else:
        progress = {}
    return RetrievalProgress(
        total_queries=int(progress.get("total_queries", 0)),
        completed_queries=int(progress.get("completed_queries", 0)),
        total_purchases=int(progress.get("total_purchases", 0)),
        processed_purchases=int(progress.get("processed_purchases", 0)),
        total_files=int(progress.get("total_files", 0)),
        processed_files=int(progress.get("processed_files", 0)),
        prepared_documents=int(progress.get("prepared_documents", 0)),
        indexed_segments=int(progress.get("indexed_segments", 0)),
    )


def _purchase_response_from_snapshot(
    snapshot: Any,
    *,
    record_from: int = 0,
) -> PurchaseSearchResponse:
    raw_items = list(getattr(snapshot, "items", None) or [])
    validated_items = [PurchaseSearchItem.model_validate(item) for item in raw_items]
    total_ready_records = len(validated_items)
    safe_record_from = min(record_from, total_ready_records) if total_ready_records else 0
    page_items = validated_items[safe_record_from : safe_record_from + _PURCHASE_SEARCH_PAGE_SIZE]
    next_record_from = safe_record_from + len(page_items)
    if next_record_from >= total_ready_records:
        next_record_from = None
    return PurchaseSearchResponse(
        run_id=str(snapshot.run_id),
        index_id=str(snapshot.index_id),
        retrieval_status=str(snapshot.status),
        retrieval_stage=str(snapshot.stage),
        message=str(snapshot.message or ""),
        progress=_retrieval_progress_from_snapshot(snapshot),
        search_urls=[str(value) for value in snapshot.request_payload.get("search_urls") or []],
        record_from=safe_record_from,
        returned_records=len(page_items),
        total_ready_records=total_ready_records,
        next_record_from=next_record_from,
        items=page_items,
    )


def _retrieval_state_from_snapshot(snapshot: Any) -> dict[str, Any]:
    progress = _retrieval_progress_from_snapshot(snapshot)
    return {
        "active_retrieval_id": str(snapshot.retrieval_id),
        "active_retrieval_request_hash": str(snapshot.request_hash),
        "active_retrieval_run_id": str(snapshot.run_id),
        "active_retrieval_index_id": str(snapshot.index_id),
        "active_retrieval_status": str(snapshot.status),
        "active_retrieval_stage": str(snapshot.stage),
        "active_retrieval_message": str(snapshot.message or ""),
        "active_retrieval_progress": progress.model_dump(),
    }


def build_sales_lead_tools(
    dependencies: SalesLeadAgentDependencies | None = None,
) -> list[Any]:
    deps = dependencies or SalesLeadAgentDependencies.from_settings(get_settings())
    yandex_web_search_tool = YandexSearchTool(
        api_key=config.YA_API_KEY or "",
        folder_id=config.YA_FOLDER_ID or "",
        max_results=5,
        summarize=True,
    )

    @tool(
        "purchase_search_tool",
        args_schema=PurchaseSearchRequest,
        description=(
            "Start or refresh a conversation-scoped procurement retrieval on EIS. Use this tool "
            "to submit a procurement search, inspect current retrieval progress, and get the "
            "index_id needed for follow-up calls to doc_search_tool. Repeated calls with "
            "the same normalized request return the latest ready subset while the background "
            "retrieval continues. The tool always returns at most 5 procurement records per call; "
            "use record_from to page through ready results."
        ),
    )
    async def purchase_search_tool(
        *,
        search_url: str | None = None,
        query_texts: list[str] | None = None,
        max_pages: int | None = None,
        record_from: int | None = None,
        runtime: ToolRuntime,
    ) -> Command:
        """Submit or refresh a background procurement retrieval for the current conversation.

        Use this tool when the task requires finding public procurement records on EIS and then
        reading or searching procurement files. The tool does not block on crawling and indexing:
        it creates or refreshes a conversation-scoped background retrieval job and returns the
        current ready subset of procurement items.

        Provide either a direct `search_url` or one or more `query_texts`. Prefer `query_texts`
        when you know the business terms but do not already have a valid EIS search URL. Prefer
        `search_url` when the exact procurement search page is already known and should be used
        without rewriting the query parameters.

        Args:
            search_url: Optional direct EIS extended-search URL. If present, it is used as-is.
            query_texts: Optional list of contextual search phrases for the `searchString` URL
                parameter. Zakupki already applies morphology and matches all words in one query
                with AND semantics. To simulate OR, pass multiple alternative search strings.
                Recommended strategy: use short procurement-style phrases or stems, not long bags
                of synonyms. For example, `страхование`, `страхованию`, `страхования` should first
                be searched as `страхован`; if that returns no results, call the tool again with a
                weaker query such as `страхов`.
            max_pages: Optional crawler page limit. This affects request identity.
            record_from: Optional zero-based offset for paging through ready procurement records.
                This affects only the returned page, not retrieval identity. The tool never returns
                more than 5 records in one call.
            runtime: Tool runtime injected by LangChain/LangGraph.

        Returns:
            A state-updating command whose tool-visible JSON payload contains `run_id`, `index_id`,
            `retrieval_status`, `retrieval_stage`, `message`, `progress`, paging fields, and the
            current ready `items` page.

        The active run identifier is taken from agent state when one already exists for this
        investigation. The model should not supply `run_id`; it is internal state carried across
        turns and included in the response only as retrieval context.

        If the conversation already has a procurement retrieval and this tool is called again with
        no new `search_url`/`query_texts`, it refreshes and returns the latest snapshot for the
        existing retrieval instead of failing on missing search input.

        If a procurement retrieval is already active for the same conversation, this tool returns
        the latest ready subset and updated progress instead of creating a new job. A new submit
        is attempted only when there is no active retrieval.
        """
        try:
            normalized_record_from = _normalize_purchase_record_from(record_from)
            retrieval_client = deps.retrieval_client or get_retrieval_service_client()
            conversation_id = _thread_id_from_runtime(runtime)
            requested_run_id = _requested_run_id_from_runtime(runtime)
            snapshot = await _resolve_existing_purchase_snapshot(
                retrieval_client=retrieval_client,
                conversation_id=conversation_id,
                runtime=runtime,
            )
            if snapshot is not None and getattr(snapshot, "status", None) in {"queued", "in_progress"}:
                response = _purchase_response_from_snapshot(snapshot, record_from=normalized_record_from)
                return Command(
                    update={
                        **_retrieval_state_from_snapshot(snapshot),
                        "messages": _tool_message(response.model_dump(), runtime),
                    }
                )
            if not _has_purchase_search_inputs(search_url=search_url, query_texts=query_texts):
                if snapshot is None:
                    raise ToolUserCorrectableError(
                        code="MISSING_SEARCH_INPUT",
                        message="purchase_search_tool requires either search_url or query_texts.",
                        suggestion=(
                            "Provide a direct search_url or one or more procurement-style search "
                            "strings and call the tool again."
                        ),
                        input_field="query_texts",
                    )
                response = _purchase_response_from_snapshot(snapshot, record_from=normalized_record_from)
                return Command(
                    update={
                        **_retrieval_state_from_snapshot(snapshot),
                        "messages": _tool_message(response.model_dump(), runtime),
                    }
                )
            try:
                snapshot = await retrieval_client.submit_purchase_search(
                    conversation_id=conversation_id,
                    requested_run_id=requested_run_id,
                    search_url=search_url,
                    query_texts=query_texts,
                    max_pages=max_pages,
                    agent_id="sales_lead_agent",
                )
            except RetrievalServiceConflictError as exc:
                raise ToolUserCorrectableError(
                    code="RETRIEVAL_ALREADY_IN_PROGRESS",
                    message=str(exc),
                    suggestion=(
                        "Ask about the current retrieval status or wait for it to finish before "
                        "starting a different procurement search."
                    ),
                    input_field="query_texts" if query_texts is not None else "search_url",
                ) from None
            except RetrievalServiceUserInputError as exc:
                raise ToolUserCorrectableError(
                    code=exc.code,
                    message=str(exc),
                    suggestion=exc.suggestion,
                    input_field=exc.input_field,
                ) from None

            response = _purchase_response_from_snapshot(snapshot, record_from=normalized_record_from)
            return Command(
                update={
                    **_retrieval_state_from_snapshot(snapshot),
                    "messages": _tool_message(response.model_dump(), runtime),
                }
            )
        except ToolUserCorrectableError:
            raise
        except Exception as exc:
            _raise_unexpected_tool_failure(
                tool_name="purchase_search_tool",
                code="PURCHASE_SEARCH_FAILED",
                message="Procurement search is unavailable",
                suggestion="Retry the same procurement search or ask about the current retrieval status.",
                exc=exc,
                input_field="query_texts" if query_texts is not None else "search_url",
            )

    @tool(
        "retrieve_page_tool",
        args_schema=RetrievePageRequest,
        description=(
            "Retrieve one exact public page URL, download its same-host attachments, prepare the "
            "content for semantic search, and return run_id/index_id for follow-up doc_search_tool calls. "
            "Do not use this for general internet search; use web_search first."
        ),
    )
    async def retrieve_page_tool(
        *,
        url: str,
        runtime: ToolRuntime,
    ) -> dict[str, Any]:
        """Fetch one exact public page URL and its same-host attachments into a searchable run.

        Internal rule: this tool does not crawl site navigation. It loads only the exact page URL
        plus downloadable same-host attachments referenced from that page.
        """
        try:
            requested_run_id = _requested_run_id_from_runtime(runtime)
            workspace = (
                deps.workspace_manager.get(requested_run_id)
                if requested_run_id
                else deps.workspace_manager.create_run()
            )
            url = _validate_retrieve_page_url(url)
            loader = AsyncWebLoader(
                url=url,
                depth=0,
                fetch_mode="playwright",
                playwright_headless=True,
                follow_download_links=True,
                max_concurrency=deps.open_source_max_concurrency,
                continue_on_error=False,
            )
            try:
                docs = await loader.load()
            except (RuntimeError, ValueError, TypeError) as exc:
                _raise_open_source_loader_error(exc)
            last_errors = getattr(loader, "last_errors", None) or []
            if last_errors and not docs:
                raise ToolUserCorrectableError(
                    code="FETCH_FAILED",
                    message="; ".join(str(item.get("error") or item) for item in last_errors),
                    suggestion="Retry the call with a different page URL.",
                    input_field="url",
                )
            if last_errors:
                logger.warning(
                    "retrieve_page_tool partial fetch warnings for %s: %s",
                    url,
                    "; ".join(str(item.get("error") or item) for item in last_errors),
                )
            if not docs:
                raise ToolUserCorrectableError(
                    code="NO_CONTENT_FETCHED",
                    message=f"No content fetched from {url}.",
                    suggestion="Check the page URL and retry.",
                    input_field="url",
                )

            pages: list[RetrievedPage] = []
            prepared_documents: list[PreparedDocument] = []
            attachments_by_parent: dict[str, list[str]] = defaultdict(list)

            for doc in docs:
                metadata = dict(doc.metadata or {})
                source_url = str(metadata.get("source") or metadata.get("url") or "").strip()
                if not source_url:
                    raise ToolUserCorrectableError(
                        code="MISSING_SOURCE_URL",
                        message="retrieve_page_tool received content without a source URL.",
                        suggestion="Retry the call with a different URL.",
                        input_field="url",
                    )
                parent_url = str(metadata.get("parent_url") or source_url).strip()
                content = str(doc.page_content or "")
                if not content.strip():
                    raise ToolUserCorrectableError(
                        code="EMPTY_FETCHED_CONTENT",
                        message=f"Fetched empty content for {source_url}.",
                        suggestion="Retry the call with a different page URL.",
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
                        RetrievedPage(
                            bundle_id=bundle_id,
                            url=source_url,
                            title=str(metadata.get("title")) if metadata.get("title") else None,
                            text_excerpt=content[:500],
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
                    RetrievedPage(
                        bundle_id=bundle_id,
                        url=source_url,
                        title=str(metadata.get("title")) if metadata.get("title") else None,
                        text_excerpt=content[:500],
                        prepared_document_ids=[doc_item.document_id for doc_item in prepared],
                    )
                )

            if not pages and not prepared_documents:
                raise ToolUserCorrectableError(
                    code="NO_SEARCHABLE_ARTIFACTS",
                    message=f"No searchable artifacts were prepared from {url}.",
                    suggestion="Retry with a different page URL.",
                    input_field="url",
                )
            for page in pages:
                page.attachments = attachments_by_parent.get(page.url, [])
            response = RetrievePageResponse(
                run_id=workspace.run_id,
                index_id=deps.document_service.shared_index_id,
                pages=pages,
                prepared_documents=prepared_documents,
            )
            return response.model_dump()
        except ToolUserCorrectableError:
            raise
        except Exception as exc:
            _raise_unexpected_tool_failure(
                tool_name="retrieve_page_tool",
                code="RETRIEVE_PAGE_FAILED",
                message=f"Failed to retrieve page {url}",
                suggestion="Retry the same URL later or use a different exact page URL.",
                exc=exc,
                input_field="url",
            )

    @tool(
        "web_search_tool",
        args_schema=WebSearchRequest,
        description=(
            "Search public web information from a single search_string and return ranked result "
            "links/snippets. Use this first for open internet research when you do not already have an exact page URL."
        ),
    )
    def web_search_tool(
        *,
        search_string: str,
    ) -> dict[str, Any]:
        """Search the open web and return a compact ranked result list for one search string."""
        try:
            normalized_query = _require_web_search_string(search_string)
            with httpx.Client(
                headers={
                    "Accept": "text/html,application/xhtml+xml",
                    "User-Agent": "Mozilla/5.0",
                },
                timeout=20.0,
                follow_redirects=True,
            ) as client:
                response = client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": normalized_query, "kl": "ru-ru"},
                )
                response.raise_for_status()
            results = [
                WebSearchResult.model_validate(item)
                for item in _extract_web_search_results(response.text, max_results=5)
            ]
            return WebSearchResponse(search_string=normalized_query, results=results).model_dump()
        except ToolUserCorrectableError:
            raise
        except Exception as exc:
            _raise_unexpected_tool_failure(
                tool_name="web_search_tool",
                code="WEB_SEARCH_FAILED",
                message=f"Web search is unavailable for query {search_string!r}",
                suggestion="Retry the same search string later or try a shorter query.",
                exc=exc,
                input_field="search_string",
            )

    @tool(
        "doc_search_tool",
        args_schema=DocSearchRequest,
        description=(
            "Search a previously prepared document index and return grounded snippets with provenance. "
            "Use this only after purchase_search_tool or retrieve_page_tool has already produced "
            "an index_id."
        ),
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

        Use this tool only after `purchase_search_tool` or `retrieve_page_tool` has already
        returned an `index_id`. This tool does not fetch new content. It searches the prepared
        document chunks and returns grounded snippets with provenance such as file path, page,
        locator, source kind, and source URL.

        Use metadata filters when the task already points to one procurement, one source URL, or
        one artifact bundle. That keeps retrieval narrower and reduces irrelevant matches.

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

        The tool returns search evidence, not a final synthesized answer. Use the returned snippets
        and provenance to support the next reasoning step or a user-facing answer.
        """
        try:
            response = deps.document_service.search(
                index_id=index_id,
                query=query,
                top_k=5 if top_k is None else top_k,
                source_kind=source_kind,  # type: ignore[arg-type]
                bundle_id=bundle_id,
                purchase_id=purchase_id,
                source_id=source_id,
            )
        except ToolUserCorrectableError:
            raise
        except Exception as exc:
            _raise_unexpected_tool_failure(
                tool_name="doc_search_tool",
                code="DOC_SEARCH_FAILED",
                message=f"Document search is unavailable for index {index_id}",
                suggestion="Retry the same document search later or use a different index_id.",
                exc=exc,
                input_field="index_id",
            )
        return response.model_dump()

    @tool(
        "read_cached_document_tool",
        args_schema=ReadCachedDocumentRequest,
        description=(
            "Read cached text content without fetching the network. Use exact document_id when you "
            "already have it from doc_search_tool or retrieve_page_tool. For procurement files that "
            "were downloaded but are not yet searchable, use bundle_id + file_name from "
            "purchase_search_tool. index_id is optional and will be reused automatically from the "
            "current conversation when omitted. The tool reads from local prepared artifacts and raw "
            "procurement download cache."
        ),
    )
    async def read_cached_document_tool(
        *,
        index_id: str | None = None,
        document_id: str | None = None,
        bundle_id: str | None = None,
        file_name: str | None = None,
        offset: int | None = None,
        max_chars: int | None = None,
        runtime: ToolRuntime | None = None,
    ) -> dict[str, Any]:
        """Read cached content from one previously prepared document.

        Use this when the investigation needs the prepared document text itself rather than
        semantic snippets. The tool reads only from local cached artifacts or indexed chunks and
        never performs a new download.

        Args:
            index_id: Optional shared index identifier returned by a previous acquisition tool.
                When omitted, the tool reuses the current runtime index context or the configured
                shared index automatically.
            document_id: Optional exact prepared document identifier returned by `doc_search_tool`
                or `retrieve_page_tool`.
            bundle_id: Optional procurement bundle identifier returned by `purchase_search_tool`.
            file_name: Optional file selector from `purchase_search_tool` when the file is already
                downloaded but not yet searchable. This may be an exact file name, a downloaded
                file path, or a short file hint when it uniquely identifies one cached file inside
                the procurement bundle.
            offset: Optional zero-based character offset into the cached content.
            max_chars: Optional response window size. Omit it to use a safe default window.

        Returns:
            A dictionary with the cached content window plus `next_offset` when more content is
            available.
        """
        try:
            requested_run_id = _requested_run_id_from_runtime(runtime)
            resolved_index_id = (
                _normalize_optional_selector(index_id)
                or _requested_index_id_from_runtime(runtime)
                or deps.document_service.shared_index_id
            )
            workspace = (
                deps.workspace_manager.get(requested_run_id)
                if isinstance(requested_run_id, str) and requested_run_id.strip()
                else None
            )
            response = deps.document_service.read_cached_document(
                index_id=resolved_index_id,
                document_id=document_id,
                workspace=workspace,
                bundle_id=bundle_id,
                file_name=file_name,
                offset=offset,
                max_chars=max_chars,
            )
        except ToolUserCorrectableError:
            raise
        except Exception as exc:
            selector = document_id or f"{bundle_id}:{file_name}"
            _raise_unexpected_tool_failure(
                tool_name="read_cached_document_tool",
                code="READ_CACHED_DOCUMENT_FAILED",
                message=f"Cached document content is unavailable for selector {selector}",
                suggestion=(
                    "Retry with a valid document_id from doc_search_tool, or use bundle_id plus a "
                    "file name, downloaded file path, or clearer file hint from purchase_search_tool."
                ),
                exc=exc,
                input_field="document_id" if document_id else "file_name",
            )
        return response.model_dump()

    @tool(
        "counterparty_scoring_tool",
        args_schema=CounterpartyScoringRequest,
        description=(
            "Fetch normalized counterparty scoring data for a supplied INN. Use this tool when you "
            "need risk, score, reliability, top factors, and optionally financial coefficients."
        ),
    )
    def counterparty_scoring_tool(
        *,
        inn: str,
        model: str | None = None,
        include_fincoefs: bool | None = None,
    ) -> dict[str, Any]:
        """Fetch counterparty scoring data.

        Use this tool when the task requires a normalized scoring snapshot for a company identified
        by INN. It retrieves risk, score, reliability, top explanatory factors, and optionally
        financial coefficients from the upstream scoring service.

        Args:
            inn: Company INN to analyze.
            model: Optional external scoring model identifier. If omitted, the configured
                default Damia model is used.
            include_fincoefs: Whether financial coefficients should also be requested.

        Returns:
            A dictionary with the normalized scoring payload for the supplied INN.

        The response is already normalized for downstream reasoning and should be used as structured
        evidence rather than treated as unparsed raw API output.
        """
        try:
            response = deps.counterparty_clients.scoring(
                inn=inn,
                model=model,
                include_fincoefs=bool(include_fincoefs),
            )
        except ToolUserCorrectableError:
            raise
        except Exception as exc:
            _raise_unexpected_tool_failure(
                tool_name="counterparty_scoring_tool",
                code="COUNTERPARTY_SCORING_FAILED",
                message=f"Scoring data is unavailable for INN {inn}",
                suggestion="Continue without scoring or retry the same INN later.",
                exc=exc,
                input_field="inn",
            )
        return response.model_dump()

    @tool(
        "counterparty_lookup_tool",
        args_schema=CounterpartyLookupRequest,
        description=(
            "Resolve the official counterparty card by INN via DaData. Use this when you need the "
            "official company name or core registration details before scoring, FSSP, or internet search. "
            "Set include_branches=true only when филиалы should be considered; otherwise the tool searches only the main legal entity."
        ),
    )
    def counterparty_lookup_tool(
        *,
        inn: str,
        include_branches: bool | None = None,
    ) -> dict[str, Any]:
        """Fetch the official counterparty card by INN via DaData.

        Use this tool when the task requires the official company name or basic registration
        details for a supplied INN. This is the preferred source for naming a counterparty before
        broader web search or deeper document retrieval. By default the tool filters to the main
        legal entity (`branch_type=MAIN`). Set `include_branches=true` only when филиалы should
        also be considered.
        """
        try:
            response = deps.counterparty_clients.lookup_party(
                inn=inn,
                include_branches=bool(include_branches),
            )
        except ToolUserCorrectableError:
            raise
        except Exception as exc:
            _raise_unexpected_tool_failure(
                tool_name="counterparty_lookup_tool",
                code="COUNTERPARTY_LOOKUP_FAILED",
                message=f"Counterparty lookup is unavailable for INN {inn}",
                suggestion="Retry the same INN later or continue without official registry data.",
                exc=exc,
                input_field="inn",
            )
        return response.model_dump()

    @tool(
        "counterparty_fssp_tool",
        args_schema=CounterpartyFSSPRequest,
        description=(
            "Fetch normalized grouped FSSP enforcement proceedings for a supplied INN. Use this tool "
            "when you need enforcement history or debt-proceeding context for a counterparty."
        ),
    )
    def counterparty_fssp_tool(
        *,
        inn: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Fetch grouped FSSP enforcement data.

        Use this tool when the task requires enforcement-proceeding context for a company
        identified by INN. It queries the upstream FSSP source and returns grouped records instead
        of raw unstructured output. The tool always requests the grouped Damia response format.

        Args:
            inn: Company INN to analyze.
            from_date: Optional lower bound for retrieval.
            to_date: Optional upper bound for retrieval.

        Returns:
            A dictionary with grouped FSSP proceedings for the supplied INN.

        The response groups proceedings by year, status, subject, and count so it can be used
        directly in downstream analysis or summaries. If the upstream FSSP source does not return
        structured data, the tool returns an empty grouped result with `message="Данные не найдены"`
        instead of raising.
        """
        try:
            response = deps.counterparty_clients.fssp(
                inn=inn,
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as exc:
            logger.info("FSSP data unavailable for inn=%s: %s", inn, exc)
            response = CounterpartyFSSPResponse(
                inn=inn,
                grouped=[],
                raw_format=1,
                message="Данные не найдены",
            )
        return response.model_dump()

    return [
        purchase_search_tool,
        retrieve_page_tool,
        doc_search_tool,
        counterparty_scoring_tool,
        counterparty_fssp_tool,
        yandex_web_search_tool,
        counterparty_lookup_tool,
        read_cached_document_tool,
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
