from __future__ import annotations

import asyncio
import hashlib
import html
import importlib.util
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
try:
    import trafilatura
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover - optional dependency
    DDGS = None

from services.kb_manager.utils.loader import load_single_document

from .business_rules import OpenSourceRule, load_open_source_rules
from .common import normalize_text
from .config import settings


LOG = logging.getLogger(__name__)

PURCHASE_SCRAPER_AVAILABLE = importlib.util.find_spec("purchase_scraper") is not None

INN_RE = re.compile(r"\b\d{10,12}\b")
AMOUNT_RE = re.compile(r"(\d[\d\s]{3,})(?:[,.]\d+)?\s*(?:руб(?:\.|лей)?|₽|rur|rub)", re.IGNORECASE)
DATE_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b")
COMPANY_RE = re.compile(r"\b(?:ООО|АО|ПАО|ИП)\s+[\"«]?[A-Za-zА-Яа-я0-9 .,_-]+[\"»]?", re.IGNORECASE)
HREF_RE = re.compile(r"""href=["']([^"'#]+)["']""", re.IGNORECASE)
SUPPORTED_FILE_TYPES = {"html", "htm", "pdf", "doc", "docx", "xls", "xlsx"}
PROCUREMENT_PATH_MARKERS = (
    "/epz/order/notice/",
    "/epz/contract/",
    "common-info.html",
    "view/common-info",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to load sales lead JSON seed '%s': %s", path, exc)
        return None


def _sanitize_filename(value: str, default: str = "document") -> str:
    candidate = re.sub(r"[^A-Za-zА-Яа-я0-9._-]+", "_", value).strip("._")
    return candidate or default


def _normalize_host(host: str | None) -> str:
    return str(host or "").strip().casefold()


def _extract_title(raw_html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
    if match:
        title = html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()
        if title:
            return title
    match = re.search(r"<h1[^>]*>(.*?)</h1>", raw_html, re.IGNORECASE | re.DOTALL)
    if match:
        title = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
        title = re.sub(r"\s+", " ", title).strip()
        if title:
            return title
    return None


def _extract_text_from_html(raw_html: str, url: str | None = None) -> str:
    if trafilatura is not None:
        extracted = trafilatura.extract(
            raw_html,
            url=url,
            include_links=False,
            deduplicate=True,
            favor_precision=True,
        )
        if extracted:
            return extracted.strip()
    fallback = re.sub(r"<[^>]+>", " ", raw_html)
    return re.sub(r"\s+", " ", html.unescape(fallback)).strip()


def _extract_amount(text: str) -> float | None:
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    try:
        return float(re.sub(r"\s+", "", match.group(1)))
    except ValueError:
        return None


def _extract_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    day, month, year = match.group(1).split(".")
    return f"{year}-{month}-{day}"


def _extract_company(text: str) -> str | None:
    match = COMPANY_RE.search(text)
    return match.group(0).strip() if match else None


def _extract_region(text: str) -> str | None:
    patterns = (
        r"(?:регион|субъект РФ|место поставки|место выполнения работ)\s*[:\\-]\s*([^\n.;]{3,120})",
        r"\b(ЦФО|СЗФО|ЮФО|СКФО|ПФО|УФО|СФО|ДФО)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_document_links(raw_html: str, base_url: str) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for href in HREF_RE.findall(raw_html):
        absolute = urljoin(base_url, html.unescape(href))
        if absolute in seen:
            continue
        seen.add(absolute)
        file_type = _guess_file_type(absolute, None)
        if file_type and file_type not in SUPPORTED_FILE_TYPES:
            continue
        if file_type or any(token in absolute.lower() for token in ("download", "attachment", "document")):
            discovered.append(
                {
                    "document_url": absolute,
                    "file_name": Path(urlparse(absolute).path).name or None,
                    "file_type": file_type,
                    "source_reference": absolute,
                }
            )
    return discovered


def _extract_by_patterns(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        try:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        except re.error:
            continue
        if match:
            candidate = next((group for group in match.groups() if isinstance(group, str) and group.strip()), match.group(0))
            normalized = re.sub(r"\s+", " ", html.unescape(candidate)).strip()
            if normalized:
                return normalized
    return None


def _extract_document_links_with_patterns(
    raw_html: str,
    base_url: str,
    patterns: list[str] | None = None,
) -> list[dict[str, Any]]:
    discovered = _extract_document_links(raw_html, base_url)
    if not patterns:
        return discovered
    filtered: list[dict[str, Any]] = []
    for item in discovered:
        url = str(item.get("document_url") or "")
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in patterns):
            filtered.append(item)
    return filtered or discovered


def _guess_file_type(url: str | None, content_type: str | None) -> str | None:
    if content_type:
        lowered = content_type.casefold()
        if "html" in lowered:
            return "html"
        if "pdf" in lowered:
            return "pdf"
        if "word" in lowered or "docx" in lowered:
            return "docx"
        if "msword" in lowered:
            return "doc"
        if "spreadsheet" in lowered or "excel" in lowered:
            return "xlsx"
    suffix = Path(urlparse(url or "").path).suffix.lower().lstrip(".")
    return suffix or None


def _normalize_query_terms(understanding: dict[str, Any]) -> list[str]:
    filters = understanding.get("filters") or {}
    parts = [
        str(filters.get("procurement_id") or ""),
        str(filters.get("company_name") or ""),
        str(filters.get("inn") or ""),
        *(str(item) for item in filters.get("topics") or []),
        *(str(item) for item in filters.get("keywords") or []),
    ]
    return [item for item in dict.fromkeys(part.strip() for part in parts if part and part.strip())]


def _match_record(record: dict[str, Any], understanding: dict[str, Any]) -> bool:
    query_text = normalize_text(understanding.get("query_text"))
    filters = understanding.get("filters") or {}
    keywords = [normalize_text(item) for item in filters.get("keywords") or [] if str(item).strip()]
    topics = [normalize_text(item) for item in filters.get("topics") or [] if str(item).strip()]
    regions = {normalize_text(item) for item in filters.get("regions") or [] if str(item).strip()}
    stop_words = {normalize_text(item) for item in filters.get("stop_words") or [] if str(item).strip()}
    required_sources = {normalize_text(item) for item in filters.get("required_sources") or [] if str(item).strip()}

    haystack = normalize_text(
        " ".join(
            [
                str(record.get("title") or ""),
                str(record.get("summary") or ""),
                str(record.get("company_name") or ""),
                str(record.get("event_type") or ""),
                str(record.get("region") or ""),
            ]
        )
    )
    if any(word and word in haystack for word in stop_words):
        return False
    if keywords and not any(word and word in haystack for word in keywords):
        return False
    if topics and not any(word and word in haystack for word in topics):
        return False
    if regions and normalize_text(record.get("region")) not in regions:
        return False
    if required_sources and normalize_text(record.get("source_type")) not in required_sources:
        return False

    inn = str(filters.get("inn") or "").strip()
    if inn and str(record.get("inn") or "").strip() != inn:
        return False

    procurement_id = str(filters.get("procurement_id") or "").strip()
    if procurement_id:
        record_id = str(record.get("source_id") or "")
        record_url = str(record.get("source_url") or "")
        if procurement_id not in record_id and procurement_id not in record_url:
            return False

    if query_text and not keywords and not topics:
        return any(token in haystack for token in query_text.split() if token)
    return True


def _coerce_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    return {"value": value}


def _build_http_client(timeout_seconds: float | None = None) -> httpx.AsyncClient:
    timeout = timeout_seconds if timeout_seconds is not None else settings.http_timeout_seconds
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        verify=settings.verify_ssl,
        headers={"User-Agent": settings.user_agent},
    )


class SeedBackedSourceCatalog:
    def __init__(self) -> None:
        seed_path = Path(settings.source_seed_path)
        seed_payload = _safe_load_json(seed_path) or {}
        if isinstance(seed_payload, list):
            self.sources = [item for item in seed_payload if isinstance(item, dict)]
            self.enrichments = {}
        else:
            self.sources = [item for item in seed_payload.get("sources", []) if isinstance(item, dict)]
            self.enrichments = dict(seed_payload.get("enrichments") or {})
        self.open_source_rules = [item for item in load_open_source_rules().rules if str(item.host).strip()]

    def filter_sources(self, source_types: Iterable[str]) -> list[dict[str, Any]]:
        allowed = {item.casefold() for item in source_types}
        return [
            dict(item)
            for item in self.sources
            if str(item.get("source_type") or "").casefold() in allowed
        ]

    def get_enrichment(self, provider: str, inn: str | None) -> dict[str, Any] | None:
        if not inn:
            return None
        provider_payload = self.enrichments.get(provider) or {}
        if isinstance(provider_payload, dict):
            value = provider_payload.get(inn)
            if isinstance(value, dict):
                return dict(value)
        return None

    def iter_whitelist_hosts(self) -> list[str]:
        return [rule.host.strip() for rule in self.open_source_rules if rule.host.strip()]

    def host_allowed(self, url: str | None) -> bool:
        if not self.open_source_rules:
            return True
        host = _normalize_host(urlparse(url or "").netloc)
        return any(host == _normalize_host(rule.host) for rule in self.open_source_rules)

    def iter_open_source_rules(self) -> list[OpenSourceRule]:
        return [
            rule
            for rule in self.open_source_rules
            if normalize_text(rule.source_type) != "procurement"
        ]

    def match_open_source_rule(self, url: str | None) -> OpenSourceRule | None:
        if not url:
            return None
        parsed = urlparse(url)
        host = _normalize_host(parsed.netloc)
        for rule in self.iter_open_source_rules():
            if host != _normalize_host(rule.host):
                continue
            if self.page_allowed(url, rule):
                return rule
        return None

    def page_allowed(self, url: str | None, rule: OpenSourceRule) -> bool:
        parsed = urlparse(url or "")
        path = parsed.path or "/"
        if rule.path_allow_patterns and not any(re.search(pattern, path, re.IGNORECASE) for pattern in rule.path_allow_patterns):
            return False
        if rule.path_deny_patterns and any(re.search(pattern, path, re.IGNORECASE) for pattern in rule.path_deny_patterns):
            return False
        return True

    def page_content_allowed(self, text: str, rule: OpenSourceRule) -> bool:
        normalized = normalize_text(text)
        if rule.required_keywords and not any(normalize_text(keyword) in normalized for keyword in rule.required_keywords):
            return False
        if rule.blocked_keywords and any(normalize_text(keyword) in normalized for keyword in rule.blocked_keywords):
            return False
        return True

    def build_open_source_queries(self, base_query: str, understanding: dict[str, Any]) -> list[str]:
        filters = understanding.get("filters") or {}
        company_name = str(filters.get("company_name") or "").strip()
        inn = str(filters.get("inn") or "").strip()
        queries: list[str] = []
        for rule in self.iter_open_source_rules():
            templates = rule.query_templates or ["site:{host} {query}"]
            for template in templates:
                query = template.format(
                    host=rule.host,
                    query=base_query,
                    company=company_name or base_query,
                    inn=inn,
                ).strip()
                if query and query not in queries:
                    queries.append(query)
        if not queries and not self.iter_open_source_rules():
            return []
        if not queries:
            queries.append(base_query)
        return queries


class ProcurementSourceAdapter:
    def __init__(self, catalog: SeedBackedSourceCatalog) -> None:
        self._catalog = catalog

    async def search(self, understanding: dict[str, Any]) -> list[dict[str, Any]]:
        live_results: list[dict[str, Any]] = []
        try:
            live_results = await self._search_live(understanding)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Procurement live adapter failed: %s", exc)
        if live_results or not settings.allow_seed_fallback:
            return live_results
        candidates = self._catalog.filter_sources({"procurement", "zakupki", "zakupki.gov.ru"})
        return [item for item in candidates if _match_record(item, understanding)]

    async def _search_live(self, understanding: dict[str, Any]) -> list[dict[str, Any]]:
        procurement_url = self._extract_direct_url(understanding)
        if procurement_url:
            urls = [procurement_url]
        else:
            urls = await self._search_procurement_urls(understanding)
        if not urls:
            return []

        async with _build_http_client() as client:
            cards = await asyncio.gather(
                *(self._fetch_procurement_card(client, url, understanding) for url in urls[: settings.procurement_results_limit]),
                return_exceptions=True,
            )
        results: list[dict[str, Any]] = []
        for card in cards:
            if isinstance(card, Exception):
                LOG.warning("Procurement card fetch failed: %s", card)
                continue
            if card and _match_record(card, understanding):
                results.append(card)
        return results

    def _extract_direct_url(self, understanding: dict[str, Any]) -> str | None:
        query_text = str(understanding.get("query_text") or "")
        match = re.search(r"https?://[^\s]+", query_text)
        if not match:
            return None
        candidate = match.group(0)
        if "zakupki.gov.ru" in _normalize_host(urlparse(candidate).netloc):
            return candidate
        return None

    async def _search_procurement_urls(self, understanding: dict[str, Any]) -> list[str]:
        query_terms = _normalize_query_terms(understanding)
        filters = understanding.get("filters") or {}
        procurement_id = str(filters.get("procurement_id") or "").strip()
        if procurement_id and procurement_id not in query_terms:
            query_terms.insert(0, procurement_id)
        search_string = " ".join(query_terms).strip() or str(understanding.get("query_text") or "").strip()
        if not search_string:
            return []

        base_url = settings.procurement_base_url.rstrip("/")
        search_url = f"{base_url}{settings.procurement_search_path}"
        discovered: list[str] = []
        seen: set[str] = set()

        async with _build_http_client() as client:
            for page_number in range(1, settings.procurement_page_depth + 1):
                params = {
                    "searchString": search_string,
                    "morphology": "on",
                    "pageNumber": str(page_number),
                    "recordsPerPage": str(settings.procurement_results_limit),
                    "sortDirection": "false",
                    "sortBy": settings.procurement_search_sort_by,
                    "showLotsInfoHidden": "false",
                    "fz44": "on",
                    "af": "on",
                }
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                for url in self._extract_procurement_links(response.text, base_url):
                    if procurement_id and procurement_id not in url:
                        continue
                    if url not in seen:
                        seen.add(url)
                        discovered.append(url)
                if len(discovered) >= settings.procurement_results_limit:
                    break
        return discovered[: settings.procurement_results_limit]

    def _extract_procurement_links(self, raw_html: str, base_url: str) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for href in HREF_RE.findall(raw_html):
            absolute = urljoin(base_url, html.unescape(href))
            parsed = urlparse(absolute)
            if "zakupki.gov.ru" not in _normalize_host(parsed.netloc):
                continue
            if not any(marker in parsed.path.lower() for marker in PROCUREMENT_PATH_MARKERS):
                continue
            normalized = absolute.split("#", 1)[0]
            if normalized not in seen:
                seen.add(normalized)
                urls.append(normalized)
        return urls

    async def _fetch_procurement_card(
        self,
        client: httpx.AsyncClient,
        url: str,
        understanding: dict[str, Any],
    ) -> dict[str, Any] | None:
        response = await client.get(url)
        response.raise_for_status()
        raw_html = response.text
        extracted_text = _extract_text_from_html(raw_html, url)
        source_id = self._extract_procurement_id(url, extracted_text, understanding)
        title = _extract_title(raw_html) or self._first_nonempty_line(extracted_text) or source_id or "Карточка закупки"
        company_name = _extract_company(extracted_text) or understanding.get("filters", {}).get("company_name")
        inn_match = INN_RE.search(extracted_text)
        documents = _extract_document_links(raw_html, url)
        return {
            "source_type": "procurement",
            "source_id": source_id,
            "source_url": url,
            "title": title,
            "summary": extracted_text[:4000],
            "company_name": company_name,
            "inn": inn_match.group(0) if inn_match else understanding.get("filters", {}).get("inn"),
            "region": _extract_region(extracted_text),
            "event_type": "procurement",
            "event_date": _extract_date(extracted_text),
            "amount": _extract_amount(extracted_text),
            "currency": "RUB",
            "object_type": "procurement",
            "documents": documents,
            "retrieved_at": _utc_now_iso(),
            "confidence": 0.82,
        }

    def _extract_procurement_id(
        self,
        url: str,
        extracted_text: str,
        understanding: dict[str, Any],
    ) -> str | None:
        parsed = urlparse(url)
        for key in ("regNumber", "purchaseNumber", "notificationNumber"):
            values = parse_qs(parsed.query).get(key)
            if values:
                return values[0]
        explicit = str(understanding.get("filters", {}).get("procurement_id") or "").strip()
        if explicit:
            return explicit
        match = re.search(r"(?:№|N)\s*([A-Za-zА-Яа-я0-9/_-]{6,})", extracted_text)
        return match.group(1) if match else None

    def _first_nonempty_line(self, text: str) -> str | None:
        for line in text.splitlines():
            normalized = line.strip()
            if normalized:
                return normalized[:300]
        return None


class OpenSourceAdapter:
    def __init__(self, catalog: SeedBackedSourceCatalog) -> None:
        self._catalog = catalog

    async def search(self, understanding: dict[str, Any]) -> list[dict[str, Any]]:
        live_results: list[dict[str, Any]] = []
        try:
            live_results = await self._search_live(understanding)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Open-source live adapter failed: %s", exc)
        if live_results or not settings.allow_seed_fallback:
            return live_results
        candidates = self._catalog.filter_sources({"open_source", "news", "website"})
        filtered = [item for item in candidates if self._catalog.host_allowed(item.get("source_url"))]
        return [item for item in filtered if _match_record(item, understanding)]

    async def _search_live(self, understanding: dict[str, Any]) -> list[dict[str, Any]]:
        base_query = " ".join(_normalize_query_terms(understanding)).strip() or str(understanding.get("query_text") or "").strip()
        if not base_query:
            return []
        queries = self._catalog.build_open_source_queries(base_query, understanding)
        if not queries:
            return []
        raw_results = await asyncio.to_thread(self._run_ddg_queries, queries)
        if not raw_results:
            return []

        limited_results = raw_results[: settings.open_search_max_pages]
        async with _build_http_client() as client:
            pages = await asyncio.gather(
                *(self._fetch_open_page(client, item) for item in limited_results),
                return_exceptions=True,
            )

        results: list[dict[str, Any]] = []
        for page in pages:
            if isinstance(page, Exception):
                LOG.warning("Open-source page fetch failed: %s", page)
                continue
            if page and _match_record(page, understanding):
                results.append(page)
        return results

    def _run_ddg_queries(self, queries: list[str]) -> list[dict[str, Any]]:
        if DDGS is None:
            raise RuntimeError("duckduckgo_search is not installed")
        aggregated: list[dict[str, Any]] = []
        seen: set[str] = set()
        with DDGS() as ddgs:
            for query in queries:
                for item in ddgs.text(query, region=settings.open_search_region, max_results=settings.open_search_results_limit):
                    if not isinstance(item, dict):
                        continue
                    href = str(item.get("href") or item.get("url") or "").strip()
                    if not href or href in seen:
                        continue
                    if self._catalog.host_allowed(href):
                        seen.add(href)
                        aggregated.append(dict(item))
        return aggregated

    async def _fetch_open_page(
        self,
        client: httpx.AsyncClient,
        item: dict[str, Any],
    ) -> dict[str, Any] | None:
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url or not self._catalog.host_allowed(url):
            return None
        rule = self._catalog.match_open_source_rule(url)
        if self._catalog.iter_open_source_rules() and rule is None:
            return None
        title = str(item.get("title") or "").strip() or None
        body = str(item.get("body") or item.get("snippet") or "").strip()
        raw_html = ""
        try:
            response = await client.get(url)
            response.raise_for_status()
            raw_html = response.text
            extracted_text = _extract_text_from_html(raw_html, url) or body
            if rule and not self._catalog.page_content_allowed(extracted_text or body, rule):
                return None
            documents = _extract_document_links_with_patterns(raw_html, url, rule.document_url_patterns if rule else None)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Open page fetch degraded for %s: %s", url, exc)
            extracted_text = body
            documents = []

        if not extracted_text and not body:
            return None
        source_text = extracted_text or body
        if rule and not self._catalog.page_content_allowed(source_text, rule):
            return None
        company_name = (
            _extract_by_patterns(raw_html or source_text, rule.company_patterns) if rule and rule.company_patterns else None
        ) or _extract_company(source_text)
        inn_match = INN_RE.search(source_text)
        extracted_title = (
            _extract_by_patterns(raw_html or source_text, rule.title_patterns) if rule and rule.title_patterns else None
        ) or title
        extracted_summary = (
            _extract_by_patterns(raw_html or source_text, rule.summary_patterns) if rule and rule.summary_patterns else None
        ) or source_text[:4000]
        source_type = rule.source_type if rule else "open_source"
        event_type = rule.event_type if rule else "open_source"
        return {
            "source_type": source_type,
            "source_id": hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
            "source_url": url,
            "title": extracted_title or url,
            "summary": extracted_summary,
            "company_name": company_name,
            "inn": inn_match.group(0) if inn_match else None,
            "region": _extract_region(source_text),
            "event_type": event_type,
            "event_date": _extract_date(source_text),
            "amount": _extract_amount(source_text),
            "currency": "RUB",
            "object_type": "web_page",
            "documents": documents,
            "retrieved_at": _utc_now_iso(),
            "confidence": 0.68,
            "tags": list(rule.tags) if rule else [],
            "metadata": {"matched_host_rule": rule.host if rule else None},
        }


class DocumentPipelineAdapter:
    async def resolve(
        self,
        *,
        source_hits: list[dict[str, Any]],
        attachments: list[dict[str, Any]],
        require_index: bool,
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for hit in source_hits:
            for raw_document in hit.get("documents") or []:
                documents.append(await self._resolve_one(raw_document, require_index=require_index))
        for attachment in attachments:
            documents.append(
                await self._resolve_one(
                    {
                        "file_name": attachment.get("filename"),
                        "file_type": Path(str(attachment.get("path") or attachment.get("filename") or "")).suffix.lower().lstrip("."),
                        "stored_path": attachment.get("path"),
                        "document_url": attachment.get("path"),
                    },
                    require_index=require_index,
                )
            )
        return documents

    def _single_segment(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        normalized = str(text or "").strip()
        if not normalized:
            return []
        return [
            {
                "page_number": int((metadata or {}).get("page_number") or 1),
                "position_start": 0,
                "position_end": len(normalized),
                "text": normalized,
                "metadata": dict(metadata or {}),
            }
        ]

    def _segments_from_loaded_docs(self, docs: list[Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        running_offset = 0
        for index, item in enumerate(docs, start=1):
            text = str(getattr(item, "page_content", "")).strip()
            if not text:
                continue
            raw_metadata = dict(getattr(item, "metadata", {}) or {})
            page_number = raw_metadata.get("page_number") or raw_metadata.get("page") or index
            position_start = raw_metadata.get("position_start")
            position_end = raw_metadata.get("position_end")
            if position_start is None:
                position_start = running_offset
            if position_end is None:
                position_end = position_start + len(text)
            running_offset = max(running_offset, int(position_end)) + 2
            segments.append(
                {
                    "page_number": int(page_number) if str(page_number).isdigit() else index,
                    "position_start": int(position_start),
                    "position_end": int(position_end),
                    "text": text,
                    "metadata": raw_metadata,
                }
            )
        return segments

    async def _resolve_one(self, raw_document: dict[str, Any], *, require_index: bool) -> dict[str, Any]:
        stored_path = raw_document.get("stored_path") or raw_document.get("path")
        document_url = raw_document.get("document_url") or raw_document.get("url") or stored_path
        inline_text = raw_document.get("text")
        file_name = raw_document.get("file_name") or raw_document.get("filename")
        file_type = str(raw_document.get("file_type") or Path(str(file_name or stored_path or document_url or "")).suffix.lower().lstrip("."))
        document: dict[str, Any] = {
            "document_url": document_url,
            "file_name": file_name,
            "file_type": file_type,
            "stored_path": stored_path,
            "parse_status": "pending",
            "index_status": "pending" if require_index else "not_requested",
            "source_reference": raw_document.get("source_reference") or document_url,
            "confidence": raw_document.get("confidence", 0.7),
            "metadata": dict(raw_document.get("metadata") or {}),
            "text": None,
            "segments": [],
        }
        if inline_text:
            text = str(inline_text).strip()
            segments = self._single_segment(text, document["metadata"])
            document["parse_status"] = "parsed"
            document["index_status"] = "ready" if require_index else "not_requested"
            document["extracted_excerpt"] = text[:1200]
            document["text"] = text
            document["segments"] = segments
            document["metadata"] = {**document["metadata"], "page_count": len(segments), "segment_count": len(segments)}
            return document

        if not stored_path and document_url and str(document_url).startswith(("http://", "https://")):
            downloaded = await self._download_remote_document(document_url, file_name=file_name)
            stored_path = downloaded["stored_path"]
            file_name = downloaded["file_name"]
            file_type = downloaded["file_type"]
            document["stored_path"] = stored_path
            document["file_name"] = file_name
            document["file_type"] = file_type
            document["metadata"] = {**document["metadata"], **dict(downloaded.get("metadata") or {})}
            if downloaded.get("inline_text"):
                text = str(downloaded["inline_text"]).strip()
                segments = self._single_segment(text, document["metadata"])
                document["parse_status"] = "parsed" if text else "empty"
                document["index_status"] = "ready" if require_index and text else "not_requested"
                document["extracted_excerpt"] = text[:1200] if text else None
                document["text"] = text
                document["segments"] = segments
                document["metadata"] = {**document["metadata"], "page_count": len(segments), "segment_count": len(segments)}
                return document

        if not stored_path:
            document["parse_status"] = "missing"
            return document

        path = Path(str(stored_path))
        if not path.exists():
            document["parse_status"] = "missing"
            return document

        effective_type = str(file_type or path.suffix.lower().lstrip(".")).casefold()
        if effective_type and effective_type not in SUPPORTED_FILE_TYPES:
            document["parse_status"] = "unsupported"
            document["index_status"] = "not_requested"
            return document

        try:
            docs = load_single_document(str(path))
        except NotImplementedError:
            document["parse_status"] = "unsupported"
            document["index_status"] = "not_requested"
            return document
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Sales lead failed to parse %s: %s", path, exc)
            document["parse_status"] = "error"
            document["metadata"] = {**document["metadata"], "error": str(exc)}
            return document

        segments = self._segments_from_loaded_docs(list(docs))
        text = "\n\n".join(str(item.get("text") or "").strip() for item in segments if str(item.get("text") or "").strip())
        document["parse_status"] = "parsed" if text else "empty"
        document["index_status"] = "ready" if require_index and text else "not_requested"
        document["extracted_excerpt"] = text[:1200] if text else None
        document["text"] = text
        document["segments"] = segments
        document["metadata"] = {
            **document["metadata"],
            "page_count": len({int(item.get('page_number') or 0) for item in segments if item.get("page_number") is not None}),
            "segment_count": len(segments),
        }
        return document

    async def _download_remote_document(self, url: str, *, file_name: str | None) -> dict[str, Any]:
        async with _build_http_client() as client:
            response = await client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type")
        file_type = _guess_file_type(url, content_type)
        final_name = file_name or Path(urlparse(url).path).name or f"document.{file_type or 'bin'}"
        final_name = _sanitize_filename(final_name, "document")

        storage_dir = Path(settings.document_store_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        stored_path = storage_dir / f"{digest}_{final_name}"
        stored_path.write_bytes(response.content)

        metadata = {
            "content_type": content_type,
            "downloaded_at": _utc_now_iso(),
            "download_url": url,
        }
        return {
            "stored_path": str(stored_path),
            "file_name": final_name,
            "file_type": file_type or stored_path.suffix.lower().lstrip("."),
            "inline_text": _extract_text_from_html(response.text, url) if file_type == "html" else None,
            "metadata": metadata,
        }


class _HttpEnrichmentAdapter:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str | None,
        api_key: str | None,
        method: str,
        inn_param: str,
        auth_header: str,
        timeout_seconds: float,
        catalog: SeedBackedSourceCatalog,
    ) -> None:
        self._provider = provider
        self._base_url = base_url
        self._api_key = api_key
        self._method = method.upper()
        self._inn_param = inn_param
        self._auth_header = auth_header
        self._timeout_seconds = timeout_seconds
        self._catalog = catalog

    async def enrich(self, inn: str | None) -> dict[str, Any]:
        if not inn:
            return {
                "provider": self._provider,
                "status": "skipped",
                "payload": {"reason": "INN not available."},
            }
        if not self._base_url:
            fallback = self._catalog.get_enrichment(self._provider.removeprefix("api_"), inn) if settings.allow_seed_fallback else None
            if fallback is not None:
                return {
                    "provider": self._provider,
                    "status": "seed_fallback",
                    "payload": fallback,
                }
            return {
                "provider": self._provider,
                "status": "not_configured",
                "payload": {"reason": "Integration endpoint is not configured."},
            }

        headers = {"User-Agent": settings.user_agent}
        if self._api_key:
            headers[self._auth_header] = self._api_key
        payload = {self._inn_param: inn}

        try:
            async with _build_http_client(self._timeout_seconds) as client:
                if self._method == "POST":
                    response = await client.post(self._base_url, json=payload, headers=headers)
                else:
                    response = await client.get(self._base_url, params=payload, headers=headers)
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s request failed for INN %s: %s", self._provider, inn, exc)
            return {
                "provider": self._provider,
                "status": "error",
                "payload": {"reason": str(exc)},
                "source_reference": self._base_url,
            }

        try:
            response_payload = response.json()
        except Exception:  # noqa: BLE001
            response_payload = {"raw_text": response.text[:4000]}
        return {
            "provider": self._provider,
            "status": "ok",
            "payload": _coerce_payload(response_payload),
            "source_reference": self._base_url,
            "retrieved_at": _utc_now_iso(),
            "confidence": 0.9,
        }


class ScoringApiAdapter(_HttpEnrichmentAdapter):
    def __init__(self, catalog: SeedBackedSourceCatalog) -> None:
        super().__init__(
            provider="api_scoring",
            base_url=settings.scoring_base_url,
            api_key=settings.scoring_api_key,
            method=settings.scoring_method,
            inn_param=settings.scoring_inn_param,
            auth_header=settings.scoring_auth_header,
            timeout_seconds=settings.scoring_timeout_seconds,
            catalog=catalog,
        )


class FsspApiAdapter(_HttpEnrichmentAdapter):
    def __init__(self, catalog: SeedBackedSourceCatalog) -> None:
        super().__init__(
            provider="api_fssp",
            base_url=settings.fssp_base_url,
            api_key=settings.fssp_api_key,
            method=settings.fssp_method,
            inn_param=settings.fssp_inn_param,
            auth_header=settings.fssp_auth_header,
            timeout_seconds=settings.fssp_timeout_seconds,
            catalog=catalog,
        )


class AdapterBundle:
    def __init__(self) -> None:
        catalog = SeedBackedSourceCatalog()
        self.procurement = ProcurementSourceAdapter(catalog)
        self.open_source = OpenSourceAdapter(catalog)
        self.documents = DocumentPipelineAdapter()
        self.scoring = ScoringApiAdapter(catalog)
        self.fssp = FsspApiAdapter(catalog)
