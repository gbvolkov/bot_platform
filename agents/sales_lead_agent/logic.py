from __future__ import annotations

import base64
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from agents.llm_utils import get_llm
from agents.utils import extract_text

from .business_rules import load_request_rules, load_scoring_rules
from .common import compute_dedup_key, normalize_text
from .config import settings


LOG = logging.getLogger(__name__)

INN_RE = re.compile(r"\b\d{10,12}\b")
LEAD_ID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
PHONE_RE = re.compile(r"(?:(?:\+7|8)[\s\-()]*)?(?:\d[\s\-()]*){10,15}")
AMOUNT_RE = re.compile(r"(\d[\d\s]{3,})(?:[,.]\d+)?\s*(?:руб(?:\.|лей)?|₽|rur|rub)", re.IGNORECASE)
COMPANY_RE = re.compile(r'\b(?:ООО|АО|ПАО|ИП)\s+["«]?[A-Za-zА-Яа-я0-9 .,_-]+["»]?', re.IGNORECASE)
PROCUREMENT_RE = re.compile(r"(?:закупк[аи]|тендер)\s*№?\s*([A-Za-zА-Яа-я0-9\-_/]+)", re.IGNORECASE)
DATE_RANGE_RE = re.compile(r"с\s*(\d{2}\.\d{2}\.\d{4})\s*по\s*(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE)
TOPIC_SPLIT_RE = re.compile(r",|/| и ", re.IGNORECASE)


class RequestUnderstandingRefinement(BaseModel):
    result_type: str | None = None
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    stop_words: list[str] = Field(default_factory=list)
    source_priority: list[str] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)
    priority: str | None = None
    company_name: str | None = None
    only_with_inn: bool | None = None
    only_with_contacts: bool | None = None


def last_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return extract_text(message)
    return ""


def _request_rules():
    return load_request_rules()


def _scoring_rules():
    return load_scoring_rules()


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = normalize_text(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _collect_keywords(text: str) -> list[str]:
    stopwords = {normalize_text(item) for item in _request_rules().stopwords}
    tokens = [normalize_text(token) for token in re.split(r"[,\n/;]", text)]
    keywords: list[str] = []
    for token in tokens:
        for piece in token.split():
            if len(piece) < 3 or piece in stopwords:
                continue
            if piece not in keywords:
                keywords.append(piece)
    return keywords[:12]


def _parse_priority(lowered: str) -> str | None:
    for marker, priority in _request_rules().priority_aliases.items():
        if normalize_text(marker) in lowered:
            return priority
    return None


def _parse_regions(lowered: str) -> list[str]:
    regions: list[str] = []
    for alias, canonical in _request_rules().region_aliases.items():
        if normalize_text(alias) in lowered and canonical not in regions:
            regions.append(canonical)
    return regions


def _split_topic_candidates(raw: str) -> list[str]:
    raw = re.split(r"\b(?:в|за|с|только|приоритет|источник|источники)\b", raw, maxsplit=1, flags=re.IGNORECASE)[0]
    return [normalize_text(piece) for piece in TOPIC_SPLIT_RE.split(raw)]


def _parse_topics(text: str) -> list[str]:
    stopwords = {normalize_text(item) for item in _request_rules().stopwords}
    topics: list[str] = []
    for pattern in _request_rules().topic_patterns:
        try:
            iterator = re.finditer(pattern, text, re.IGNORECASE)
        except re.error:
            continue
        for match in iterator:
            for piece in _split_topic_candidates(match.group(1)):
                if piece and piece not in stopwords and piece not in topics:
                    topics.append(piece)
    return topics[:8]


def _parse_stop_words(text: str) -> list[str]:
    stop_words: list[str] = []
    for pattern in _request_rules().stop_word_patterns:
        try:
            iterator = re.finditer(pattern, text, re.IGNORECASE)
        except re.error:
            continue
        for match in iterator:
            for piece in TOPIC_SPLIT_RE.split(match.group(1)):
                cleaned = normalize_text(piece)
                if cleaned and cleaned not in stop_words:
                    stop_words.append(cleaned)
    return stop_words[:8]


def _parse_feedback_status(lowered: str) -> str | None:
    for marker, status in _request_rules().feedback_aliases.items():
        if normalize_text(marker) in lowered:
            return status
    return None


def _parse_period(query_text: str, lowered: str) -> tuple[str | None, str | None]:
    now = datetime.now(timezone.utc)
    range_match = DATE_RANGE_RE.search(query_text)
    if range_match:
        start = datetime.strptime(range_match.group(1), "%d.%m.%Y").replace(tzinfo=timezone.utc)
        end = datetime.strptime(range_match.group(2), "%d.%m.%Y").replace(tzinfo=timezone.utc) + timedelta(days=1)
        return start.isoformat(), end.isoformat()
    period_aliases = _request_rules().period_aliases
    if any(normalize_text(alias) in lowered for alias in period_aliases.get("week", [])):
        return (now - timedelta(days=7)).isoformat(), now.isoformat()
    if any(normalize_text(alias) in lowered for alias in period_aliases.get("month", [])):
        return (now - timedelta(days=30)).isoformat(), now.isoformat()
    if any(normalize_text(alias) in lowered for alias in period_aliases.get("today", [])):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.isoformat(), now.isoformat()
    if any(normalize_text(alias) in lowered for alias in period_aliases.get("yesterday", [])):
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=1)
        return start.isoformat(), end.isoformat()
    return None, None


def _parse_source_preferences(lowered: str) -> tuple[list[str], list[str]]:
    source_priority: list[str] = []
    required_sources: list[str] = []
    for source_name, markers in _request_rules().source_priority_keywords.items():
        if any(normalize_text(marker) in lowered for marker in markers):
            source_priority.append(source_name)
            if source_name == "procurement":
                required_sources.append(source_name)
    if not source_priority:
        source_priority = ["procurement", "open_source"]
    return source_priority, required_sources


def _parse_company_name(text: str) -> str | None:
    match = COMPANY_RE.search(text)
    return match.group(0).strip() if match else None


def _pick_result_type(lowered: str, *, feedback_status: str | None, procurement_id: str | None, inn: str | None, company_name: str | None) -> str:
    triggers = _request_rules().result_type_triggers
    if feedback_status and any(normalize_text(item) in lowered for item in triggers.get("feedback", [])):
        return "feedback"
    if any(normalize_text(item) in lowered for item in triggers.get("index_lookup", [])):
        return "index_lookup"
    if any(normalize_text(item) in lowered for item in triggers.get("export", [])):
        return "export"
    if any(normalize_text(item) in lowered for item in triggers.get("digest", [])):
        return "digest"
    if any(normalize_text(item) in lowered for item in triggers.get("contacts", [])):
        return "contacts"
    if any(normalize_text(item) in lowered for item in triggers.get("shortlist", [])):
        return "shortlist"
    if procurement_id or "закуп" in lowered or "тендер" in lowered:
        return "procurement_review"
    if inn or "инн" in lowered:
        return "company_check"
    if company_name:
        return "lead_card"
    return "search"


def _finalize_understanding(*, query_text: str, filters: dict[str, Any], result_type: str, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    use_existing_store = result_type in {"export", "digest", "shortlist", "lead_card", "contacts", "company_check", "index_lookup", "feedback"}
    needs_source_collection = result_type not in {"feedback", "index_lookup", "export"} or bool(filters.get("procurement_id") or attachments)
    if attachments:
        needs_source_collection = True
    needs_documents = bool(filters.get("procurement_id") or attachments or result_type in {"procurement_review", "company_check", "lead_card"})
    needs_enrichment = bool(filters.get("inn") or result_type in {"company_check", "lead_card"})
    return {
        "query_text": query_text,
        "result_type": result_type,
        "filters": filters,
        "use_existing_store": use_existing_store,
        "needs_source_collection": needs_source_collection,
        "needs_documents": needs_documents,
        "needs_enrichment": needs_enrichment,
        "needs_export": result_type in {"export", "shortlist", "digest"},
        "needs_index_search": result_type == "index_lookup",
        "needs_feedback": result_type == "feedback",
    }


def parse_request(text: str, metadata: dict[str, Any] | None, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    del metadata
    query_text = text.strip()
    lowered = normalize_text(query_text)
    procurement_match = PROCUREMENT_RE.search(query_text)
    procurement_id = procurement_match.group(1) if procurement_match else None
    inn_match = INN_RE.search(query_text)
    inn = inn_match.group(0) if inn_match else None
    lead_id_match = LEAD_ID_RE.search(query_text)
    lead_id = lead_id_match.group(0) if lead_id_match else None
    company_name = _parse_company_name(query_text)
    feedback_status = _parse_feedback_status(lowered)
    filters = {
        "keywords": _collect_keywords(query_text),
        "regions": _parse_regions(lowered),
        "topics": _parse_topics(query_text),
        "stop_words": _parse_stop_words(query_text),
        "source_priority": [],
        "required_sources": [],
        "inn": inn,
        "company_name": company_name,
        "procurement_id": procurement_id,
        "period_from": None,
        "period_to": None,
        "priority": _parse_priority(lowered),
        "only_with_inn": any(normalize_text(token) in lowered for token in _request_rules().only_with_inn_markers),
        "only_with_contacts": any(normalize_text(token) in lowered for token in _request_rules().only_with_contacts_markers),
        "lead_id": lead_id,
        "feedback_status": feedback_status,
        "feedback_comment": query_text if feedback_status else None,
    }
    filters["period_from"], filters["period_to"] = _parse_period(query_text, lowered)
    filters["source_priority"], filters["required_sources"] = _parse_source_preferences(lowered)
    result_type = _pick_result_type(lowered, feedback_status=feedback_status, procurement_id=procurement_id, inn=inn, company_name=company_name)
    return _finalize_understanding(query_text=query_text, filters=filters, result_type=result_type, attachments=attachments)


def _merge_filter_list(existing: list[str], candidate: list[str]) -> list[str]:
    return _unique([*(existing or []), *(candidate or [])])


def refine_request_understanding(understanding: dict[str, Any], *, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    if not settings.request_understanding_llm_enabled:
        return understanding
    query_text = str(understanding.get("query_text") or "").strip()
    if len(query_text) < 12:
        return understanding
    attachment_names = [str(item.get("filename") or item.get("path") or "") for item in attachments if item]
    prompt = (
        "Ты уточняешь пользовательский запрос для предпродажного sales lead агента.\n"
        "Верни только структурированные поля запроса без выдумывания фактов о компаниях.\n"
        "Выбери тип ответа из: search, procurement_review, company_check, lead_card, contacts, digest, shortlist, export, index_lookup, feedback.\n"
        f"Текст запроса:\n{query_text}\n"
        f"Приложения:\n{attachment_names or ['нет']}\n"
    )
    try:
        llm = get_llm(
            model=settings.request_understanding_llm_model,
            provider=settings.request_understanding_llm_provider,
            temperature=0,
            streaming=False,
            max_tool_calls=1,
        )
        structured = llm.with_structured_output(RequestUnderstandingRefinement)
        refined = structured.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("Semantic request understanding fallback skipped: %s", exc)
        return understanding
    if isinstance(refined, dict):
        refined = RequestUnderstandingRefinement.model_validate(refined)
    if not isinstance(refined, RequestUnderstandingRefinement):
        return understanding

    filters = dict(understanding.get("filters") or {})
    filters["keywords"] = _merge_filter_list(filters.get("keywords") or [], refined.keywords)
    filters["topics"] = _merge_filter_list(filters.get("topics") or [], refined.topics)
    filters["regions"] = _merge_filter_list(filters.get("regions") or [], refined.regions)
    filters["stop_words"] = _merge_filter_list(filters.get("stop_words") or [], refined.stop_words)
    filters["source_priority"] = _merge_filter_list(filters.get("source_priority") or [], refined.source_priority)
    filters["required_sources"] = _merge_filter_list(filters.get("required_sources") or [], refined.required_sources)
    if not filters.get("priority") and refined.priority:
        filters["priority"] = refined.priority
    if not filters.get("company_name") and refined.company_name:
        filters["company_name"] = refined.company_name
    if filters.get("only_with_inn") is not True and refined.only_with_inn is True:
        filters["only_with_inn"] = True
    if filters.get("only_with_contacts") is not True and refined.only_with_contacts is True:
        filters["only_with_contacts"] = True
    if not filters.get("source_priority"):
        filters["source_priority"] = ["procurement", "open_source"]

    result_type = str(understanding.get("result_type") or "search")
    if result_type == "search" and refined.result_type:
        result_type = refined.result_type
    return _finalize_understanding(query_text=query_text, filters=filters, result_type=result_type, attachments=attachments)


def build_task_plan(understanding: dict[str, Any], attachments: list[dict[str, Any]]) -> dict[str, Any]:
    filters = understanding.get("filters") or {}
    result_type = understanding.get("result_type")
    return {
        "use_existing_store": bool(understanding.get("use_existing_store")),
        "collect_sources": bool(understanding.get("needs_source_collection")),
        "resolve_documents": bool(understanding.get("needs_documents") or attachments),
        "extract_facts": not bool(understanding.get("needs_feedback")),
        "persist": result_type not in {"export", "feedback", "index_lookup"},
        "export": bool(understanding.get("needs_export")),
        "enrich": bool(understanding.get("needs_enrichment")),
        "require_index": result_type in {"procurement_review", "lead_card", "company_check", "contacts", "index_lookup"},
        "query_index": bool(understanding.get("needs_index_search")),
        "apply_feedback": bool(understanding.get("needs_feedback")),
        "source_priority": list(filters.get("source_priority") or ["procurement", "open_source"]),
    }


def parse_datetime_filter(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_domain(source_url: str | None) -> str | None:
    host = urlparse(source_url or "").netloc
    return host or None


def extract_contacts(text: str, source_reference: str | None) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for email in EMAIL_RE.findall(text):
        key = (email, None)
        if key in seen:
            continue
        seen.add(key)
        contacts.append(
            {
                "contact_name": None,
                "contact_role": None,
                "contact_email": email,
                "contact_phone": None,
                "contact_source": source_reference,
                "contact_confidence": 0.75,
                "source_reference": source_reference,
            }
        )
    for phone in PHONE_RE.findall(text):
        normalized_phone = re.sub(r"\D+", "", phone)
        if len(normalized_phone) < 10:
            continue
        key = (None, normalized_phone)
        if key in seen:
            continue
        seen.add(key)
        contacts.append(
            {
                "contact_name": None,
                "contact_role": None,
                "contact_email": None,
                "contact_phone": phone.strip(),
                "contact_source": source_reference,
                "contact_confidence": 0.6,
                "source_reference": source_reference,
            }
        )
    return contacts


def _extract_amount(text: str) -> float | None:
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    raw = re.sub(r"\s+", "", match.group(1))
    try:
        return float(raw)
    except ValueError:
        return None


def _merge_contact_lists(*contact_lists: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for contact_list in contact_lists:
        for contact in contact_list:
            key = (contact.get("contact_email"), contact.get("contact_phone"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(contact))
    return merged


def build_leads_from_sources(*, understanding: dict[str, Any], source_hits: list[dict[str, Any]], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lead_cards: list[dict[str, Any]] = []
    docs_by_source: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        source_ref = str(document.get("source_reference") or document.get("document_url") or document.get("stored_path") or "")
        docs_by_source.setdefault(source_ref, []).append(document)

    if not source_hits and documents:
        source_hits = [
            {
                "source_type": "attachment",
                "source_id": None,
                "source_url": document.get("document_url"),
                "title": document.get("file_name"),
                "summary": document.get("extracted_excerpt"),
                "documents": [document],
            }
            for document in documents
        ]

    for source in source_hits:
        source_reference = source.get("source_url") or source.get("source_id") or source.get("title")
        related_docs = list(source.get("documents") or [])
        related_docs.extend(docs_by_source.get(str(source_reference or ""), []))
        doc_text = "\n\n".join(
            str(item.get("text") or item.get("extracted_excerpt") or "").strip()
            for item in related_docs
            if str(item.get("text") or item.get("extracted_excerpt") or "").strip()
        )
        source_text = "\n".join(filter(None, [str(source.get("title") or ""), str(source.get("summary") or ""), doc_text]))
        source_inn_match = INN_RE.search(source_text)
        source_contacts = _merge_contact_lists(source.get("contacts") or [], extract_contacts(source_text, source_reference))
        amount = source.get("amount")
        if amount in {None, ""}:
            amount = _extract_amount(source_text)
        company_name = (
            source.get("company_name")
            or _parse_company_name(source_text)
            or understanding.get("filters", {}).get("company_name")
            or str(source.get("title") or "Unknown company")
        )
        manual_review_required = any(item.get("parse_status") in {"unsupported", "error"} for item in related_docs)
        lead = {
            "company_name": company_name,
            "inn": source.get("inn") or (source_inn_match.group(0) if source_inn_match else understanding.get("filters", {}).get("inn")),
            "ogrn": source.get("ogrn"),
            "region": source.get("region"),
            "website": source.get("website") or _extract_domain(source.get("source_url")),
            "source_type": source.get("source_type"),
            "source_id": source.get("source_id"),
            "source_url": source.get("source_url"),
            "event_type": source.get("event_type") or understanding.get("result_type"),
            "event_title": source.get("title"),
            "event_date": source.get("event_date"),
            "event_summary": source.get("summary") or source_text[:1200],
            "amount": amount,
            "currency": source.get("currency") or ("RUB" if amount else None),
            "object_type": source.get("object_type"),
            "source_reference": source_reference,
            "retrieved_at": source.get("retrieved_at"),
            "confidence": source.get("confidence", 0.7),
            "facts": {
                "company_name": company_name,
                "inn": source.get("inn") or (source_inn_match.group(0) if source_inn_match else None),
                "event_summary": source.get("summary") or source_text[:400],
                "source_domain": _extract_domain(source.get("source_url")),
            },
            "contacts": source_contacts,
            "documents": [
                {
                    "document_url": item.get("document_url"),
                    "file_name": item.get("file_name"),
                    "file_type": item.get("file_type"),
                    "stored_path": item.get("stored_path"),
                    "parse_status": item.get("parse_status"),
                    "index_status": item.get("index_status"),
                    "source_reference": item.get("source_reference"),
                    "confidence": item.get("confidence"),
                    "extracted_excerpt": item.get("extracted_excerpt"),
                    "text": item.get("text"),
                    "segments": [dict(segment) for segment in list(item.get("segments") or [])],
                    "metadata": dict(item.get("metadata") or {}),
                }
                for item in related_docs
            ],
            "sources": [
                {
                    "source_type": source.get("source_type") or "unknown",
                    "source_id": source.get("source_id"),
                    "source_url": source.get("source_url"),
                    "source_reference": source_reference,
                    "confidence": source.get("confidence", 0.7),
                    "is_primary": True,
                    "metadata": {"title": source.get("title")},
                }
            ],
            "enrichments": [],
            "missing_data": [],
            "manual_review_required": manual_review_required,
            "workflow_status": "new",
            "feedback_status": "pending",
            "tags": list(dict.fromkeys([*(source.get("tags") or []), *(understanding.get("filters", {}).get("topics") or [])])),
            "digest_included": False,
        }
        lead["dedup_key"] = compute_dedup_key(lead)
        lead_cards.append(lead)
    return deduplicate_leads(lead_cards)


def deduplicate_leads(lead_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for lead in lead_cards:
        key = str(lead.get("dedup_key") or compute_dedup_key(lead))
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(lead)
            merged[key]["dedup_key"] = key
            continue
        for field in ("inn", "region", "website", "amount", "currency", "event_title", "event_summary"):
            if not existing.get(field) and lead.get(field):
                existing[field] = lead.get(field)
        existing["contacts"] = _merge_contact_lists(existing.get("contacts") or [], lead.get("contacts") or [])
        existing["documents"] = list(existing.get("documents") or []) + list(lead.get("documents") or [])
        existing["sources"] = list(existing.get("sources") or []) + list(lead.get("sources") or [])
        existing["tags"] = list(dict.fromkeys([*(existing.get("tags") or []), *(lead.get("tags") or [])]))
        facts = dict(existing.get("facts") or {})
        facts.update(dict(lead.get("facts") or {}))
        existing["facts"] = facts
        existing["manual_review_required"] = bool(existing.get("manual_review_required") or lead.get("manual_review_required"))
    return list(merged.values())


def apply_enrichment_to_leads(
    lead_cards: list[dict[str, Any]],
    scoring_results: dict[str, dict[str, Any]],
    fssp_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for lead in lead_cards:
        enriched = dict(lead)
        enrichments = list(lead.get("enrichments") or [])
        inn = str(lead.get("inn") or "").strip()
        if inn:
            if inn in scoring_results:
                enrichments.append(scoring_results[inn])
            if inn in fssp_results:
                enrichments.append(fssp_results[inn])
        enriched["enrichments"] = enrichments
        updated.append(enriched)
    return updated


def score_leads(lead_cards: list[dict[str, Any]], keywords: list[str]) -> list[dict[str, Any]]:
    rules = _scoring_rules()
    weights = rules.weights
    thresholds = rules.thresholds
    amount_thresholds = rules.amount_thresholds
    insufficient_data_rules = rules.insufficient_data
    rationale_messages = rules.rationale_messages
    next_steps = rules.next_steps

    scored: list[dict[str, Any]] = []
    normalized_keywords = [normalize_text(item) for item in keywords if normalize_text(item)]
    for lead in lead_cards:
        payload = dict(lead)
        score = 0.0
        rationale: list[str] = []
        summary = normalize_text(f"{lead.get('event_title') or ''} {lead.get('event_summary') or ''}")
        if normalized_keywords and any(keyword in summary for keyword in normalized_keywords):
            score += float(weights.get("keyword_match", 30))
            rationale.append(rationale_messages.get("keyword_match", "событие совпадает с тематикой запроса"))
        if lead.get("amount"):
            amount = float(lead["amount"])
            if amount >= float(amount_thresholds.get("large", 10_000_000)):
                score += float(weights.get("large_amount", 20))
                rationale.append(rationale_messages.get("large_amount", "крупный масштаб события"))
            elif amount >= float(amount_thresholds.get("medium", 1_000_000)):
                score += float(weights.get("medium_amount", 10))
                rationale.append(rationale_messages.get("medium_amount", "есть подтвержденная сумма"))
        if lead.get("inn"):
            score += float(weights.get("identified_inn", 15))
            rationale.append(rationale_messages.get("identified_inn", "компания идентифицирована по ИНН"))
        if lead.get("contacts"):
            score += float(weights.get("contacts_found", 10))
            rationale.append(rationale_messages.get("contacts_found", "найдены публичные контакты"))

        enrichments = list(lead.get("enrichments") or [])
        if any(item.get("status") == "ok" for item in enrichments):
            score += float(weights.get("external_enrichment_ok", 10))
            rationale.append(rationale_messages.get("external_enrichment_ok", "получено внешнее обогащение"))

        risk_flags: list[str] = []
        for enrichment in enrichments:
            payload_data = enrichment.get("payload") or {}
            if bool(payload_data.get("high_risk")) or bool(payload_data.get("manual_review_required")):
                risk_flags.append(enrichment.get("provider") or "external_api")

        missing_data: list[str] = []
        if not lead.get("inn"):
            missing_data.append("inn")
        if not lead.get("contacts"):
            missing_data.append("contacts")
        if not lead.get("event_summary"):
            missing_data.append("event_summary")
        if any(item.get("parse_status") in {"unsupported", "error"} for item in lead.get("documents") or []):
            missing_data.append("document_review")

        manual_review_required = bool(lead.get("manual_review_required")) or bool(risk_flags)
        if risk_flags:
            score -= float(weights.get("risk_penalty", 15))
            rationale.append(rationale_messages.get("risk_penalty", "есть риск-сигналы, требуется ручная проверка"))

        missing_fields_min = float(insufficient_data_rules.get("missing_fields_min", 2))
        score_max = float(insufficient_data_rules.get("score_max", 35))
        if len(missing_data) >= missing_fields_min and score < score_max:
            priority = "insufficient_data"
        elif score >= float(thresholds.get("high", 60)):
            priority = "high"
        elif score >= float(thresholds.get("medium", 35)):
            priority = "medium"
        elif score > 0:
            priority = "low"
        else:
            priority = "insufficient_data"

        next_step = next_steps.get("default", "Связаться по найденным контактам и подтвердить коммерческий интерес.")
        if priority == "insufficient_data":
            next_step = next_steps.get("insufficient_data", "Собрать недостающие идентификаторы и контакты перед передачей в продажи.")
        elif manual_review_required:
            next_step = next_steps.get("manual_review", "Перед передачей в продажи требуется ручная проверка рисков и источников.")

        payload["lead_score"] = score
        payload["lead_priority"] = priority
        payload["rationale"] = "; ".join(rationale) if rationale else "недостаточно подтверждающих факторов"
        payload["missing_data"] = missing_data
        payload["manual_review_required"] = manual_review_required
        payload["facts"] = {**dict(payload.get("facts") or {}), "recommended_next_step": next_step}
        scored.append(payload)
    return scored


def compose_response(
    *,
    understanding: dict[str, Any],
    persisted_leads: list[dict[str, Any]],
    export_record: dict[str, Any] | None,
    summary_export_record: dict[str, Any] | None,
    feedback_result: dict[str, Any] | None,
    index_hits: list[dict[str, Any]],
    errors: list[str],
) -> str:
    if feedback_result:
        saved = feedback_result.get("saved", True)
        if saved:
            lines = [f"Обратная связь сохранена для лида {feedback_result.get('lead_id')}: {feedback_result.get('status')}."]
        else:
            lines = [f"Не удалось сохранить обратную связь: {feedback_result.get('message') or 'лид не найден'}."]
        if feedback_result.get("comment"):
            lines.append(f"Комментарий: {feedback_result.get('comment')}")
        if errors:
            lines.append("Частичные ошибки: " + "; ".join(errors))
        return "\n".join(lines)

    if understanding.get("result_type") == "index_lookup":
        if not index_hits:
            message = "По уже проиндексированным материалам совпадения не найдены."
            if errors:
                message += " Частичные ошибки: " + "; ".join(errors)
            return message
        lines = [f"Найдено совпадений в индексе: {len(index_hits)}."]
        for index, hit in enumerate(index_hits[:10], start=1):
            lines.append(f"{index}. Лид {hit.get('lead_id')} | {hit.get('company_name')} | score={round(float(hit.get('score') or 0), 2)}")
            lines.append(f"   Фрагмент: {hit.get('snippet')}")
            page_number = hit.get("page_number")
            if page_number is not None:
                lines.append(f"   Страница: {page_number}")
            lines.append(f"   Источник: {hit.get('source_reference') or hit.get('source_url') or 'не указан'}")
        if errors:
            lines.append("Частичные ошибки: " + "; ".join(errors))
        return "\n".join(lines)

    if errors and not persisted_leads:
        return "Во время обработки запроса возникли ошибки: " + "; ".join(errors)
    if not persisted_leads:
        return "По текущему запросу подходящие лиды не найдены. Можно сузить или уточнить фильтры, указать ИНН/закупку точнее или запросить выгрузку пустой выборки."

    result_type = understanding.get("result_type")
    if result_type in {"digest", "shortlist", "export", "search"}:
        lines = [f"Найдено лидов: {len(persisted_leads)}."]
        for index, lead in enumerate(persisted_leads[:10], start=1):
            lines.append(
                f"{index}. {lead.get('company_name')} | lead_id: {lead.get('id') or 'n/a'} | "
                f"приоритет: {lead.get('lead_priority')} | "
                f"основание: {lead.get('rationale') or lead.get('event_summary') or 'без пояснения'} | "
                f"источник: {lead.get('source_url') or lead.get('source_reference') or 'не указан'}"
            )
        if export_record:
            lines.append(f"Табличная выгрузка подготовлена: {export_record.get('filename')}.")
        if summary_export_record:
            lines.append(f"Текстовая сводка для продавца подготовлена: {summary_export_record.get('filename')}.")
        if errors:
            lines.append("Частичные ошибки: " + "; ".join(errors))
        return "\n".join(lines)

    focus_lead = persisted_leads[0]
    lines = [
        f"Лид: {focus_lead.get('id') or 'n/a'}",
        f"Компания: {focus_lead.get('company_name')}",
        f"Приоритет: {focus_lead.get('lead_priority')} (score={focus_lead.get('lead_score')})",
        f"Причина: {focus_lead.get('rationale')}",
        f"Событие: {focus_lead.get('event_title') or focus_lead.get('event_summary') or 'не указано'}",
        f"ИНН: {focus_lead.get('inn') or 'не найден'}",
        f"Источник: {focus_lead.get('source_url') or focus_lead.get('source_reference') or 'не указан'}",
    ]
    contacts = focus_lead.get("contacts") or []
    if result_type == "contacts":
        if contacts:
            lines.append("Публичные контакты:")
            for contact in contacts[:10]:
                lines.append(
                    f"- {contact.get('contact_email') or contact.get('contact_phone') or 'контакт без реквизита'} "
                    f"(источник: {contact.get('contact_source') or contact.get('source_reference') or 'не указан'})"
                )
        else:
            lines.append("Публичные контакты не найдены.")
    enrichments = focus_lead.get("enrichments") or []
    if enrichments:
        lines.append("Внешнее обогащение:")
        for enrichment in enrichments:
            lines.append(f"- {enrichment.get('provider')}: {enrichment.get('status')}")
    if focus_lead.get("documents"):
        lines.append("Документы:")
        for document in (focus_lead.get("documents") or [])[:10]:
            metadata = dict(document.get("metadata") or {})
            page_count = metadata.get("page_count")
            page_label = f", pages={page_count}" if page_count else ""
            lines.append(
                f"- {document.get('file_name') or document.get('document_url') or 'без названия'} "
                f"[{document.get('parse_status')}/{document.get('index_status')}{page_label}]"
            )
    if focus_lead.get("missing_data"):
        lines.append("Пробелы в данных: " + ", ".join(focus_lead.get("missing_data") or []))
    if errors:
        lines.append("Частичные ошибки: " + "; ".join(errors))
    return "\n".join(lines)


def build_export_attachment(path: str) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    data = base64.b64encode(file_path.read_bytes()).decode("ascii")
    ext = file_path.suffix.lower()
    if ext == ".csv":
        mime_type = "text/csv"
        file_format = "csv"
    elif ext == ".txt":
        mime_type = "text/plain"
        file_format = "text"
    else:
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        file_format = "excel"
    return {
        "type": "file",
        "format": file_format,
        "filename": file_path.name,
        "mime_type": mime_type,
        "data": data,
    }
