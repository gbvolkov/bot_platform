from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from rank_bm25 import BM25Okapi
from sqlalchemy import Select, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .common import compute_dedup_key, normalize_text
from .config import settings
from .models import (
    Lead,
    LeadContact,
    LeadDocument,
    LeadEnrichment,
    LeadExport,
    LeadFact,
    LeadFeedback,
    LeadIndexChunk,
    LeadSource,
)
from .schemas import (
    LeadContactView,
    LeadDocumentView,
    LeadEnrichmentView,
    LeadExportRequest,
    LeadExportView,
    LeadFactView,
    LeadFeedbackCreate,
    LeadFeedbackView,
    LeadIndexHitView,
    LeadSourceView,
    LeadView,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _chunk_text(text: str) -> list[dict[str, Any]]:
    normalized = str(text or "").strip()
    if not normalized:
        return []

    chunk_size = settings.index_chunk_size
    overlap = min(settings.index_chunk_overlap, max(chunk_size - 50, 0))
    step = max(chunk_size - overlap, 1)
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "normalized_text": normalize_text(chunk_text),
                    "position_start": start,
                    "position_end": end,
                }
            )
            chunk_index += 1
        if end >= len(normalized):
            break
        start += step
    return chunks


def _extract_chunk_entities(text: str) -> dict[str, Any]:
    inns = sorted(set(re.findall(r"\b\d{10,12}\b", text)))
    emails = sorted(set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", text)))
    phones = sorted(set(re.findall(r"(?:(?:\+7|8)[\s\-()]*)?(?:\d[\s\-()]*){10,15}", text)))
    return {
        "inns": inns,
        "emails": emails,
        "phones": phones,
    }


def _serialize_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return {"items": value}
    return {"value": value}


def _feedback_to_view(feedback: LeadFeedback) -> LeadFeedbackView:
    return LeadFeedbackView(
        id=feedback.id,
        lead_id=feedback.lead_id,
        user_id=feedback.user_id,
        status=feedback.status,
        comment=feedback.comment,
        created_at=feedback.created_at,
    )


def _source_to_view(source: LeadSource) -> LeadSourceView:
    return LeadSourceView(
        id=source.id,
        source_type=source.source_type,
        source_id=source.source_id,
        source_url=source.source_url,
        source_reference=source.source_reference,
        retrieved_at=source.retrieved_at,
        confidence=source.confidence,
        is_primary=source.is_primary,
        metadata=dict(source.metadata_json or {}),
    )


def _document_to_view(document: LeadDocument) -> LeadDocumentView:
    return LeadDocumentView(
        id=document.id,
        document_url=document.document_url,
        file_name=document.file_name,
        file_type=document.file_type,
        stored_path=document.stored_path,
        parse_status=document.parse_status,
        index_status=document.index_status,
        source_reference=document.source_reference,
        retrieved_at=document.retrieved_at,
        confidence=document.confidence,
        extracted_excerpt=document.extracted_excerpt,
        metadata=dict(document.metadata_json or {}),
    )


def _fact_to_view(fact: LeadFact) -> LeadFactView:
    return LeadFactView(
        id=fact.id,
        field_name=fact.field_name,
        value=dict(fact.value_json or {}),
        source_reference=fact.source_reference,
        retrieved_at=fact.retrieved_at,
        confidence=fact.confidence,
    )


def _contact_to_view(contact: LeadContact) -> LeadContactView:
    return LeadContactView(
        id=contact.id,
        contact_name=contact.contact_name,
        contact_role=contact.contact_role,
        contact_email=contact.contact_email,
        contact_phone=contact.contact_phone,
        contact_source=contact.contact_source,
        contact_confidence=contact.contact_confidence,
        source_reference=contact.source_reference,
        retrieved_at=contact.retrieved_at,
        metadata=dict(contact.metadata_json or {}),
    )


def _enrichment_to_view(enrichment: LeadEnrichment) -> LeadEnrichmentView:
    return LeadEnrichmentView(
        id=enrichment.id,
        provider=enrichment.provider,
        status=enrichment.status,
        payload=dict(enrichment.payload_json or {}),
        source_reference=enrichment.source_reference,
        retrieved_at=enrichment.retrieved_at,
        confidence=enrichment.confidence,
    )


def _index_hit_to_view(
    *,
    chunk: LeadIndexChunk,
    lead: Lead,
    document: LeadDocument | None,
    score: float,
) -> LeadIndexHitView:
    snippet = chunk.text[:500]
    return LeadIndexHitView(
        lead_id=lead.id,
        company_name=lead.company_name,
        lead_priority=lead.lead_priority,
        score=float(score),
        snippet=snippet,
        source_reference=chunk.source_reference or lead.source_reference,
        source_url=lead.source_url,
        document_id=document.id if document is not None else None,
        document_name=document.file_name if document is not None else None,
        chunk_index=chunk.chunk_index,
        page_number=chunk.page_number,
        position_start=chunk.position_start,
        position_end=chunk.position_end,
        metadata=dict(chunk.metadata_json or {}),
    )


def _lead_to_view(lead: Lead) -> LeadView:
    return LeadView(
        id=lead.id,
        created_by_user_id=lead.created_by_user_id,
        dedup_key=lead.dedup_key,
        source_type=lead.source_type,
        source_id=lead.source_id,
        source_url=lead.source_url,
        company_name=lead.company_name,
        inn=lead.inn,
        ogrn=lead.ogrn,
        region=lead.region,
        website=lead.website,
        event_type=lead.event_type,
        event_title=lead.event_title,
        event_date=lead.event_date,
        event_summary=lead.event_summary,
        amount=lead.amount,
        currency=lead.currency,
        object_type=lead.object_type,
        lead_priority=lead.lead_priority,
        lead_score=lead.lead_score,
        rationale=lead.rationale,
        missing_data=list(lead.missing_data or []),
        manual_review_required=lead.manual_review_required,
        workflow_status=lead.workflow_status,
        feedback_status=lead.feedback_status,
        tags=list(lead.tags or []),
        digest_included=lead.digest_included,
        source_reference=lead.source_reference,
        retrieved_at=lead.retrieved_at,
        confidence=lead.confidence,
        metadata=dict(lead.metadata_json or {}),
        created_at=lead.created_at,
        updated_at=lead.updated_at,
        sources=[_source_to_view(item) for item in list(lead.sources or [])],
        documents=[_document_to_view(item) for item in list(lead.documents or [])],
        facts=[_fact_to_view(item) for item in list(lead.facts or [])],
        contacts=[_contact_to_view(item) for item in list(lead.contacts or [])],
        enrichments=[_enrichment_to_view(item) for item in list(lead.enrichments or [])],
    )


def _export_to_view(export: LeadExport) -> LeadExportView:
    return LeadExportView(
        id=export.id,
        requested_by=export.requested_by,
        format=export.format,
        filename=export.filename,
        path=export.path,
        filters=dict(export.filters_json or {}),
        row_count=export.row_count,
        created_at=export.created_at,
    )


def _lead_query() -> Select[tuple[Lead]]:
    return (
        select(Lead)
        .options(
            selectinload(Lead.sources),
            selectinload(Lead.documents),
            selectinload(Lead.facts),
            selectinload(Lead.contacts),
            selectinload(Lead.enrichments),
            selectinload(Lead.feedback_entries),
            selectinload(Lead.index_chunks),
        )
        .order_by(Lead.updated_at.desc())
    )


def _apply_lead_filters(
    stmt: Select[tuple[Lead]],
    *,
    user_id: str,
    period_from: Optional[datetime] = None,
    period_to: Optional[datetime] = None,
    region: Optional[str] = None,
    priority: Optional[str] = None,
    source_type: Optional[str] = None,
    inn: Optional[str] = None,
    company_name: Optional[str] = None,
    procurement_id: Optional[str] = None,
    query_text: Optional[str] = None,
    only_with_inn: bool = False,
    only_with_contacts: bool = False,
) -> Select[tuple[Lead]]:
    conditions = [or_(Lead.created_by_user_id == user_id, Lead.created_by_user_id.is_(None))]
    if period_from is not None:
        conditions.append(Lead.created_at >= period_from)
    if period_to is not None:
        conditions.append(Lead.created_at <= period_to)
    if region:
        conditions.append(Lead.region == region)
    if priority:
        conditions.append(Lead.lead_priority == priority)
    if source_type:
        conditions.append(Lead.source_type == source_type)
    if inn:
        conditions.append(Lead.inn == inn)
    if company_name:
        conditions.append(Lead.company_name.contains(company_name))
    if procurement_id:
        conditions.append(
            or_(
                Lead.source_id.contains(procurement_id),
                Lead.source_url.contains(procurement_id),
                Lead.event_title.contains(procurement_id),
            )
        )
    if only_with_inn:
        conditions.append(Lead.inn.is_not(None))
    if only_with_contacts:
        stmt = stmt.join(Lead.contacts).where(
            or_(LeadContact.contact_email.is_not(None), LeadContact.contact_phone.is_not(None))
        )
    normalized_query = normalize_text(query_text)
    if normalized_query:
        tokens = [item for item in normalized_query.split() if len(item) >= 3]
        token_conditions = [
            or_(
                Lead.company_name.contains(token),
                Lead.event_title.contains(token),
                Lead.event_summary.contains(token),
            )
            for token in tokens
        ]
        if token_conditions:
            conditions.append(and_(*token_conditions))
    return stmt.where(and_(*conditions))


async def list_leads(
    session: AsyncSession,
    *,
    user_id: str,
    period_from: Optional[datetime] = None,
    period_to: Optional[datetime] = None,
    region: Optional[str] = None,
    priority: Optional[str] = None,
    source_type: Optional[str] = None,
    inn: Optional[str] = None,
    company_name: Optional[str] = None,
    procurement_id: Optional[str] = None,
    query_text: Optional[str] = None,
    only_with_inn: bool = False,
    only_with_contacts: bool = False,
    limit: int = 100,
) -> list[LeadView]:
    stmt = _apply_lead_filters(
        _lead_query(),
        user_id=user_id,
        period_from=period_from,
        period_to=period_to,
        region=region,
        priority=priority,
        source_type=source_type,
        inn=inn,
        company_name=company_name,
        procurement_id=procurement_id,
        query_text=query_text,
        only_with_inn=only_with_inn,
        only_with_contacts=only_with_contacts,
    ).limit(limit)
    result = await session.scalars(stmt)
    return [_lead_to_view(lead) for lead in result.unique().all()]


async def list_lead_entities(
    session: AsyncSession,
    *,
    user_id: str,
    period_from: Optional[datetime] = None,
    period_to: Optional[datetime] = None,
    region: Optional[str] = None,
    priority: Optional[str] = None,
    source_type: Optional[str] = None,
    inn: Optional[str] = None,
    company_name: Optional[str] = None,
    procurement_id: Optional[str] = None,
    query_text: Optional[str] = None,
    only_with_inn: bool = False,
    only_with_contacts: bool = False,
    limit: int = 100,
) -> list[Lead]:
    stmt = _apply_lead_filters(
        _lead_query(),
        user_id=user_id,
        period_from=period_from,
        period_to=period_to,
        region=region,
        priority=priority,
        source_type=source_type,
        inn=inn,
        company_name=company_name,
        procurement_id=procurement_id,
        query_text=query_text,
        only_with_inn=only_with_inn,
        only_with_contacts=only_with_contacts,
    ).limit(limit)
    result = await session.scalars(stmt)
    return result.unique().all()


async def get_lead(
    session: AsyncSession,
    *,
    lead_id: str,
    user_id: str,
) -> LeadView:
    lead = await session.scalar(_apply_lead_filters(_lead_query().where(Lead.id == lead_id), user_id=user_id))
    if lead is None:
        raise KeyError(f"Lead '{lead_id}' not found.")
    return _lead_to_view(lead)


async def add_feedback(
    session: AsyncSession,
    *,
    lead_id: str,
    user_id: str,
    payload: LeadFeedbackCreate,
) -> LeadFeedbackView:
    lead = await session.get(Lead, lead_id)
    if lead is None:
        raise KeyError(f"Lead '{lead_id}' not found.")
    feedback = LeadFeedback(lead_id=lead_id, user_id=user_id, status=payload.status, comment=payload.comment)
    lead.feedback_status = payload.status
    session.add(feedback)
    await session.commit()
    await session.refresh(feedback)
    return _feedback_to_view(feedback)


async def resolve_feedback_target(
    session: AsyncSession,
    *,
    user_id: str,
    lead_id: str | None = None,
    inn: str | None = None,
    company_name: str | None = None,
    procurement_id: str | None = None,
) -> LeadView | None:
    stmt = _apply_lead_filters(
        _lead_query(),
        user_id=user_id,
        inn=inn,
        company_name=company_name,
        procurement_id=procurement_id,
    )
    if lead_id:
        stmt = stmt.where(Lead.id == lead_id)
    lead = await session.scalar(stmt.limit(1))
    return _lead_to_view(lead) if lead is not None else None


async def search_lead_index(
    session: AsyncSession,
    *,
    user_id: str,
    query_text: str,
    period_from: Optional[datetime] = None,
    period_to: Optional[datetime] = None,
    region: Optional[str] = None,
    priority: Optional[str] = None,
    source_type: Optional[str] = None,
    inn: Optional[str] = None,
    company_name: Optional[str] = None,
    procurement_id: Optional[str] = None,
    only_with_inn: bool = False,
    only_with_contacts: bool = False,
    limit: int | None = None,
) -> list[LeadIndexHitView]:
    normalized_query = normalize_text(query_text)
    tokens = [item for item in normalized_query.split() if item]
    if not tokens:
        return []

    stmt = (
        select(LeadIndexChunk, Lead, LeadDocument)
        .join(Lead, Lead.id == LeadIndexChunk.lead_id)
        .outerjoin(LeadDocument, LeadDocument.id == LeadIndexChunk.document_id)
        .order_by(Lead.updated_at.desc())
    )
    stmt = _apply_lead_filters(
        stmt,
        user_id=user_id,
        period_from=period_from,
        period_to=period_to,
        region=region,
        priority=priority,
        source_type=source_type,
        inn=inn,
        company_name=company_name,
        procurement_id=procurement_id,
        only_with_inn=only_with_inn,
        only_with_contacts=only_with_contacts,
        query_text=None,
    )
    rows = (await session.execute(stmt)).all()
    if not rows:
        return []

    corpus = [str(chunk.normalized_text or "").split() for chunk, _lead, _document in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokens)
    ranked = sorted(
        ((float(score), chunk, lead, document) for (chunk, lead, document), score in zip(rows, scores, strict=False)),
        key=lambda item: item[0],
        reverse=True,
    )
    result_limit = limit or settings.index_search_limit
    hits: list[LeadIndexHitView] = []
    for score, chunk, lead, document in ranked:
        if score <= 0:
            continue
        hits.append(_index_hit_to_view(chunk=chunk, lead=lead, document=document, score=score))
        if len(hits) >= result_limit:
            break
    return hits


def _lead_export_rows(leads: Sequence[Lead]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lead in leads:
        contact_names = ", ".join(filter(None, [item.contact_name for item in lead.contacts]))
        emails = ", ".join(filter(None, [item.contact_email for item in lead.contacts]))
        phones = ", ".join(filter(None, [item.contact_phone for item in lead.contacts]))
        rows.append(
            {
                "lead_id": lead.id,
                "company_name": lead.company_name,
                "inn": lead.inn,
                "ogrn": lead.ogrn,
                "region": lead.region,
                "website": lead.website,
                "event_type": lead.event_type,
                "event_title": lead.event_title,
                "event_date": lead.event_date.isoformat() if lead.event_date else None,
                "event_summary": lead.event_summary,
                "amount": lead.amount,
                "currency": lead.currency,
                "lead_priority": lead.lead_priority,
                "lead_score": lead.lead_score,
                "rationale": lead.rationale,
                "source_type": lead.source_type,
                "source_url": lead.source_url,
                "contact_names": contact_names,
                "contact_emails": emails,
                "contact_phones": phones,
                "manual_review_required": lead.manual_review_required,
                "workflow_status": lead.workflow_status,
                "feedback_status": lead.feedback_status,
            }
        )
    return rows


def _lead_seller_summary(leads: Sequence[Lead]) -> str:
    lines: list[str] = [f"Кандидатов в работе: {len(leads)}", ""]
    for index, lead in enumerate(leads, start=1):
        emails = ", ".join(filter(None, [item.contact_email for item in lead.contacts]))
        phones = ", ".join(filter(None, [item.contact_phone for item in lead.contacts]))
        next_step = None
        for fact in lead.facts:
            if fact.field_name == "recommended_next_step":
                next_step = dict(fact.value_json or {}).get("value")
                break
        lines.extend(
            [
                f"{index}. {lead.company_name}",
                f"lead_id: {lead.id}",
                f"приоритет: {lead.lead_priority} | score: {lead.lead_score}",
                f"событие: {lead.event_title or lead.event_summary or 'не указано'}",
                f"основание: {lead.rationale or 'без пояснения'}",
                f"ИНН: {lead.inn or 'не найден'}",
                f"контакты: {emails or phones or 'не найдены'}",
                f"источник: {lead.source_url or lead.source_reference or 'не указан'}",
                f"следующий шаг: {next_step or 'требуется ручная оценка продавцом'}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _write_export_file(*, export_format: str, rows: Sequence[Dict[str, Any]], leads: Sequence[Lead]) -> tuple[str, Path]:
    export_dir = Path(settings.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().strftime("%Y%m%d%H%M%S")
    normalized_format = export_format.lower()
    if normalized_format not in {"xlsx", "csv", "txt"}:
        normalized_format = "xlsx"
    filename = f"sales_lead_export_{timestamp}.{normalized_format}"
    path = export_dir / filename
    if normalized_format == "txt":
        path.write_text(_lead_seller_summary(leads), encoding="utf-8")
    else:
        dataframe = pd.DataFrame(list(rows))
        if normalized_format == "csv":
            dataframe.to_csv(path, index=False)
        else:
            dataframe.to_excel(path, index=False)
    return filename, path


def _iter_document_segments(document: Dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = list(document.get("segments") or [])
    if raw_segments:
        normalized: list[dict[str, Any]] = []
        for index, segment in enumerate(raw_segments):
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            page_number = segment.get("page_number")
            if page_number is None:
                page_number = index + 1
            position_start = segment.get("position_start")
            position_end = segment.get("position_end")
            if position_start is None:
                position_start = 0
            if position_end is None:
                position_end = int(position_start) + len(text)
            normalized.append(
                {
                    "page_number": int(page_number),
                    "position_start": int(position_start),
                    "position_end": int(position_end),
                    "text": text,
                    "metadata": dict(segment.get("metadata") or {}),
                }
            )
        return normalized

    fallback_text = str(document.get("text") or "").strip()
    if not fallback_text:
        return []
    page_number = document.get("metadata", {}).get("page_number") or 1
    return [
        {
            "page_number": int(page_number),
            "position_start": 0,
            "position_end": len(fallback_text),
            "text": fallback_text,
            "metadata": dict(document.get("metadata") or {}),
        }
    ]


def _iter_segment_chunks(segment: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = _chunk_text(str(segment.get("text") or ""))
    segment_start = int(segment.get("position_start") or 0)
    for chunk in chunks:
        chunk["position_start"] = segment_start + int(chunk["position_start"])
        chunk["position_end"] = segment_start + int(chunk["position_end"])
    return chunks


async def create_export(
    session: AsyncSession,
    *,
    request: LeadExportRequest,
    user_id: str,
) -> LeadExportView:
    leads = await list_lead_entities(
        session,
        user_id=user_id,
        period_from=request.period_from,
        period_to=request.period_to,
        region=request.region,
        priority=request.priority,
        source_type=request.source_type,
        only_with_inn=request.only_with_inn,
        only_with_contacts=request.only_with_contacts,
        limit=request.limit,
    )
    rows = _lead_export_rows(leads)
    filename, path = _write_export_file(export_format=request.format, rows=rows, leads=leads)
    export = LeadExport(
        requested_by=user_id,
        format=request.format.lower(),
        filename=filename,
        path=str(path),
        filters_json=request.model_dump(mode="json"),
        row_count=len(rows),
    )
    session.add(export)
    await session.commit()
    await session.refresh(export)
    return _export_to_view(export)


async def persist_leads(
    session: AsyncSession,
    *,
    lead_cards: Sequence[Dict[str, Any]],
    created_by_user_id: Optional[str],
) -> list[LeadView]:
    persisted_ids: list[str] = []
    for payload in lead_cards:
        dedup_key = str(payload.get("dedup_key") or compute_dedup_key(payload))
        stmt = (
            select(Lead)
            .where(
                or_(
                    Lead.dedup_key == dedup_key,
                    and_(
                        Lead.inn.is_not(None),
                        Lead.inn == payload.get("inn"),
                        Lead.company_name == payload.get("company_name"),
                    ),
                )
            )
            .limit(1)
        )
        lead = await session.scalar(stmt)
        if lead is None:
            lead = Lead(company_name=str(payload.get("company_name") or "Unknown company"))
            session.add(lead)
            await session.flush()

        lead.created_by_user_id = payload.get("created_by_user_id") or created_by_user_id
        lead.dedup_key = dedup_key
        lead.source_type = payload.get("source_type")
        lead.source_id = payload.get("source_id")
        lead.source_url = payload.get("source_url")
        lead.company_name = str(payload.get("company_name") or lead.company_name)
        lead.inn = payload.get("inn")
        lead.ogrn = payload.get("ogrn")
        lead.region = payload.get("region")
        lead.website = payload.get("website")
        lead.event_type = payload.get("event_type")
        lead.event_title = payload.get("event_title")
        lead.event_date = _normalize_datetime(payload.get("event_date"))

        lead.event_summary = payload.get("event_summary")
        amount = payload.get("amount")
        lead.amount = float(amount) if amount not in {None, ""} else None
        lead.currency = payload.get("currency")
        lead.object_type = payload.get("object_type")
        lead.lead_priority = str(payload.get("lead_priority") or "insufficient_data")
        lead.lead_score = float(payload.get("lead_score") or 0.0)
        lead.rationale = payload.get("rationale")
        lead.missing_data = list(payload.get("missing_data") or [])
        lead.manual_review_required = bool(payload.get("manual_review_required"))
        lead.workflow_status = str(payload.get("workflow_status") or "new")
        lead.feedback_status = str(payload.get("feedback_status") or lead.feedback_status or "pending")
        lead.tags = list(payload.get("tags") or [])
        lead.digest_included = bool(payload.get("digest_included"))
        lead.source_reference = payload.get("source_reference")
        lead.retrieved_at = _normalize_datetime(payload.get("retrieved_at"))

        lead.confidence = float(payload.get("confidence")) if payload.get("confidence") is not None else None
        lead.metadata_json = dict(payload.get("metadata") or {})

        await session.execute(LeadSource.__table__.delete().where(LeadSource.lead_id == lead.id))
        await session.execute(LeadIndexChunk.__table__.delete().where(LeadIndexChunk.lead_id == lead.id))
        await session.execute(LeadDocument.__table__.delete().where(LeadDocument.lead_id == lead.id))
        await session.execute(LeadFact.__table__.delete().where(LeadFact.lead_id == lead.id))
        await session.execute(LeadContact.__table__.delete().where(LeadContact.lead_id == lead.id))
        await session.execute(LeadEnrichment.__table__.delete().where(LeadEnrichment.lead_id == lead.id))

        for source in payload.get("sources") or []:
            session.add(
                LeadSource(
                    lead_id=lead.id,
                    source_type=str(source.get("source_type") or lead.source_type or "unknown"),
                    source_id=source.get("source_id"),
                    source_url=source.get("source_url"),
                    source_reference=source.get("source_reference"),
                    retrieved_at=lead.retrieved_at,
                    confidence=source.get("confidence"),
                    is_primary=bool(source.get("is_primary")),
                    metadata_json=dict(source.get("metadata") or {}),
                )
            )

        for document in payload.get("documents") or []:
            document_model = LeadDocument(
                lead_id=lead.id,
                document_url=document.get("document_url"),
                file_name=document.get("file_name"),
                file_type=document.get("file_type"),
                stored_path=document.get("stored_path"),
                parse_status=str(document.get("parse_status") or "pending"),
                index_status=str(document.get("index_status") or "pending"),
                source_reference=document.get("source_reference"),
                retrieved_at=lead.retrieved_at,
                confidence=document.get("confidence"),
                extracted_excerpt=document.get("extracted_excerpt"),
                metadata_json=dict(document.get("metadata") or {}),
            )
            session.add(document_model)
            await session.flush()

            segments = _iter_document_segments(document)
            if document_model.index_status == "ready" and segments:
                chunk_index = 0
                for segment in segments:
                    for chunk in _iter_segment_chunks(segment):
                        session.add(
                            LeadIndexChunk(
                                lead_id=lead.id,
                                document_id=document_model.id,
                                chunk_index=chunk_index,
                                text=chunk["text"],
                                normalized_text=chunk["normalized_text"],
                                source_reference=document.get("source_reference") or document.get("document_url"),
                                page_number=segment.get("page_number"),
                                position_start=chunk["position_start"],
                                position_end=chunk["position_end"],
                                metadata_json={
                                    **dict(document.get("metadata") or {}),
                                    **dict(segment.get("metadata") or {}),
                                    "entities": _extract_chunk_entities(chunk["text"]),
                                },
                            )
                        )
                        chunk_index += 1

        for field_name, fact_value in (payload.get("facts") or {}).items():
            session.add(
                LeadFact(
                    lead_id=lead.id,
                    field_name=field_name,
                    value_json=_serialize_value(fact_value),
                    source_reference=payload.get("source_reference"),
                    retrieved_at=lead.retrieved_at,
                    confidence=lead.confidence,
                )
            )

        for contact in payload.get("contacts") or []:
            session.add(
                LeadContact(
                    lead_id=lead.id,
                    contact_name=contact.get("contact_name"),
                    contact_role=contact.get("contact_role"),
                    contact_email=contact.get("contact_email"),
                    contact_phone=contact.get("contact_phone"),
                    contact_source=contact.get("contact_source"),
                    contact_confidence=contact.get("contact_confidence"),
                    source_reference=contact.get("source_reference"),
                    retrieved_at=lead.retrieved_at,
                    metadata_json=dict(contact.get("metadata") or {}),
                )
            )

        for enrichment in payload.get("enrichments") or []:
            session.add(
                LeadEnrichment(
                    lead_id=lead.id,
                    provider=str(enrichment.get("provider") or "unknown"),
                    status=str(enrichment.get("status") or "pending"),
                    payload_json=dict(enrichment.get("payload") or {}),
                    source_reference=enrichment.get("source_reference"),
                    retrieved_at=lead.retrieved_at,
                    confidence=enrichment.get("confidence"),
                )
            )

        persisted_ids.append(lead.id)

    await session.commit()

    views: list[LeadView] = []
    for lead_id in persisted_ids:
        reloaded = await session.scalar(_lead_query().where(Lead.id == lead_id))
        if reloaded is not None:
            views.append(_lead_to_view(reloaded))
    return views
