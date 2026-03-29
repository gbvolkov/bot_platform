from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .db import AsyncSessionFactory
from .models import SalesLeadRetrievalEvent, SalesLeadRetrievalJob
from .schemas import RetrievalProgress, RetrievalSnapshot, RetrievalStatus


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _read_json_payload(path_value: str | None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _write_json_payload(path_value: str | None, payload: list[dict[str, Any]]) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_payload(
    path_value: str | None,
    *,
    request_payload: dict[str, Any],
    status: str,
    stage: str,
    message: str,
    progress: dict[str, Any],
    error_text: str | None,
) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "search_urls": request_payload.get("search_urls") or [],
                "status": status,
                "stage": stage,
                "message": message,
                "progress": progress,
                "error_text": error_text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class SubmissionLookup:
    active: RetrievalSnapshot | None
    matching: RetrievalSnapshot | None


class ActiveRetrievalConflictError(Exception):
    def __init__(self, snapshot: RetrievalSnapshot) -> None:
        super().__init__(snapshot.message or "Another procurement retrieval is already running.")
        self.snapshot = snapshot


def _snapshot_from_job(job: SalesLeadRetrievalJob, *, include_payloads: bool) -> RetrievalSnapshot:
    return RetrievalSnapshot(
        retrieval_id=job.id,
        conversation_id=job.conversation_id,
        request_hash=job.request_hash,
        run_id=job.run_id,
        index_id=job.index_id,
        status=job.status,  # type: ignore[arg-type]
        stage=job.stage,
        message=job.message or "",
        completion_announced=bool(job.completion_announced),
        snapshot_updated_at=_utc_iso(job.snapshot_updated_at),
        request_payload=dict(job.request_payload_json or {}),
        progress=RetrievalProgress.model_validate(dict(job.progress_json or {})),
        items=_read_json_payload(job.items_snapshot_path) if include_payloads else [],
        prepared_documents=_read_json_payload(job.documents_snapshot_path) if include_payloads else [],
        error_text=job.error_text,
    )


class SalesLeadRetrievalStore:
    def _default_progress(self) -> dict[str, int]:
        return RetrievalProgress().model_dump()

    async def lookup_submission(
        self,
        *,
        conversation_id: str,
        request_hash: str,
        include_payloads: bool = True,
    ) -> SubmissionLookup:
        async with AsyncSessionFactory() as session:
            active_job = await self._get_active_job(session=session, conversation_id=conversation_id)
            matching_job = await self._get_job_by_request(
                session=session,
                conversation_id=conversation_id,
                request_hash=request_hash,
            )
            return SubmissionLookup(
                active=_snapshot_from_job(active_job, include_payloads=include_payloads) if active_job else None,
                matching=_snapshot_from_job(matching_job, include_payloads=include_payloads) if matching_job else None,
            )

    async def create_job(
        self,
        *,
        retrieval_id: str,
        conversation_id: str,
        agent_id: str,
        request_hash: str,
        request_payload: dict[str, Any],
        run_id: str,
        index_id: str,
        items_snapshot_path: str,
        documents_snapshot_path: str,
        summary_snapshot_path: str | None,
        message: str,
    ) -> RetrievalSnapshot:
        progress = self._default_progress()
        _write_json_payload(items_snapshot_path, [])
        _write_json_payload(documents_snapshot_path, [])
        _write_summary_payload(
            summary_snapshot_path,
            request_payload=request_payload,
            status="queued",
            stage="queued",
            message=message,
            progress=progress,
            error_text=None,
        )
        async with AsyncSessionFactory() as session:
            job = SalesLeadRetrievalJob(
                id=retrieval_id,
                conversation_id=conversation_id,
                active_conversation_id=conversation_id,
                agent_id=agent_id,
                request_hash=request_hash,
                request_payload_json=request_payload,
                run_id=run_id,
                index_id=index_id,
                status="queued",
                stage="queued",
                message=message,
                progress_json=progress,
                items_snapshot_path=items_snapshot_path,
                documents_snapshot_path=documents_snapshot_path,
                summary_snapshot_path=summary_snapshot_path,
                ready_items_count=0,
                ready_documents_count=0,
                indexed_segments_count=0,
                completion_announced=False,
                snapshot_updated_at=_utc_now(),
                last_heartbeat_at=_utc_now(),
            )
            session.add(job)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                matching_job = await self._get_job_by_request(
                    session=session,
                    conversation_id=conversation_id,
                    request_hash=request_hash,
                )
                if matching_job is not None:
                    return _snapshot_from_job(matching_job, include_payloads=True)
                active_job = await self._get_active_job(session=session, conversation_id=conversation_id)
                if active_job is not None:
                    raise ActiveRetrievalConflictError(
                        _snapshot_from_job(active_job, include_payloads=True)
                    ) from None
                raise
            await session.refresh(job)
            return _snapshot_from_job(job, include_payloads=True)

    async def get_latest_for_conversation(
        self,
        *,
        conversation_id: str,
        include_payloads: bool = False,
    ) -> RetrievalSnapshot | None:
        async with AsyncSessionFactory() as session:
            active_job = await self._get_active_job(session=session, conversation_id=conversation_id)
            if active_job is not None:
                return _snapshot_from_job(active_job, include_payloads=include_payloads)
            stmt = (
                select(SalesLeadRetrievalJob)
                .where(SalesLeadRetrievalJob.conversation_id == conversation_id)
                .order_by(SalesLeadRetrievalJob.updated_at.desc(), SalesLeadRetrievalJob.created_at.desc())
                .limit(1)
            )
            job = await session.scalar(stmt)
            if job is None:
                return None
            return _snapshot_from_job(job, include_payloads=include_payloads)

    async def get_retrieval(
        self,
        *,
        retrieval_id: str,
        include_payloads: bool = False,
    ) -> RetrievalSnapshot | None:
        async with AsyncSessionFactory() as session:
            job = await session.get(SalesLeadRetrievalJob, retrieval_id)
            if job is None:
                return None
            return _snapshot_from_job(job, include_payloads=include_payloads)

    async def claim_next_queued_job(self) -> RetrievalSnapshot | None:
        async with AsyncSessionFactory() as session:
            stmt = (
                select(SalesLeadRetrievalJob)
                .where(SalesLeadRetrievalJob.status == "queued")
                .order_by(SalesLeadRetrievalJob.created_at.asc())
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            job = await session.scalar(stmt)
            if job is None:
                return None
            now = _utc_now()
            job.status = "in_progress"
            job.stage = "starting"
            job.message = "Procurement retrieval worker started."
            job.started_at = now
            job.last_heartbeat_at = now
            job.snapshot_updated_at = now
            await session.commit()
            await session.refresh(job)
            return _snapshot_from_job(job, include_payloads=False)

    async def update_job(
        self,
        retrieval_id: str,
        *,
        status: RetrievalStatus | None = None,
        stage: str | None = None,
        message: str | None = None,
        progress: dict[str, Any] | None = None,
        items: list[dict[str, Any]] | None = None,
        prepared_documents: list[dict[str, Any]] | None = None,
        error_text: str | None = None,
        completion_announced: bool | None = None,
        clear_active: bool = False,
        finished: bool = False,
    ) -> RetrievalSnapshot:
        async with AsyncSessionFactory() as session:
            job = await session.get(SalesLeadRetrievalJob, retrieval_id)
            if job is None:
                raise KeyError(f"Unknown retrieval job '{retrieval_id}'")
            now = _utc_now()
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if message is not None:
                job.message = message
            if progress is not None:
                job.progress_json = dict(progress)
                job.ready_items_count = len(items) if items is not None else int(progress.get("total_purchases", job.ready_items_count))
                job.ready_documents_count = (
                    len(prepared_documents) if prepared_documents is not None else int(progress.get("prepared_documents", job.ready_documents_count))
                )
                job.indexed_segments_count = int(progress.get("indexed_segments", job.indexed_segments_count))
            if items is not None:
                _write_json_payload(job.items_snapshot_path, items)
                job.ready_items_count = len(items)
            if prepared_documents is not None:
                _write_json_payload(job.documents_snapshot_path, prepared_documents)
                job.ready_documents_count = len(prepared_documents)
            if error_text is not None:
                job.error_text = error_text
            if completion_announced is not None:
                job.completion_announced = completion_announced
            if clear_active:
                job.active_conversation_id = None
            if finished:
                job.finished_at = now
            job.snapshot_updated_at = now
            job.last_heartbeat_at = now
            await session.commit()
            _write_summary_payload(
                job.summary_snapshot_path,
                request_payload=dict(job.request_payload_json or {}),
                status=job.status,
                stage=job.stage,
                message=job.message or "",
                progress=dict(job.progress_json or {}),
                error_text=job.error_text,
            )
            await session.refresh(job)
            return _snapshot_from_job(job, include_payloads=True)

    async def append_event(
        self,
        retrieval_id: str,
        *,
        stage: str,
        message: str,
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        async with AsyncSessionFactory() as session:
            session.add(
                SalesLeadRetrievalEvent(
                    retrieval_id=retrieval_id,
                    stage=stage,
                    level=level,
                    message=message,
                    payload_json=payload or {},
                )
            )
            await session.commit()

    async def mark_announced(self, retrieval_id: str) -> None:
        await self.update_job(retrieval_id, completion_announced=True)

    async def _get_active_job(
        self,
        *,
        session: AsyncSession,
        conversation_id: str,
    ) -> SalesLeadRetrievalJob | None:
        stmt = select(SalesLeadRetrievalJob).where(
            SalesLeadRetrievalJob.active_conversation_id == conversation_id
        )
        return await session.scalar(stmt)

    async def _get_job_by_request(
        self,
        *,
        session: AsyncSession,
        conversation_id: str,
        request_hash: str,
    ) -> SalesLeadRetrievalJob | None:
        stmt = select(SalesLeadRetrievalJob).where(
            SalesLeadRetrievalJob.conversation_id == conversation_id,
            SalesLeadRetrievalJob.request_hash == request_hash,
        )
        return await session.scalar(stmt)
