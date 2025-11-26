from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

from .config import settings
from .models import EnqueuePayload, JobStage, QueueEvent


def _now_ts() -> float:
    return time.time()


class RedisTaskQueue:
    """Thin wrapper around Redis primitives to orchestrate agent jobs."""

    def __init__(self) -> None:
        self._redis = Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        self._queue_key = settings.queue_key
        self._status_prefix = settings.status_prefix
        self._channel_prefix = settings.channel_prefix
        self._job_ttl = settings.job_ttl_seconds
        self._sse_heartbeat_interval = settings.sse_heartbeat_seconds
        self._worker_heartbeat_interval = settings.worker_heartbeat_seconds
        self._stale_after = settings.heartbeat_stale_after_seconds
        self._watchdog_interval = settings.watchdog_interval_seconds
        self._active_jobs_key = f"{self._status_prefix}active_jobs"

    @property
    def redis(self) -> Redis:
        return self._redis

    def _status_key(self, job_id: str) -> str:
        return f"{self._status_prefix}{job_id}"

    def _channel(self, job_id: str) -> str:
        return f"{self._channel_prefix}{job_id}"

    async def startup(self) -> None:
        # Lazily verify connectivity.
        await self._redis.ping()
        logger.debug("RedisTaskQueue connected to %s", settings.redis_url)

    async def shutdown(self) -> None:
        await self._redis.aclose()
        logger.debug("RedisTaskQueue connection closed")

    async def enqueue(self, payload: EnqueuePayload) -> None:
        job_key = self._status_key(payload.job_id)
        created_at = _now_ts()
        status_payload = {
            "status": "queued",
            "created_at": str(created_at),
            "updated_at": str(created_at),
            "conversation_id": payload.conversation_id,
            "model": payload.model,
            "user_id": payload.user_id,
        }
        await self._redis.hset(job_key, mapping=status_payload)
        await self._redis.expire(job_key, self._job_ttl)
        await self._redis.rpush(self._queue_key, payload.model_dump_json(exclude_none=True))
        logger.debug("Enqueued job_id=%s conversation_id=%s", payload.job_id, payload.conversation_id)
        await self.publish_event(
            QueueEvent(job_id=payload.job_id, type="status", status="queued"),
        )

    async def publish_event(self, event: QueueEvent) -> None:
        await self._redis.publish(self._channel(event.job_id), event.model_dump_json(exclude_none=True))
        logger.debug("Published event job_id=%s type=%s status=%s", event.job_id, event.type, event.status)

    async def mark_status(self, job_id: str, status: JobStage, extra: Optional[Dict[str, Any]] = None) -> None:
        job_key = self._status_key(job_id)
        now_ts = _now_ts()
        mapping: Dict[str, str] = {
            "status": status,
            "updated_at": str(now_ts),
            "last_heartbeat": str(now_ts),
        }
        if extra:
            mapping.update({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value) for key, value in extra.items()})
        await self._redis.hset(job_key, mapping=mapping)
        await self._redis.expire(job_key, self._job_ttl)

    async def store_result(self, job_id: str, result: Dict[str, Any]) -> None:
        await self.mark_status(job_id, "completed", {"result": result})
        await self.clear_active_job(job_id)
        logger.debug("Stored result for job_id=%s", job_id)

    async def store_failure(self, job_id: str, error: str) -> None:
        await self.mark_status(job_id, "failed", {"error": error})
        await self.clear_active_job(job_id)
        logger.debug("Stored failure for job_id=%s error=%s", job_id, error)

    async def register_active_job(self, job_id: str) -> None:
        now_ts = _now_ts()
        job_key = self._status_key(job_id)
        await self._redis.hset(job_key, mapping={"last_heartbeat": str(now_ts)})
        await self._redis.zadd(self._active_jobs_key, {job_id: now_ts})
        await self._redis.expire(job_key, self._job_ttl)
        logger.debug("Registered active job_id=%s at %s", job_id, now_ts)

    async def clear_active_job(self, job_id: str) -> None:
        await self._redis.zrem(self._active_jobs_key, job_id)
        logger.debug("Cleared active job_id=%s", job_id)

    async def update_heartbeat(self, job_id: str, status: Optional[str] = None) -> None:
        job_key = self._status_key(job_id)
        now_ts = _now_ts()
        mapping: Dict[str, str] = {"last_heartbeat": str(now_ts), "updated_at": str(now_ts)}
        if status:
            mapping["status"] = status
        await self._redis.hset(job_key, mapping=mapping)
        await self._redis.zadd(self._active_jobs_key, {job_id: now_ts})
        await self._redis.expire(job_key, self._job_ttl)
        logger.debug("Heartbeat recorded job_id=%s status=%s ts=%s", job_id, status, now_ts)

    async def fail_job_if_active(self, job_id: str, reason: str) -> bool:
        status = await self._redis.hget(self._status_key(job_id), "status")
        if status in {"completed", "failed", "interrupted", None}:
            await self.clear_active_job(job_id)
            return False
        await self.store_failure(job_id, reason)
        await self.publish_event(
            QueueEvent(job_id=job_id, type="failed", status="failed", error=reason),
        )
        logger.warning("Marked job_id=%s failed due to reason=%s", job_id, reason)
        return True

    async def fail_stale_jobs(self) -> list[str]:
        if self._stale_after <= 0:
            return []
        cutoff = _now_ts() - self._stale_after
        stale_ids = await self._redis.zrangebyscore(self._active_jobs_key, min="-inf", max=cutoff)
        failed: list[str] = []
        for job_id in stale_ids:
            if await self.fail_job_if_active(job_id, "Heartbeat timeout exceeded"):
                failed.append(job_id)
        if failed:
            logger.warning("Stale heartbeat detected for job_ids=%s", ", ".join(failed))
        return failed

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        job_key = self._status_key(job_id)
        raw = await self._redis.hgetall(job_key)
        decoded: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in {"result", "metadata"}:
                try:
                    decoded[key] = json.loads(value)
                except json.JSONDecodeError:
                    decoded[key] = value
            elif key == "error":
                decoded[key] = value
            elif key in {"created_at", "updated_at", "last_heartbeat"}:
                decoded[key] = float(value)
            else:
                decoded[key] = value
        return decoded

    async def pop_job(self, timeout: int = 5) -> Optional[EnqueuePayload]:
        """Blocking pop used by workers."""
        item = await self._redis.blpop(self._queue_key, timeout=timeout)
        if not item:
            return None
        _, data = item
        return EnqueuePayload.model_validate_json(data)

    @asynccontextmanager
    async def subscribe(self, job_id: str) -> AsyncIterator[PubSub]:
        channel = self._channel(job_id)
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            yield pubsub
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def iter_events(
        self,
        job_id: str,
        *,
        include_status_snapshot: bool = False,
    ) -> AsyncIterator[QueueEvent]:
        current_status: Optional[JobStage] = None
        async with self.subscribe(job_id) as pubsub:
            if include_status_snapshot:
                snapshot = await self.get_status(job_id)
                stage = snapshot.get("status") or "queued"
                current_status = stage  # type: ignore[assignment]
                yield QueueEvent(job_id=job_id, type="status", status=stage)
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    await asyncio.sleep(0.05)
                    continue
                if message["type"] != "message":
                    continue
                data = message["data"]
                try:
                    event = QueueEvent.model_validate_json(data)
                except ValueError:
                    continue
                if event.type == "status" and event.status:
                    current_status = event.status  # type: ignore[assignment]
                elif event.type == "completed":
                    current_status = "completed"
                elif event.type == "failed":
                    current_status = "failed"
                elif event.type == "interrupt":
                    current_status = "interrupted"
                yield event
                if event.type in {"completed", "failed", "interrupt"}:
                    break

    async def wait_for_completion(self, job_id: str, timeout: Optional[float] = None) -> QueueEvent:
        """Convenience helper for non-streaming callers."""
        if timeout is None:
            timeout = settings.completion_wait_timeout_seconds

        async def _wait() -> QueueEvent:
            async for event in self.iter_events(job_id):
                if event.type in {"completed", "failed", "interrupt"}:
                    return event
            # Fallback for defensive completeness.
            return QueueEvent(job_id=job_id, type="failed", status="failed", error="No terminal event received.")

        return await asyncio.wait_for(_wait(), timeout=timeout)
logger = logging.getLogger("task_queue.redis")
