from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Optional
import contextlib

from services.bot_client import BotServiceClient

from .config import settings as queue_settings
from .models import EnqueuePayload, QueueEvent
from .redis_queue import RedisTaskQueue

logger = logging.getLogger("task_queue.worker")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _chunk_text(value: str, limit: int) -> list[str]:
    if not value:
        return []
    return [value[i : i + limit] for i in range(0, len(value), limit)]


async def _heartbeat_loop(queue: RedisTaskQueue, job_id: str, status_fn) -> None:
    interval = queue_settings.worker_heartbeat_seconds
    if interval <= 0:
        return
    try:
        while True:
            await asyncio.sleep(interval)
            status = status_fn()
            await queue.update_heartbeat(job_id, status=status)
            await queue.publish_event(QueueEvent(job_id=job_id, type="heartbeat", status=status))
            logger.debug("Heartbeat sent job_id=%s status=%s", job_id, status)
    except asyncio.CancelledError:
        logger.debug("Heartbeat loop cancelled job_id=%s", job_id)
        pass


async def _process_job(
    *,
    payload: EnqueuePayload,
    client: BotServiceClient,
    queue: RedisTaskQueue,
) -> None:
    logger.debug(
        "Processing job job_id=%s conversation_id=%s user_id=%s text_chars=%d attachments=%d",
        payload.job_id,
        payload.conversation_id,
        payload.user_id,
        len(payload.text or ""),
        len(payload.attachments or []) if payload.attachments else 0,
    )
    job_stage = {"value": "running"}

    def current_status() -> str:
        return job_stage["value"]

    await queue.mark_status(payload.job_id, "running")
    await queue.publish_event(QueueEvent(job_id=payload.job_id, type="status", status="running"))
    await queue.register_active_job(payload.job_id)
    await queue.update_heartbeat(payload.job_id, status=current_status())

    heartbeat_task = asyncio.create_task(_heartbeat_loop(queue, payload.job_id, current_status))

    soft_timeout = queue_settings.bot_request_timeout_seconds if queue_settings.bot_request_timeout_seconds > 0 else None
    start_time = time.monotonic()
    warn_emitted = False
    interval = max(queue_settings.worker_heartbeat_seconds, 1)

    job_future = asyncio.create_task(
        client.send_message(
            conversation_id=payload.conversation_id,
            user_id=payload.user_id,
            user_role=payload.user_role,
            text=payload.text,
            attachments=payload.attachments,
            metadata=payload.metadata,
        )
    )

    try:
        response = None
        while True:
            try:
                response = await asyncio.wait_for(asyncio.shield(job_future), timeout=interval)
                break
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start_time
                logger.debug(
                    "Job %s still running after %.2fs (heartbeat interval %ss)",
                    payload.job_id,
                    elapsed,
                    interval,
                )
                await queue.update_heartbeat(payload.job_id, status=current_status())
                if soft_timeout and elapsed > soft_timeout and not warn_emitted:
                    logger.warning(
                        "Job %s exceeded configured bot request timeout (%ss) but is still running",
                        payload.job_id,
                        soft_timeout,
                    )
                    warn_emitted = True
                continue

        if response is None:
            raise RuntimeError("Agent invocation returned no response")

        agent_message = response.get("agent_message", {}) or {}
        raw_text = agent_message.get("raw_text") or ""
        logger.debug("Agent response job_id=%s raw_text_chars=%d", payload.job_id, len(raw_text))

        if raw_text:
            await queue.mark_status(payload.job_id, "streaming")
            job_stage["value"] = "streaming"
            await queue.publish_event(QueueEvent(job_id=payload.job_id, type="status", status="streaming"))
            await queue.update_heartbeat(payload.job_id, status=current_status())
            for chunk in _chunk_text(raw_text, queue_settings.chunk_char_limit):
                logger.debug("Publishing chunk job_id=%s size=%d", payload.job_id, len(chunk))
                await queue.publish_event(QueueEvent(job_id=payload.job_id, type="chunk", content=chunk))
                await queue.update_heartbeat(payload.job_id, status=current_status())
        else:
            logger.debug("No content to stream job_id=%s", payload.job_id)

        metadata = {
            "conversation_id": payload.conversation_id,
            "content": raw_text,
            "response": response,
        }

        await queue.store_result(payload.job_id, metadata)
        await queue.publish_event(
            QueueEvent(job_id=payload.job_id, type="completed", status="completed", metadata=metadata)
        )
        job_stage["value"] = "completed"
        await queue.update_heartbeat(payload.job_id, status=current_status())
        logger.info("Job %s completed", payload.job_id)
    except Exception as exc:
        logger.exception("Job %s failed", payload.job_id)
        error_detail = str(exc).strip() or repr(exc)
        job_stage["value"] = "failed"
        error_message = f"Agent invocation failed: {exc.__class__.__name__}: {error_detail}"
        await queue.store_failure(payload.job_id, error_message)
        await queue.publish_event(
            QueueEvent(job_id=payload.job_id, type="failed", status="failed", error=error_message)
        )
        await queue.update_heartbeat(payload.job_id, status=current_status())
    finally:
        if not job_future.done():
            job_future.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await job_future
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

async def _watchdog_loop(queue: RedisTaskQueue, stop_event: asyncio.Event) -> None:
    interval = queue_settings.watchdog_interval_seconds
    if interval <= 0 or queue_settings.heartbeat_stale_after_seconds <= 0:
        return
    logger.info(
        "Watchdog started interval=%ss stale_after=%ss",
        interval,
        queue_settings.heartbeat_stale_after_seconds,
    )
    try:
        while not stop_event.is_set():
            failed_jobs = await queue.fail_stale_jobs()
            if failed_jobs:
                logger.warning("Watchdog marked stale jobs as failed: %s", ", ".join(failed_jobs))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
    finally:
        logger.info("Watchdog stopped")


async def _worker_loop(stop_event: asyncio.Event) -> None:
    queue = RedisTaskQueue()
    client = BotServiceClient(
        base_url=str(queue_settings.bot_service_base_url),
        request_timeout=queue_settings.bot_request_timeout_seconds,
        connect_timeout=queue_settings.bot_connect_timeout_seconds,
    )
    await asyncio.gather(queue.startup(), client.startup())

    watchdog_task = asyncio.create_task(_watchdog_loop(queue, stop_event))
    logger.info(
        "Worker started; awaiting jobs on %s (request_timeout=%s, connect_timeout=%s)",
        queue_settings.queue_key,
        queue_settings.bot_request_timeout_seconds,
        queue_settings.bot_connect_timeout_seconds,
    )
    try:
        while not stop_event.is_set():
            job: Optional[EnqueuePayload] = await queue.pop_job(timeout=5)
            if job is None:
                logger.debug("No job available; continuing poll")
                continue
            logger.debug("Dequeued job job_id=%s", job.job_id)
            await _process_job(payload=job, client=client, queue=queue)
    finally:
        stop_event.set()
        watchdog_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog_task
        await asyncio.gather(client.shutdown(), queue.shutdown())
        logger.info("Worker shutdown complete")


def run() -> None:
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())

    try:
        loop.run_until_complete(_worker_loop(stop_event))
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    run()
