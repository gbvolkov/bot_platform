from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any, Dict, List, Optional
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


def _extract_response_attachments(agent_message: Dict[str, Any]) -> List[Dict[str, Any]]:
    attachments: List[Dict[str, Any]] = []
    metadata = agent_message.get("metadata")
    if isinstance(metadata, dict):
        meta_attachments = metadata.get("attachments")
        if isinstance(meta_attachments, list):
            attachments.extend(item for item in meta_attachments if isinstance(item, dict))
    if attachments:
        return attachments

    content = agent_message.get("content")
    parts = None
    if isinstance(content, dict):
        if content.get("type") == "segments":
            parts = content.get("parts")
    elif isinstance(content, list):
        parts = content
    if not isinstance(parts, list):
        return attachments

    for piece in parts:
        if isinstance(piece, dict) and piece.get("type") in {"file", "image", "audio", "video", "attachment"}:
            attachments.append(piece)
    return attachments


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
    job_future: Optional[asyncio.Task] = None

    streamed_already = False
    try:
        response = None
        if payload.stream:
            streamed_parts: List[str] = []
            stream_started = False
            final_event: Optional[Dict[str, Any]] = None

            async for event in client.send_message_stream(
                conversation_id=payload.conversation_id,
                user_id=payload.user_id,
                user_role=payload.user_role,
                text=payload.text,
                raw_user_text=payload.raw_user_text,
                attachments=payload.attachments,
                metadata=payload.metadata,
            ):
                event_type = event.get("type")
                if event_type == "chunk":
                    content = event.get("content") or ""
                    if not content:
                        continue
                    if not stream_started:
                        await queue.mark_status(payload.job_id, "streaming")
                        job_stage["value"] = "streaming"
                        await queue.publish_event(
                            QueueEvent(job_id=payload.job_id, type="status", status="streaming")
                        )
                        stream_started = True
                        streamed_already = True
                    streamed_parts.append(content)
                    logger.debug("Publishing chunk job_id=%s size=%d", payload.job_id, len(content))
                    await queue.publish_event(
                        QueueEvent(job_id=payload.job_id, type="chunk", content=content)
                    )
                    await queue.update_heartbeat(payload.job_id, status=current_status())
                elif event_type in {"completed", "interrupt", "failed"}:
                    final_event = event
                    break

                elapsed = time.monotonic() - start_time
                if soft_timeout and elapsed > soft_timeout and not warn_emitted:
                    logger.warning(
                        "Job %s exceeded configured bot request timeout (%ss) but is still running",
                        payload.job_id,
                        soft_timeout,
                    )
                    warn_emitted = True

            if final_event is None:
                raise RuntimeError("Stream ended without terminal event")

            if final_event.get("type") == "failed":
                error_message = final_event.get("error") or "Agent execution failed."
                job_stage["value"] = "failed"
                await queue.store_failure(payload.job_id, error_message)
                await queue.publish_event(
                    QueueEvent(job_id=payload.job_id, type="failed", status="failed", error=error_message)
                )
                await queue.update_heartbeat(payload.job_id, status=current_status())
                return

            raw_text = final_event.get("content") or "".join(streamed_parts)
            metadata = final_event.get("metadata") or {}
            if not stream_started and raw_text:
                await queue.mark_status(payload.job_id, "streaming")
                job_stage["value"] = "streaming"
                await queue.publish_event(
                    QueueEvent(job_id=payload.job_id, type="status", status="streaming")
                )
                for chunk in _chunk_text(raw_text, queue_settings.chunk_char_limit):
                    await queue.publish_event(
                        QueueEvent(job_id=payload.job_id, type="chunk", content=chunk)
                    )
                await queue.update_heartbeat(payload.job_id, status=current_status())
                streamed_already = True

            if final_event.get("type") == "interrupt":
                if raw_text and "content" not in metadata:
                    metadata = {**metadata, "content": raw_text}
                job_stage["value"] = "interrupted"
                await queue.mark_status(payload.job_id, "interrupted", {"result": metadata})
                await queue.publish_event(
                    QueueEvent(
                        job_id=payload.job_id,
                        type="interrupt",
                        status="interrupted",
                        metadata=metadata,
                    )
                )
                await queue.clear_active_job(payload.job_id)
                logger.info("Job %s interrupted; awaiting user input", payload.job_id)
                return

            response = {
                "agent_message": {
                    "raw_text": raw_text,
                    "metadata": metadata,
                }
            }
        else:
            job_future = asyncio.create_task(
                client.send_message(
                    conversation_id=payload.conversation_id,
                    user_id=payload.user_id,
                    user_role=payload.user_role,
                    text=payload.text,
                    raw_user_text=payload.raw_user_text,
                    attachments=payload.attachments,
                    metadata=payload.metadata,
                )
            )

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
        agent_status = agent_message.get("metadata", {}).get("agent_status") if agent_message else None
        logger.debug(
            "Agent message job_id=%s status=%s keys=%s metadata=%s",
            payload.job_id,
            agent_status,
            list(agent_message.keys()),
            agent_message.get("metadata"),
        )
        logger.debug(
            "Agent response job_id=%s raw_text_chars=%d status=%s",
            payload.job_id,
            len(raw_text),
            agent_status,
        )
        attachments = _extract_response_attachments(agent_message) if agent_message else []

        if agent_status == "interrupted":
            metadata = agent_message.get("metadata") or {}
            if raw_text and "content" not in metadata:
                metadata = {**metadata, "content": raw_text}
            job_stage["value"] = "interrupted"
            await queue.mark_status(payload.job_id, "interrupted", {"result": metadata})
            await queue.publish_event(
                QueueEvent(
                    job_id=payload.job_id,
                    type="interrupt",
                    status="interrupted",
                    metadata=metadata,
                )
            )
            await queue.clear_active_job(payload.job_id)
            logger.info("Job %s interrupted; awaiting user input", payload.job_id)
            return

        if raw_text and not streamed_already:
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
        if attachments:
            metadata["attachments"] = attachments

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
        if job_future is not None and not job_future.done():
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
