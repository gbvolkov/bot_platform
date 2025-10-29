"""Notification helpers for dispatching KB reload events to interested parties."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Callable, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

from .models import WebhookRegistration


@dataclass(frozen=True)
class KBReloadContext:
    """Context payload describing why a reload is requested."""

    reason: str
    source: str
    document_ids: Sequence[str] = field(default_factory=tuple)
    initiated_by: Optional[str] = None
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


ReloadCallback = Callable[[KBReloadContext], None]


class ReloadBroadcaster:
    """In-memory broadcaster that notifies registered listeners sequentially."""

    def __init__(self) -> None:
        self._listeners: Dict[str, ReloadCallback] = {}
        self._lock = RLock()

    def register(self, listener_id: str, callback: ReloadCallback) -> None:
        with self._lock:
            self._listeners[listener_id] = callback
        logger.debug("Registered KB reload listener '%s'", listener_id)

    def unregister(self, listener_id: str) -> None:
        with self._lock:
            removed = self._listeners.pop(listener_id, None)
        if removed:
            logger.debug("Removed KB reload listener '%s'", listener_id)

    def broadcast(self, context: KBReloadContext) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        with self._lock:
            listeners = dict(self._listeners)
        for listener_id, callback in listeners.items():
            try:
                callback(context)
            except Exception:  # pragma: no cover - external callbacks
                logger.exception("KB reload listener '%s' failed", listener_id)
                results[listener_id] = False
            else:
                results[listener_id] = True
        return results

    def listeners(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._listeners)


_default_broadcaster = ReloadBroadcaster()


def get_broadcaster() -> ReloadBroadcaster:
    """Return the module-level broadcaster instance."""
    return _default_broadcaster


def register_reload_listener(listener_id: str, callback: ReloadCallback) -> None:
    """Register a listener on the default broadcaster."""
    _default_broadcaster.register(listener_id, callback)


def unregister_reload_listener(listener_id: str) -> None:
    """Remove a listener from the default broadcaster."""
    _default_broadcaster.unregister(listener_id)


class WebhookRegistry:
    """Registry that stores external webhook subscribers."""

    def __init__(self) -> None:
        self._webhooks: Dict[str, WebhookRegistration] = {}
        self._lock = RLock()

    def register(self, registration: WebhookRegistration) -> None:
        with self._lock:
            self._webhooks[registration.listener_id] = registration
        logger.debug("Registered KB webhook '%s' -> %s", registration.listener_id, registration.url)

    def unregister(self, listener_id: str) -> None:
        with self._lock:
            removed = self._webhooks.pop(listener_id, None)
        if removed:
            logger.debug("Removed KB webhook '%s'", listener_id)

    def list(self) -> Tuple[WebhookRegistration, ...]:
        with self._lock:
            return tuple(self._webhooks.values())

    def dispatch(self, context: KBReloadContext) -> Dict[str, bool]:
        """POST the reload context to all registered webhook endpoints."""
        with self._lock:
            webhooks = dict(self._webhooks)
        results: Dict[str, bool] = {}

        if not webhooks:
            return results

        if requests is None:  # pragma: no cover - dependency absent
            logger.warning("The 'requests' package is not available; skipping webhook delivery.")
            for listener_id in webhooks:
                results[listener_id] = False
            return results

        payload = {
            "reason": context.reason,
            "source": context.source,
            "document_ids": list(context.document_ids),
            "initiated_by": context.initiated_by,
            "requested_at": context.requested_at.isoformat(),
        }

        for listener_id, registration in webhooks.items():
            headers = {"Content-Type": "application/json"}
            headers.update(registration.headers)
            if registration.secret:
                headers.setdefault("X-KB-Secret", registration.secret)

            try:
                response = requests.post(registration.url, json=payload, headers=headers, timeout=10)
                success = response.status_code < 400
                if not success:
                    logger.warning(
                        "Webhook '%s' returned non-success status %s",
                        listener_id,
                        response.status_code,
                    )
            except Exception:  # pragma: no cover - network operations
                logger.exception("Webhook '%s' delivery failed", listener_id)
                success = False
            results[listener_id] = success
        return results


_default_webhook_registry = WebhookRegistry()


def get_webhook_registry() -> WebhookRegistry:
    """Return the module-level webhook registry."""
    return _default_webhook_registry


def register_webhook_listener(registration: WebhookRegistration) -> None:
    """Register a webhook listener on the default registry."""
    _default_webhook_registry.register(registration)


def unregister_webhook_listener(listener_id: str) -> None:
    """Remove a webhook listener from the default registry."""
    _default_webhook_registry.unregister(listener_id)
