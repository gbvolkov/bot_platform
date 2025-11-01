from __future__ import annotations

import logging
from typing import Any

from uvicorn.protocols.http.h11_impl import H11Protocol


class LoggingH11Protocol(H11Protocol):
    """Custom h11 protocol that logs raw bytes for malformed requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._raw_buffer: bytes = b""

    def data_received(self, data: bytes) -> None:  # type: ignore[override]
        if data:
            # Keep the tail (last 2048 bytes) to avoid unbounded growth.
            self._raw_buffer = (self._raw_buffer + data)[-2048:]
        super().data_received(data)

    def handle_invalid_request(self, exc: Exception) -> None:  # type: ignore[override]
        peer = None
        if self.transport:
            peer = self.transport.get_extra_info("peername")
        snippet = self._raw_buffer.decode("utf-8", errors="replace")
        logging.getLogger("uvicorn.error").warning(
            "Invalid HTTP request received from %s: %s | raw=%r",
            peer,
            exc,
            snippet,
        )
        super().handle_invalid_request(exc)
