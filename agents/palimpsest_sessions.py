from __future__ import annotations

"""Backward-compatible imports for Palimpsest session helpers.

New guarded agents should import from platform_guardrails directly. This module is
kept so existing agents continue to resolve their legacy imports unchanged.
"""

from platform_guardrails.middleware import PalimpsestSessionMiddleware
from platform_guardrails.privacy import (
    PalimpsestSessionManager,
    _call_text_transform,
    anonymize_with_session,
    clone_message_with_transform,
    content_is_reset,
    map_strings as _map_strings,
    state_has_reset_message,
    thread_id_from_config,
    thread_id_from_runtime,
    transform_content,
)

__all__ = [
    "PalimpsestSessionManager",
    "PalimpsestSessionMiddleware",
    "_call_text_transform",
    "_map_strings",
    "anonymize_with_session",
    "clone_message_with_transform",
    "content_is_reset",
    "state_has_reset_message",
    "thread_id_from_config",
    "thread_id_from_runtime",
    "transform_content",
]
