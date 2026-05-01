from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .decisions import GuardrailDecision


_SENSITIVE_EXACT_KEYS = {
    "base64",
    "base64_data",
    "content",
    "data",
    "input",
    "output",
    "prompt",
    "prompts",
    "raw_text",
    "response",
}
_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|authorization|credential)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
_LONG_SECRET_RE = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")


def _summarize_string(value: str) -> dict[str, Any]:
    return {
        "redacted": True,
        "chars": len(value),
    }


def redact_text(value: str) -> str:
    text = _EMAIL_RE.sub("<redacted-email>", value)
    text = _PHONE_RE.sub("<redacted-phone>", text)
    text = _LONG_SECRET_RE.sub("<redacted-secret>", text)
    return text


def redact_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, str):
        if key and _is_sensitive_key(key):
            return _summarize_string(value)
        redacted = redact_text(value)
        if redacted != value:
            return redacted
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for item_key, item_value in value.items():
            key_text = str(item_key)
            if _is_sensitive_key(key_text):
                if isinstance(item_value, str):
                    result[key_text] = _summarize_string(item_value)
                elif isinstance(item_value, (list, dict)):
                    result[key_text] = {
                        "redacted": True,
                        "type": type(item_value).__name__,
                    }
                else:
                    result[key_text] = "<redacted>"
            else:
                result[key_text] = redact_value(item_value, key=key_text)
        return result
    if isinstance(value, list):
        return [redact_value(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, key=key) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_EXACT_KEYS or bool(_SENSITIVE_KEY_RE.search(normalized))


def _generation_texts(response: Any) -> list[str]:
    generations = getattr(response, "generations", None)
    if not isinstance(generations, list):
        return []
    texts: list[str] = []
    for generation_group in generations:
        if not isinstance(generation_group, Iterable):
            continue
        for generation in generation_group:
            text = getattr(generation, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return texts


class RedactingJSONFileTracer(BaseCallbackHandler):
    """Local JSONL callback tracer that avoids persisting raw prompts and outputs."""

    def __init__(self, path: str = "traces.jsonl") -> None:
        target = Path(path)
        if target.parent and str(target.parent) not in {"", "."}:
            target.parent.mkdir(parents=True, exist_ok=True)
        self.f = target.open("a", encoding="utf-8")
        self._token_count = 0

    def _write(self, payload: dict[str, Any]) -> None:
        self.f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.f.flush()

    def on_llm_start(self, serialized, prompts, **kwargs):
        safe_prompts = list(prompts or [])
        self._write(
            {
                "ts": time.time(),
                "type": "llm_start",
                "model": redact_value(serialized),
                "prompt_count": len(safe_prompts),
                "prompt_chars": [len(str(prompt)) for prompt in safe_prompts],
            }
        )

    def on_llm_end(self, response, **kwargs):
        texts = _generation_texts(response)
        self._write(
            {
                "ts": time.time(),
                "type": "llm_end",
                "response_count": len(texts),
                "response_chars": [len(text) for text in texts],
            }
        )

    def on_llm_new_token(self, token: str, **kwargs):
        self._token_count += 1
        self._write(
            {
                "ts": time.time(),
                "type": "llm_token",
                "idx": self._token_count,
                "token_chars": len(token or ""),
            }
        )

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = None
        if isinstance(serialized, dict):
            tool_name = serialized.get("name")
        self._write(
            {
                "ts": time.time(),
                "type": "tool_start",
                "tool": tool_name,
                "input_chars": len(str(input_str or "")),
            }
        )

    def on_tool_end(self, output, **kwargs):
        content = getattr(output, "content", output)
        self._write(
            {
                "ts": time.time(),
                "type": "tool_end",
                "output": {
                    "redacted": True,
                    "chars": len(str(content or "")),
                    "tool_call_id": getattr(output, "tool_call_id", None),
                },
            }
        )


class GuardrailEventLogger:
    """Structured JSONL logger for guardrail decisions with redacted metadata."""

    def __init__(self, path: str | None = None) -> None:
        self._file = None
        if path:
            target = Path(path)
            if target.parent and str(target.parent) not in {"", "."}:
                target.parent.mkdir(parents=True, exist_ok=True)
            self._file = target.open("a", encoding="utf-8")

    def log_decision(self, decision: GuardrailDecision) -> None:
        if self._file is None:
            return
        payload = {
            "ts": time.time(),
            "type": "guardrail_decision",
            "decision": redact_value(decision),
        }
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()


__all__ = [
    "GuardrailEventLogger",
    "RedactingJSONFileTracer",
    "redact_text",
    "redact_value",
]
