from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatibleJsonClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        temperature: float = 0.2,
        timeout: int = 240,
        retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        if "api.openai.com" in self.base_url and not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for api.openai.com")

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for json_mode in (True, False):
            try:
                return self._complete_json(system=system, user=user, json_mode=json_mode)
            except RuntimeError as exc:
                last_error = exc
                message = str(exc).lower()
                if json_mode and "response_format" in message:
                    continue
                if json_mode and "json_object" in message:
                    continue
                raise
        raise RuntimeError(f"LLM request failed: {last_error}")

    def _complete_json(self, *, system: str, user: str, json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(endpoint, data=request_body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
                parsed = parse_json_object(content)
                return parsed
            except urllib.error.HTTPError as exc:
                body = _read_http_error_body(exc)
                last_error = RuntimeError(f"HTTP Error {exc.code}: {exc.reason}; body={body}")
                if exc.code == 400:
                    break
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
            except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
        mode = "with response_format" if json_mode else "without response_format"
        raise RuntimeError(f"LLM request failed {mode}: {last_error}")


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - error body is best effort.
        return "<unreadable>"
    return body[:2000] or "<empty>"


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM returned JSON, but not an object")
    return data
