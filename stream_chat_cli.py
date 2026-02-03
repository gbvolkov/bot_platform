from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import httpx

DEFAULT_BASE_URL = os.environ.get("OPENAI_PROXY_BASE_URL", "http://127.0.0.1:8084")
DEFAULT_BOT_SERVICE_URL = os.environ.get("BOT_SERVICE_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get("OPENAI_PROXY_MODEL", "simple_agent")
DEFAULT_USER = os.environ.get("OPENAI_PROXY_USER", "cli-user")
DEFAULT_BACKEND = os.environ.get("STREAM_CHAT_BACKEND", "auto")

EXIT_COMMANDS = {"exit", "/exit", "/quit", "quit"}
RESET_COMMANDS = {"/reset", "reset"}
MULTILINE_START_COMMANDS = {"/multi", "/multiline", "<<<"}
MULTILINE_END_COMMANDS = {"/send", "/end", ">>>"}
MULTILINE_CANCEL_COMMANDS = {"/cancel", "/abort"}


def _stream_chat(
    *,
    base_url: str,
    payload: dict,
    show_status: bool,
) -> tuple[Optional[str], str]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
    assistant_chunks: list[str] = []
    conversation_id: Optional[str] = None

    with httpx.stream("POST", url, headers=headers, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if line.startswith(":"):
                if show_status:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "error" in event:
                err = event["error"]
                message = err.get("message") if isinstance(err, dict) else str(err)
                sys.stderr.write(f"\n[error] {message}\n")
                sys.stderr.flush()
                continue
            if conversation_id is None:
                conversation_id = event.get("conversation_id")
            choices = event.get("choices") or []
            delta_content = None
            if choices:
                delta = choices[0].get("delta") or {}
                delta_content = delta.get("content")
                if delta_content:
                    sys.stdout.write(delta_content)
                    sys.stdout.flush()
                    assistant_chunks.append(delta_content)
            if show_status and not delta_content:
                agent_status = event.get("agent_status")
                if agent_status:
                    sys.stdout.write(f"\n[{agent_status}]\n")
                    sys.stdout.flush()

    assistant_text = "".join(assistant_chunks).strip()
    return conversation_id, assistant_text


def _ensure_bot_conversation(
    *,
    base_url: str,
    model: str,
    user: str,
    conversation_id: Optional[str],
) -> Optional[str]:
    headers = {"X-User-Id": user}
    timeout = httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=None)
    if conversation_id:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            resp = client.get(f"/api/conversations/{conversation_id}", headers=headers)
            if resp.status_code == 404:
                raise RuntimeError("Conversation not found for bot_service backend.")
            resp.raise_for_status()
            return conversation_id

    payload = {"agent_id": model}
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        resp = client.post("/api/conversations/", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        conv_id = data.get("id")
        status = data.get("status")
        if status != "active":
            for _ in range(30):
                time.sleep(1.0)
                check = client.get(f"/api/conversations/{conv_id}", headers=headers)
                if check.status_code == 404:
                    break
                check.raise_for_status()
                detail = check.json()
                if detail.get("status") == "active":
                    break
        return conv_id


def _stream_chat_bot_service(
    *,
    base_url: str,
    model: str,
    user: str,
    conversation_id: Optional[str],
    user_text: str,
    show_status: bool,
) -> tuple[Optional[str], str]:
    conversation_id = _ensure_bot_conversation(
        base_url=base_url,
        model=model,
        user=user,
        conversation_id=conversation_id,
    )
    headers = {"X-User-Id": user, "Accept": "text/event-stream", "Content-Type": "application/json"}
    payload = {
        "payload": {
            "type": "text",
            "text": user_text,
            "metadata": {"raw_user_text": user_text},
        }
    }
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
    assistant_chunks: list[str] = []
    custom_chunks: list[str] = []

    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/api/conversations/{conversation_id}/messages",
        headers=headers,
        params={"stream": "true"},
        json=payload,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if line.startswith(":"):
                if show_status:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "chunk":
                content = event.get("content") or ""
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    assistant_chunks.append(content)
            elif event_type == "custom":
                payload_item = event.get("data") or {}
                if isinstance(payload_item, dict):
                    custom_type = payload_item.get("type")
                    if custom_type in {"artifact_delta", "user_delta"}:
                        text = payload_item.get("text") or ""
                        if text:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                            custom_chunks.append(text)
                    elif custom_type == "done":
                        if show_status:
                            sys.stdout.write("\n[done]\n")
                            sys.stdout.flush()
            elif event_type in {"completed", "interrupt"}:
                content = event.get("content") or ""
                if content and not assistant_chunks and not custom_chunks:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    assistant_chunks.append(content)
                if show_status:
                    sys.stdout.write(f"\n[{event_type}]\n")
                    sys.stdout.flush()
            elif event_type == "failed":
                err = event.get("error") or "Agent failed."
                sys.stderr.write(f"\n[error] {err}\n")
                sys.stderr.flush()
                break

    combined = "".join(custom_chunks or assistant_chunks).strip()
    return conversation_id, combined


def _send_message(
    *,
    base_url: str,
    model: str,
    user: Optional[str],
    conversation_id: Optional[str],
    messages: list[dict],
    show_status: bool,
) -> tuple[Optional[str], str]:
    payload: dict = {"model": model, "messages": messages, "stream": True}
    if user:
        payload["user"] = user
    if conversation_id:
        payload["conversation_id"] = conversation_id

    try:
        new_conversation_id, assistant_text = _stream_chat(
            base_url=base_url,
            payload=payload,
            show_status=show_status,
        )
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        sys.stderr.write(f"\nRequest failed ({exc.response.status_code}): {body}\n")
        sys.stderr.flush()
        return conversation_id, ""

    if new_conversation_id:
        conversation_id = new_conversation_id
    return conversation_id, assistant_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive streaming client for openai_proxy (/v1/chat/completions)."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Proxy base URL.")
    parser.add_argument(
        "--bot-service-url",
        default=DEFAULT_BOT_SERVICE_URL,
        help="Bot service base URL.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Agent id to call.")
    parser.add_argument("--user", default=DEFAULT_USER, help="User id for proxy.")
    parser.add_argument(
        "--backend",
        choices=["auto", "openai_proxy", "bot_service"],
        default=DEFAULT_BACKEND,
        help="Streaming backend: openai_proxy, bot_service, or auto.",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Reuse existing conversation id (optional).",
    )
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Print agent status events and heartbeat dots.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    conversation_id: Optional[str] = args.conversation_id
    backend = args.backend
    if backend == "auto":
        backend = "bot_service" if args.model == "new_theodor_agent" else "openai_proxy"
    user_id = args.user or DEFAULT_USER
    if backend == "bot_service":
        print(f"Streaming chat via bot_service {args.bot_service_url} (model={args.model}).")
    else:
        print(f"Streaming chat via {args.base_url} (model={args.model}).")
    print("Type /exit to quit, /reset to clear local history, /multi for multiline (end with /send).\n")

    try:
        while True:
            user_input = input("You> ").strip()
            if not user_input:
                continue
            user_input_lower = user_input.lower()
            if user_input_lower in EXIT_COMMANDS:
                print("Goodbye!")
                return 0
            if user_input_lower in RESET_COMMANDS:
                messages = [{"role": "system", "content": args.system}] if args.system else []
                conversation_id = None
                print("Local history cleared.\n")
                continue
            if user_input_lower in MULTILINE_START_COMMANDS:
                user_input = _collect_multiline_input()
                if user_input is None:
                    print("Goodbye!")
                    return 0
                if not user_input:
                    print("Canceled.\n")
                    continue

            messages.append({"role": "user", "content": user_input})
            if backend == "bot_service":
                conversation_id, assistant_text = _stream_chat_bot_service(
                    base_url=args.bot_service_url,
                    model=args.model,
                    user=user_id,
                    conversation_id=conversation_id,
                    user_text=user_input,
                    show_status=args.show_status,
                )
            else:
                conversation_id, assistant_text = _send_message(
                    base_url=args.base_url,
                    model=args.model,
                    user=user_id,
                    conversation_id=conversation_id,
                    messages=messages,
                    show_status=args.show_status,
                )
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
            print("\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0


def _collect_multiline_input() -> Optional[str]:
    print("Multiline mode: finish with /send, cancel with /cancel.")
    lines: list[str] = []
    while True:
        line = input("...> ")
        cmd = line.strip().lower()
        if cmd in EXIT_COMMANDS:
            return None
        if cmd in MULTILINE_CANCEL_COMMANDS:
            return ""
        if cmd in MULTILINE_END_COMMANDS:
            break
        lines.append(line)
    return "\n".join(lines).strip()


if __name__ == "__main__":
    raise SystemExit(main())
