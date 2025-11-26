from __future__ import annotations

import json
import sys

import httpx

BASE_URL = "http://localhost:8000/api"
USER_ID = "b20c5f6d-5811-4279-9357-644293e81805"
USER_ROLE = "default"


def _headers() -> dict[str, str]:
    return {"X-User-Id": USER_ID, "X-User-Role": USER_ROLE}


def _p(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload)


def main() -> int:
    client = httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))

    # 1) Create conversation
    conv_resp = client.post(f"{BASE_URL}/conversations/", json={"agent_id": "theodor_agent"}, headers=_headers())
    _p("create_conversation status", conv_resp.status_code)
    conv_data = conv_resp.json()
    _p("create_conversation body", conv_data)
    conv_id = conv_data["id"]

    # 2) Send initial message (expect interrupt/options)
    user_text = "Привет!\nУ меня есть идея стартапа: Uber для выгула собак.\nПомоги мне проработать её."
    payload1 = {"payload": {"type": "text", "text": user_text, "metadata": {"raw_user_text": user_text}}}
    msg1 = client.post(f"{BASE_URL}/conversations/{conv_id}/messages", json=payload1, headers=_headers())
    _p("send_message_1 status", msg1.status_code)
    _p("send_message_1 body", msg1.json())

    # 3) Inspect conversation (metadata/pending_interrupt)
    conv_detail_1 = client.get(f"{BASE_URL}/conversations/{conv_id}", headers=_headers())
    _p("conversation_after_msg1 status", conv_detail_1.status_code)
    _p("conversation_after_msg1 body", conv_detail_1.json())

    # 4) Resume with a user choice
    choice_text = "Вариант B"
    payload2 = {"payload": {"type": "text", "text": choice_text, "metadata": {"raw_user_text": choice_text}}}
    msg2 = client.post(f"{BASE_URL}/conversations/{conv_id}/messages", json=payload2, headers=_headers())
    _p("send_message_2 status", msg2.status_code)
    _p("send_message_2 body", msg2.json())

    # 5) Inspect conversation again to confirm pending_interrupt handling
    conv_detail_2 = client.get(f"{BASE_URL}/conversations/{conv_id}", headers=_headers())
    _p("conversation_after_msg2 status", conv_detail_2.status_code)
    _p("conversation_after_msg2 body", conv_detail_2.json())

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        _p("error", str(exc))
        sys.exit(1)
