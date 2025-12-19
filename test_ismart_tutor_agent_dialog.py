from __future__ import annotations

import sys
import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from agents.ismart_tutor_agent import agent as tutor_agent


def _safe_str(value: Any) -> str:
    text = str(value)
    encoding = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    try:
        text.encode(encoding)
        return text
    except Exception:
        return text.encode("ascii", "backslashreplace").decode("ascii")


def safe_print(*args: Any, **kwargs: Any) -> None:
    print(*(_safe_str(arg) for arg in args), **kwargs)


def _last_ai_text(result: Dict[str, Any]) -> str:
    messages = result.get("messages") or []
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return str(first.get("text") or "")
            return str(first)
        return str(content)
    return ""


def _hint_is_ready(graph, cfg: Dict[str, Any]) -> bool:
    state = graph.get_state(cfg).values
    return not state.get("needs_person_info") and not state.get("needs_question") and bool(
        (state.get("hint_raw") or "").strip()
    )


def run_dialog() -> None:
    graph = tutor_agent.initialize_agent()
    cfg = {
        "configurable": {
            "thread_id": f"ismart_tutor_sim_{uuid.uuid4().hex}",
            "user_id": "local_user",
            "user_role": "default",
        }
    }

    safe_print("iSmart tutor dialog simulation")
    safe_print("- Type your question/task.")
    safe_print("- If asked for profile, provide the missing fields.")
    safe_print("- Type RESET to reset the dialog, EXIT to quit.")

    while True:
        user_text = input("\nUSER: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            return

        msg_type = "reset" if user_text.strip().upper() == "RESET" else "text"
        payload = {"messages": [HumanMessage(content=[{"type": msg_type, "text": user_text}])]}

        result = graph.invoke(payload, config=cfg)
        assistant_text = _last_ai_text(result)
        safe_print("ASSISTANT:", assistant_text)

        if msg_type != "reset" and _hint_is_ready(graph, cfg):
            return


if __name__ == "__main__":
    run_dialog()

