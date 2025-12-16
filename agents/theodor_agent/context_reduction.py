from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


ARTIFACT_SUMMARY_TAG = "theodor_artifact_summary"


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content") or ""
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    return ""


def is_tool_related_message(msg: Any) -> bool:
    if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
        return True
    if isinstance(msg, AIMessage):
        return bool(
            getattr(msg, "tool_calls", None)
            or getattr(msg, "invalid_tool_calls", None)
            or (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls")
        )
    return False


def is_artifact_summary_message(msg: Any) -> bool:
    if not isinstance(msg, SystemMessage):
        return False
    meta = getattr(msg, "additional_kwargs", {}) or {}
    return meta.get("type") == ARTIFACT_SUMMARY_TAG


def last_summary_index(messages: Iterable[Any], *, exclude_artifact_id: Optional[int] = None) -> int:
    idx = -1
    for i, msg in enumerate(messages):
        if not is_artifact_summary_message(msg):
            continue
        meta = getattr(msg, "additional_kwargs", {}) or {}
        if exclude_artifact_id is not None and meta.get("artifact_id") == exclude_artifact_id:
            continue
        idx = i
    return idx


def summarize_artifact_discussion(
    *,
    model: BaseChatModel,
    artifact_id: int,
    artifact_name: str,
    user_prompt: str,
    selected_option_label: str,
    selected_option_text: str,
    user_notes: List[str],
    max_chars: int = 1200,
) -> str:
    system = SystemMessage(
        content=(
            "Create a compact long-term memory note for a finished artifact discussion.\n"
            "Focus on user wishes, constraints, decisions, and outcomes that matter for future artifacts.\n"
            "Rules:\n"
            "- Do NOT paste or quote the artifact text.\n"
            "- Do NOT include tool calls, tool results, URLs, or internal reasoning.\n"
            "- Output 4-8 bullet points, concise, no extra prose.\n"
            "- If something is unknown, omit it.\n"
        )
    )

    notes_block = "\n".join(f"- {note}" for note in user_notes if note.strip()) or "- (no additional user notes)"
    human = HumanMessage(
        content=(
            f"Artifact: {artifact_id} — {artifact_name}\n\n"
            f"Overall user prompt:\n{user_prompt.strip()}\n\n"
            f"Selected option: {selected_option_label}\n"
            f"Selected option (short): {selected_option_text.strip()}\n\n"
            "User messages during this artifact:\n"
            f"{notes_block}\n"
        )
    )

    try:
        result = model.invoke([system, human])
        summary = extract_text(getattr(result, "content", result))
    except Exception:  # noqa: BLE001
        logging.exception("Artifact summarization failed (artifact_id=%s).", artifact_id)
        summary = ""

    summary = summary.strip()
    if not summary:
        # Deterministic fallback (still avoids artifact text).
        fallback_bits: List[str] = []
        if selected_option_label.strip():
            fallback_bits.append(f"- Selected option: {selected_option_label.strip()}")
        for note in user_notes[-5:]:
            note = note.strip()
            if note:
                fallback_bits.append(f"- User note: {note}")
        summary = "\n".join(fallback_bits).strip()

    if max_chars > 0 and len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


def build_pruned_history(
    *,
    messages: List[Any],
    keep_first_user_messages: int = 5,
    keep_last_messages: int = 5,
    drop_summary_for_artifact_id: Optional[int] = None,
) -> List[Any]:
    non_tool = [m for m in (messages or []) if not is_tool_related_message(m)]

    keep_indices: set[int] = set()

    # Keep all existing artifact summaries (optionally drop the current one to replace it).
    for idx, msg in enumerate(non_tool):
        if not is_artifact_summary_message(msg):
            continue
        meta = getattr(msg, "additional_kwargs", {}) or {}
        if drop_summary_for_artifact_id is not None and meta.get("artifact_id") == drop_summary_for_artifact_id:
            continue
        keep_indices.add(idx)

    # Keep first N user messages (from the whole conversation).
    kept_user = 0
    for idx, msg in enumerate(non_tool):
        if getattr(msg, "type", "") != "human":
            continue
        keep_indices.add(idx)
        kept_user += 1
        if kept_user >= keep_first_user_messages:
            break

    # Keep last N messages, excluding summary messages so they don't consume the "tail" budget.
    tail_candidates = [idx for idx, m in enumerate(non_tool) if not is_artifact_summary_message(m)]
    tail = tail_candidates[-keep_last_messages:] if keep_last_messages > 0 else []
    keep_indices.update(tail)

    return [non_tool[idx] for idx in sorted(keep_indices)]
