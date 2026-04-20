"""Уточнение поискового запроса для follow-up вопросов в режиме KPI."""
from __future__ import annotations

import re
from typing import Sequence

DialogMessage = tuple[str, str, str]

_INDEX_QUERY_RE = re.compile(r"\b([1-9]\d?)\b")
_BULLET_ITEM_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)(.+?)\s*$")
_QUOTED_KPI_RE = re.compile(r"[«\"]([^«»\"\n]{3,200})[»\"]")
_SHORT_REFERENCE_RE = re.compile(
    r"\b(?:что\s+такое|что\s+за|расшифруй|объясни|расскажи|как\s+считается|как\s+он\s+считается|"
    r"как\s+она\s+считается|как\s+это\s+считается|этот|эта|это|он|она|его|ее|её|второй|третий|"
    r"четвертый|четвёртый|пятый|шестой|седьмой|восьмой|девятый|десятый|показатель|кпэ)\b",
    re.IGNORECASE,
)
_METHODOLOGY_RE = re.compile(
    r"\b(?:методик\w*|формул\w*|расч[её]т\w*|рассчит\w*|детализац\w*|"
    r"подробн\w*|нюанс\w*)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_FOLLOW_UP_RE = re.compile(
    r"^\s*(?:да|ага|угу|ок(?:ей)?|хорошо|давай|можно|нужно|нужна|нужен|"
    r"нужны|покаж(?:и|ите))\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_METHODOLOGY_OFFER_RE = re.compile(
    r"методик\w*[^\n]*kpi[^\n]*списк|могу[^\n]*показать[^\n]*методик\w*",
    re.IGNORECASE,
)

_ORDINAL_INDEXES = {
    "первый": 1,
    "второй": 2,
    "третий": 3,
    "четвертый": 4,
    "четвёртый": 4,
    "пятый": 5,
    "шестой": 6,
    "седьмой": 7,
    "восьмой": 8,
    "девятый": 9,
    "десятый": 10,
}


def build_kpi_search_query(query: str, dialog_messages: Sequence[DialogMessage]) -> str:
    """Добавляет в retrieval-запрос KPI из истории, если текущий вопрос слишком ссылочный."""
    normalized_query = (query or "").strip()
    if not normalized_query or not _looks_like_reference_question(normalized_query):
        return normalized_query

    history = _drop_current_user_message(dialog_messages, normalized_query)
    if not history:
        return normalized_query

    kpi_name = _resolve_kpi_name(normalized_query, history)
    if kpi_name:
        return f"{normalized_query}\nУточняемый KPI: {kpi_name}"

    last_assistant_message = _get_last_message(history, "assistant")
    if not last_assistant_message:
        return normalized_query

    snippet = last_assistant_message.strip()
    if len(snippet) > 900:
        snippet = snippet[:900].rsplit(" ", 1)[0].rstrip() + "..."
    return f"{normalized_query}\nПредыдущий ответ ассистента:\n{snippet}"


def resolve_kpi_reference(query: str, dialog_messages: Sequence[DialogMessage]) -> str | None:
    """Возвращает точное имя KPI из истории для ссылочного follow-up вопроса."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return None
    if not (_looks_like_reference_question(normalized_query) or is_methodology_follow_up(normalized_query, dialog_messages)):
        return None

    history = _drop_current_user_message(dialog_messages, normalized_query)
    if not history:
        return None

    return _resolve_kpi_name(normalized_query, history)


def extract_last_listed_kpis(dialog_messages: Sequence[DialogMessage]) -> list[str]:
    """Извлекает KPI из последнего ответа ассистента со списком."""
    last_assistant_message = _get_last_message(dialog_messages, "assistant")
    if not last_assistant_message:
        return []
    return _extract_kpi_items(last_assistant_message)


def is_methodology_follow_up(query: str, dialog_messages: Sequence[DialogMessage]) -> bool:
    """Определяет свободный follow-up про методику/формулу после ответа со списком KPI."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return False
    if _METHODOLOGY_RE.search(normalized_query):
        return True
    if not _AFFIRMATIVE_FOLLOW_UP_RE.match(normalized_query):
        return False

    last_assistant_message = _get_last_message(dialog_messages, "assistant")
    if not last_assistant_message:
        return False
    if not extract_last_listed_kpis(dialog_messages):
        return False
    return bool(_METHODOLOGY_OFFER_RE.search(last_assistant_message))


def _looks_like_reference_question(query: str) -> bool:
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if len(tokens) > 8:
        return False
    return bool(
        _INDEX_QUERY_RE.search(query)
        or _SHORT_REFERENCE_RE.search(query)
        or _METHODOLOGY_RE.search(query)
    )


def _drop_current_user_message(
    dialog_messages: Sequence[DialogMessage],
    current_query: str,
) -> list[DialogMessage]:
    remaining: list[DialogMessage] = []
    skipped = False
    current_query = current_query.strip()

    for message_text, role, timestamp in dialog_messages:
        if not skipped and role == "user" and (message_text or "").strip() == current_query:
            skipped = True
            continue
        remaining.append((message_text, role, timestamp))

    return remaining


def _resolve_kpi_name(query: str, history: Sequence[DialogMessage]) -> str | None:
    requested_index = _extract_requested_index(query)
    last_assistant_message = _get_last_message(history, "assistant")
    items = _extract_kpi_items(last_assistant_message) if last_assistant_message else []

    if requested_index and items:
        if 1 <= requested_index <= len(items):
            return items[requested_index - 1]

    if len(items) == 1 and (_METHODOLOGY_RE.search(query) or _AFFIRMATIVE_FOLLOW_UP_RE.match(query)):
        return items[0]

    if last_assistant_message and not items:
        quoted = _extract_quoted_kpi_name(last_assistant_message)
        if quoted:
            return quoted

    last_user_message = _get_last_message(history, "user")
    if last_user_message:
        return _extract_quoted_kpi_name(last_user_message)

    return None


def _extract_requested_index(query: str) -> int | None:
    digit_match = _INDEX_QUERY_RE.search(query)
    if digit_match:
        return int(digit_match.group(1))

    lowered = query.lower()
    for word, index in _ORDINAL_INDEXES.items():
        if word in lowered:
            return index
    return None


def _extract_kpi_items(text: str) -> list[str]:
    items: list[str] = []
    for line in (text or "").splitlines():
        match = _BULLET_ITEM_RE.match(line)
        if not match:
            continue
        item = match.group(1).strip()
        item = re.split(r"\s+[—-]\s+", item, maxsplit=1)[0].strip()
        item = re.split(r"\s+\(", item, maxsplit=1)[0].strip()
        item = item.rstrip(" .;:")
        if item:
            items.append(item)
    return items


def _extract_quoted_kpi_name(text: str) -> str | None:
    match = _QUOTED_KPI_RE.search(text or "")
    if not match:
        return None
    return match.group(1).strip().rstrip(" .;:")


def _get_last_message(history: Sequence[DialogMessage], role: str) -> str | None:
    for message_text, message_role, _ in history:
        if message_role == role and (message_text or "").strip():
            return message_text
    return None
