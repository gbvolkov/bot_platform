"""Пост-обработка ответов для режима kpi_table."""
from __future__ import annotations

import re

_SOURCE_HEADER_RE = re.compile(
    r"^\s*(?:[*_`#>\-]+\s*)?(?:источники|источник(?:и)? данных|ссылки|sources?|references)\s*:?\s*$",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_PRODUCT_MENTION_RE = re.compile(
    r"\bпо\s+продукту\s+[\"'«»]?kpi_table[\"'«»]?\b",
    re.IGNORECASE,
)
_PRODUCT_TOKEN_RE = re.compile(r"[\"'«»]?kpi_table[\"'«»]?", re.IGNORECASE)
_NET_COMMISSION_RE = re.compile(
    r"нетто-?\s*комиссионн(?:ое|ого|ому|ым|ом)\s+вознаграждени",
    re.IGNORECASE,
)
_INCOME_RE = re.compile(
    r"\b(?:доход(?:а|ом|у)?|заработ(?:ок|ка|ком)|зарплат(?:а|ой|ы|у)?|"
    r"преми(?:я|и|ей|ю)|чист(?:ый|ого)\s+заработок)\b",
    re.IGNORECASE,
)
_NEGATED_INCOME_RE = re.compile(
    r"\bне\s+(?:доход(?:а|ом|у)?|заработ(?:ок|ка|ком)|зарплат(?:а|ой|ы|у)?|преми(?:я|и|ей|ю))\b",
    re.IGNORECASE,
)
_BLOCK_SPLIT_RE = re.compile(r"(\n+|[.!?]+\s+)")
_LEADING_KPI_INTRO_RE = re.compile(
    r"^\s*Я виртуальный помощник по KPI[^\n]*\n?",
    re.IGNORECASE,
)
_LEADING_HELP_LINE_RE = re.compile(
    r"^\s*Могу помочь[^\n]*\n?",
    re.IGNORECASE,
)
_LEADING_SPECIFY_LINE_RE = re.compile(
    r"^\s*Если вы укажете[^\n]*\n?",
    re.IGNORECASE,
)
_INSUFFICIENT_INFO_RE = re.compile(
    r"^\s*В доступных данных недостаточно информации[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ASK_FOR_DETAILS_RE = re.compile(
    r"^\s*Пожалуйста,\s*уточните[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_AMBIGUOUS_ROLE_RE = re.compile(
    r"^\s*Для должности [^\n]*перечень KPI зависит от [^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_FULL_LIST_RE = re.compile(
    r"^\s*Привожу полный перечень[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\.|[-*•])\s+", re.MULTILINE)

_SAFE_NET_COMMISSION_TEXT = (
    "Показатель «Нетто-комиссионное вознаграждение» в KPI-контексте нужно трактовать "
    "как расход компании на выплаты внешним агентам или посредникам, а не как доход "
    "или заработок сотрудника."
)
_AMBIGUOUS_ROLE_TEXT = "Для этой должности KPI зависят от подразделения."


def strip_sources_footer(text: str) -> str:
    """Удаляет хвостовой блок ссылок/источников в конце ответа."""
    if not text:
        return text

    lines = text.splitlines()
    if not lines:
        return text

    # Удаляем нижний блок, если он начинается с заголовка "Источники"/"Ссылки".
    lookback_limit = max(0, len(lines) - 10)
    for idx in range(len(lines) - 1, lookback_limit - 1, -1):
        if _SOURCE_HEADER_RE.match(lines[idx].strip()):
            lines = lines[:idx]
            break

    # Удаляем висящие URL в самом конце, даже если заголовка не было.
    end = len(lines)
    while end > 0:
        stripped = lines[end - 1].strip()
        if not stripped:
            end -= 1
            continue
        if _SOURCE_HEADER_RE.match(stripped) or _URL_RE.search(stripped):
            end -= 1
            continue
        break

    return "\n".join(lines[:end]).rstrip()


def normalize_kpi_wording(text: str) -> str:
    """Убирает продуктные формулировки для режима kpi_table."""
    if not text:
        return text

    normalized = _PRODUCT_MENTION_RE.sub("по KPI и методике их расчета", text)
    normalized = _PRODUCT_TOKEN_RE.sub("данным KPI", normalized)
    return normalized


def simplify_kpi_response(text: str) -> str:
    """Убирает шаблонную болтовню и делает ответ для KPI режима компактнее."""
    if not text:
        return text

    simplified = text.strip()

    if "\n" in simplified and _LEADING_KPI_INTRO_RE.match(simplified):
        for pattern in (
            _LEADING_KPI_INTRO_RE,
            _LEADING_HELP_LINE_RE,
            _LEADING_SPECIFY_LINE_RE,
        ):
            simplified = pattern.sub("", simplified, count=1).lstrip()

    simplified = _INSUFFICIENT_INFO_RE.sub(
        "Чтобы назвать ваши KPI, нужно уточнить должность и подразделение.",
        simplified,
    )
    simplified = _ASK_FOR_DETAILS_RE.sub(
        "Уточните отдел, подразделение или центр ответственности, и я назову точные KPI.",
        simplified,
    )
    simplified = _AMBIGUOUS_ROLE_RE.sub(_AMBIGUOUS_ROLE_TEXT, simplified)
    simplified = _FULL_LIST_RE.sub("", simplified)
    simplified = _MULTI_BLANK_RE.sub("\n\n", simplified)

    if _AMBIGUOUS_ROLE_TEXT in simplified and _LIST_MARKER_RE.search(simplified):
        return (
            _AMBIGUOUS_ROLE_TEXT
            + "\nУточните отдел, подразделение или направление, и я назову точные KPI."
        )

    return simplified.strip()


def enforce_kpi_guardrails(text: str) -> str:
    """Исправляет опасные трактовки KPI, которые нельзя выдавать пользователю."""
    if not text or not _NET_COMMISSION_RE.search(text):
        return text

    parts = _BLOCK_SPLIT_RE.split(text)
    corrected: list[str] = []
    net_window = 0
    replaced = False

    for index, part in enumerate(parts):
        if index % 2 == 1:
            corrected.append(part)
            continue

        current = part
        if not current:
            corrected.append(current)
            continue

        has_net_commission = bool(_NET_COMMISSION_RE.search(current))
        if has_net_commission:
            net_window = 2

        if (
            (has_net_commission or net_window > 0)
            and _INCOME_RE.search(current)
            and not _NEGATED_INCOME_RE.search(current)
        ):
            bullet_prefix = ""
            bullet_match = re.match(r"^(\s*(?:[-*•]\s+)?)", current)
            if bullet_match:
                bullet_prefix = bullet_match.group(1)
            current = bullet_prefix + _SAFE_NET_COMMISSION_TEXT
            replaced = True
            net_window = 0
        elif net_window > 0:
            net_window -= 1

        corrected.append(current)

    corrected_text = "".join(corrected)
    if replaced:
        return corrected_text

    if _INCOME_RE.search(corrected_text) and not _NEGATED_INCOME_RE.search(corrected_text):
        return corrected_text.rstrip() + "\n\n" + _SAFE_NET_COMMISSION_TEXT
    return corrected_text
