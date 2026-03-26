from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agents.utils import build_internal_invoke_config, extract_text

from .documents import GazDocumentsClient
from .logic import (
    build_followup,
    build_shortlist,
    classify_branch,
    clean_text,
    family_label,
    filter_sales_context,
    update_provisional_recommendations,
)
from .schemas import (
    ComparisonDigestResult,
    ComparisonProductDigest,
    ProductDimensionBaseline,
    ProductSnapshotEntry,
    ProductSnapshotResult,
    SalesDirectionDigest,
    SalesLandscapeResult,
    SourceCandidateRef,
)


LOG = logging.getLogger(__name__)
_VALID_SEARCH_INTENTS = {"overview", "compare", "specs", "financing", "objection"}
_VALID_BRANCHES = {
    "tco",
    "configuration",
    "comparison",
    "service_risk",
    "internal_approval",
    "passenger_route",
    "special_body",
    "special_conditions",
}
_COMPOSITE_SEARCH_LIMIT = 3
_COMPOSITE_READ_LIMIT = 4
_FAMILY_ALIASES = {
    "gazelle_next": "gazelle_next",
    "gazelle next": "gazelle_next",
    "gazelle неxt": "gazelle_next",
    "gazelle некст": "gazelle_next",
    "газель next": "gazelle_next",
    "газель некст": "gazelle_next",
    "gazelle_nn": "gazelle_nn",
    "gazelle nn": "gazelle_nn",
    "gazelle нн": "gazelle_nn",
    "газель nn": "gazelle_nn",
    "газель нн": "gazelle_nn",
    "gazelle_city": "gazelle_city",
    "gazelle city": "gazelle_city",
    "газель city": "gazelle_city",
    "газель сити": "gazelle_city",
    "sobol_nn": "sobol_nn",
    "sobol nn": "sobol_nn",
    "sobol нн": "sobol_nn",
    "соболь nn": "sobol_nn",
    "соболь нн": "sobol_nn",
    "sobol_4x4": "sobol_4x4",
    "sobol 4x4": "sobol_4x4",
    "sobol 4*4": "sobol_4x4",
    "соболь 4x4": "sobol_4x4",
    "соболь 4*4": "sobol_4x4",
    "соболь 4х4": "sobol_4x4",
    "sobol_business": "sobol_business",
    "sobol business": "sobol_business",
    "соболь бизнес": "sobol_business",
    "gazon_next": "gazon_next",
    "gazon next": "gazon_next",
    "gazon некст": "gazon_next",
    "газон next": "gazon_next",
    "газон некст": "gazon_next",
    "valdai": "valdai",
    "валдай": "valdai",
    "валдай next": "valdai",
    "валдай 8": "valdai",
    "sadko": "sadko",
    "садко": "sadko",
    "садко next": "sadko",
    "садко 9": "sadko",
    "vector_next": "vector_next",
    "vector next": "vector_next",
    "вектор next": "vector_next",
    "вектор некст": "vector_next",
    "citymax": "citymax",
    "city max": "citymax",
    "ситимакс": "citymax",
    "ситимакс 8": "citymax",
    "ситимакс 9": "citymax",
    "паз": "paz",
    "паз 3205": "paz",
    "паз 4234": "paz",
    "paz": "paz",
    "sat": "sat",
}
_UNAVAILABLE_STATUS = "unavailable"
_PARTIAL_STATUS = "partial"
_FAMILY_DISPLAY_LABELS = {
    "gazelle_next": "Газель NEXT",
    "gazelle_nn": "Газель NN",
    "gazelle_city": "Газель City",
    "sobol_nn": "Соболь NN",
    "sobol_4x4": "Соболь 4x4",
    "sobol_business": "Соболь Бизнес",
    "gazon_next": "Газон NEXT",
    "valdai": "Валдай",
    "sadko": "Садко",
    "vector_next": "Вектор NEXT",
    "citymax": "Ситимакс",
    "paz": "ПАЗ",
    "sat": "SAT",
}
_PRODUCT_TERM_GROUPS = {
    "газель": ("gazelle_next", "gazelle_nn", "gazelle_city"),
    "gazelle": ("gazelle_next", "gazelle_nn", "gazelle_city"),
    "соболь": ("sobol_nn", "sobol_4x4"),
    "sobol": ("sobol_nn", "sobol_4x4"),
}


def _normalize_family_text(value: Any) -> str:
    text = clean_text(value).casefold().replace("ё", "е").replace("_", " ").replace("-", " ")
    text = re.sub(r"(\d)\s*[xх*]\s*(\d)", r"\1x\2", text)
    return re.sub(r"\s+", " ", text).strip()


_NORMALIZED_FAMILY_ALIASES = {_normalize_family_text(key): value for key, value in _FAMILY_ALIASES.items()}
_KNOWN_FAMILY_IDS = tuple(dict.fromkeys(_NORMALIZED_FAMILY_ALIASES.values()))
_PRODUCT_TERM_TO_FAMILY_IDS: Dict[str, Tuple[str, ...]] = {
    alias: (family_id,) for alias, family_id in _NORMALIZED_FAMILY_ALIASES.items()
}
for alias, family_ids in _PRODUCT_TERM_GROUPS.items():
    _PRODUCT_TERM_TO_FAMILY_IDS[_normalize_family_text(alias)] = tuple(family_ids)
_FAMILY_ALIAS_PATTERNS: Tuple[Tuple[str, str, re.Pattern[str]], ...] = tuple(
    (
        alias,
        family_id,
        re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.IGNORECASE),
    )
    for alias, family_id in sorted(_NORMALIZED_FAMILY_ALIASES.items(), key=lambda item: (-len(item[0]), item[0]))
)
_PRODUCT_TERM_PATTERNS: Tuple[Tuple[str, Tuple[str, ...], re.Pattern[str]], ...] = tuple(
    (
        alias,
        family_ids,
        re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.IGNORECASE),
    )
    for alias, family_ids in sorted(_PRODUCT_TERM_TO_FAMILY_IDS.items(), key=lambda item: (-len(item[0]), item[0]))
)
_DIMENSION_ALIASES = {
    "dimensions": ("dimension", "dimensions", "size", "length", "width", "height", "габар", "длина", "ширина", "высота"),
    "payload": ("payload", "capacity", "грузопод", "нагруз", "масса"),
    "volume": ("volume", "объем", "m3", "м3"),
    "power": ("power", "horsepower", "hp", "л.с", "мощн"),
    "engine": ("engine", "torque", "двигат", "крут"),
    "fuel": ("fuel", "consumption", "расход", "l/100", "л/100"),
    "versions": ("version", "versions", "trim", "комплект", "верси", "модифик"),
    "finance": ("leasing", "credit", "finance", "лизинг", "кредит", "финанс"),
    "fit": ("fit", "use case", "task", "задач", "сценар", "маршрут"),
}
_GROUP_TRADEOFFS = {
    "ru": {
        "light_cargo": [
            "Strong for frequent city work, but final body type and payload margin still need narrowing.",
            "Usually the best starting direction for mixed short-haul delivery when heavy payload headroom is not the main constraint.",
        ],
        "medium_cargo": [
            "Better when volume or payload reserve matters, though it is typically less city-friendly than light cargo directions.",
            "Stronger for heavier logistics than for very light urban jobs.",
        ],
        "offroad_special": [
            "Best when terrain access or a specialized body matters more than a generic van format.",
            "The special-body type should be narrowed early, otherwise the comparison stays too wide.",
        ],
        "passenger": [
            "Selection depends more on route format, seating capacity, and saloon layout than on a raw model list.",
            "For passenger tasks, route profile and capacity are more important than discussing narrow trims too early.",
        ],
    },
    "en": {
        "light_cargo": [
            "Strong for frequent city work, but final body type and payload margin still need narrowing.",
            "Usually the best starting direction for mixed short-haul delivery when heavy payload headroom is not the main constraint.",
        ],
        "medium_cargo": [
            "Better when volume or payload reserve matters, though it is typically less city-friendly than light cargo directions.",
            "Stronger for heavier logistics than for very light urban jobs.",
        ],
        "offroad_special": [
            "Best when terrain access or a specialized body matters more than a generic van format.",
            "The special-body type should be narrowed early, otherwise the comparison stays too wide.",
        ],
        "passenger": [
            "Selection depends more on route format, seating capacity, and saloon layout than on a raw model list.",
            "For passenger tasks, route profile and capacity are more important than discussing narrow trims too early.",
        ],
    },
}
_GENERIC_ASSUMPTIONS = {
    "ru": "Точная модификация ещё не зафиксирована, поэтому digest опирается на базовые исполнения и верхнеуровневые материалы с самым сильным сигналом.",
    "en": "The exact modification is not fixed yet, so the digest is based on baseline variants and high-signal top-level materials.",
}
_DEFAULT_NARROWING = {
    "ru": "Следующий лучший шаг сужения — один недорогой уточняющий вопрос: тип кузова, приоритет объёма против грузоподъёмности или доминирующий маршрутный сценарий.",
    "en": "The next best narrowing step is one cheap clarification: body type, volume vs payload priority, or the dominant route pattern.",
}
_FINANCE_FALLBACK = {
    "ru": "Финансовое сравнение становится заметно конкретнее после сужения до 2–3 направлений и более понятной конфигурации.",
    "en": "The finance comparison becomes more concrete after narrowing to 2-3 directions and a clearer configuration.",
}


def parse_allowed_product_names_env(raw_value: str | None) -> List[str] | None:
    raw_text = clean_text(raw_value)
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("GAZ_AGENT_ALLOWED_PRODUCT_NAMES must be a valid JSON array of strings.") from exc
    if not isinstance(parsed, list):
        raise ValueError("GAZ_AGENT_ALLOWED_PRODUCT_NAMES must be a JSON array of strings.")

    allowed_family_ids: List[str] = []
    for item in parsed:
        if not isinstance(item, str) or not clean_text(item):
            raise ValueError("GAZ_AGENT_ALLOWED_PRODUCT_NAMES must contain only non-empty strings.")
        family_id = _normalize_family_token(item)
        if family_id not in _KNOWN_FAMILY_IDS:
            raise ValueError(f"Unknown product family in GAZ_AGENT_ALLOWED_PRODUCT_NAMES: {item}")
        if family_id not in allowed_family_ids:
            allowed_family_ids.append(family_id)
    return allowed_family_ids or None


def format_allowed_product_names(allowed_family_ids: Sequence[str] | None) -> List[str]:
    labels: List[str] = []
    for family_id in allowed_family_ids or ():
        label = _family_display_label(family_id)
        if label and label not in labels:
            labels.append(label)
    return labels


def _tool_message(content: Any, runtime: ToolRuntime | None) -> List[ToolMessage]:
    tool_call_id = runtime.tool_call_id if runtime else None
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return [ToolMessage(content=content, tool_call_id=tool_call_id)]



def _clamp_int(value: int | None, *, lower: int, upper: int, default: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(parsed, upper))



def _debug_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)



def _chunk_summaries(payload: Any) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []

    def add_summary(name: str, value: Any) -> None:
        text = _debug_text(value)
        summaries.append({"chunk": name, "preview": text[:64], "length": len(text)})

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    add_summary(f"{key}[{index}]", item)
            else:
                add_summary(key, value)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            add_summary(f"payload[{index}]", item)
    else:
        add_summary("payload", payload)
    return summaries



def _debug_tool_call(tool_name: str, args: Dict[str, Any]) -> None:
    print(f"[gaz_tool] {tool_name} args={json.dumps(args, ensure_ascii=False, default=str)}", flush=True)



def _debug_tool_result(tool_name: str, payload: Any) -> None:
    print(
        f"[gaz_tool] {tool_name} result_chunks={json.dumps(_chunk_summaries(payload), ensure_ascii=False, default=str)}",
        flush=True,
    )



def _append_tool_call(state: Dict[str, Any], tool_name: str) -> List[str]:
    existing = list(state.get("tool_calls_this_turn") or [])
    existing.append(tool_name)
    return existing


def _derive_child_thread_id(parent_thread_id: str, suffix: str) -> str:
    thread_id = clean_text(parent_thread_id)
    return f"{thread_id}{suffix}" if thread_id else suffix.lstrip(":")


def _extract_last_ai_text(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    for message in reversed(list(response.get("messages") or [])):
        if isinstance(message, AIMessage):
            return extract_text(message)
    return ""



def _normalize_guard_text(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value).lower()).strip()



def _build_focus_key(candidate_id: str, focus: str) -> str:
    return f"{_normalize_guard_text(candidate_id)}::{_normalize_guard_text(focus)}"



def _build_search_key(query: str, intent: str, families: Sequence[str], competitor: str) -> str:
    normalized_families = sorted(_normalize_guard_text(item) for item in families if _normalize_guard_text(item))
    parts = [
        _normalize_guard_text(intent),
        _normalize_guard_text(query),
        "|".join(normalized_families),
        _normalize_guard_text(competitor),
    ]
    return "||".join(parts)



def _append_runtime_warning(state: Dict[str, Any], *, stage: str, code: str, detail: str = "") -> List[Dict[str, Any]]:
    warnings = list(state.get("runtime_warnings") or [])
    warning: Dict[str, Any] = {"stage": stage, "code": code}
    if detail:
        warning["detail"] = detail
    warnings.append(warning)
    return warnings



def _append_tool_limit_hit(
    state: Dict[str, Any],
    *,
    tool_name: str,
    reason: str,
    candidate_id: str = "",
    focus_key: str = "",
    search_key: str = "",
) -> List[Dict[str, Any]]:
    hits = list(state.get("tool_limit_hits") or [])
    hit: Dict[str, Any] = {"tool_name": tool_name, "reason": reason}
    if candidate_id:
        hit["candidate_id"] = candidate_id
    if focus_key:
        hit["focus_key"] = focus_key
    if search_key:
        hit["search_key"] = search_key
    hits.append(hit)
    return hits



def _locale_key(locale: str) -> str:
    return "en" if clean_text(locale).lower().startswith("en") else "ru"



def _normalize_family_token(value: Any) -> str:
    text = _normalize_family_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text in _NORMALIZED_FAMILY_ALIASES:
        return _NORMALIZED_FAMILY_ALIASES[text]
    compact = text.replace(" ", "_")
    if compact in _KNOWN_FAMILY_IDS:
        return compact
    return compact


def _family_display_label(family_id: str) -> str:
    return _FAMILY_DISPLAY_LABELS.get(family_id, family_label(family_id))


def _resolve_product_term_value(value: Any) -> List[str]:
    text = _normalize_family_text(value)
    if not text:
        return []
    direct = _PRODUCT_TERM_TO_FAMILY_IDS.get(text)
    if direct:
        return list(dict.fromkeys(direct))
    resolved: List[str] = []
    for _alias, family_ids, pattern in _PRODUCT_TERM_PATTERNS:
        if not pattern.search(text):
            continue
        for family_id in family_ids:
            if family_id not in resolved:
                resolved.append(family_id)
    return resolved


def _resolve_requested_product_terms(
    requested_product_terms: Sequence[str] | None,
    *,
    fallback_text: str = "",
) -> Dict[str, Any]:
    explicit_terms = [clean_text(item) for item in requested_product_terms or [] if clean_text(item)]
    resolved_family_ids: List[str] = []
    unresolved_terms: List[str] = []
    if explicit_terms:
        for term in explicit_terms:
            resolved = _resolve_product_term_value(term)
            if resolved:
                for family_id in resolved:
                    if family_id not in resolved_family_ids:
                        resolved_family_ids.append(family_id)
            else:
                unresolved_terms.append(term)
        source_terms = explicit_terms
    else:
        source_terms = [clean_text(fallback_text)] if clean_text(fallback_text) else []
        for term in source_terms:
            for family_id in _resolve_product_term_value(term):
                if family_id not in resolved_family_ids:
                    resolved_family_ids.append(family_id)
    return {
        "explicit_terms": explicit_terms,
        "source_terms": source_terms,
        "resolved_family_ids": resolved_family_ids,
        "resolved_family_labels": [_family_display_label(family_id) for family_id in resolved_family_ids],
        "unresolved_terms": _dedupe_preserve(unresolved_terms),
    }


def _find_family_alias_matches(value: Any) -> List[Tuple[str, str]]:
    text = _normalize_family_text(value)
    matches: List[Tuple[str, str]] = []
    seen: set[str] = set()
    if not text:
        return matches
    for alias, family_id, pattern in _FAMILY_ALIAS_PATTERNS:
        if family_id in seen:
            continue
        if pattern.search(text):
            matches.append((family_id, alias))
            seen.add(family_id)
    return matches


def _extract_family_mentions_from_text(value: Any) -> List[str]:
    return [family_id for family_id, _alias in _find_family_alias_matches(value)]


def _normalize_family_list(values: Sequence[str] | None) -> List[str]:
    normalized: List[str] = []
    for value in values or []:
        family = _normalize_family_token(value)
        if family and family not in normalized:
            normalized.append(family)
    return normalized



def _candidate_families(candidate: Dict[str, Any]) -> List[str]:
    families: List[str] = []
    for item in ((candidate.get("metadata") or {}).get("product_families") or []):
        family = _normalize_family_token(item)
        if family and family not in families:
            families.append(family)
    if families:
        return families
    for family in _extract_family_mentions_from_text(candidate.get("title")):
        if family not in families:
            families.append(family)
    return families



def _candidate_matches_family(candidate: Dict[str, Any], family: str) -> bool:
    return family in _candidate_families(candidate)



def _candidate_ref(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return SourceCandidateRef(
        candidate_id=str(candidate.get("candidate_id") or ""),
        title=str(candidate.get("title") or candidate.get("candidate_id") or ""),
        doc_kind=str(candidate.get("doc_kind") or "general"),
    ).model_dump()


def _dedupe_preserve(values: Sequence[str] | None) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = clean_text(value)
        if not text:
            continue
        marker = text.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(text)
    return result


def _localized_family_filter_message(locale: str, unsupported_names: Sequence[str] | None, *, partial: bool) -> str:
    names = ", ".join(_dedupe_preserve(unsupported_names))
    if _locale_key(locale) == "en":
        if partial:
            return f"Information is available only for supported product families. Not available for: {names}."
        return f"Information is not available for the following product families: {names}."
    if partial:
        return f"Информация доступна только по поддерживаемым продуктовым семействам. Недоступно для: {names}."
    return f"Информация недоступна для следующих продуктовых семейств: {names}."


def _family_filter_notice(locale: str, unsupported_names: Sequence[str] | None, *, partial: bool) -> Dict[str, Any] | None:
    normalized_names = _dedupe_preserve(unsupported_names)
    if not normalized_names:
        return None
    return {
        "status": _PARTIAL_STATUS if partial else _UNAVAILABLE_STATUS,
        "message": _localized_family_filter_message(locale, normalized_names, partial=partial),
        "unsupported_product_names": normalized_names,
    }


def _apply_family_filter_notice(payload: Dict[str, Any], locale: str, unsupported_names: Sequence[str] | None) -> Dict[str, Any]:
    notice = _family_filter_notice(locale, unsupported_names, partial=True)
    if notice:
        payload["availability_notice"] = notice
    return payload


def _filter_candidates_for_requested_families(
    candidates: Sequence[Dict[str, Any]] | None,
    requested_family_ids: Sequence[str] | None,
    *,
    keep_unknown: bool = False,
) -> List[Dict[str, Any]]:
    normalized_ids = _normalize_family_list(requested_family_ids)
    if not normalized_ids:
        return list(candidates or [])
    requested = set(normalized_ids)
    filtered: List[Dict[str, Any]] = []
    for candidate in candidates or []:
        families = _candidate_families(candidate)
        if not families:
            if keep_unknown:
                filtered.append(candidate)
            continue
        if requested.intersection(families):
            filtered.append(candidate)
    return filtered


def _filter_baseline_groups_for_requested_families(
    baseline: Dict[str, Any],
    requested_family_ids: Sequence[str] | None,
) -> Dict[str, Any]:
    normalized_ids = _normalize_family_list(requested_family_ids)
    if not normalized_ids:
        return dict(baseline)
    requested = set(normalized_ids)
    product_groups: List[Dict[str, Any]] = []
    for group in baseline.get("product_groups") or []:
        families = list(group.get("families") or [])
        kept_families = [family for family in families if _normalize_family_token(family) in requested]
        if not kept_families:
            continue
        updated_group = dict(group)
        updated_group["families"] = kept_families
        product_groups.append(updated_group)
    updated_baseline = dict(baseline)
    updated_baseline["product_groups"] = product_groups
    return updated_baseline


def _sanitize_competitor_signal(raw_value: Any, requested_family_ids: Sequence[str] | None) -> str:
    text = clean_text(raw_value)
    if not text:
        return ""
    normalized_requested = _normalize_family_list(requested_family_ids)
    if not normalized_requested:
        return text
    requested = set(normalized_requested)
    matches = _find_family_alias_matches(text)
    if matches:
        kept = [_family_display_label(family_id) for family_id, _alias in matches if family_id in requested]
        return ", ".join(_dedupe_preserve(kept))
    normalized = _normalize_family_token(text)
    return text if normalized in requested else ""


def _sanitize_text_terms(text: str, terms_to_strip: Sequence[str] | None) -> str:
    sanitized = clean_text(text)
    for term in sorted(_dedupe_preserve(terms_to_strip), key=lambda item: (-len(item), item.casefold())):
        sanitized = re.sub(
            rf"(?<!\w){re.escape(term)}(?!\w)",
            " ",
            sanitized,
            flags=re.IGNORECASE,
        )
    sanitized = re.sub(r"\b(and|or|и|или)\b\s+(?=(on|for|about|по)\b)", " ", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\b(and|or|и|или)\b(?=\s*(?:[,;:/-]|$))", " ", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s*[,;:/]\s*", " ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" ,;:/-")
    return sanitized


def _build_information_tool_family_guard(
    locale: str,
    allowed_family_ids: Sequence[str] | None,
    *,
    families: Sequence[str] | None = None,
    competitor: Any = "",
    question: str = "",
    query: str = "",
    topic: str = "",
    use_case: str = "",
    focus: str = "",
    problem_focus: str = "",
    state: Dict[str, Any] | None = None,
    candidate: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_allowed = _normalize_family_list(allowed_family_ids)
    if not normalized_allowed:
        return {
            "filter_enabled": False,
            "mode": "pass",
            "has_explicit_request": False,
            "supported_family_ids": [],
            "unsupported_names": [],
            "unsupported_terms": [],
        }

    allowed = set(normalized_allowed)
    state = state or {}
    supported_family_ids: List[str] = []
    unsupported_names: List[str] = []
    unsupported_terms: List[str] = []
    has_explicit_request = False

    def add_supported(family_id: str) -> None:
        if family_id and family_id not in supported_family_ids:
            supported_family_ids.append(family_id)

    def add_unsupported(name: Any) -> None:
        text = clean_text(name)
        if text and text not in unsupported_names:
            unsupported_names.append(text)

    def add_unsupported_term(value: Any) -> None:
        text = clean_text(value)
        if text and text not in unsupported_terms:
            unsupported_terms.append(text)

    def handle_structured_value(raw_value: Any, *, treat_unknown_as_unsupported: bool) -> None:
        nonlocal has_explicit_request
        text = clean_text(raw_value)
        if not text:
            return
        has_explicit_request = True
        normalized = _normalize_family_token(text)
        if normalized in _KNOWN_FAMILY_IDS:
            if normalized in allowed:
                add_supported(normalized)
            else:
                add_unsupported(_family_display_label(normalized) if "_" in text else text)
                add_unsupported_term(text)
                add_unsupported_term(_family_display_label(normalized))
            return
        if treat_unknown_as_unsupported:
            add_unsupported(text)
            add_unsupported_term(text)

    for raw_family in families or []:
        handle_structured_value(raw_family, treat_unknown_as_unsupported=True)

    for raw_family in ((candidate or {}).get("metadata") or {}).get("product_families") or []:
        handle_structured_value(raw_family, treat_unknown_as_unsupported=True)

    handle_structured_value(competitor, treat_unknown_as_unsupported=True)
    handle_structured_value((state.get("slots") or {}).get("competitor"), treat_unknown_as_unsupported=True)

    text_sources = (
        question,
        query,
        topic,
        use_case,
        focus,
        problem_focus,
        state.get("problem_summary"),
        (candidate or {}).get("title"),
    )
    for raw_text in text_sources:
        for family_id, alias in _find_family_alias_matches(raw_text):
            has_explicit_request = True
            if family_id in allowed:
                add_supported(family_id)
            else:
                add_unsupported(_family_display_label(family_id))
                add_unsupported_term(alias)
                add_unsupported_term(_family_display_label(family_id))

    mode = "pass"
    if has_explicit_request and unsupported_names and not supported_family_ids:
        mode = _UNAVAILABLE_STATUS
    elif has_explicit_request and unsupported_names and supported_family_ids:
        mode = _PARTIAL_STATUS

    return {
        "filter_enabled": True,
        "mode": mode,
        "has_explicit_request": has_explicit_request,
        "supported_family_ids": supported_family_ids,
        "unsupported_names": _dedupe_preserve(unsupported_names),
        "unsupported_terms": _dedupe_preserve(unsupported_terms),
    }


def _family_filter_block_update(
    state: Dict[str, Any],
    runtime: ToolRuntime | None,
    *,
    locale: str,
    tool_name: str,
    unsupported_names: Sequence[str] | None,
) -> Dict[str, Any]:
    payload = _family_filter_notice(locale, unsupported_names, partial=False) or {
        "status": _UNAVAILABLE_STATUS,
        "message": "Information is unavailable.",
    }
    return {
        "runtime_warnings": _append_runtime_warning(
            state,
            stage=f"gaz:{tool_name}",
            code="unsupported_product_family",
            detail=", ".join(_dedupe_preserve(unsupported_names)),
        ),
        "tool_calls_this_turn": _append_tool_call(state, tool_name),
        "messages": _tool_message(payload, runtime),
    }



def _source_candidate_refs(candidates: Sequence[Dict[str, Any]] | None, limit: int = 6) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates or []:
        candidate_id = clean_text(candidate.get("candidate_id"))
        if not candidate_id or candidate_id in seen:
            continue
        refs.append(_candidate_ref(candidate))
        seen.add(candidate_id)
        if len(refs) >= limit:
            break
    return refs



def _merge_candidates(existing: Sequence[Dict[str, Any]] | None, new: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for candidate in list(existing or []) + list(new or []):
        candidate_id = clean_text(candidate.get("candidate_id"))
        if not candidate_id:
            continue
        if candidate_id not in merged:
            order.append(candidate_id)
            merged[candidate_id] = dict(candidate)
    return [merged[candidate_id] for candidate_id in order]



def _merge_reads(existing: Sequence[Dict[str, Any]] | None, new: Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}
    order: List[tuple[str, str]] = []
    for item in list(existing or []) + list(new or []):
        key = (clean_text(item.get("candidate_id")), clean_text(item.get("focus")))
        if not key[0]:
            continue
        if key not in merged:
            order.append(key)
            merged[key] = dict(item)
    return [merged[key] for key in order]



def _pick_top_families(candidates: Sequence[Dict[str, Any]] | None, explicit_families: Sequence[str] | None, limit: int) -> List[str]:
    normalized = [family for family in _normalize_family_list(explicit_families) if family]
    if normalized:
        return normalized[: max(1, limit)]
    counter: Counter[str] = Counter()
    for candidate in candidates or []:
        for family in _candidate_families(candidate):
            counter[family] += 1
    return [family for family, _count in counter.most_common(max(1, limit))]



def _pick_candidates_for_family(candidates: Sequence[Dict[str, Any]] | None, family: str, limit: int) -> List[Dict[str, Any]]:
    matches = [candidate for candidate in candidates or [] if _candidate_matches_family(candidate, family)]
    return matches[: max(1, limit)]



def _infer_axes(query: str, dimensions: Sequence[str] | None) -> List[str]:
    explicit = [clean_text(item) for item in dimensions or [] if clean_text(item)]
    if explicit:
        return explicit[:4]
    lowered = clean_text(query).lower()
    axes: List[str] = []
    for axis, aliases in _DIMENSION_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            axes.append(axis)
    if axes:
        return axes[:4]
    return ["fit", "specs", "finance"]



def _split_excerpt_text(text: str) -> List[str]:
    parts = re.split(r"[\r\n]+|(?<=[.!?])\s+", clean_text(text))
    cleaned: List[str] = []
    for part in parts:
        normalized = re.sub(r"\s+", " ", part).strip(" -?")
        if len(normalized) >= 12:
            cleaned.append(normalized)
    return cleaned



def _extract_focus_points(reads: Sequence[Dict[str, Any]] | None, keywords: Sequence[str] | None, limit: int = 3) -> List[str]:
    scored: Dict[str, int] = {}
    lowered_keywords = [clean_text(item).lower() for item in keywords or [] if clean_text(item)]
    for read in reads or []:
        for excerpt in read.get("excerpts") or []:
            excerpt_text = clean_text(excerpt.get("excerpt"))
            for part in _split_excerpt_text(excerpt_text):
                lowered = part.lower()
                score = 1
                for keyword in lowered_keywords:
                    if keyword in lowered:
                        score += 2
                if re.search(r"\d", lowered):
                    score += 1
                if score > 1:
                    scored[part] = max(scored.get(part, 0), score)
    ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return [text[:220] for text, _score in ordered[: max(1, limit)]]



def _summarize_candidate_rationales(candidates: Sequence[Dict[str, Any]] | None, limit: int = 2) -> List[str]:
    values: List[str] = []
    for candidate in candidates or []:
        rationale = clean_text(candidate.get("rationale"))
        if rationale and rationale not in values:
            values.append(rationale)
        if len(values) >= limit:
            break
    return values



def _recommended_next_narrowing(locale: str, state: Dict[str, Any]) -> str:
    locale_key = _locale_key(locale)
    slots = state.get("slots") or {}
    if not clean_text(slots.get("body_type")) and clean_text(slots.get("transport_type")) != "passenger":
        return {
            "ru": "Следующий лучший шаг сужения — выбрать тип кузова или надстройки, чтобы перейти от общих направлений к рабочим вариантам и документам.",
            "en": "The next best narrowing step is the body or superstructure type so the conversation can move from broad directions to workable options.",
        }[locale_key]
    if not clean_text(slots.get("capacity_or_payload")):
        return {
            "ru": "Следующий лучший шаг — понять, что важнее: объём, грузоподъёмность или универсальность под смешанные задачи.",
            "en": "The next best narrowing step is whether volume, payload, or mixed-use flexibility matters most.",
        }[locale_key]
    return _DEFAULT_NARROWING[locale_key]



def _derive_financial_angle(locale: str, finance_options: Sequence[str] | None, candidates: Sequence[Dict[str, Any]] | None) -> str:
    options = [clean_text(item) for item in finance_options or [] if clean_text(item)]
    if any(clean_text(candidate.get("doc_kind")) in {"tco", "approval"} for candidate in candidates or []):
        if options:
            return options[0]
    if len(options) >= 2:
        return options[1]
    if options:
        return options[0]
    return _FINANCE_FALLBACK[_locale_key(locale)]



def _search_with_trace(
    docs_client: GazDocumentsClient,
    trace: Dict[str, Any],
    *,
    query: str,
    intent: str,
    families: Sequence[str] | None,
    competitor: str,
    top_k: int,
) -> Dict[str, Any]:
    response = docs_client.search_sales_materials(
        query=query,
        intent=intent,
        families=list(families or []),
        competitor=competitor,
        top_k=top_k,
    )
    trace.setdefault("search_calls", []).append(
        {
            "intent": intent,
            "query": query,
            "families": list(families or []),
            "competitor": competitor,
            "top_k": top_k,
            "candidate_ids": [candidate.get("candidate_id") for candidate in response.get("candidates") or [] if candidate.get("candidate_id")],
        }
    )
    return response



def _read_with_trace(
    docs_client: GazDocumentsClient,
    trace: Dict[str, Any],
    *,
    candidate_id: str,
    focus: str,
    max_segments: int,
) -> Dict[str, Any]:
    response = docs_client.read_material(candidate_id=candidate_id, focus=focus, max_segments=max_segments)
    trace.setdefault("read_calls", []).append(
        {
            "candidate_id": candidate_id,
            "focus": focus,
            "max_segments": max_segments,
            "excerpt_count": len(response.get("excerpts") or []),
        }
    )
    return response



def _composite_update(
    state: Dict[str, Any],
    *,
    tool_name: str,
    digest_key: str,
    digest_payload: Dict[str, Any],
    query: str,
    new_candidates: Sequence[Dict[str, Any]] | None,
    new_reads: Sequence[Dict[str, Any]] | None,
    research_layer: str,
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    merged_candidates = _merge_candidates(state.get("material_candidates") or [], new_candidates or [])
    merged_reads = _merge_reads(state.get("material_reads") or [], new_reads or [])
    allowed_ids = [candidate.get("candidate_id") for candidate in merged_candidates if candidate.get("candidate_id")]
    research_status = dict(state.get("research_status") or {})
    queries = list(research_status.get("queries") or [])
    if clean_text(query) and clean_text(query) not in queries:
        queries.append(clean_text(query))
    documents_touched = list(research_status.get("documents_touched") or [])
    for item in merged_reads:
        candidate_id = clean_text(item.get("candidate_id"))
        if candidate_id and candidate_id not in documents_touched:
            documents_touched.append(candidate_id)
    composite_traces = list(state.get("composite_tool_traces") or [])
    composite_traces.append(trace)
    digest_history = list(state.get(digest_key) or [])
    digest_history.append(digest_payload)
    return {
        digest_key: digest_history,
        "composite_tool_traces": composite_traces,
        "material_candidates": merged_candidates,
        "material_reads": merged_reads,
        "allowed_material_ids": allowed_ids,
        "provisional_recommendations": update_provisional_recommendations(merged_candidates, state.get("provisional_recommendations") or []),
        "research_status": {
            **research_status,
            "queries": queries,
            "candidate_count": len(merged_candidates),
            "documents_touched": documents_touched,
            "has_prior_search": bool(research_status.get("has_prior_search")) or bool(trace.get("search_calls")),
            "has_prior_read": bool(research_status.get("has_prior_read")) or bool(new_reads),
            "last_composite_tool": tool_name,
        },
        "research_layer": research_layer,
        "tool_calls_this_turn": _append_tool_call(state, tool_name),
    }



def _landscape_digest(
    locale: str,
    state: Dict[str, Any],
    docs_client: GazDocumentsClient,
    topic: str,
    audience: str,
    use_case: str,
    focus: str,
    *,
    allowed_family_ids: Sequence[str] | None = None,
    requested_family_ids: Sequence[str] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    query = " ".join(part for part in [clean_text(topic), clean_text(use_case), clean_text(focus)] if part)
    baseline = filter_sales_context(locale, topic=query, max_groups=4)
    target_family_ids = _normalize_family_list(requested_family_ids) or _normalize_family_list(allowed_family_ids)
    baseline = _filter_baseline_groups_for_requested_families(baseline, target_family_ids)
    finance_options = list(baseline.get("finance_options") or [])[:3]
    trace: Dict[str, Any] = {"tool_name": "get_sales_landscape", "query": query, "focus": clean_text(focus)}
    responses: List[Dict[str, Any]] = []
    if query:
        responses.append(_search_with_trace(docs_client, trace, query=query, intent="overview", families=[], competitor="", top_k=6))
        lowered = query.lower()
        if len(responses) < _COMPOSITE_SEARCH_LIMIT and any(term in lowered for term in ("finance", "leasing", "credit", "лизинг", "кредит", "финанс")):
            responses.append(_search_with_trace(docs_client, trace, query=query, intent="financing", families=[], competitor="", top_k=4))
    candidate_pool = _merge_candidates([], [candidate for response in responses for candidate in response.get("candidates") or []])
    candidate_pool = _filter_candidates_for_requested_families(
        candidate_pool,
        target_family_ids,
        keep_unknown=not _normalize_family_list(requested_family_ids),
    )
    read_responses: List[Dict[str, Any]] = []
    for group in (baseline.get("product_groups") or [])[: min(2, _COMPOSITE_READ_LIMIT)]:
        group_families = {_normalize_family_token(item) for item in group.get("families") or []}
        matching = [candidate for candidate in candidate_pool if group_families.intersection(_candidate_families(candidate))]
        if not matching:
            continue
        read_responses.append(
            _read_with_trace(
                docs_client,
                trace,
                candidate_id=matching[0].get("candidate_id"),
                focus=clean_text(focus) or "main characteristics, use cases, and tradeoffs for this direction",
                max_segments=2,
            )
        )
        if len(read_responses) >= _COMPOSITE_READ_LIMIT:
            break

    directions: List[Dict[str, Any]] = []
    for group in (baseline.get("product_groups") or [])[:4]:
        group_families = {_normalize_family_token(item) for item in group.get("families") or []}
        matching = [candidate for candidate in candidate_pool if group_families.intersection(_candidate_families(candidate))]
        matching_ids = {candidate.get("candidate_id") for candidate in matching if candidate.get("candidate_id")}
        matching_reads = [item for item in read_responses if item.get("candidate_id") in matching_ids]
        tradeoffs = list(_GROUP_TRADEOFFS[_locale_key(locale)].get(group.get("group_id"), []))[:2]
        directions.append(
            SalesDirectionDigest(
                group_id=str(group.get("group_id") or "direction"),
                title=str(group.get("title") or "Direction"),
                families=list(group.get("families") or []),
                main_characteristics=list(group.get("main_characteristics") or [])[:3],
                typical_use_cases=list(group.get("main_characteristics") or [])[:2],
                financial_angle=_derive_financial_angle(locale, finance_options, matching),
                key_tradeoffs=tradeoffs,
                evidence_highlights=_extract_focus_points(matching_reads, [clean_text(focus), clean_text(use_case), clean_text(topic)], limit=2),
                source_candidates=[SourceCandidateRef(**item) for item in _source_candidate_refs(matching, limit=3)],
            ).model_dump()
        )

    payload = SalesLandscapeResult(
        topic=clean_text(topic),
        audience=clean_text(audience),
        use_case=clean_text(use_case),
        focus=clean_text(focus),
        directions=directions,
        finance_options=finance_options,
        recommended_next_narrowing=_recommended_next_narrowing(locale, state),
        source_candidates=[SourceCandidateRef(**item) for item in _source_candidate_refs(candidate_pool, limit=6)],
    ).model_dump()
    update = _composite_update(
        state,
        tool_name="get_sales_landscape",
        digest_key="sales_digests",
        digest_payload=payload,
        query=query,
        new_candidates=candidate_pool,
        new_reads=read_responses,
        research_layer="sales_landscape",
        trace=trace,
    )
    update["sales_context_baseline"] = {
        "topic": clean_text(topic),
        "audience": clean_text(audience),
        "use_case": clean_text(use_case),
        "product_groups": baseline.get("product_groups") or [],
        "finance_options": finance_options,
    }
    return payload, update



def _comparison_assumptions(locale: str, explicit_families: Sequence[str] | None, dimensions: Sequence[str] | None) -> List[str]:
    locale_key = _locale_key(locale)
    assumptions = [_GENERIC_ASSUMPTIONS[locale_key]]
    if not explicit_families:
        assumptions.append({
            "ru": "Список сравниваемых направлений был выведен из контекста запроса и найденных материалов, а не из полностью зафиксированного списка клиента.",
            "en": "The compared families were inferred from context and retrieved materials, not from a fully fixed client list.",
        }[locale_key])
    if not dimensions:
        assumptions.append({
            "ru": "Оси сравнения были выведены из запроса и sales-контекста, без полностью зафиксированной конфигурации.",
            "en": "The comparison axes were inferred from the request and sales context without a narrowly fixed configuration.",
        }[locale_key])
    return assumptions



def _compare_digest(
    locale: str,
    state: Dict[str, Any],
    docs_client: GazDocumentsClient,
    query: str,
    families: Sequence[str] | None,
    competitor: str,
    dimensions: Sequence[str] | None,
    top_families: int,
    *,
    allowed_family_ids: Sequence[str] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    query_text = clean_text(query) or clean_text(state.get("problem_summary"))
    allowed_ids = _normalize_family_list(allowed_family_ids)
    allowed_set = set(allowed_ids)
    normalized_families = _normalize_family_list(families)
    if allowed_set:
        normalized_families = [family for family in normalized_families if family in allowed_set]
    axes = _infer_axes(query_text, dimensions)
    trace: Dict[str, Any] = {"tool_name": "compare_product_directions", "query": query_text, "axes": axes}
    responses: List[Dict[str, Any]] = []
    responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="compare", families=normalized_families, competitor=clean_text(competitor), top_k=6))
    if clean_text(competitor) and len(responses) < _COMPOSITE_SEARCH_LIMIT:
        responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="objection", families=normalized_families, competitor=clean_text(competitor), top_k=4))
    if axes and len(responses) < _COMPOSITE_SEARCH_LIMIT:
        responses.append(_search_with_trace(docs_client, trace, query=f"{query_text} {' '.join(axes)}", intent="specs", families=normalized_families, competitor=clean_text(competitor), top_k=4))
    candidate_pool = _merge_candidates([], [candidate for response in responses for candidate in response.get("candidates") or []])
    candidate_pool = _filter_candidates_for_requested_families(
        candidate_pool,
        normalized_families or allowed_ids,
        keep_unknown=not normalized_families,
    )
    state_recommendations = _normalize_family_list(state.get("provisional_recommendations") or [])
    if allowed_set:
        state_recommendations = [family for family in state_recommendations if family in allowed_set]
    selected_families = _pick_top_families(candidate_pool, normalized_families or state_recommendations, top_families)
    if allowed_set:
        selected_families = [family for family in selected_families if family in allowed_set]
    if normalized_families:
        selected_families = [family for family in selected_families if family in normalized_families]
    read_responses: List[Dict[str, Any]] = []
    for family in selected_families:
        for candidate in _pick_candidates_for_family(candidate_pool, family, limit=2):
            if len(read_responses) >= _COMPOSITE_READ_LIMIT:
                break
            read_responses.append(
                _read_with_trace(
                    docs_client,
                    trace,
                    candidate_id=candidate.get("candidate_id"),
                    focus=f"comparison differences for {family_label(family)} with emphasis on {', '.join(axes)}; {query_text}",
                    max_segments=2,
                )
            )
        if len(read_responses) >= _COMPOSITE_READ_LIMIT:
            break

    finance_options = list(filter_sales_context(locale, topic=query_text, max_groups=3).get("finance_options") or [])[:2]
    products_compared: List[Dict[str, Any]] = []
    high_level_differences: List[str] = []
    for family in selected_families:
        family_candidates = _pick_candidates_for_family(candidate_pool, family, limit=3)
        family_ids = {candidate.get("candidate_id") for candidate in family_candidates if candidate.get("candidate_id")}
        family_reads = [item for item in read_responses if item.get("candidate_id") in family_ids]
        differentiators = _extract_focus_points(family_reads, axes + [clean_text(competitor), query_text], limit=3)
        if not differentiators:
            differentiators = _summarize_candidate_rationales(family_candidates, limit=2)
        product = ComparisonProductDigest(
            family_id=family,
            label=family_label(family),
            main_use_cases=_summarize_candidate_rationales(family_candidates, limit=2),
            differentiators=differentiators,
            financial_angle=_derive_financial_angle(locale, finance_options, family_candidates),
            source_candidates=[SourceCandidateRef(**item) for item in _source_candidate_refs(family_candidates, limit=3)],
        ).model_dump()
        products_compared.append(product)
        high_level_differences.append(f"{family_label(family)}: {'; '.join(differentiators[:2]) if differentiators else _FINANCE_FALLBACK[_locale_key(locale)]}")

    payload = ComparisonDigestResult(
        query=query_text,
        products_compared=products_compared,
        comparison_axes=axes,
        high_level_differences=high_level_differences,
        assumptions=_comparison_assumptions(locale, normalized_families, dimensions),
        source_candidates=[SourceCandidateRef(**item) for item in _source_candidate_refs(candidate_pool, limit=6)],
    ).model_dump()
    update = _composite_update(
        state,
        tool_name="compare_product_directions",
        digest_key="comparison_digests",
        digest_payload=payload,
        query=query_text,
        new_candidates=candidate_pool,
        new_reads=read_responses,
        research_layer="comparison_digest",
        trace=trace,
    )
    if selected_families:
        update["provisional_recommendations"] = list(dict.fromkeys(selected_families + list(state.get("provisional_recommendations") or [])))[:3]
    return payload, update



def _snapshot_assumptions(locale: str, explicit_families: Sequence[str] | None, dimensions: Sequence[str] | None) -> List[str]:
    locale_key = _locale_key(locale)
    assumptions = [_GENERIC_ASSUMPTIONS[locale_key]]
    if not explicit_families:
        assumptions.append({
            "ru": "Направления для snapshot выведены из найденных материалов, поэтому финальный набор моделей ещё полезно уточнить одним следующим вопросом.",
            "en": "The snapshot directions were inferred from retrieved materials, so the final model set still benefits from one more narrowing step.",
        }[locale_key])
    if not dimensions:
        assumptions.append({
            "ru": "Это показывает наиболее вероятные характеристики под ваш запрос, а не полный технический паспорт каждой модификации.",
            "en": "This highlights the most likely characteristics for the ask, not a full technical passport of every modification.",
        }[locale_key])
    return assumptions



def _snapshot_digest(
    locale: str,
    state: Dict[str, Any],
    docs_client: GazDocumentsClient,
    query: str,
    families: Sequence[str] | None,
    dimensions: Sequence[str] | None,
    competitor: str,
    max_products: int,
    *,
    allowed_family_ids: Sequence[str] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    query_text = clean_text(query) or clean_text(state.get("problem_summary"))
    allowed_ids = _normalize_family_list(allowed_family_ids)
    allowed_set = set(allowed_ids)
    normalized_families = _normalize_family_list(families)
    if allowed_set:
        normalized_families = [family for family in normalized_families if family in allowed_set]
    axes = _infer_axes(query_text, dimensions)
    trace: Dict[str, Any] = {"tool_name": "collect_product_snapshot", "query": query_text, "dimensions": axes}
    responses: List[Dict[str, Any]] = []
    responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="specs", families=normalized_families, competitor=clean_text(competitor), top_k=6))
    if clean_text(competitor) and len(responses) < _COMPOSITE_SEARCH_LIMIT:
        responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="compare", families=normalized_families, competitor=clean_text(competitor), top_k=4))
    if not normalized_families and len(responses) < _COMPOSITE_SEARCH_LIMIT:
        responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="overview", families=[], competitor=clean_text(competitor), top_k=4))
    candidate_pool = _merge_candidates([], [candidate for response in responses for candidate in response.get("candidates") or []])
    candidate_pool = _filter_candidates_for_requested_families(
        candidate_pool,
        normalized_families or allowed_ids,
        keep_unknown=not normalized_families,
    )
    state_recommendations = _normalize_family_list(state.get("provisional_recommendations") or [])
    if allowed_set:
        state_recommendations = [family for family in state_recommendations if family in allowed_set]
    selected_families = _pick_top_families(candidate_pool, normalized_families or state_recommendations, max_products)
    if allowed_set:
        selected_families = [family for family in selected_families if family in allowed_set]
    if normalized_families:
        selected_families = [family for family in selected_families if family in normalized_families]
    read_responses: List[Dict[str, Any]] = []
    for family in selected_families:
        for candidate in _pick_candidates_for_family(candidate_pool, family, limit=2):
            if len(read_responses) >= _COMPOSITE_READ_LIMIT:
                break
            read_responses.append(
                _read_with_trace(
                    docs_client,
                    trace,
                    candidate_id=candidate.get("candidate_id"),
                    focus=f"collect numeric or baseline characteristics for {family_label(family)}: {', '.join(axes)}; {query_text}",
                    max_segments=3,
                )
            )
        if len(read_responses) >= _COMPOSITE_READ_LIMIT:
            break

    products: List[Dict[str, Any]] = []
    baselines: List[Dict[str, Any]] = []
    for family in selected_families:
        family_candidates = _pick_candidates_for_family(candidate_pool, family, limit=3)
        family_ids = {candidate.get("candidate_id") for candidate in family_candidates if candidate.get("candidate_id")}
        family_reads = [item for item in read_responses if item.get("candidate_id") in family_ids]
        facts = _extract_focus_points(family_reads, axes + [query_text, clean_text(competitor)], limit=4)
        if not facts:
            facts = _summarize_candidate_rationales(family_candidates, limit=2)
        products.append(
            ProductSnapshotEntry(
                family_id=family,
                label=family_label(family),
                facts=facts,
                source_candidates=[SourceCandidateRef(**item) for item in _source_candidate_refs(family_candidates, limit=3)],
            ).model_dump()
        )
        for axis in axes[:3]:
            evidence = _extract_focus_points(family_reads, [axis], limit=2)
            if evidence:
                baselines.append(
                    ProductDimensionBaseline(
                        family_id=family,
                        label=family_label(family),
                        dimension=axis,
                        evidence=evidence,
                    ).model_dump()
                )

    payload = ProductSnapshotResult(
        query=query_text,
        dimensions_requested=axes,
        products=products,
        value_ranges_or_baselines=baselines,
        assumptions=_snapshot_assumptions(locale, normalized_families, dimensions),
        source_candidates=[SourceCandidateRef(**item) for item in _source_candidate_refs(candidate_pool, limit=6)],
    ).model_dump()
    update = _composite_update(
        state,
        tool_name="collect_product_snapshot",
        digest_key="product_snapshots",
        digest_payload=payload,
        query=query_text,
        new_candidates=candidate_pool,
        new_reads=read_responses,
        research_layer="product_snapshot",
        trace=trace,
    )
    if selected_families:
        update["provisional_recommendations"] = list(dict.fromkeys(selected_families + list(state.get("provisional_recommendations") or [])))[:3]
    return payload, update



def build_classify_problem_branch_tool(locale: str = "ru"):
    @tool("classify_problem_branch", parse_docstring=True)
    def classify_problem_branch(reasoning_note: str = "", runtime: ToolRuntime = None) -> Command:
        """Classify the current customer case into one active business branch or a conflict.

        Use this tool when the conversation already contains enough problem context to decide
        which sales branch is currently dominant, such as configuration, comparison,
        passenger route, special body, or service-risk handling.

        Args:
            reasoning_note: Optional short note explaining why branch locking is needed now or
                what ambiguity should be resolved.
        """
        _debug_tool_call("classify_problem_branch", {"reasoning_note": reasoning_note})
        state = runtime.state if runtime else {}
        active_branch, branch_conflict, reasoning = classify_branch(
            state.get("slots") or {},
            state.get("problem_summary") or "",
        )
        payload: Dict[str, Any] = {
            "active_branch": None if branch_conflict else active_branch,
            "branch_conflict": branch_conflict,
            "tool_calls_this_turn": _append_tool_call(state, "classify_problem_branch"),
        }
        content = {
            "status": "conflict" if branch_conflict else "locked",
            "branch": active_branch,
            "branches": branch_conflict,
            "reasoning": reasoning,
        }
        _debug_tool_result("classify_problem_branch", content)
        payload["messages"] = _tool_message(content, runtime)
        return Command(update=payload)

    return classify_problem_branch



def build_sales_catalog_overview_tool(locale: str = "ru"):
    @tool("get_sales_catalog_overview", parse_docstring=True)
    def get_sales_catalog_overview(
        topic: str = "",
        audience: str = "",
        use_case: str = "",
        runtime: ToolRuntime = None,
    ) -> Command:
        """Return a broad product and financing overview for early sales discovery.

        Use this tool when the customer ask is still high level and you need a broad portfolio
        view instead of narrow product facts. It is especially useful for early discovery,
        initial qualification, and finance-aware top-of-funnel conversations.

        Args:
            topic: Main topic or commercial need to orient the overview around.
            audience: Intended customer segment or stakeholder, if known.
            use_case: Practical task, route, or workload the vehicle should cover.
        """
        _debug_tool_call(
            "get_sales_catalog_overview",
            {"topic": topic, "audience": audience, "use_case": use_case},
        )
        context = filter_sales_context(locale, topic=" ".join([clean_text(topic), clean_text(use_case)]), max_groups=3)
        payload = {
            "topic": clean_text(topic),
            "audience": clean_text(audience),
            "use_case": clean_text(use_case),
            "product_groups": context.get("product_groups") or [],
            "finance_options": context.get("finance_options") or [],
        }
        state = runtime.state if runtime else {}
        _debug_tool_result("get_sales_catalog_overview", payload)
        return Command(
            update={
                "sales_context_baseline": payload,
                "research_layer": "portfolio_baseline",
                "tool_calls_this_turn": _append_tool_call(state, "get_sales_catalog_overview"),
                "messages": _tool_message(payload, runtime),
            }
        )

    return get_sales_catalog_overview



def build_sales_landscape_tool(
    locale: str,
    docs_client: GazDocumentsClient,
    allowed_family_ids: Sequence[str] | None = None,
):
    @tool("get_sales_landscape", parse_docstring=True)
    def get_sales_landscape(
        topic: str = "",
        audience: str = "",
        use_case: str = "",
        focus: str = "",
        runtime: ToolRuntime = None,
    ) -> Command:
        """Build a broad multi-direction sales digest with supporting materials.

        Use this tool when you need a curated landscape of likely product directions plus the
        strongest supporting sales materials. It is best for broad commercial guidance before
        the conversation narrows to one concrete model or exact specification.

        Args:
            topic: Main topic or vehicle need to explore.
            audience: Customer segment or stakeholder perspective to optimize for.
            use_case: Operational scenario, route type, or business task behind the request.
            focus: Optional emphasis such as payload, dimensions, finance, passenger use, or
                objections.
        """
        state = runtime.state if runtime else {}
        _debug_tool_call(
            "get_sales_landscape",
            {"topic": topic, "audience": audience, "use_case": use_case, "focus": focus},
        )
        guard = _build_information_tool_family_guard(
            locale,
            allowed_family_ids,
            topic=topic,
            use_case=use_case,
            focus=focus,
            state=state,
        )
        if guard["mode"] == _UNAVAILABLE_STATUS:
            payload = _family_filter_notice(locale, guard["unsupported_names"], partial=False) or {"status": _UNAVAILABLE_STATUS}
            _debug_tool_result("get_sales_landscape", payload)
            return Command(
                update=_family_filter_block_update(
                    state,
                    runtime,
                    locale=locale,
                    tool_name="get_sales_landscape",
                    unsupported_names=guard["unsupported_names"],
                )
            )
        payload, update = _landscape_digest(
            locale,
            state,
            docs_client,
            topic,
            audience,
            use_case,
            focus,
            allowed_family_ids=allowed_family_ids,
            requested_family_ids=guard["supported_family_ids"],
        )
        if guard["mode"] == _PARTIAL_STATUS:
            payload = _apply_family_filter_notice(payload, locale, guard["unsupported_names"])
            digests = list(update.get("sales_digests") or [])
            if digests:
                digests[-1] = payload
                update["sales_digests"] = digests
        _debug_tool_result("get_sales_landscape", payload)
        update["messages"] = _tool_message(payload, runtime)
        return Command(update=update)

    return get_sales_landscape



def build_compare_product_directions_tool(
    locale: str,
    docs_client: GazDocumentsClient,
    allowed_family_ids: Sequence[str] | None = None,
):
    @tool("compare_product_directions", parse_docstring=True)
    def compare_product_directions(
        query: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
        dimensions: Sequence[str] | None = None,
        top_families: int = 3,
        runtime: ToolRuntime = None,
    ) -> Command:
        """Return a multi-product comparison digest for likely directions or competitor context.

        Use this tool when the customer asks to compare several GAZ directions, or when a
        competitor is mentioned and you need a structured multi-family comparison rather than a
        single-product answer.

        Args:
            query: Natural-language comparison request in business terms.
            families: Optional candidate product families to compare, such as Gazelle NEXT,
                Sobol NN, or Gazon NEXT.
            competitor: Optional competitor model, brand, or alternative solution mentioned by
                the customer.
            dimensions: Optional comparison axes such as dimensions, payload, power, engine,
                fuel, versions, or finance.
            top_families: Maximum number of product families to include in the digest after
                internal narrowing.
        """
        state = runtime.state if runtime else {}
        clamped_top_families = _clamp_int(top_families, lower=1, upper=4, default=3)
        _debug_tool_call(
            "compare_product_directions",
            {
                "query": query,
                "families": list(families or []),
                "competitor": competitor,
                "dimensions": list(dimensions or []),
                "top_families": clamped_top_families,
            },
        )
        guard = _build_information_tool_family_guard(
            locale,
            allowed_family_ids,
            families=families,
            competitor=competitor,
            query=query,
            state=state,
        )
        if guard["mode"] == _UNAVAILABLE_STATUS:
            payload = _family_filter_notice(locale, guard["unsupported_names"], partial=False) or {"status": _UNAVAILABLE_STATUS}
            _debug_tool_result("compare_product_directions", payload)
            return Command(
                update=_family_filter_block_update(
                    state,
                    runtime,
                    locale=locale,
                    tool_name="compare_product_directions",
                    unsupported_names=guard["unsupported_names"],
                )
            )
        effective_families: Sequence[str] | None = (
            guard["supported_family_ids"] if guard["has_explicit_request"] and guard["supported_family_ids"] else families
        )
        effective_competitor = (
            _sanitize_competitor_signal(competitor, guard["supported_family_ids"] or allowed_family_ids)
            if guard["has_explicit_request"]
            else competitor
        )
        payload, update = _compare_digest(
            locale,
            state,
            docs_client,
            query,
            effective_families,
            effective_competitor,
            dimensions,
            clamped_top_families,
            allowed_family_ids=allowed_family_ids,
        )
        if guard["mode"] == _PARTIAL_STATUS:
            payload = _apply_family_filter_notice(payload, locale, guard["unsupported_names"])
            digests = list(update.get("comparison_digests") or [])
            if digests:
                digests[-1] = payload
                update["comparison_digests"] = digests
        _debug_tool_result("compare_product_directions", payload)
        update["messages"] = _tool_message(payload, runtime)
        return Command(update=update)

    return compare_product_directions



def build_collect_product_snapshot_tool(
    locale: str,
    docs_client: GazDocumentsClient,
    allowed_family_ids: Sequence[str] | None = None,
):
    @tool("collect_product_snapshot", parse_docstring=True)
    def collect_product_snapshot(
        query: str,
        families: Sequence[str] | None = None,
        dimensions: Sequence[str] | None = None,
        competitor: str = "",
        max_products: int = 3,
        runtime: ToolRuntime = None,
    ) -> Command:
        """Return a structured multi-product snapshot for numeric or technical asks.

        Use this tool when the customer needs a compact fact pack across a few likely products,
        especially for dimensions, payload, engine, seating, or other technical attributes.

        Args:
            query: Natural-language request describing the technical or numeric need.
            families: Optional product families to prioritize in the snapshot.
            dimensions: Optional list of requested dimensions or technical axes to emphasize.
            competitor: Optional competitor that should stay in view while selecting products.
            max_products: Maximum number of product entries to return in the snapshot.
        """
        state = runtime.state if runtime else {}
        clamped_max_products = _clamp_int(max_products, lower=1, upper=4, default=3)
        _debug_tool_call(
            "collect_product_snapshot",
            {
                "query": query,
                "families": list(families or []),
                "dimensions": list(dimensions or []),
                "competitor": competitor,
                "max_products": clamped_max_products,
            },
        )
        guard = _build_information_tool_family_guard(
            locale,
            allowed_family_ids,
            families=families,
            competitor=competitor,
            query=query,
            state=state,
        )
        if guard["mode"] == _UNAVAILABLE_STATUS:
            payload = _family_filter_notice(locale, guard["unsupported_names"], partial=False) or {"status": _UNAVAILABLE_STATUS}
            _debug_tool_result("collect_product_snapshot", payload)
            return Command(
                update=_family_filter_block_update(
                    state,
                    runtime,
                    locale=locale,
                    tool_name="collect_product_snapshot",
                    unsupported_names=guard["unsupported_names"],
                )
            )
        effective_families: Sequence[str] | None = (
            guard["supported_family_ids"] if guard["has_explicit_request"] and guard["supported_family_ids"] else families
        )
        effective_competitor = (
            _sanitize_competitor_signal(competitor, guard["supported_family_ids"] or allowed_family_ids)
            if guard["has_explicit_request"]
            else competitor
        )
        payload, update = _snapshot_digest(
            locale,
            state,
            docs_client,
            query,
            effective_families,
            dimensions,
            effective_competitor,
            clamped_max_products,
            allowed_family_ids=allowed_family_ids,
        )
        if guard["mode"] == _PARTIAL_STATUS:
            payload = _apply_family_filter_notice(payload, locale, guard["unsupported_names"])
            digests = list(update.get("product_snapshots") or [])
            if digests:
                digests[-1] = payload
                update["product_snapshots"] = digests
        _debug_tool_result("collect_product_snapshot", payload)
        update["messages"] = _tool_message(payload, runtime)
        return Command(update=update)

    return collect_product_snapshot



def build_search_sales_materials_tool(
    locale: str,
    docs_client: GazDocumentsClient,
    allowed_family_ids: Sequence[str] | None = None,
):
    @tool("search_sales_materials", parse_docstring=True)
    def search_sales_materials(
        query: str,
        intent: str = "overview",
        families: Sequence[str] | None = None,
        competitor: str = "",
        top_k: int = 4,
        runtime: ToolRuntime = None,
    ) -> Command:
        """Search official sales materials before reading any specific document in depth.

        Use this tool to discover candidate brochures, presentations, battlecards, and other
        approved materials that are relevant to the current ask. Prefer this before
        `read_material` when you still need to identify the right sources.

        Args:
            query: Natural-language search request describing the customer problem or sales need.
            intent: Search mode. Supported values are overview, compare, specs, financing, and
                objection.
            families: Optional product families to bias the search toward.
            competitor: Optional competitor or alternative the customer mentioned.
            top_k: Maximum number of material candidates to return.
        """
        state = runtime.state if runtime else {}
        resolved_intent = clean_text(intent).lower() or "overview"
        if resolved_intent not in _VALID_SEARCH_INTENTS:
            resolved_intent = "overview"
        normalized_families = _normalize_family_list(families)
        clamped_top_k = _clamp_int(top_k, lower=1, upper=6, default=4)
        _debug_tool_call(
            "search_sales_materials",
            {
                "query": query,
                "intent": resolved_intent,
                "families": normalized_families,
                "competitor": competitor,
                "top_k": clamped_top_k,
            },
        )
        guard = _build_information_tool_family_guard(
            locale,
            allowed_family_ids,
            families=families,
            competitor=competitor,
            query=query,
            state=state,
        )
        if guard["mode"] == _UNAVAILABLE_STATUS:
            payload = _family_filter_notice(locale, guard["unsupported_names"], partial=False) or {"status": _UNAVAILABLE_STATUS}
            _debug_tool_result("search_sales_materials", payload)
            return Command(
                update=_family_filter_block_update(
                    state,
                    runtime,
                    locale=locale,
                    tool_name="search_sales_materials",
                    unsupported_names=guard["unsupported_names"],
                )
            )
        search_query = clean_text(query) or clean_text(state.get("problem_summary"))
        effective_families = (
            guard["supported_family_ids"] if guard["has_explicit_request"] and guard["supported_family_ids"] else normalized_families
        )
        competitor_source = clean_text(competitor) or clean_text((state.get("slots") or {}).get("competitor"))
        resolved_competitor = (
            _sanitize_competitor_signal(competitor_source, guard["supported_family_ids"] or allowed_family_ids)
            if guard["has_explicit_request"]
            else competitor_source
        )
        search_key = _build_search_key(search_query, resolved_intent, effective_families, resolved_competitor)
        existing_search_keys = list(state.get("search_keys_this_turn") or [])
        if search_key in existing_search_keys:
            content = {"status": "blocked", "reason": "duplicate_search_attempt", "query": search_query, "intent": resolved_intent}
            _debug_tool_result("search_sales_materials", content)
            return Command(
                update={
                    "sales_loop_guard_reason": "duplicate_search_attempt",
                    "tool_limit_hits": _append_tool_limit_hit(
                        state,
                        tool_name="search_sales_materials",
                        reason="duplicate_search_attempt",
                        search_key=search_key,
                    ),
                    "runtime_warnings": _append_runtime_warning(
                        state,
                        stage="gaz:search_sales_materials",
                        code="duplicate_search_attempt",
                        detail=search_query,
                    ),
                    "tool_calls_this_turn": _append_tool_call(state, "search_sales_materials"),
                    "messages": _tool_message(content, runtime),
                }
            )

        response = docs_client.search_sales_materials(
            query=search_query,
            intent=resolved_intent,
            families=effective_families,
            competitor=resolved_competitor,
            top_k=clamped_top_k,
        )
        candidates = _filter_candidates_for_requested_families(
            response.get("candidates") or [],
            effective_families or allowed_family_ids,
            keep_unknown=not bool(effective_families),
        )
        allowed_ids = [item.get("candidate_id") for item in candidates if item.get("candidate_id")]
        research_status = dict(state.get("research_status") or {})
        queries = list(research_status.get("queries") or [])
        if clean_text(query):
            queries.append(clean_text(query))
        response_payload = dict(response)
        response_payload["candidates"] = candidates
        if guard["mode"] == _PARTIAL_STATUS:
            response_payload = _apply_family_filter_notice(response_payload, locale, guard["unsupported_names"])
        payload = {
            "docs_status": {"service_available": True, "collection_available": True},
            "search_query": search_query,
            "material_candidates": candidates,
            "allowed_material_ids": allowed_ids,
            "provisional_recommendations": update_provisional_recommendations(candidates, state.get("provisional_recommendations") or []),
            "research_status": {
                **research_status,
                "queries": queries,
                "last_intent": resolved_intent,
                "candidate_count": len(candidates),
                "documents_touched": list(research_status.get("documents_touched") or []),
                "has_prior_search": True,
            },
            "research_layer": "broad_search",
            "search_keys_this_turn": [*existing_search_keys, search_key],
            "tool_calls_this_turn": _append_tool_call(state, "search_sales_materials"),
            "messages": _tool_message(response_payload, runtime),
        }
        _debug_tool_result("search_sales_materials", response_payload)
        return Command(update=payload)

    return search_sales_materials



def build_read_material_tool(
    locale: str,
    docs_client: GazDocumentsClient,
    allowed_family_ids: Sequence[str] | None = None,
):
    @tool("read_material", parse_docstring=True)
    def read_material(candidate_id: str, focus: str, max_segments: int = 3, runtime: ToolRuntime = None) -> Command:
        """Read a targeted excerpt from a previously surfaced material candidate.

        Use this tool only after `search_sales_materials` or `get_branch_pack` has returned
        candidate documents. It reads focused excerpts instead of the whole material and is
        best for extracting evidence for one concrete point.

        Args:
            candidate_id: Identifier of a previously surfaced material candidate that is allowed
                in the current turn.
            focus: Specific fact, objection, question, or topic to extract from the material.
            max_segments: Maximum number of focused excerpt segments to return.
        """
        state = runtime.state if runtime else {}
        allowed_ids = list(state.get("allowed_material_ids") or [])
        clamped_max_segments = _clamp_int(max_segments, lower=1, upper=5, default=3)
        _debug_tool_call(
            "read_material",
            {
                "candidate_id": candidate_id,
                "focus": focus,
                "max_segments": clamped_max_segments,
                "allowed_material_ids": allowed_ids,
            },
        )
        focus_text = clean_text(focus)
        selected_candidate = next(
            (item for item in state.get("material_candidates") or [] if clean_text(item.get("candidate_id")) == clean_text(candidate_id)),
            {},
        )
        guard = _build_information_tool_family_guard(
            locale,
            allowed_family_ids,
            focus=focus_text,
            state=state,
            candidate=selected_candidate,
        )
        focus_key = _build_focus_key(candidate_id, focus_text)
        existing_focus_keys = list(state.get("read_focus_keys_this_turn") or [])
        read_attempts = dict(state.get("read_attempts_by_candidate") or {})
        candidate_reads = int(read_attempts.get(candidate_id) or 0)
        if candidate_id not in allowed_ids:
            content = {"status": "blocked", "reason": "candidate_not_allowed", "candidate_id": candidate_id}
            _debug_tool_result("read_material", content)
            return Command(
                update={
                    "sales_loop_guard_reason": "candidate_not_allowed",
                    "runtime_warnings": _append_runtime_warning(
                        state,
                        stage="gaz:read_material",
                        code="candidate_not_allowed",
                        detail=candidate_id,
                    ),
                    "tool_calls_this_turn": _append_tool_call(state, "read_material"),
                    "messages": _tool_message(content, runtime),
                }
            )
        if guard["mode"] == _UNAVAILABLE_STATUS:
            payload = _family_filter_notice(locale, guard["unsupported_names"], partial=False) or {"status": _UNAVAILABLE_STATUS}
            _debug_tool_result("read_material", payload)
            return Command(
                update=_family_filter_block_update(
                    state,
                    runtime,
                    locale=locale,
                    tool_name="read_material",
                    unsupported_names=guard["unsupported_names"],
                )
            )
        if focus_key in existing_focus_keys:
            content = {"status": "blocked", "reason": "duplicate_read_attempt", "candidate_id": candidate_id, "focus": focus_text}
            _debug_tool_result("read_material", content)
            return Command(
                update={
                    "sales_loop_guard_reason": "duplicate_read_attempt",
                    "tool_limit_hits": _append_tool_limit_hit(
                        state,
                        tool_name="read_material",
                        reason="duplicate_read_attempt",
                        candidate_id=candidate_id,
                        focus_key=focus_key,
                    ),
                    "runtime_warnings": _append_runtime_warning(
                        state,
                        stage="gaz:read_material",
                        code="duplicate_read_attempt",
                        detail=f"{candidate_id}:{focus_text}",
                    ),
                    "tool_calls_this_turn": _append_tool_call(state, "read_material"),
                    "messages": _tool_message(content, runtime),
                }
            )
        if candidate_reads >= 2:
            content = {"status": "blocked", "reason": "candidate_read_budget_exhausted", "candidate_id": candidate_id}
            _debug_tool_result("read_material", content)
            return Command(
                update={
                    "sales_loop_guard_reason": "candidate_read_budget_exhausted",
                    "tool_limit_hits": _append_tool_limit_hit(
                        state,
                        tool_name="read_material",
                        reason="candidate_read_budget_exhausted",
                        candidate_id=candidate_id,
                    ),
                    "runtime_warnings": _append_runtime_warning(
                        state,
                        stage="gaz:read_material",
                        code="candidate_read_budget_exhausted",
                        detail=candidate_id,
                    ),
                    "tool_calls_this_turn": _append_tool_call(state, "read_material"),
                    "messages": _tool_message(content, runtime),
                }
            )

        response = docs_client.read_material(
            candidate_id=candidate_id,
            focus=focus_text,
            max_segments=clamped_max_segments,
        )
        if guard["mode"] == _PARTIAL_STATUS:
            response = _apply_family_filter_notice(dict(response), locale, guard["unsupported_names"])
        research_status = dict(state.get("research_status") or {})
        documents_touched = list(research_status.get("documents_touched") or [])
        if candidate_id not in documents_touched:
            documents_touched.append(candidate_id)
        new_reads = list(state.get("material_reads") or [])
        new_reads.append(response)
        read_attempts[candidate_id] = candidate_reads + 1
        _debug_tool_result("read_material", response)
        return Command(
            update={
                "material_reads": new_reads,
                "read_attempts_by_candidate": read_attempts,
                "read_focus_keys_this_turn": [*existing_focus_keys, focus_key],
                "research_status": {
                    **research_status,
                    "documents_touched": documents_touched,
                    "has_prior_read": True,
                },
                "research_layer": "targeted_read",
                "tool_calls_this_turn": _append_tool_call(state, "read_material"),
                "messages": _tool_message(response, runtime),
            }
        )

    return read_material



def build_branch_pack_tool(
    locale: str,
    docs_client: GazDocumentsClient,
    allowed_family_ids: Sequence[str] | None = None,
):
    @tool("get_branch_pack", parse_docstring=True)
    def get_branch_pack(branch: str, problem_focus: str = "", top_k: int = 4, runtime: ToolRuntime = None) -> Command:
        """Fetch a deeper branch-focused material pack when broad search is not enough.

        Use this tool after the active business branch is already known and you need more
        focused materials than a generic sales search provides.

        Args:
            branch: Active business branch to load materials for. Supported values are tco,
                configuration, comparison, service_risk, internal_approval, passenger_route,
                special_body, and special_conditions.
            problem_focus: Optional one-line description of the exact customer issue inside that
                branch.
            top_k: Maximum number of branch-specific material candidates to return.
        """
        state = runtime.state if runtime else {}
        resolved_branch = clean_text(branch)
        clamped_top_k = _clamp_int(top_k, lower=1, upper=5, default=4)
        _debug_tool_call(
            "get_branch_pack",
            {"branch": resolved_branch, "problem_focus": problem_focus, "top_k": clamped_top_k},
        )
        guard = _build_information_tool_family_guard(
            locale,
            allowed_family_ids,
            problem_focus=problem_focus,
            state=state,
        )
        if resolved_branch not in _VALID_BRANCHES:
            content = {"status": "blocked", "reason": "invalid_branch", "branch": resolved_branch}
            _debug_tool_result("get_branch_pack", content)
            return Command(update={"messages": _tool_message(content, runtime)})
        if guard["mode"] == _UNAVAILABLE_STATUS:
            payload = _family_filter_notice(locale, guard["unsupported_names"], partial=False) or {"status": _UNAVAILABLE_STATUS}
            _debug_tool_result("get_branch_pack", payload)
            return Command(
                update=_family_filter_block_update(
                    state,
                    runtime,
                    locale=locale,
                    tool_name="get_branch_pack",
                    unsupported_names=guard["unsupported_names"],
                )
            )
        response = docs_client.get_branch_pack(
            branch=resolved_branch,
            slots=state.get("slots") or {},
            problem_summary=clean_text(problem_focus) or clean_text(state.get("problem_summary")),
            top_k=clamped_top_k,
        )
        candidates = _filter_candidates_for_requested_families(
            response.get("candidates") or [],
            guard["supported_family_ids"] or allowed_family_ids,
            keep_unknown=not bool(guard["supported_family_ids"]),
        )
        allowed_ids = [item.get("candidate_id") for item in candidates if item.get("candidate_id")]
        research_status = dict(state.get("research_status") or {})
        response_payload = dict(response)
        response_payload["candidates"] = candidates
        if guard["mode"] == _PARTIAL_STATUS:
            response_payload = _apply_family_filter_notice(response_payload, locale, guard["unsupported_names"])
        payload = {
            "active_branch": resolved_branch,
            "branch_conflict": [],
            "material_candidates": candidates,
            "allowed_material_ids": allowed_ids,
            "provisional_recommendations": update_provisional_recommendations(candidates, state.get("provisional_recommendations") or []),
            "research_status": {
                **research_status,
                "last_branch_pack": resolved_branch,
                "candidate_count": len(candidates),
                "documents_touched": list(research_status.get("documents_touched") or []),
            },
            "research_layer": "branch_pack",
            "tool_calls_this_turn": _append_tool_call(state, "get_branch_pack"),
            "messages": _tool_message(response_payload, runtime),
        }
        _debug_tool_result("get_branch_pack", response_payload)
        return Command(update=payload)

    return get_branch_pack


def build_query_pricing_bi_tool(
    locale: str,
    pricing_bi_agent: Any,
    pricing_bi_configurable: Dict[str, Any],
    thread_suffix: str,
    allowed_family_ids: Sequence[str] | None = None,
):
    @tool("query_pricing_bi", parse_docstring=True)
    def query_pricing_bi(
        question: str,
        requested_product_terms: List[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> Command:
        """Query the internal pricing BI agent for exact price, option, and maintenance facts.

        Use this tool for business questions about model price ranges, option availability,
        option pricing, trims, and maintenance cost or service intervals. The question should
        stay in business language and should not mention database tables, columns, or SQL.
        When possible, also pass product terms from the user's request. Umbrella terms such as
        "Соболь" or "Газель" are allowed here; the tool will normalize and expand them to the
        concrete internal product families before querying BI.

        Args:
            question: Business-language question for the pricing BI agent, such as a request
                about model price, configuration content, option status, option price, or
                maintenance cost.
            requested_product_terms: Optional list of product terms from the user's request.
                Pass names like "Газель NN", "Газель NEXT", or broader terms like "Соболь".
                The tool will normalize and expand them internally.
        """
        state = runtime.state if runtime else {}
        resolved_question = clean_text(question)
        normalized_requested_terms = [clean_text(item) for item in requested_product_terms or [] if clean_text(item)]
        _debug_tool_call(
            "query_pricing_bi",
            {"question": resolved_question, "requested_product_terms": normalized_requested_terms},
        )

        if not resolved_question:
            payload = {
                "status": "error",
                "error_code": "empty_question",
                "message": "Pricing BI question is empty.",
            }
            _debug_tool_result("query_pricing_bi", payload)
            return Command(
                update={
                    "runtime_warnings": _append_runtime_warning(
                        state,
                        stage="gaz:query_pricing_bi",
                        code="empty_question",
                    ),
                    "tool_calls_this_turn": _append_tool_call(state, "query_pricing_bi"),
                    "messages": _tool_message(payload, runtime),
                }
            )
        resolved_scope = _resolve_requested_product_terms(normalized_requested_terms, fallback_text=resolved_question)
        resolved_family_ids = list(resolved_scope.get("resolved_family_ids") or [])
        resolved_family_labels = list(resolved_scope.get("resolved_family_labels") or [])
        unresolved_terms = list(resolved_scope.get("unresolved_terms") or [])

        normalized_allowed = _normalize_family_list(allowed_family_ids)
        allowed_set = set(normalized_allowed)
        effective_family_ids = list(resolved_family_ids)
        unsupported_names: List[str] = []
        partial_filter = False
        if normalized_allowed and resolved_family_ids:
            effective_family_ids = [family_id for family_id in resolved_family_ids if family_id in allowed_set]
            unsupported_names.extend(
                _family_display_label(family_id) for family_id in resolved_family_ids if family_id not in allowed_set
            )
            if effective_family_ids and len(effective_family_ids) != len(resolved_family_ids):
                partial_filter = True
        if normalized_allowed and normalized_requested_terms and unresolved_terms:
            unsupported_names.extend(unresolved_terms)
            if effective_family_ids:
                partial_filter = True
        unsupported_names = _dedupe_preserve(unsupported_names)

        if normalized_allowed and not effective_family_ids and (resolved_family_ids or unresolved_terms):
            payload = _family_filter_notice(
                locale,
                unsupported_names or unresolved_terms or normalized_requested_terms,
                partial=False,
            ) or {"status": _UNAVAILABLE_STATUS}
            _debug_tool_result("query_pricing_bi", payload)
            return Command(
                update=_family_filter_block_update(
                    state,
                    runtime,
                    locale=locale,
                    tool_name="query_pricing_bi",
                    unsupported_names=unsupported_names or unresolved_terms or normalized_requested_terms,
                )
            )

        forwarded_question = resolved_question
        scope_labels = [_family_display_label(family_id) for family_id in effective_family_ids] or resolved_family_labels
        if scope_labels:
            scope_prefix = (
                f"Work only within these product families: {', '.join(scope_labels)}. "
                if _locale_key(locale) == "en"
                else f"Работай только по следующим продуктовым семействам: {', '.join(scope_labels)}. "
            )
            forwarded_question = f"{scope_prefix}{forwarded_question}"

        internal_config = build_internal_invoke_config(
            runtime.config if runtime else None,
            extra_tags=["gaz:pricing_bi"],
        )
        parent_configurable = dict(internal_config.get("configurable") or {})
        configurable = dict(pricing_bi_configurable)
        configurable.update(parent_configurable)
        configurable["thread_id"] = _derive_child_thread_id(str(parent_configurable.get("thread_id") or ""), thread_suffix)
        internal_config["configurable"] = configurable

        try:
            response = pricing_bi_agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": forwarded_question,
                                }
                            ]
                        )
                    ]
                },
                config=internal_config,
                context=getattr(runtime, "context", None),
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Pricing BI tool failed: %s", exc)
            payload = {
                "status": "error",
                "error_code": "pricing_bi_invoke_failed",
                "message": str(exc),
                "question": resolved_question,
            }
            _debug_tool_result("query_pricing_bi", payload)
            return Command(
                update={
                    "runtime_warnings": _append_runtime_warning(
                        state,
                        stage="gaz:query_pricing_bi",
                        code="pricing_bi_invoke_failed",
                        detail=str(exc),
                    ),
                    "tool_calls_this_turn": _append_tool_call(state, "query_pricing_bi"),
                    "messages": _tool_message(payload, runtime),
                }
            )

        answer = clean_text(_extract_last_ai_text(response))
        if not answer:
            payload = {
                "status": "error",
                "error_code": "pricing_bi_empty_response",
                "message": "Pricing BI returned an empty response.",
                "question": resolved_question,
            }
            _debug_tool_result("query_pricing_bi", payload)
            return Command(
                update={
                    "runtime_warnings": _append_runtime_warning(
                        state,
                        stage="gaz:query_pricing_bi",
                        code="pricing_bi_empty_response",
                    ),
                    "tool_calls_this_turn": _append_tool_call(state, "query_pricing_bi"),
                    "messages": _tool_message(payload, runtime),
                }
            )

        payload = {
            "status": "ok",
            "question": resolved_question,
            "answer": answer,
        }
        if partial_filter:
            payload = _apply_family_filter_notice(payload, locale, unsupported_names)
        _debug_tool_result("query_pricing_bi", payload)
        return Command(
            update={
                "tool_calls_this_turn": _append_tool_call(state, "query_pricing_bi"),
                "messages": _tool_message(payload, runtime),
            }
        )

    return query_pricing_bi


def build_solution_shortlist_tool():
    @tool("build_solution_shortlist")
    def build_solution_shortlist(runtime: ToolRuntime = None) -> Command:
        """Build a deterministic shortlist from the currently surfaced materials.

        Use this tool after enough materials have already been surfaced and narrowed. It turns
        the current material context into a concise shortlist without performing any new search.
        """
        state = runtime.state if runtime else {}
        shortlist = [item.model_dump() for item in build_shortlist(state.get("active_branch"), state.get("slots") or {}, state.get("material_candidates") or [])]
        payload = {"shortlist": shortlist}
        _debug_tool_call("build_solution_shortlist", {})
        _debug_tool_result("build_solution_shortlist", payload)
        return Command(
            update={
                "shortlist": shortlist,
                "research_layer": "shortlist",
                "tool_calls_this_turn": _append_tool_call(state, "build_solution_shortlist"),
                "messages": _tool_message(payload, runtime),
            }
        )

    return build_solution_shortlist



def build_followup_pack_tool():
    @tool("build_followup_pack")
    def build_followup_pack(runtime: ToolRuntime = None) -> Command:
        """Build a deterministic next-step package from the current material context.

        Use this tool after the relevant materials and branch context are already in place and
        you need a structured follow-up package rather than more retrieval.
        """
        state = runtime.state if runtime else {}
        followup = build_followup(
            state.get("active_branch"),
            state.get("slots") or {},
            state.get("material_candidates") or [],
            state.get("material_reads") or [],
        ).model_dump()
        payload = {"followup_pack": followup}
        _debug_tool_call("build_followup_pack", {})
        _debug_tool_result("build_followup_pack", payload)
        return Command(
            update={
                "followup_pack": followup,
                "research_layer": "followup",
                "tool_calls_this_turn": _append_tool_call(state, "build_followup_pack"),
                "messages": _tool_message(payload, runtime),
            }
        )

    return build_followup_pack
