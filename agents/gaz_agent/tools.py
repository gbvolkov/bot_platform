from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Sequence

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

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
    "газель next": "gazelle_next",
    "gazelle_nn": "gazelle_nn",
    "gazelle nn": "gazelle_nn",
    "газель nn": "gazelle_nn",
    "gazelle_city": "gazelle_city",
    "gazelle city": "gazelle_city",
    "газель city": "gazelle_city",
    "sobol_nn": "sobol_nn",
    "sobol nn": "sobol_nn",
    "соболь nn": "sobol_nn",
    "sobol_business": "sobol_business",
    "sobol business": "sobol_business",
    "соболь бизнес": "sobol_business",
    "gazon_next": "gazon_next",
    "gazon next": "gazon_next",
    "газон next": "gazon_next",
    "valdai": "valdai",
    "валдай": "valdai",
    "sadko": "sadko",
    "садко": "sadko",
    "vector_next": "vector_next",
    "vector next": "vector_next",
    "вектор next": "vector_next",
    "citymax": "citymax",
    "city max": "citymax",
    "паз": "paz",
    "paz": "paz",
    "sat": "sat",
}
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
    text = clean_text(value).lower().replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if text in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[text]
    compact = text.replace(" ", "_")
    if compact in _FAMILY_ALIASES.values():
        return compact
    return compact



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
    return families



def _candidate_matches_family(candidate: Dict[str, Any], family: str) -> bool:
    return family in _candidate_families(candidate)



def _candidate_ref(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return SourceCandidateRef(
        candidate_id=str(candidate.get("candidate_id") or ""),
        title=str(candidate.get("title") or candidate.get("candidate_id") or ""),
        doc_kind=str(candidate.get("doc_kind") or "general"),
    ).model_dump()



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



def _landscape_digest(locale: str, state: Dict[str, Any], docs_client: GazDocumentsClient, topic: str, audience: str, use_case: str, focus: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    query = " ".join(part for part in [clean_text(topic), clean_text(use_case), clean_text(focus)] if part)
    baseline = filter_sales_context(locale, topic=query, max_groups=4)
    finance_options = list(baseline.get("finance_options") or [])[:3]
    trace: Dict[str, Any] = {"tool_name": "get_sales_landscape", "query": query, "focus": clean_text(focus)}
    responses: List[Dict[str, Any]] = []
    if query:
        responses.append(_search_with_trace(docs_client, trace, query=query, intent="overview", families=[], competitor="", top_k=6))
        lowered = query.lower()
        if len(responses) < _COMPOSITE_SEARCH_LIMIT and any(term in lowered for term in ("finance", "leasing", "credit", "лизинг", "кредит", "финанс")):
            responses.append(_search_with_trace(docs_client, trace, query=query, intent="financing", families=[], competitor="", top_k=4))
    candidate_pool = _merge_candidates([], [candidate for response in responses for candidate in response.get("candidates") or []])
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



def _compare_digest(locale: str, state: Dict[str, Any], docs_client: GazDocumentsClient, query: str, families: Sequence[str] | None, competitor: str, dimensions: Sequence[str] | None, top_families: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    query_text = clean_text(query) or clean_text(state.get("problem_summary"))
    normalized_families = _normalize_family_list(families)
    axes = _infer_axes(query_text, dimensions)
    trace: Dict[str, Any] = {"tool_name": "compare_product_directions", "query": query_text, "axes": axes}
    responses: List[Dict[str, Any]] = []
    responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="compare", families=normalized_families, competitor=clean_text(competitor), top_k=6))
    if clean_text(competitor) and len(responses) < _COMPOSITE_SEARCH_LIMIT:
        responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="objection", families=normalized_families, competitor=clean_text(competitor), top_k=4))
    if axes and len(responses) < _COMPOSITE_SEARCH_LIMIT:
        responses.append(_search_with_trace(docs_client, trace, query=f"{query_text} {' '.join(axes)}", intent="specs", families=normalized_families, competitor=clean_text(competitor), top_k=4))
    candidate_pool = _merge_candidates([], [candidate for response in responses for candidate in response.get("candidates") or []])
    selected_families = _pick_top_families(candidate_pool, normalized_families or state.get("provisional_recommendations") or [], top_families)
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



def _snapshot_digest(locale: str, state: Dict[str, Any], docs_client: GazDocumentsClient, query: str, families: Sequence[str] | None, dimensions: Sequence[str] | None, competitor: str, max_products: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    query_text = clean_text(query) or clean_text(state.get("problem_summary"))
    normalized_families = _normalize_family_list(families)
    axes = _infer_axes(query_text, dimensions)
    trace: Dict[str, Any] = {"tool_name": "collect_product_snapshot", "query": query_text, "dimensions": axes}
    responses: List[Dict[str, Any]] = []
    responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="specs", families=normalized_families, competitor=clean_text(competitor), top_k=6))
    if clean_text(competitor) and len(responses) < _COMPOSITE_SEARCH_LIMIT:
        responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="compare", families=normalized_families, competitor=clean_text(competitor), top_k=4))
    if not normalized_families and len(responses) < _COMPOSITE_SEARCH_LIMIT:
        responses.append(_search_with_trace(docs_client, trace, query=query_text, intent="overview", families=[], competitor=clean_text(competitor), top_k=4))
    candidate_pool = _merge_candidates([], [candidate for response in responses for candidate in response.get("candidates") or []])
    selected_families = _pick_top_families(candidate_pool, normalized_families or state.get("provisional_recommendations") or [], max_products)
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
    @tool("classify_problem_branch")
    def classify_problem_branch(reasoning_note: str = "", runtime: ToolRuntime = None) -> Command:
        """Classify the current customer case into one active business branch or a conflict."""
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
    @tool("get_sales_catalog_overview")
    def get_sales_catalog_overview(
        topic: str = "",
        audience: str = "",
        use_case: str = "",
        runtime: ToolRuntime = None,
    ) -> Command:
        """Return a broad product and financing overview for sales discovery."""
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



def build_sales_landscape_tool(locale: str, docs_client: GazDocumentsClient):
    @tool("get_sales_landscape")
    def get_sales_landscape(
        topic: str = "",
        audience: str = "",
        use_case: str = "",
        focus: str = "",
        runtime: ToolRuntime = None,
    ) -> Command:
        """Build a broad multi-direction sales digest with supporting materials."""
        state = runtime.state if runtime else {}
        _debug_tool_call(
            "get_sales_landscape",
            {"topic": topic, "audience": audience, "use_case": use_case, "focus": focus},
        )
        payload, update = _landscape_digest(locale, state, docs_client, topic, audience, use_case, focus)
        _debug_tool_result("get_sales_landscape", payload)
        update["messages"] = _tool_message(payload, runtime)
        return Command(update=update)

    return get_sales_landscape



def build_compare_product_directions_tool(locale: str, docs_client: GazDocumentsClient):
    @tool("compare_product_directions")
    def compare_product_directions(
        query: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
        dimensions: Sequence[str] | None = None,
        top_families: int = 3,
        runtime: ToolRuntime = None,
    ) -> Command:
        """Return a multi-product comparison digest for likely directions or competitor context."""
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
        payload, update = _compare_digest(locale, state, docs_client, query, families, competitor, dimensions, clamped_top_families)
        _debug_tool_result("compare_product_directions", payload)
        update["messages"] = _tool_message(payload, runtime)
        return Command(update=update)

    return compare_product_directions



def build_collect_product_snapshot_tool(locale: str, docs_client: GazDocumentsClient):
    @tool("collect_product_snapshot")
    def collect_product_snapshot(
        query: str,
        families: Sequence[str] | None = None,
        dimensions: Sequence[str] | None = None,
        competitor: str = "",
        max_products: int = 3,
        runtime: ToolRuntime = None,
    ) -> Command:
        """Return a structured multi-product snapshot for numeric or technical asks."""
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
        payload, update = _snapshot_digest(locale, state, docs_client, query, families, dimensions, competitor, clamped_max_products)
        _debug_tool_result("collect_product_snapshot", payload)
        update["messages"] = _tool_message(payload, runtime)
        return Command(update=update)

    return collect_product_snapshot



def build_search_sales_materials_tool(docs_client: GazDocumentsClient):
    @tool("search_sales_materials")
    def search_sales_materials(
        query: str,
        intent: str = "overview",
        families: Sequence[str] | None = None,
        competitor: str = "",
        top_k: int = 4,
        runtime: ToolRuntime = None,
    ) -> Command:
        """Search broad official sales materials before deeper reads."""
        state = runtime.state if runtime else {}
        resolved_intent = clean_text(intent).lower() or "overview"
        if resolved_intent not in _VALID_SEARCH_INTENTS:
            resolved_intent = "overview"
        normalized_families = [clean_text(item) for item in families or [] if clean_text(item)]
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
        search_query = clean_text(query) or clean_text(state.get("problem_summary"))
        resolved_competitor = clean_text(competitor) or clean_text((state.get("slots") or {}).get("competitor"))
        search_key = _build_search_key(search_query, resolved_intent, normalized_families, resolved_competitor)
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
            families=normalized_families,
            competitor=resolved_competitor,
            top_k=clamped_top_k,
        )
        candidates = list(response.get("candidates") or [])
        allowed_ids = [item.get("candidate_id") for item in candidates if item.get("candidate_id")]
        research_status = dict(state.get("research_status") or {})
        queries = list(research_status.get("queries") or [])
        if clean_text(query):
            queries.append(clean_text(query))
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
            "messages": _tool_message(response, runtime),
        }
        _debug_tool_result("search_sales_materials", response)
        return Command(update=payload)

    return search_sales_materials



def build_read_material_tool(docs_client: GazDocumentsClient):
    @tool("read_material")
    def read_material(candidate_id: str, focus: str, max_segments: int = 3, runtime: ToolRuntime = None) -> Command:
        """Read a targeted excerpt from a previously surfaced material candidate."""
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



def build_branch_pack_tool(docs_client: GazDocumentsClient):
    @tool("get_branch_pack")
    def get_branch_pack(branch: str, problem_focus: str = "", top_k: int = 4, runtime: ToolRuntime = None) -> Command:
        """Fetch a deeper, branch-focused material pack when broad search is not enough."""
        state = runtime.state if runtime else {}
        resolved_branch = clean_text(branch)
        clamped_top_k = _clamp_int(top_k, lower=1, upper=5, default=4)
        _debug_tool_call(
            "get_branch_pack",
            {"branch": resolved_branch, "problem_focus": problem_focus, "top_k": clamped_top_k},
        )
        if resolved_branch not in _VALID_BRANCHES:
            content = {"status": "blocked", "reason": "invalid_branch", "branch": resolved_branch}
            _debug_tool_result("get_branch_pack", content)
            return Command(update={"messages": _tool_message(content, runtime)})
        response = docs_client.get_branch_pack(
            branch=resolved_branch,
            slots=state.get("slots") or {},
            problem_summary=clean_text(problem_focus) or clean_text(state.get("problem_summary")),
            top_k=clamped_top_k,
        )
        candidates = list(response.get("candidates") or [])
        allowed_ids = [item.get("candidate_id") for item in candidates if item.get("candidate_id")]
        research_status = dict(state.get("research_status") or {})
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
            "messages": _tool_message(response, runtime),
        }
        _debug_tool_result("get_branch_pack", response)
        return Command(update=payload)

    return get_branch_pack



def build_solution_shortlist_tool():
    @tool("build_solution_shortlist")
    def build_solution_shortlist(runtime: ToolRuntime = None) -> Command:
        """Build a deterministic shortlist from the currently surfaced materials."""
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
        """Build a deterministic next-step package from the current material context."""
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
