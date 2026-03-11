from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schemas import AnswerDepth, ClientIntent, CustomerTemperature, FollowupPack, GazBranch, GazStage, ShortlistEntry

_SPECIAL_BODY_TERMS = {
    "refrigerator",
    "refrigerated",
    "ref",
    "isothermal",
    "reefer",
    "fridge",
    "tow",
    "towing",
    "garbage",
    "vacuum",
    "tank",
    "tipper",
    "lift",
    "kmu",
    "platform",
    "crane",
    "эваку",
    "рефриж",
    "изотерм",
    "мусоров",
    "вакуум",
    "цистерн",
    "самосвал",
    "кму",
    "автовыш",
    "платформ",
}

_ROLE_APPROVAL_TERMS = ("approval", "procurement", "finance", "director", "закуп", "соглас", "фин", "руковод")
_TCO_TERMS = ("tco", "econom", "cost", "ownership", "окуп", "стоим", "эконом")
_SERVICE_TERMS = ("service", "downtime", "reliab", "quality", "простой", "сервис", "надеж", "качеств")
_CONFIGURATION_TERMS = ("config", "body", "base", "комплект", "конфиг", "кузов", "исполн")
_PASSENGER_TERMS = ("route", "capacity", "пассаж", "маршрут", "вместим")
_SPECIAL_CONDITION_TERMS = ("offroad", "4x4", "severe", "harsh", "municipal", "внедор", "суров", "тяжел")
_FINANCE_TERMS = ("finance", "financing", "leasing", "credit", "лизинг", "кредит", "финанс")
_COMPARE_TERMS = ("compare", "versus", "vs", "сравн", "отлич", "разниц", "лучше", "хуже")
_SPECS_TERMS = ("spec", "version", "size", "payload", "power", "engine", "dimension", "габар", "грузопод", "мощн", "двигат", "комплект", "верси", "цифр", "ttx", "ттх")
_MATERIALS_TERMS = ("material", "deck", "presentation", "doc", "pdf", "материал", "презентац", "документ")
_NEXT_STEP_TERMS = ("next step", "what next", "send", "proposal", "commercial offer", "дальше", "следующ", "пришл", "предложен")
_RECOMMEND_TERMS = ("recommend", "advise", "what should i take", "что взять", "посовет", "как думаешь")
_OBJECTION_TERMS = ("дорого", "expensive", "не устраивает", "туфта", "чушь", "не хочу", "не надо")
_COMPETITOR_RISK_TERMS = ("sollers", "atlant", "уйду", "куплю у", "конкур", "competitor")
_IMPATIENCE_TERMS = ("короче", "faster", "быстрее", "меньше текста", "хватит", "просто ответь", "just answer")
_CONCRETE_NUMBER_TERMS = ("конкретн", "точн", "exact", "specific", "numbers", "цифр")
_COMPARISON_TABLE_TERMS = ("таблиц", "table")
_VERSION_TERMS = ("верс", "trim", "modification", "модифик")



_DISCOVERY_SLOT_PRIORITY = (
    "decision_criterion",
    "transport_type",
    "operating_context",
    "body_type",
    "capacity_or_payload",
    "decision_role",
)

_FAMILY_LABELS = {
    "gazelle_next": "Gazelle Next",
    "gazelle_nn": "Gazelle NN",
    "gazelle_city": "Gazelle City",
    "sobol_nn": "Sobol NN",
    "sobol_business": "Sobol Business",
    "gazon_next": "Gazon Next",
    "valdai": "Valdai",
    "sadko": "Sadko",
    "vector_next": "Vector Next",
    "citymax": "Citymax",
    "paz": "PAZ",
    "sat": "SAT",
}

_SALES_CONTEXT = {
    "ru": {
        "product_groups": [
            {
                "group_id": "light_cargo",
                "title": "Легкие коммерческие фургоны и шасси",
                "families": ["Газель NN", "Газель NEXT", "Соболь NN"],
                "main_characteristics": [
                    "городские и пригородные рейсы",
                    "частые короткие доставки и дистрибуция",
                    "широкий выбор фургонов, бортовых версий и шасси под надстройки",
                ],
            },
            {
                "group_id": "medium_cargo",
                "title": "Среднетоннажные шасси и грузовые платформы",
                "families": ["Газон NEXT", "Valdai"],
                "main_characteristics": [
                    "больше объем и запас по нагрузке",
                    "межгород, распределительная логистика и более тяжелые рейсы",
                    "подходят под разные грузовые надстройки",
                ],
            },
            {
                "group_id": "offroad_special",
                "title": "Внедорожные и специальные исполнения",
                "families": ["Садко", "Соболь 4x4", "спецнадстройки"],
                "main_characteristics": [
                    "сложные дорожные условия и муниципальные задачи",
                    "шасси под рефрижераторы, эвакуаторы, КМУ и другие надстройки",
                    "когда важны проходимость или специализированный кузов",
                ],
            },
            {
                "group_id": "passenger",
                "title": "Пассажирские решения",
                "families": ["Газель City", "Vector Next", "Citymax"],
                "main_characteristics": [
                    "городские и пригородные пассажирские маршруты",
                    "разная вместимость и формат салона",
                    "для маршрутных, корпоративных и муниципальных задач",
                ],
            },
        ],
        "finance_options": [
            "лизинг и кредитные сценарии без обещания точных ставок до уточнения клиента и конфигурации",
            "фирменные финансовые программы и акционные предложения, если они подходят под модель и период",
            "сравнение ежемесячной нагрузки и стоимости владения после сужения до 2-3 направлений",
        ],
    },
    "en": {
        "product_groups": [
            {
                "group_id": "light_cargo",
                "title": "Light commercial vans and chassis",
                "families": ["Gazelle NN", "Gazelle NEXT", "Sobol NN"],
                "main_characteristics": [
                    "city and suburban routes",
                    "frequent short-haul delivery and distribution",
                    "wide range of vans, flatbeds, and chassis for body builders",
                ],
            },
            {
                "group_id": "medium_cargo",
                "title": "Medium-duty chassis and cargo platforms",
                "families": ["Gazon Next", "Valdai"],
                "main_characteristics": [
                    "more volume and payload headroom",
                    "intercity distribution and heavier jobs",
                    "fit for multiple cargo body directions",
                ],
            },
            {
                "group_id": "offroad_special",
                "title": "Off-road and special-purpose directions",
                "families": ["Sadko", "Sobol 4x4", "special bodies"],
                "main_characteristics": [
                    "difficult road conditions and municipal work",
                    "chassis for refrigerated, tow, crane, and other body options",
                    "when terrain or a specialized body matters",
                ],
            },
            {
                "group_id": "passenger",
                "title": "Passenger solutions",
                "families": ["Gazelle City", "Vector Next", "Citymax"],
                "main_characteristics": [
                    "urban and suburban passenger routes",
                    "different seating and saloon formats",
                    "for route, corporate, and municipal passenger use",
                ],
            },
        ],
        "finance_options": [
            "leasing and credit scenarios without promising exact rates before narrowing the client and configuration",
            "branded finance programs and promotional conditions when they match the model and time period",
            "monthly burden and ownership-cost comparison after narrowing to 2-3 directions",
        ],
    },
}



def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize(value: Any) -> str:
    return clean_text(value).lower()


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def merge_slots(existing: Mapping[str, Any] | None, updates: Mapping[str, Any] | None) -> Dict[str, Any]:
    merged = dict(existing or {})
    for key, value in dict(updates or {}).items():
        if key == "special_conditions":
            current = list(merged.get(key) or [])
            for item in value or []:
                item_text = clean_text(item)
                if item_text and item_text not in current:
                    current.append(item_text)
            merged[key] = current
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def merge_flags(existing: Mapping[str, Any] | None, updates: Mapping[str, Any] | None) -> Dict[str, Any]:
    merged = dict(existing or {})
    for key, value in dict(updates or {}).items():
        if isinstance(value, bool):
            merged[key] = value
    return merged


def compute_missing_slots(slots: Mapping[str, Any] | None) -> List[str]:
    data = dict(slots or {})
    missing: List[str] = []
    for key in ("customer_goal", "transport_type", "decision_criterion"):
        if not clean_text(data.get(key)):
            missing.append(key)
    has_context = any(
        [
            clean_text(data.get("route_type")),
            clean_text(data.get("body_type")),
            clean_text(data.get("competitor")),
            bool(data.get("special_conditions")),
        ]
    )
    if not has_context:
        missing.append("operating_context")
    return missing


def prioritize_missing_slots(missing_slots: Sequence[str] | None) -> List[str]:
    current = list(missing_slots or [])
    ranked = [key for key in _DISCOVERY_SLOT_PRIORITY if key in current]
    ranked.extend(key for key in current if key not in ranked)
    return ranked


def should_use_discovery_agent(
    intent_flags: Mapping[str, Any] | None,
    missing_slots: Sequence[str] | None,
) -> bool:
    flags = dict(intent_flags or {})
    return bool(missing_slots or flags.get("requested_portfolio_overview") or flags.get("requested_financing"))


def build_sales_context(locale: str | None = None) -> Dict[str, Any]:
    locale_key = "en" if clean_text(locale).lower().startswith("en") else "ru"
    return copy.deepcopy(_SALES_CONTEXT[locale_key])


def filter_sales_context(locale: str | None = None, topic: str = "", max_groups: int = 3) -> Dict[str, Any]:
    context = build_sales_context(locale)
    topic_text = _normalize(topic)
    if not topic_text:
        return context

    terms = [term for term in re.split(r"\W+", topic_text) if len(term) >= 3]
    if not terms:
        return context

    scored_groups = []
    for group in context["product_groups"]:
        haystack = " ".join(
            [
                _normalize(group.get("title")),
                " ".join(_normalize(item) for item in group.get("families") or []),
                " ".join(_normalize(item) for item in group.get("main_characteristics") or []),
            ]
        )
        score = 0
        for term in terms:
            if term in _normalize(group.get("title")):
                score += 3
            elif term in " ".join(_normalize(item) for item in group.get("families") or []):
                score += 2
            elif term in haystack:
                score += 1
        if score > 0:
            scored_groups.append((score, group))

    if not scored_groups:
        return context

    scored_groups.sort(key=lambda item: item[0], reverse=True)
    context["product_groups"] = [group for _score, group in scored_groups[: max(1, min(max_groups, len(scored_groups)))]]
    return context


def build_sales_context_baseline(
    locale: str | None = None,
    slots: Mapping[str, Any] | None = None,
    problem_summary: str = "",
    current_intent: str | None = None,
) -> Dict[str, Any]:
    data = dict(slots or {})
    topic_parts = [
        problem_summary,
        data.get("customer_goal"),
        data.get("transport_type"),
        data.get("route_type"),
        data.get("body_type"),
        data.get("competitor"),
        data.get("decision_criterion"),
    ]
    topic = " ".join(clean_text(part) for part in topic_parts if clean_text(part))
    max_groups = 4 if (current_intent or "overview") == "overview" else 3
    baseline = filter_sales_context(locale, topic=topic, max_groups=max_groups)
    baseline = copy.deepcopy(baseline)
    transport_type = _normalize(data.get("transport_type"))
    if transport_type == "passenger":
        passenger_groups = [group for group in baseline.get("product_groups") or [] if group.get("group_id") == "passenger"]
        if passenger_groups:
            baseline["product_groups"] = passenger_groups
    elif transport_type == "cargo":
        cargo_groups = [group for group in baseline.get("product_groups") or [] if group.get("group_id") != "passenger"]
        if cargo_groups:
            baseline["product_groups"] = cargo_groups[:max_groups]
    if current_intent == "financing":
        baseline["product_groups"] = list(baseline.get("product_groups") or [])[:2]
    return baseline


def infer_discovery_focus_area(
    intent_flags: Mapping[str, Any] | None,
    slots: Mapping[str, Any] | None,
    problem_summary: str = "",
) -> str:
    flags = dict(intent_flags or {})
    data = dict(slots or {})
    summary_text = _normalize(problem_summary)
    body_type = _normalize(data.get("body_type"))
    transport_type = _normalize(data.get("transport_type"))
    special_conditions = " ".join(_normalize(item) for item in data.get("special_conditions") or [])
    criterion = _normalize(data.get("decision_criterion"))
    text = " ".join([summary_text, body_type, transport_type, special_conditions, criterion])

    if flags.get("requested_financing") or flags.get("requested_price") or _contains_any(text, _FINANCE_TERMS):
        return "finance"
    if clean_text(data.get("competitor")) or flags.get("requested_competitor_comparison"):
        return "comparison"
    if transport_type == "passenger" or _contains_any(text, _PASSENGER_TERMS):
        return "passenger"
    if _contains_any(text, _SERVICE_TERMS):
        return "service"
    if _contains_any(body_type, tuple(_SPECIAL_BODY_TERMS)) or _contains_any(special_conditions, _SPECIAL_CONDITION_TERMS):
        return "special"
    return "portfolio"


def classify_branch(slots: Mapping[str, Any] | None, problem_summary: str = "") -> Tuple[Optional[GazBranch], List[GazBranch], str]:
    data = dict(slots or {})
    criterion = (_normalize(data.get("decision_criterion")) + " " + _normalize(problem_summary)).strip()
    role = _normalize(data.get("decision_role"))
    goal = _normalize(data.get("customer_goal"))
    transport_type = _normalize(data.get("transport_type"))
    body_type = _normalize(data.get("body_type"))
    competitor = _normalize(data.get("competitor"))
    special_conditions = " ".join(_normalize(item) for item in data.get("special_conditions") or [])

    candidates: List[GazBranch] = []
    reasoning: List[str] = []

    if competitor:
        candidates.append("comparison")
        reasoning.append("competitor mentioned")
    if _contains_any(f"{criterion} {goal}", _TCO_TERMS):
        candidates.append("tco")
        reasoning.append("economics or TCO language present")
    if _contains_any(f"{criterion} {goal}", _SERVICE_TERMS):
        candidates.append("service_risk")
        reasoning.append("service or downtime concern present")
    if _contains_any(f"{role} {criterion}", _ROLE_APPROVAL_TERMS):
        candidates.append("internal_approval")
        reasoning.append("decision role suggests internal approval")
    if transport_type == "passenger" or _contains_any(f"{criterion} {goal}", _PASSENGER_TERMS):
        candidates.append("passenger_route")
        reasoning.append("passenger route selection context")
    if body_type and _contains_any(body_type, tuple(_SPECIAL_BODY_TERMS)):
        candidates.extend(["special_body", "configuration"])
        reasoning.append("special body requirement detected")
    if _contains_any(special_conditions, _SPECIAL_CONDITION_TERMS):
        candidates.append("special_conditions")
        reasoning.append("special operating condition detected")
    if _contains_any(f"{criterion} {goal}", _CONFIGURATION_TERMS) or body_type or transport_type == "cargo":
        candidates.append("configuration")
        reasoning.append("configuration fit is relevant")

    unique = list(dict.fromkeys(candidates))
    if not unique:
        return "unknown_selection", [], "insufficient routing signal"

    if "internal_approval" in unique and len(unique) > 1:
        other = next(branch for branch in unique if branch != "internal_approval")
        return None, ["internal_approval", other], "; ".join(reasoning)
    if "comparison" in unique and "configuration" in unique:
        return None, ["comparison", "configuration"], "; ".join(reasoning)
    if "tco" in unique and "service_risk" in unique:
        return None, ["tco", "service_risk"], "; ".join(reasoning)
    if "special_conditions" in unique and "special_body" in unique:
        return None, ["special_conditions", "special_body"], "; ".join(reasoning)

    priority: Sequence[GazBranch] = (
        "special_body",
        "special_conditions",
        "passenger_route",
        "comparison",
        "tco",
        "service_risk",
        "internal_approval",
        "configuration",
        "unknown_selection",
    )
    active = next((branch for branch in priority if branch in unique), "unknown_selection")
    return active, [], "; ".join(reasoning)


def infer_client_intent(intent_flags: Mapping[str, Any] | None, last_user_text: str = "") -> ClientIntent:
    flags = dict(intent_flags or {})
    text = _normalize(last_user_text)
    if flags.get("requested_financing") or flags.get("requested_price") or _contains_any(text, _FINANCE_TERMS):
        return "financing"
    if flags.get("requested_competitor_comparison") or flags.get("threatened_competitor_switch") or _contains_any(text, _COMPETITOR_RISK_TERMS):
        return "objection"
    if flags.get("requested_comparison_table") or _contains_any(text, _COMPARE_TERMS) or _contains_any(text, _COMPARISON_TABLE_TERMS):
        return "compare"
    if flags.get("requested_versions") or _contains_any(text, _VERSION_TERMS):
        return "specs"
    if flags.get("requested_specs") or flags.get("requested_concrete_numbers") or _contains_any(text, _SPECS_TERMS) or _contains_any(text, _CONCRETE_NUMBER_TERMS):
        return "specs"
    if flags.get("requested_materials") or _contains_any(text, _MATERIALS_TERMS):
        return "materials"
    if flags.get("asks_for_recommendation") or _contains_any(text, _RECOMMEND_TERMS):
        return "recommendation"
    if _contains_any(text, _NEXT_STEP_TERMS):
        return "next_step"
    return "overview"


def infer_customer_temperature(intent_flags: Mapping[str, Any] | None, last_user_text: str = "") -> CustomerTemperature:
    flags = dict(intent_flags or {})
    text = _normalize(last_user_text)
    if flags.get("threatened_competitor_switch") or _contains_any(text, _COMPETITOR_RISK_TERMS):
        return "competitor_risk"
    if flags.get("expressed_friction") or flags.get("challenged_questions") or _contains_any(text, _OBJECTION_TERMS):
        return "irritated"
    if flags.get("expressed_impatience") or _contains_any(text, _IMPATIENCE_TERMS):
        return "impatient"
    return "neutral"


def derive_answer_depth(intent: ClientIntent, intent_flags: Mapping[str, Any] | None, branch_hint: Optional[str] = None) -> AnswerDepth:
    flags = dict(intent_flags or {})
    if intent == "overview":
        return "broad"
    if intent == "financing":
        return "bounded"
    if intent in {"compare", "specs", "materials"}:
        return "bounded"
    if intent == "objection":
        if flags.get("requested_competitor_comparison") or flags.get("threatened_competitor_switch"):
            return "justified"
        return "bounded"
    if intent in {"recommendation", "next_step"}:
        return "justified" if branch_hint else "bounded"
    return "bounded"


def clamp_answer_depth(
    intent: ClientIntent,
    proposed_depth: AnswerDepth | str,
    *,
    has_prior_search: bool = False,
    has_prior_read: bool = False,
    has_branch_basis: bool = False,
) -> AnswerDepth:
    depth = str(proposed_depth or "broad")
    if depth not in {"broad", "bounded", "justified", "deep_research"}:
        depth = "broad"
    if depth == "deep_research":
        depth = "justified"

    if intent == "overview":
        return "broad"
    if intent == "financing":
        return "bounded"
    if intent in {"compare", "specs", "materials"}:
        if not (has_prior_search and has_prior_read):
            return "bounded"
        return "justified" if depth == "justified" else "bounded"
    if intent == "objection":
        if depth == "justified" and (has_prior_search or has_prior_read):
            return "justified"
        return "bounded"
    if intent in {"recommendation", "next_step"}:
        if depth == "justified" and has_branch_basis:
            return "justified"
        return "bounded"
    return "bounded"


def derive_hitl_trigger_kind(intent: ClientIntent, intent_flags: Mapping[str, Any] | None) -> Optional[str]:
    flags = dict(intent_flags or {})
    if flags.get("requested_materials") or intent == "materials":
        return "document_package_wait"
    if flags.get("requested_comparison_table") or flags.get("requested_competitor_comparison"):
        return "deep_comparison_wait"
    return None


def evaluate_hitl_gate(
    intent: ClientIntent,
    customer_temperature: CustomerTemperature,
    intent_flags: Mapping[str, Any] | None,
    research_status: Mapping[str, Any] | None,
    *,
    research_wait_rejected: bool = False,
    has_material_candidates: bool = False,
    has_material_reads: bool = False,
) -> Dict[str, Any]:
    status = dict(research_status or {})
    has_prior_search = bool(status.get("has_prior_search")) or has_material_candidates
    has_prior_read = bool(status.get("has_prior_read")) or has_material_reads
    trigger_kind = derive_hitl_trigger_kind(intent, intent_flags)
    blocked_by_temperature = customer_temperature in {"irritated", "competitor_risk"}
    blocked_by_first_turn_budget = not has_prior_search and not has_prior_read
    blocked_by_missing_prior_search = has_prior_search and not has_prior_read
    hitl_eligible = bool(trigger_kind) and not research_wait_rejected
    needs_wait = bool(
        hitl_eligible
        and not blocked_by_temperature
        and not blocked_by_first_turn_budget
        and not blocked_by_missing_prior_search
    )
    return {
        "hitl_eligible": hitl_eligible,
        "needs_hitl_wait_confirmation": needs_wait,
        "hitl_blocked_by_temperature": blocked_by_temperature,
        "hitl_blocked_by_first_turn_budget": blocked_by_first_turn_budget,
        "hitl_blocked_by_missing_prior_search": blocked_by_missing_prior_search,
        "hitl_trigger_kind": trigger_kind,
    }


def derive_work_mode(intent: ClientIntent, answer_depth: AnswerDepth) -> GazStage:
    if intent == "next_step":
        return "FOLLOWUP"
    if intent == "recommendation":
        return "RECOMMEND"
    if answer_depth in {"justified", "deep_research"} or intent in {"compare", "specs", "materials", "objection"}:
        return "RESEARCH"
    return "SELL"


def build_allowed_tool_names(intent: ClientIntent, answer_depth: AnswerDepth, work_mode: GazStage) -> List[str]:
    allowed = ["get_sales_catalog_overview"]
    if intent in {"overview", "financing", "recommendation", "next_step"}:
        allowed.append("get_sales_landscape")
    if intent in {"compare", "objection", "recommendation"}:
        allowed.append("compare_product_directions")
    if intent in {"specs", "recommendation"}:
        allowed.append("collect_product_snapshot")
    if intent in {"financing", "materials", "compare", "specs", "objection", "recommendation", "next_step"}:
        allowed.extend(["search_sales_materials", "read_material"])
    if answer_depth in {"justified", "deep_research"} or intent in {"recommendation", "next_step"}:
        allowed.extend(["classify_problem_branch", "get_branch_pack"])
    if work_mode in {"RECOMMEND", "FOLLOWUP"} or intent in {"recommendation", "next_step"}:
        allowed.extend(["build_solution_shortlist", "build_followup_pack"])
    return list(dict.fromkeys(allowed))


def derive_research_layer(
    *,
    has_sales_digest: bool = False,
    has_comparison_digest: bool = False,
    has_product_snapshot: bool = False,
    has_material_candidates: bool,
    has_material_reads: bool,
    has_branch_pack: bool,
    has_shortlist: bool,
    has_followup: bool,
) -> str:
    if has_followup:
        return "followup"
    if has_shortlist:
        return "shortlist"
    if has_branch_pack:
        return "branch_pack"
    if has_product_snapshot:
        return "product_snapshot"
    if has_comparison_digest:
        return "comparison_digest"
    if has_material_reads:
        return "targeted_read"
    if has_sales_digest:
        return "sales_landscape"
    if has_material_candidates:
        return "broad_search"
    return "portfolio_baseline"



def select_active_tool_names(
    intent: ClientIntent,
    answer_depth: AnswerDepth,
    work_mode: GazStage,
    planned_tools: Sequence[str] | None = None,
    *,
    has_sales_digest: bool = False,
    has_comparison_digest: bool = False,
    has_product_snapshot: bool = False,
    has_material_candidates: bool = False,
    has_material_reads: bool = False,
    has_branch_pack: bool = False,
    has_shortlist: bool = False,
    has_followup: bool = False,
) -> List[str]:
    planned = list(planned_tools or build_allowed_tool_names(intent, answer_depth, work_mode))
    active: List[str] = []

    if intent == "overview":
        for tool_name in ("get_sales_catalog_overview", "get_sales_landscape"):
            if tool_name in planned:
                active.append(tool_name)
        return list(dict.fromkeys(active))

    if intent == "financing":
        if not has_sales_digest:
            for tool_name in ("get_sales_catalog_overview", "get_sales_landscape", "search_sales_materials"):
                if tool_name in planned:
                    active.append(tool_name)
            return list(dict.fromkeys(active))
        for tool_name in ("get_sales_landscape", "search_sales_materials"):
            if tool_name in planned:
                active.append(tool_name)
        if has_material_candidates and "read_material" in planned:
            active.append("read_material")
        return list(dict.fromkeys(active))

    if intent in {"compare", "objection"}:
        if not has_comparison_digest:
            if "compare_product_directions" in planned:
                active.append("compare_product_directions")
            return list(dict.fromkeys(active))
        for tool_name in ("compare_product_directions", "search_sales_materials"):
            if tool_name in planned:
                active.append(tool_name)
        if has_material_candidates and "read_material" in planned:
            active.append("read_material")
        if answer_depth in {"justified", "deep_research"}:
            if "classify_problem_branch" in planned:
                active.append("classify_problem_branch")
            if "get_branch_pack" in planned:
                active.append("get_branch_pack")
        return list(dict.fromkeys(active))

    if intent == "specs":
        if not has_product_snapshot:
            if "collect_product_snapshot" in planned:
                active.append("collect_product_snapshot")
            return list(dict.fromkeys(active))
        for tool_name in ("collect_product_snapshot", "search_sales_materials"):
            if tool_name in planned:
                active.append(tool_name)
        if has_material_candidates and "read_material" in planned:
            active.append("read_material")
        return list(dict.fromkeys(active))

    if intent == "materials":
        if not has_material_candidates:
            if "search_sales_materials" in planned:
                active.append("search_sales_materials")
            return list(dict.fromkeys(active))
        for tool_name in ("search_sales_materials", "read_material"):
            if tool_name in planned:
                active.append(tool_name)
        if answer_depth in {"justified", "deep_research"}:
            if "classify_problem_branch" in planned:
                active.append("classify_problem_branch")
            if "get_branch_pack" in planned:
                active.append("get_branch_pack")
        return list(dict.fromkeys(active))

    if intent in {"recommendation", "next_step"}:
        if not (has_sales_digest or has_comparison_digest or has_product_snapshot or has_material_candidates):
            for tool_name in ("get_sales_landscape", "compare_product_directions", "collect_product_snapshot"):
                if tool_name in planned:
                    active.append(tool_name)
            if intent == "recommendation" and "classify_problem_branch" in planned:
                active.append("classify_problem_branch")
        else:
            for tool_name in ("get_sales_landscape", "compare_product_directions", "collect_product_snapshot", "search_sales_materials"):
                if tool_name in planned:
                    active.append(tool_name)
            if has_material_candidates and "read_material" in planned:
                active.append("read_material")
        if answer_depth in {"justified", "deep_research"}:
            if "classify_problem_branch" in planned:
                active.append("classify_problem_branch")
            if "get_branch_pack" in planned:
                active.append("get_branch_pack")
        if intent == "recommendation" or work_mode == "RECOMMEND" or has_shortlist:
            if "build_solution_shortlist" in planned:
                active.append("build_solution_shortlist")
        if intent == "next_step" or work_mode == "FOLLOWUP" or has_followup:
            if "build_followup_pack" in planned:
                active.append("build_followup_pack")
        return list(dict.fromkeys(active))

    for tool_name in ("get_sales_catalog_overview", "get_sales_landscape", "search_sales_materials"):
        if tool_name in planned:
            active.append(tool_name)
    return list(dict.fromkeys(active))


def normalize_provisional_recommendations(values: Sequence[str] | None) -> List[str]:
    cleaned: List[str] = []
    for value in values or []:
        item = clean_text(value)
        if item and item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= 3:
            break
    return cleaned


def update_provisional_recommendations(
    candidates: Sequence[Mapping[str, Any]] | None,
    existing: Sequence[str] | None = None,
) -> List[str]:
    counter: Counter[str] = Counter(existing or [])
    for candidate in candidates or []:
        metadata = candidate.get("metadata") or {}
        for family in metadata.get("product_families") or []:
            if clean_text(family):
                counter[str(family)] += 1
    return [family for family, _count in counter.most_common(3)]


def is_affirmative(value: Any) -> bool:
    text = _normalize(value)
    return text in {"да", "ok", "okay", "yes", "угу", "ага", "подожду", "жду", "согласен", "подтверждаю"}


def is_negative(value: Any) -> bool:
    text = _normalize(value)
    return text in {"нет", "no", "не надо", "не ждать", "не хочу", "без ожидания", "не согласен"}


def extract_conclusions(answer: str) -> List[str]:
    text = clean_text(answer)
    if not text:
        return []
    lines = [line.strip("-â€¢ ") for line in text.splitlines() if clean_text(line)]
    if len(lines) >= 2:
        return lines[:4]
    parts = [part.strip() for part in text.split(".") if clean_text(part)]
    return parts[:4] or [text]


def build_shortlist(
    branch: str | None,
    slots: Mapping[str, Any] | None,
    material_candidates: Sequence[Mapping[str, Any]] | None,
) -> List[ShortlistEntry]:
    slot_data = dict(slots or {})
    candidates = list(material_candidates or [])
    counter: Counter[str] = Counter()
    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        for family in metadata.get("product_families") or []:
            if clean_text(family):
                counter[str(family)] += 1
    families = [family for family, _count in counter.most_common(3)]

    result: List[ShortlistEntry] = []
    for family in families[:3]:
        result.append(
            ShortlistEntry(
                family_id=family,
                fit_reason=_fit_reason(family, slot_data, branch),
                risk_note=_risk_note(family, slot_data),
            )
        )
    return result


def build_followup(
    branch: str | None,
    slots: Mapping[str, Any] | None,
    material_candidates: Sequence[Mapping[str, Any]] | None,
    material_reads: Sequence[Mapping[str, Any]] | None,
) -> FollowupPack:
    slot_data = dict(slots or {})
    role = clean_text(slot_data.get("decision_role")) or None
    recommended_action = _recommended_action(branch, role)

    reads_by_id = {item.get("candidate_id"): item for item in material_reads or [] if item.get("candidate_id")}
    documents = []
    for candidate in material_candidates or []:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id:
            continue
        documents.append(
            {
                "candidate_id": candidate_id,
                "title": candidate.get("title") or candidate_id,
                "why_it_matters": _followup_reason(candidate, reads_by_id.get(candidate_id), branch),
            }
        )
        if len(documents) >= 3:
            break

    return FollowupPack(
        decision_role=role,
        recommended_action=recommended_action,
        documents=documents,
    )


def family_label(family_id: str) -> str:
    return _FAMILY_LABELS.get(family_id, family_id.replace("_", " ").title())


def _fit_reason(family: str, slots: Mapping[str, Any], branch: str | None) -> str:
    label = family_label(family)
    body_type = clean_text(slots.get("body_type"))
    competitor = clean_text(slots.get("competitor"))
    route_type = clean_text(slots.get("route_type"))
    criterion = clean_text(slots.get("decision_criterion"))
    if body_type:
        return f"{label} matches the current body or configuration requirement: {body_type}."
    if competitor:
        return f"{label} is a focused direction for a grounded comparison against {competitor}."
    if route_type:
        return f"{label} fits the operating profile for {route_type} use."
    if criterion:
        return f"{label} aligns with the current decision criterion: {criterion}."
    if branch:
        return f"{label} fits the current {branch.replace('_', ' ')} branch better than a broad catalog search."
    return f"{label} fits the current problem framing."


def _risk_note(family: str, slots: Mapping[str, Any]) -> Optional[str]:
    special_conditions = " ".join(str(item).lower() for item in slots.get("special_conditions") or [])
    if family in {"gazelle_next", "gazelle_nn"} and ("4x4" in special_conditions or "Ð²Ð½ÐµÐ´Ð¾Ñ€" in special_conditions):
        return "Check operating-condition margin before treating it as the final direction."
    if family == "vector_next" and clean_text(slots.get("capacity_or_payload")):
        return "Validate seat or payload margin against the final route profile."
    return None


def _recommended_action(branch: str | None, role: Optional[str]) -> str:
    role_text = _normalize(role)
    if branch == "internal_approval" or _contains_any(role_text, _ROLE_APPROVAL_TERMS):
        return "send_internal_approval_pack"
    if branch == "comparison":
        return "prepare_comparison_pack"
    if branch == "tco":
        return "prepare_tco_review"
    if branch == "service_risk":
        return "send_service_materials"
    return "continue_configuration_selection"


def _followup_reason(
    candidate: Mapping[str, Any],
    read_result: Optional[Mapping[str, Any]],
    branch: str | None,
) -> str:
    if read_result and read_result.get("focus"):
        return f"Supports the current {branch or 'selection'} discussion for focus: {read_result['focus']}."
    rationale = clean_text(candidate.get("rationale"))
    if rationale:
        return rationale[0].upper() + rationale[1:]
    return f"Keeps the next step grounded in the active {branch or 'selection'} branch."


__all__ = [
    "build_allowed_tool_names",
    "build_followup",
    "build_sales_context",
    "build_sales_context_baseline",
    "build_shortlist",
    "classify_branch",
    "clean_text",
    "compute_missing_slots",
    "derive_answer_depth",
    "clamp_answer_depth",
    "derive_research_layer",
    "derive_work_mode",
    "extract_conclusions",
    "evaluate_hitl_gate",
    "family_label",
    "derive_hitl_trigger_kind",
    "filter_sales_context",
    "infer_client_intent",
    "infer_customer_temperature",
    "infer_discovery_focus_area",
    "is_affirmative",
    "is_negative",
    "merge_flags",
    "merge_slots",
    "normalize_provisional_recommendations",
    "prioritize_missing_slots",
    "select_active_tool_names",
    "should_use_discovery_agent",
    "update_provisional_recommendations",
]

