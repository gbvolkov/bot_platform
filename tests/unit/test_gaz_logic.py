from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from agents.gaz_agent.agent import (
    _compact_payload,
    _is_question_policy_violation,
    _is_soft_question_validation,
    _question_guidance,
    _structured_response_format,
    create_answer_plan_node,
    create_response_finalize_node,
    create_sales_response_node,
    create_turn_intent_node,
    route_after_answer_plan,
)
from agents.gaz_agent.logic import (
    build_allowed_tool_names,
    build_followup,
    build_sales_context,
    build_sales_context_baseline,
    build_shortlist,
    clamp_answer_depth,
    classify_branch,
    compute_missing_slots,
    derive_answer_depth,
    derive_hitl_trigger_kind,
    derive_research_layer,
    derive_work_mode,
    evaluate_hitl_gate,
    filter_sales_context,
    infer_client_intent,
    infer_customer_temperature,
    infer_discovery_focus_area,
    is_affirmative,
    is_negative,
    prioritize_missing_slots,
    select_active_tool_names,
    update_provisional_recommendations,
)
from agents.gaz_agent.locales import get_locale, resolve_locale
from agents.gaz_agent.prompts import get_prompt, get_text
from agents.gaz_agent.schemas import AnswerPlanResult
from agents.gaz_agent.tools import (
    _compare_digest,
    _landscape_digest,
    _snapshot_digest,
    build_read_material_tool,
    build_search_sales_materials_tool,
)


class DummyPlanner:
    def __init__(self, structured_response):
        self.structured_response = structured_response

    def invoke(self, state, config=None, context=None):
        return {"structured_response": dict(self.structured_response)}


class DummyValidatorSequence:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, state, config=None, context=None):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return {"structured_response": dict(self.responses[index])}


class DummyRepairAgent:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def invoke(self, state, config=None, context=None):
        self.calls += 1
        return {"messages": [AIMessage(content=self.answer)]}



class DummyAgentSequence:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, state, config=None, context=None):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return dict(self.responses[index])


class DummyFailingRepairAgent:
    def __init__(self):
        self.calls = 0

    def invoke(self, state, config=None, context=None):
        self.calls += 1
        raise RuntimeError("repair failed")


class DummyLoopAgent:
    def __init__(self):
        self.calls = 0

    def invoke(self, state, config=None, context=None):
        self.calls += 1
        raise GraphRecursionError("loop")


def _merge_command_state(base_state, command):
    merged = dict(base_state)
    merged.update(command.update)
    return merged


def _runtime(state):
    return SimpleNamespace(state=state, tool_call_id="test-tool-call")

class DummyDocsClient:
    def __init__(self, estimate_response=None):
        self.estimate_response = estimate_response or {
            "requires_hitl_wait_confirmation": False,
            "estimated_remaining_cost": "low",
            "rationale": "Live turn should be able to answer from a bounded search/read path.",
        }
        self.calls = []

    def estimate_research_cost(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.estimate_response)


class DummyCompositeDocsClient:
    def __init__(self):
        self.search_calls = []
        self.read_calls = []
        self._candidates = [
            {
                "candidate_id": "cand_gnn_cfg",
                "title": "Gazelle NN specs",
                "doc_kind": "configuration",
                "rationale": "fits city mixed cargo work",
                "metadata": {"product_families": ["gazelle_nn"]},
            },
            {
                "candidate_id": "cand_gnext_cmp",
                "title": "Gazelle NEXT comparison",
                "doc_kind": "comparison",
                "rationale": "good baseline for value comparison",
                "metadata": {"product_families": ["gazelle_next"]},
            },
            {
                "candidate_id": "cand_sobol_cfg",
                "title": "Sobol NN options",
                "doc_kind": "configuration",
                "rationale": "compact direction for tighter city use",
                "metadata": {"product_families": ["sobol_nn"]},
            },
            {
                "candidate_id": "cand_finance",
                "title": "Finance program deck",
                "doc_kind": "approval",
                "rationale": "supports financing discussion",
                "metadata": {"product_families": ["gazelle_nn", "gazelle_next"]},
            },
        ]

    def search_sales_materials(self, **kwargs):
        self.search_calls.append(dict(kwargs))
        query = (kwargs.get("query") or "").lower()
        families = list(kwargs.get("families") or [])
        candidates = list(self._candidates)
        if families:
            candidates = [item for item in candidates if set(item["metadata"]["product_families"]).intersection(families)]
        if "sobol" in query or "соболь" in query:
            sobol = [item for item in candidates if "sobol_nn" in item["metadata"]["product_families"]]
            rest = [item for item in candidates if "sobol_nn" not in item["metadata"]["product_families"]]
            candidates = sobol + rest
        return {"candidates": candidates[: kwargs.get("top_k", 4)]}

    def read_material(self, candidate_id, focus, max_segments=3):
        self.read_calls.append({"candidate_id": candidate_id, "focus": focus, "max_segments": max_segments})
        title_map = {
            "cand_gnn_cfg": "Gazelle NN specs",
            "cand_gnext_cmp": "Gazelle NEXT comparison",
            "cand_sobol_cfg": "Sobol NN options",
            "cand_finance": "Finance program deck",
        }
        excerpt_map = {
            "cand_gnn_cfg": "Payload up to 1.5t, city-focused chassis, engine options and baseline dimensions.",
            "cand_gnext_cmp": "Useful value baseline, broad configuration choice, stable comparison position against alternatives.",
            "cand_sobol_cfg": "Compact format for tighter city work, lower footprint, practical mixed urban use.",
            "cand_finance": "Leasing and finance scenarios should be compared after narrowing to 2-3 directions.",
        }
        return {
            "candidate_id": candidate_id,
            "title": title_map[candidate_id],
            "focus": focus,
            "excerpts": [{"excerpt": excerpt_map[candidate_id], "relevance_reason": "match", "metadata": {}}],
            "metadata": {"doc_kind": "configuration"},
        }


COMPOSITE_STATE = {
    "locale": "ru",
    "slots": {"transport_type": "cargo"},
    "problem_summary": "городские смешанные перевозки",
    "provisional_recommendations": [],
    "research_status": {},
    "material_candidates": [],
    "material_reads": [],
    "sales_digests": [],
    "comparison_digests": [],
    "product_snapshots": [],
    "composite_tool_traces": [],
}


def test_compute_missing_slots_requires_context():
    missing = compute_missing_slots(
        {
            "customer_goal": "Need a new vehicle",
            "transport_type": "cargo",
            "decision_criterion": "configuration",
        }
    )
    assert missing == ["operating_context"]


def test_prioritize_missing_slots_prefers_decision_then_context():
    missing = prioritize_missing_slots(["operating_context", "decision_criterion", "transport_type"])
    assert missing == ["decision_criterion", "transport_type", "operating_context"]


def test_sales_context_has_products_and_finance():
    context = build_sales_context("ru")
    assert context["product_groups"]
    assert context["finance_options"]
    assert any(group["families"] for group in context["product_groups"])


def test_sales_context_can_be_filtered_by_topic():
    context = filter_sales_context("ru", topic="пассажирский маршрут")
    assert context["product_groups"]
    assert any("пассаж" in group["title"].lower() for group in context["product_groups"])


def test_infer_discovery_focus_area_prefers_finance_then_special_cases():
    assert infer_discovery_focus_area({"requested_financing": True}, {}, "") == "finance"
    assert infer_discovery_focus_area({}, {"transport_type": "passenger"}, "") == "passenger"
    assert infer_discovery_focus_area({}, {"competitor": "Atlant"}, "") == "comparison"


def test_infer_client_intent_for_specs_and_objection():
    assert infer_client_intent({"requested_specs": True}, "") == "specs"
    assert infer_client_intent({"threatened_competitor_switch": True}, "") == "objection"
    assert infer_client_intent({}, "Покажи сравнительную таблицу") == "compare"


def test_infer_customer_temperature_detects_competitor_then_irritation():
    assert infer_customer_temperature({"threatened_competitor_switch": True}, "") == "competitor_risk"
    assert infer_customer_temperature({"expressed_friction": True}, "") == "irritated"
    assert infer_customer_temperature({"expressed_impatience": True}, "") == "impatient"


def test_answer_depth_and_work_mode_are_sales_first():
    assert derive_answer_depth("overview", {}) == "broad"
    assert derive_answer_depth("specs", {"requested_concrete_numbers": True}) == "bounded"
    assert derive_answer_depth("objection", {"requested_competitor_comparison": True}) == "justified"
    assert derive_work_mode("recommendation", "bounded") == "RECOMMEND"
    assert derive_work_mode("compare", "bounded") == "RESEARCH"


def test_clamp_answer_depth_requires_prior_research_or_branch_basis():
    assert clamp_answer_depth("compare", "deep_research", has_prior_search=False, has_prior_read=False) == "bounded"
    assert clamp_answer_depth("compare", "justified", has_prior_search=True, has_prior_read=True) == "justified"
    assert clamp_answer_depth("recommendation", "justified", has_branch_basis=False) == "bounded"
    assert clamp_answer_depth("recommendation", "justified", has_branch_basis=True) == "justified"


def test_hitl_gate_requires_explicit_trigger_and_prior_bounded_research():
    assert derive_hitl_trigger_kind("materials", {"requested_materials": True}) == "document_package_wait"
    assert derive_hitl_trigger_kind("compare", {"requested_comparison_table": True}) == "deep_comparison_wait"

    first_turn_gate = evaluate_hitl_gate(
        "materials",
        "neutral",
        {"requested_materials": True},
        {"has_prior_search": False, "has_prior_read": False},
    )
    assert first_turn_gate["hitl_eligible"] is True
    assert first_turn_gate["needs_hitl_wait_confirmation"] is False
    assert first_turn_gate["hitl_blocked_by_first_turn_budget"] is True

    blocked_gate = evaluate_hitl_gate(
        "materials",
        "irritated",
        {"requested_materials": True},
        {"has_prior_search": True, "has_prior_read": True},
    )
    assert blocked_gate["needs_hitl_wait_confirmation"] is False
    assert blocked_gate["hitl_blocked_by_temperature"] is True

    eligible_gate = evaluate_hitl_gate(
        "materials",
        "neutral",
        {"requested_materials": True},
        {"has_prior_search": True, "has_prior_read": True},
    )
    assert eligible_gate["needs_hitl_wait_confirmation"] is True
    assert eligible_gate["hitl_blocked_by_missing_prior_search"] is False


def test_allowed_tool_names_expand_with_answer_depth():
    broad_tools = build_allowed_tool_names("overview", "broad", "SELL")
    assert "get_sales_catalog_overview" in broad_tools
    assert "get_sales_landscape" in broad_tools
    assert "read_material" not in broad_tools

    justified_tools = build_allowed_tool_names("recommendation", "justified", "RECOMMEND")
    assert "compare_product_directions" in justified_tools
    assert "collect_product_snapshot" in justified_tools
    assert "read_material" in justified_tools
    assert "get_branch_pack" in justified_tools
    assert "build_solution_shortlist" in justified_tools
    assert "build_followup_pack" in justified_tools


def test_affirmative_and_negative_helpers():
    assert is_affirmative("да")
    assert is_affirmative("yes")
    assert is_negative("нет")
    assert is_negative("no")


def test_update_provisional_recommendations_prefers_frequent_families():
    recommendations = update_provisional_recommendations(
        [
            {"metadata": {"product_families": ["gazelle_next", "gazelle_nn"]}},
            {"metadata": {"product_families": ["gazelle_next"]}},
        ]
    )
    assert recommendations[0] == "gazelle_next"


def test_classify_branch_detects_comparison_configuration_conflict():
    active_branch, branch_conflict, reasoning = classify_branch(
        {
            "customer_goal": "Need the right refrigerated body and want to compare with Atlant",
            "transport_type": "cargo",
            "body_type": "refrigerator",
            "competitor": "sollers_atlant",
            "decision_criterion": "configuration",
        }
    )
    assert active_branch is None
    assert set(branch_conflict) == {"comparison", "configuration"}
    assert reasoning


def test_classify_branch_returns_unknown_when_signal_is_insufficient():
    active_branch, branch_conflict, _reasoning = classify_branch({})
    assert active_branch == "unknown_selection"
    assert branch_conflict == []


def test_build_shortlist_prefers_material_product_families():
    shortlist = build_shortlist(
        "configuration",
        {"transport_type": "cargo", "body_type": "refrigerator"},
        [
            {"metadata": {"product_families": ["gazelle_next", "gazelle_nn"]}},
            {"metadata": {"product_families": ["gazelle_next"]}},
        ],
    )
    assert shortlist[0].family_id == "gazelle_next"
    assert len(shortlist) <= 3
    assert shortlist[0].fit_reason


def test_build_followup_is_role_aware():
    pack = build_followup(
        "internal_approval",
        {"decision_role": "finance director"},
        [
            {
                "candidate_id": "cand_1",
                "title": "Approval deck",
                "rationale": "supports internal approval reasoning",
            }
        ],
        [],
    )
    assert pack.recommended_action == "send_internal_approval_pack"
    assert pack.documents[0].why_it_matters


def test_locale_resolution_accepts_regional_variants():
    assert resolve_locale("ru-RU") == "ru"
    assert resolve_locale("en_US") == "en"
    assert resolve_locale("de-DE") == "ru"


def test_russian_locale_bundle_is_available():
    locale = get_locale("ru")
    assert "Здравствуйте" in locale["agent"]["opening_message"]
    assert "коммерческому транспорту ГАЗ" in get_prompt("ru", "system")
    assert "текущий запрос клиента" in get_prompt("ru", "turn_intent_extractor")
    assert "get_sales_landscape" in get_prompt("ru", "tool_landscape")
    assert "compare_product_directions" in get_prompt("ru", "tool_compare_directions")
    assert "collect_product_snapshot" in get_prompt("ru", "tool_snapshot")
    assert "немного подождать" in get_text("ru", "document_package_wait_question")
    assert "Deep research" not in get_text("ru", "document_package_wait_content")
    assert "{reason}" not in get_text("ru", "deep_comparison_wait_content")


def test_build_sales_context_baseline_filters_by_problem_summary():
    baseline = build_sales_context_baseline(
        "ru",
        {"transport_type": "passenger"},
        "Нужен пассажирский маршрут по городу",
        "overview",
    )
    assert baseline["product_groups"]
    assert any("пассаж" in group["title"].lower() for group in baseline["product_groups"])


def test_derive_research_layer_progresses_from_composite_to_followup():
    assert derive_research_layer(
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=False,
        has_material_reads=False,
        has_branch_pack=False,
        has_shortlist=False,
        has_followup=False,
    ) == "portfolio_baseline"
    assert derive_research_layer(
        has_sales_digest=True,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=False,
        has_material_reads=False,
        has_branch_pack=False,
        has_shortlist=False,
        has_followup=False,
    ) == "sales_landscape"
    assert derive_research_layer(
        has_sales_digest=False,
        has_comparison_digest=True,
        has_product_snapshot=False,
        has_material_candidates=True,
        has_material_reads=False,
        has_branch_pack=False,
        has_shortlist=False,
        has_followup=False,
    ) == "comparison_digest"
    assert derive_research_layer(
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=True,
        has_material_candidates=True,
        has_material_reads=True,
        has_branch_pack=False,
        has_shortlist=False,
        has_followup=False,
    ) == "product_snapshot"
    assert derive_research_layer(
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=True,
        has_material_reads=True,
        has_branch_pack=True,
        has_shortlist=False,
        has_followup=False,
    ) == "branch_pack"
    assert derive_research_layer(
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=True,
        has_material_reads=True,
        has_branch_pack=True,
        has_shortlist=True,
        has_followup=False,
    ) == "shortlist"
    assert derive_research_layer(
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=True,
        has_material_reads=True,
        has_branch_pack=True,
        has_shortlist=True,
        has_followup=True,
    ) == "followup"


def test_select_active_tool_names_for_overview_prefers_landscape():
    planned = build_allowed_tool_names("overview", "broad", "SELL")
    active = select_active_tool_names(
        "overview",
        "broad",
        "SELL",
        planned_tools=planned,
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=False,
        has_material_reads=False,
        has_branch_pack=False,
        has_shortlist=False,
        has_followup=False,
    )
    assert active == ["get_sales_catalog_overview", "get_sales_landscape"]


def test_select_active_tool_names_for_compare_starts_with_composite_comparison():
    planned = build_allowed_tool_names("compare", "justified", "RESEARCH")
    active = select_active_tool_names(
        "compare",
        "justified",
        "RESEARCH",
        planned_tools=planned,
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=False,
        has_material_reads=False,
        has_branch_pack=False,
        has_shortlist=False,
        has_followup=False,
    )
    assert active == ["compare_product_directions"]


def test_select_active_tool_names_for_specs_starts_with_snapshot():
    planned = build_allowed_tool_names("specs", "justified", "RESEARCH")
    active = select_active_tool_names(
        "specs",
        "justified",
        "RESEARCH",
        planned_tools=planned,
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=False,
        has_material_reads=False,
        has_branch_pack=False,
        has_shortlist=False,
        has_followup=False,
    )
    assert active == ["collect_product_snapshot"]


def test_select_active_tool_names_for_recommendation_prefers_composites_before_narrow_reads():
    planned = build_allowed_tool_names("recommendation", "justified", "RECOMMEND")
    active = select_active_tool_names(
        "recommendation",
        "justified",
        "RECOMMEND",
        planned_tools=planned,
        has_sales_digest=False,
        has_comparison_digest=False,
        has_product_snapshot=False,
        has_material_candidates=False,
        has_material_reads=False,
        has_branch_pack=False,
        has_shortlist=False,
        has_followup=False,
    )
    assert "get_sales_landscape" in active
    assert "compare_product_directions" in active
    assert "collect_product_snapshot" in active
    assert "read_material" not in active


def test_answer_plan_node_clamps_first_turn_and_skips_estimate_for_mixed_use():
    planner = DummyPlanner(
        {
            "current_client_intent": "compare",
            "answer_depth": "deep_research",
            "customer_temperature": "neutral",
            "clarification_allowed": True,
            "search_query": "городские смешанные заказы",
            "provisional_recommendations": [],
        }
    )
    docs_client = DummyDocsClient()
    node = create_answer_plan_node(planner, docs_client)
    result = node(
        {
            "locale": "ru",
            "intent_flags": {},
            "last_user_text": "Ну там по-разному бывает. Разные заказы же.",
            "problem_summary": "городские смешанные заказы",
            "slots": {"transport_type": "cargo"},
            "provisional_recommendations": [],
            "research_status": {},
            "material_candidates": [],
            "material_reads": [],
            "sales_digests": [],
            "comparison_digests": [],
            "product_snapshots": [],
            "shortlist": [],
            "followup_pack": {},
        },
        config={},
        runtime=SimpleNamespace(context=None),
    )
    assert result["planned_answer_depth"] == "deep_research"
    assert result["answer_depth"] == "bounded"
    assert result["needs_hitl_wait_confirmation"] is False
    assert docs_client.calls == []
    assert route_after_answer_plan(result) == "sales_response"


def test_provider_then_tool_uses_auto_schema_response_format():
    response_format = _structured_response_format(AnswerPlanResult, "provider_then_tool")
    assert response_format is AnswerPlanResult


def test_answer_plan_node_ignores_invalid_branch_hint_from_planner():
    planner = DummyPlanner(
        {
            "current_client_intent": "financing",
            "answer_depth": "bounded",
            "customer_temperature": "neutral",
            "clarification_allowed": True,
            "search_query": "?????? ??? ?????????? ???????",
            "branch_hint": "financing",
            "provisional_recommendations": ["gazelle_nn"],
        }
    )
    docs_client = DummyDocsClient()
    node = create_answer_plan_node(planner, docs_client)
    result = node(
        {
            "locale": "ru",
            "intent_flags": {"requested_financing": True},
            "last_user_text": "????? ?????? ?? ?????????? ???????.",
            "problem_summary": "????? ?????? ?? ?????????? ???????",
            "slots": {"transport_type": "cargo"},
            "provisional_recommendations": [],
            "research_status": {},
            "material_candidates": [],
            "material_reads": [],
            "sales_digests": [],
            "comparison_digests": [],
            "product_snapshots": [],
            "shortlist": [],
            "followup_pack": {},
        },
        config={},
        runtime=SimpleNamespace(context=None),
    )
    assert result["current_client_intent"] == "financing"
    assert result["active_branch"] is None
    assert result["answer_depth"] == "bounded"


def test_answer_plan_node_only_enables_hitl_after_prior_search_and_read():
    planner = DummyPlanner(
        {
            "current_client_intent": "materials",
            "answer_depth": "deep_research",
            "customer_temperature": "neutral",
            "clarification_allowed": False,
            "search_query": "пакет материалов для выбора городского фургона",
            "provisional_recommendations": ["gazelle_nn"],
        }
    )
    docs_client = DummyDocsClient(
        {
            "requires_hitl_wait_confirmation": True,
            "estimated_remaining_cost": "high",
            "rationale": "Deep research requested across a wide candidate set.",
        }
    )
    node = create_answer_plan_node(planner, docs_client)
    result = node(
        {
            "locale": "ru",
            "intent_flags": {"requested_materials": True},
            "last_user_text": "Пришлите нормальный пакет материалов и сравнение.",
            "problem_summary": "нужен пакет материалов по городскому фургону",
            "slots": {"transport_type": "cargo", "competitor": "sollers_atlant"},
            "provisional_recommendations": [],
            "research_status": {"has_prior_search": True, "has_prior_read": True},
            "material_candidates": [],
            "material_reads": [],
            "sales_digests": [],
            "comparison_digests": [],
            "product_snapshots": [],
            "shortlist": [],
            "followup_pack": {},
        },
        config={},
        runtime=SimpleNamespace(context=None),
    )
    assert docs_client.calls
    assert result["answer_depth"] == "justified"
    assert result["needs_hitl_wait_confirmation"] is True
    assert result["hitl_trigger_kind"] == "document_package_wait"
    assert route_after_answer_plan(result) == "hitl_wait"


def test_landscape_digest_aggregates_multi_direction_sales_summary():
    docs_client = DummyCompositeDocsClient()
    payload, update = _landscape_digest(
        "ru",
        dict(COMPOSITE_STATE),
        docs_client,
        topic="городская доставка",
        audience="owner",
        use_case="смешанные заказы по городу",
        focus="финансирование и типовые отличия",
    )
    assert payload["topic"] == "городская доставка"
    assert 1 <= len(payload["directions"]) <= 4
    assert payload["finance_options"]
    assert payload["recommended_next_narrowing"]
    assert payload["source_candidates"]
    assert update["sales_digests"]
    assert update["composite_tool_traces"]
    assert update["research_layer"] == "sales_landscape"
    assert 1 <= len(docs_client.search_calls) <= 3
    assert 1 <= len(docs_client.read_calls) <= 4


def test_compare_digest_builds_multi_product_comparison_digest():
    docs_client = DummyCompositeDocsClient()
    payload, update = _compare_digest(
        "ru",
        dict(COMPOSITE_STATE),
        docs_client,
        query="сравни Газель NN, Газель NEXT и Соболь для города против Atlant",
        families=["gazelle_nn", "gazelle_next", "sobol_nn"],
        competitor="Atlant",
        dimensions=["payload", "power", "finance"],
        top_families=3,
    )
    assert payload["products_compared"]
    assert len(payload["products_compared"]) >= 2
    assert payload["comparison_axes"]
    assert payload["high_level_differences"]
    assert payload["source_candidates"]
    assert update["comparison_digests"]
    assert update["research_layer"] == "comparison_digest"
    assert 1 <= len(docs_client.search_calls) <= 3
    assert 1 <= len(docs_client.read_calls) <= 4


def test_snapshot_digest_returns_ranges_or_baselines_for_multi_product_specs():
    docs_client = DummyCompositeDocsClient()
    payload, update = _snapshot_digest(
        "ru",
        dict(COMPOSITE_STATE),
        docs_client,
        query="дай размеры, мощность и грузоподъемность по нескольким вариантам",
        families=["gazelle_nn", "gazelle_next", "sobol_nn"],
        dimensions=["dimensions", "power", "payload"],
        competitor="",
        max_products=3,
    )
    assert payload["products"]
    assert payload["value_ranges_or_baselines"]
    assert payload["assumptions"]
    assert payload["source_candidates"]
    assert update["product_snapshots"]
    assert update["research_layer"] == "product_snapshot"
    assert 1 <= len(docs_client.search_calls) <= 3
    assert 1 <= len(docs_client.read_calls) <= 4


def test_compact_payload_reuses_latest_composite_digests():
    state = {
        **dict(COMPOSITE_STATE),
        "stage": "SELL",
        "current_client_intent": "compare",
        "answer_depth": "bounded",
        "customer_temperature": "neutral",
        "sales_digests": [
            {
                "topic": "городская доставка",
                "directions": [{"title": "Light cargo", "families": ["gazelle_nn"], "main_characteristics": ["маневренность"], "key_tradeoffs": ["payload margin"]}],
                "finance_options": ["лизинг"],
                "recommended_next_narrowing": "тип кузова",
            }
        ],
        "comparison_digests": [
            {
                "query": "сравни Газель NN и NEXT",
                "products_compared": [{"family_id": "gazelle_nn", "label": "Gazelle NN", "differentiators": ["city chassis"]}],
                "comparison_axes": ["payload", "power"],
                "high_level_differences": ["Gazelle NN: city chassis"],
                "assumptions": ["baseline variants"],
            }
        ],
        "product_snapshots": [
            {
                "query": "мощность и габариты",
                "dimensions_requested": ["power", "dimensions"],
                "products": [{"family_id": "gazelle_nn", "label": "Gazelle NN", "facts": ["Payload up to 1.5t"]}],
                "value_ranges_or_baselines": [{"family_id": "gazelle_nn", "label": "Gazelle NN", "dimension": "payload", "evidence": ["Payload up to 1.5t"]}],
                "assumptions": ["baseline"],
            }
        ],
    }
    payload = _compact_payload(state)
    assert payload["sales_digest"]["directions"]
    assert payload["comparison_digest"]["products_compared"]
    assert payload["product_snapshot"]["products"]


def test_is_question_policy_violation_detects_question_budget_feedback():
    assert _is_question_policy_violation("More than one clarifying question asked at the end.") is True
    assert _is_question_policy_violation("Draft answer repeats email request twice.") is True
    assert _is_question_policy_violation("Unsupported document claim about payload.") is False


def test_soft_question_validation_is_question_only():
    assert _is_soft_question_validation({"violations": ["More than one clarifying question asked at the end."]}) is True
    assert _is_soft_question_validation({"violations": ["More than one clarifying question asked at the end.", "Unsupported document claim."]}) is False


def test_question_guidance_prefers_suggested_fix_then_locale_default():
    assert _question_guidance("en", {"suggested_fix": "Ask for email only once next turn."}) == "Ask for email only once next turn."
    assert _question_guidance("en", {"suggested_fix": None}) == get_prompt("en", "question_budget_guidance")


def test_response_finalize_repairs_invalid_answer_without_breaking_dialog():
    validator = DummyValidatorSequence([
        {
            "is_valid": False,
            "violations": ["Unsupported exact numeric claim in the draft answer."],
            "suggested_fix": "Remove unsupported exact numbers and keep only careful orientation.",
        },
        {
            "is_valid": True,
            "violations": [],
            "suggested_fix": None,
        },
    ])
    repair_agent = DummyRepairAgent("Safer rewritten answer without unsupported exact numbers.")
    node = create_response_finalize_node(validator, repair_agent)
    state = {
        "locale": "en",
        "stage": "SELL",
        "messages": [AIMessage(content="Bad answer with exact 1.9 t claim.")],
        "allowed_tool_names": [],
        "tool_calls_this_turn": [],
        "sales_context_baseline": {"product_groups": [{"title": "Light commercial vans and chassis"}]},
    }

    result = node(state, {}, SimpleNamespace())

    assert validator.calls == 2
    assert repair_agent.calls == 1
    assert any(isinstance(item, AIMessage) and "Safer rewritten answer" in item.content for item in result["messages"])
    assert "sales_answer_repaired_after_validation" in result["trace"][-1]["policy_checks_passed"]


def test_response_finalize_keeps_repaired_answer_when_repair_still_fails_validation():
    validator = DummyValidatorSequence([
        {
            "is_valid": False,
            "violations": ["Unsupported exact numeric claim in the draft answer."],
            "suggested_fix": "Remove unsupported exact numbers and keep only careful orientation.",
        },
        {
            "is_valid": False,
            "violations": ["Unsupported exact numeric claim in the repaired answer."],
            "suggested_fix": None,
        },
    ])
    repair_agent = DummyRepairAgent("Still bad answer with exact 1.9 t claim.")
    node = create_response_finalize_node(validator, repair_agent)
    state = {
        "locale": "en",
        "stage": "SELL",
        "messages": [AIMessage(content="Bad answer with exact 1.9 t claim.")],
        "allowed_tool_names": [],
        "tool_calls_this_turn": [],
        "sales_context_baseline": {"product_groups": [{"title": "Light commercial vans and chassis"}]},
    }

    result = node(state, {}, SimpleNamespace())

    assert validator.calls == 2
    assert repair_agent.calls == 1
    assert any(isinstance(item, AIMessage) and "Still bad answer with exact 1.9 t claim." in item.content for item in result["messages"])
    assert "sales_answer_repair_attempt_failed_but_sent_as_is" in result["trace"][-1]["policy_checks_passed"]

def test_turn_intent_node_falls_back_when_structured_output_missing():
    extractor = DummyAgentSequence([{}, {}])
    node = create_turn_intent_node(extractor)
    state = {
        "locale": "en",
        "last_user_text": "Compare the practical difference between Gazelle NN and Sobol.",
        "slots": {"transport_type": "cargo"},
        "intent_flags": {"requested_specs": False, "requested_comparison_table": True},
    }

    result = node(state, {}, SimpleNamespace())

    assert extractor.calls == 2
    assert result["current_client_intent"] == "compare"
    assert result["runtime_warnings"]
    assert result["llm_retry_instruction"] == ""


def test_answer_plan_node_falls_back_when_structured_output_missing():
    planner = DummyAgentSequence([{}, {}])
    docs_client = DummyDocsClient()
    node = create_answer_plan_node(planner, docs_client)
    state = {
        "locale": "en",
        "last_user_text": "Show what you can offer for city delivery.",
        "problem_summary": "city delivery",
        "slots": {"transport_type": "cargo"},
        "intent_flags": {"requested_portfolio_overview": True},
        "current_client_intent": "overview",
        "research_status": {},
        "material_candidates": [],
        "material_reads": [],
        "sales_digests": [],
        "comparison_digests": [],
        "product_snapshots": [],
        "shortlist": [],
        "followup_pack": {},
    }

    result = node(state, {}, SimpleNamespace())

    assert planner.calls == 2
    assert result["stage"] == "SELL"
    assert result["answer_depth"] == "broad"
    assert result["runtime_warnings"]
    assert docs_client.calls == []


def test_response_finalize_skips_validation_when_validator_returns_no_structured_response():
    validator = DummyValidatorSequence([{}, {}])
    repair_agent = DummyRepairAgent("unused")
    node = create_response_finalize_node(validator, repair_agent)
    state = {
        "locale": "en",
        "stage": "SELL",
        "messages": [AIMessage(content="Usable draft answer.")],
        "allowed_tool_names": [],
        "tool_calls_this_turn": [],
        "sales_context_baseline": {"product_groups": [{"title": "Light commercial vans and chassis"}]},
    }

    result = node(state, {}, SimpleNamespace())

    assert validator.calls == 2
    assert repair_agent.calls == 0
    assert "validator_nonblocking_warning_recorded" in result["trace"][-1]["policy_checks_passed"]
    assert result["runtime_warnings"]


def test_sales_response_node_uses_continue_agent_when_main_agent_returns_no_answer():
    sales_agent = DummyAgentSequence([{}, {}])
    continue_agent = DummyAgentSequence([{"messages": [AIMessage(content="Here is a short practical answer.")]}])
    node = create_sales_response_node(sales_agent, continue_agent)
    state = {
        "locale": "en",
        "stage": "SELL",
        "current_client_intent": "overview",
        "answer_depth": "broad",
    }

    result = node(state, {}, SimpleNamespace())

    assert sales_agent.calls == 2
    assert continue_agent.calls == 1
    assert any(isinstance(item, AIMessage) and "short practical answer" in item.content for item in result["messages"])
    assert result["runtime_warnings"]

def test_ru_locale_has_repair_and_question_budget_prompts():
    locale = get_locale("ru")
    assert locale["prompts"]["sales_repair_agent"]
    assert locale["prompts"]["question_budget_guidance"]


def test_search_sales_materials_blocks_duplicate_query_in_same_turn():
    docs_client = DummyCompositeDocsClient()
    tool = build_search_sales_materials_tool(docs_client)
    base_state = {
        "problem_summary": "????????? ????????",
        "slots": {"transport_type": "cargo"},
        "provisional_recommendations": [],
        "research_status": {},
        "search_keys_this_turn": [],
        "tool_calls_this_turn": [],
    }

    first = tool.func(
        query="???????? ?????? NN",
        intent="specs",
        families=["gazelle_nn"],
        competitor="",
        top_k=4,
        runtime=_runtime(base_state),
    )
    state_after_first = _merge_command_state(base_state, first)
    second = tool.func(
        query="???????? ?????? NN",
        intent="specs",
        families=["gazelle_nn"],
        competitor="",
        top_k=4,
        runtime=_runtime(state_after_first),
    )

    assert len(docs_client.search_calls) == 1
    assert second.update["sales_loop_guard_reason"] == "duplicate_search_attempt"
    assert second.update["tool_limit_hits"][-1]["reason"] == "duplicate_search_attempt"


def test_read_material_blocks_duplicate_focus_in_same_turn():
    docs_client = DummyCompositeDocsClient()
    tool = build_read_material_tool(docs_client)
    base_state = {
        "allowed_material_ids": ["cand_gnn_cfg"],
        "read_attempts_by_candidate": {},
        "read_focus_keys_this_turn": [],
        "research_status": {},
        "material_reads": [],
        "tool_calls_this_turn": [],
    }

    first = tool.func(
        candidate_id="cand_gnn_cfg",
        focus="????? ???????? ?????????",
        max_segments=3,
        runtime=_runtime(base_state),
    )
    state_after_first = _merge_command_state(base_state, first)
    second = tool.func(
        candidate_id="cand_gnn_cfg",
        focus="????? ???????? ?????????",
        max_segments=3,
        runtime=_runtime(state_after_first),
    )

    assert len(docs_client.read_calls) == 1
    assert second.update["sales_loop_guard_reason"] == "duplicate_read_attempt"
    assert second.update["tool_limit_hits"][-1]["reason"] == "duplicate_read_attempt"


def test_read_material_blocks_third_read_of_same_candidate():
    docs_client = DummyCompositeDocsClient()
    tool = build_read_material_tool(docs_client)
    base_state = {
        "allowed_material_ids": ["cand_gnn_cfg"],
        "read_attempts_by_candidate": {},
        "read_focus_keys_this_turn": [],
        "research_status": {},
        "material_reads": [],
        "tool_calls_this_turn": [],
    }

    first = tool.func(
        candidate_id="cand_gnn_cfg",
        focus="????? ???????? ?????????",
        max_segments=3,
        runtime=_runtime(base_state),
    )
    state_after_first = _merge_command_state(base_state, first)
    second = tool.func(
        candidate_id="cand_gnn_cfg",
        focus="????? ????????",
        max_segments=3,
        runtime=_runtime(state_after_first),
    )
    state_after_second = _merge_command_state(state_after_first, second)
    third = tool.func(
        candidate_id="cand_gnn_cfg",
        focus="????? ???????????? ????????",
        max_segments=3,
        runtime=_runtime(state_after_second),
    )

    assert len(docs_client.read_calls) == 2
    assert third.update["sales_loop_guard_reason"] == "candidate_read_budget_exhausted"
    assert third.update["tool_limit_hits"][-1]["reason"] == "candidate_read_budget_exhausted"


def test_sales_response_node_falls_back_to_continue_after_graph_recursion_limit():
    sales_agent = DummyLoopAgent()
    continue_agent = DummyAgentSequence([{
        "messages": [AIMessage(content="????????? ??????? ???????? ????????????, ? ?????? SLA ????? ???????????? ????????, ?? ?? ???????? ? ??????? ????????? ?????? ??? ????? ???????????????.")]
    }])
    node = create_sales_response_node(sales_agent, continue_agent)
    state = {
        "locale": "ru",
        "stage": "RESEARCH",
        "current_client_intent": "materials",
        "answer_depth": "bounded",
        "runtime_warnings": [],
        "messages": [],
    }

    result = node(state, {}, SimpleNamespace())

    assert sales_agent.calls == 1
    assert continue_agent.calls == 1
    assert any(isinstance(item, AIMessage) and "?????? SLA" in item.content for item in result["messages"])
    codes = {item["code"] for item in result["runtime_warnings"]}
    assert "graph_recursion_limit" in codes
    assert "sales_response_fallback_to_continue" in codes


def test_response_finalize_keeps_original_answer_when_repair_agent_fails():
    validator = DummyValidatorSequence([
        {
            "is_valid": False,
            "violations": ["Unsupported exact numeric claim in the draft answer."],
            "suggested_fix": "Remove unsupported exact numbers and keep only careful orientation.",
        }
    ])
    repair_agent = DummyFailingRepairAgent()
    node = create_response_finalize_node(validator, repair_agent)
    state = {
        "locale": "en",
        "stage": "SELL",
        "messages": [AIMessage(content="Bad answer with exact 1.9 t claim.")],
        "allowed_tool_names": [],
        "tool_calls_this_turn": [],
        "sales_context_baseline": {"product_groups": [{"title": "Light commercial vans and chassis"}]},
    }

    result = node(state, {}, SimpleNamespace())

    assert repair_agent.calls == 1
    assert "messages" not in result
    assert "sales_answer_kept_as_is_after_failed_repair_attempt" in result["trace"][-1]["policy_checks_passed"]
    codes = {item["code"] for item in result["runtime_warnings"]}
    assert "repair_path_failed" in codes
