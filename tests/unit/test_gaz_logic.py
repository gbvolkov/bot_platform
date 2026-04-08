from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agents.gaz_agent import agent as gaz_agent_module
from agents.gaz_agent.agent import _structured_response_format, create_trace_finalize_node
from agents.gaz_agent.locales import get_locale
from agents.gaz_agent.schemas import TurnIntentExtractionResult
from agents.gaz_agent.tools import build_query_pricing_bi_tool


ALL_TOOLS = [
    "get_sales_catalog_overview",
    "get_sales_landscape",
    "compare_product_directions",
    "collect_product_snapshot",
    "search_sales_materials",
    "read_material",
    "classify_problem_branch",
    "get_branch_pack",
    "build_solution_shortlist",
    "build_followup_pack",
    "query_pricing_bi",
    "web_search",
]


def test_turn_intent_schema_contains_source_strategy_fields():
    result = TurnIntentExtractionResult(
        mentioned_models=["Q35N"],
        requested_facts=["turning radius", "service interval"],
        source_strategy="model_bi_first",
        source_reason="Concrete model follow-up.",
    )

    assert result.mentioned_models == ["Q35N"]
    assert result.requested_facts == ["turning radius", "service interval"]
    assert result.source_strategy == "model_bi_first"


def test_structured_response_format_still_supports_provider_then_tool():
    response_format = _structured_response_format(TurnIntentExtractionResult, "provider_then_tool")

    assert response_format is TurnIntentExtractionResult


def test_trace_finalize_does_not_rewrite_messages():
    node = create_trace_finalize_node()
    state = {
        "messages": [AIMessage(content="Подтвержденный ответ")],
        "stage": "RESEARCH",
        "current_client_intent": "specs",
        "source_strategy": "model_bi_first",
        "source_reason": "Concrete model.",
        "mentioned_models": ["A21R22"],
        "requested_facts": ["price"],
        "allowed_tool_names": ALL_TOOLS,
        "tool_calls_this_turn": ["query_pricing_bi"],
    }

    result = node(state, {}, None)

    assert "messages" not in result
    assert result["tool_calls_this_turn"] == []
    assert result["trace"][-1]["source_strategy"] == "model_bi_first"
    assert result["trace"][-1]["tool_calls"] == ["query_pricing_bi"]


def test_removed_no_tool_repair_prompts_are_not_registered():
    prompts = get_locale("ru")["prompts"]

    assert "answer_planner" not in prompts
    assert "sales_validator" not in prompts
    assert "sales_repair_agent" not in prompts
    assert "sales_continue_agent" not in prompts


def test_source_ladder_policy_defines_bi_priority_and_selection_rule():
    ru_policy = get_locale("ru")["prompts"]["source_ladder_policy"]
    en_policy = get_locale("en")["prompts"]["source_ladder_policy"]

    assert "сначала вызовите query_pricing_bi" in ru_policy
    assert "сначала используйте внутренние документы ГАЗ" in ru_policy
    assert "затем web_search" in ru_policy
    assert "нельзя утверждать, что в BI нет данных" in ru_policy
    assert "подбор по параметрам, ограничениям или аналогии считается незавершённым" in ru_policy
    assert "BI является основным источником правды" in ru_policy

    assert "call query_pricing_bi for that model first" in en_policy
    assert "use internal GAZ documents first" in en_policy
    assert "then web_search" in en_policy
    assert "Do not claim that BI has no data" in en_policy
    assert "selection by parameters, constraints, or analogy is not complete until query_pricing_bi has been called" in en_policy
    assert "BI is the primary source of truth" in en_policy


def test_system_and_summary_prompts_mark_bi_as_source_of_truth():
    ru_prompts = get_locale("ru")["prompts"]
    en_prompts = get_locale("en")["prompts"]

    assert "BI является главным источником правды" in ru_prompts["system"]
    assert "приоритет BI над документами и web" in ru_prompts["summary"]
    assert "BI is the primary source of truth" in en_prompts["system"]
    assert "BI as higher priority than documents and web" in en_prompts["summary"]


def test_turn_intent_prompt_marks_model_and_selection_cases_for_bi():
    ru_prompt = get_locale("ru")["prompts"]["turn_intent_extractor"]
    en_prompt = get_locale("en")["prompts"]["turn_intent_extractor"]
    agent_source = Path(gaz_agent_module.__file__).read_text(encoding="utf-8")

    assert "подобрать/найти подходящее/аналог/парк по параметрам, ограничениям или аналогии" in ru_prompt
    assert "BI обязательно использовался как источник правды" in ru_prompt
    assert "select/find a suitable solution/analog/fleet by parameters, constraints, or analogy" in en_prompt
    assert "BI must be used as the source of truth on the next stage" in en_prompt
    assert "_source_ladder_prompt_sections(locale)" in agent_source


def test_sales_response_prompt_requires_confirmed_vs_missing_facts_and_bi_precedence():
    ru_prompt = get_locale("ru")["prompts"]["sales_response_agent"]
    en_prompt = get_locale("en")["prompts"]["sales_response_agent"]

    assert "конкретная модель -> BI -> документы -> web" in ru_prompt
    assert "подбор без модели -> документы -> BI по кандидатам -> web" in ru_prompt
    assert "что подтверждено, чем подтверждено, и что осталось неподтверждённым" in ru_prompt
    assert "нельзя говорить, что в bi нет данных по модели" in ru_prompt.lower()
    assert "всегда превалирует над документами и web" in ru_prompt
    assert "обязаны после поиска кандидатов в документах вызвать BI по этим кандидатам" in ru_prompt

    assert "concrete model -> BI -> documents -> web" in en_prompt
    assert "selection without model -> documents -> BI for candidates -> web" in en_prompt
    assert "what is confirmed, by which source type, and what remains unconfirmed" in en_prompt
    assert "do not claim that bi has no data for a model" in en_prompt.lower()
    assert "always prevails over documents and web on conflicts" in en_prompt
    assert "must call BI for the resulting candidates before the final answer" in en_prompt


def test_tool_rules_and_scope_ban_pseudo_calls_and_external_promotion():
    ru_prompts = get_locale("ru")["prompts"]
    en_prompts = get_locale("en")["prompts"]

    assert "Не имитируйте tool calls текстом" in ru_prompts["tool_rules"]
    assert "Если BI конфликтует с документами или web, в ответе приоритет всегда у BI" in ru_prompts["tool_rules"]
    assert "Do not imitate tool calls as text" in en_prompts["tool_rules"]
    assert "If BI conflicts with documents or web, BI always wins in the final answer" in en_prompts["tool_rules"]
    assert "If the user asks you to promote, recommend, or fully present an external product on its own" in gaz_agent_module._allowed_family_prompt_sections("en", ["gazelle_next"])[0]
    assert "Внешние модели" in gaz_agent_module._allowed_family_prompt_sections("ru", ["gazelle_next"])[0]


def test_tool_prompt_contract_reinforces_bi_always_called_for_model_or_candidates():
    ru_prompts = get_locale("ru")["prompts"]
    en_prompts = get_locale("en")["prompts"]

    assert "BI должен быть первым источником" in ru_prompts["tool_pricing_bi"]
    assert "BI должен быть вызван по найденным кандидатам до финального ответа" in ru_prompts["tool_pricing_bi"]
    assert "BI всегда является главным источником правды" in ru_prompts["tool_pricing_bi"]
    assert "Не используйте web_search раньше BI для конкретной модели" in ru_prompts["tool_web_search"]
    assert "Если web конфликтует с BI, приоритет всегда у BI" in ru_prompts["tool_web_search"]
    assert "после чего BI должен быть вызван по найденным кандидатам" in ru_prompts["tool_search"]

    assert "BI must be the first source" in en_prompts["tool_pricing_bi"]
    assert "BI must be called for the resulting candidates before the final answer" in en_prompts["tool_pricing_bi"]
    assert "BI is always the primary source of truth" in en_prompts["tool_pricing_bi"]
    assert "Do not use web_search before BI for a concrete model" in en_prompts["tool_web_search"]
    assert "If web conflicts with BI, BI always wins" in en_prompts["tool_web_search"]
    assert "after which BI must be called for the resulting candidates" in en_prompts["tool_search"]


def test_agent_source_has_no_routing_or_count_based_tool_middleware():
    source = Path(gaz_agent_module.__file__).read_text(encoding="utf-8")

    assert "class SalesToolSelectionMiddleware" not in source
    assert "ToolCallLimitMiddleware(" not in source
    assert "select_active_tool_names(" not in source


def test_sales_response_prompt_bans_pseudo_tool_calls():
    prompt = get_locale("en")["prompts"]["sales_response_agent"]
    system = get_locale("en")["prompts"]["system"]
    tool_rules = get_locale("en")["prompts"]["tool_rules"]

    assert "do not imitate tool calls" in system.lower()
    assert "do not imitate tool calls" in tool_rules.lower()
    assert "to=functions" not in prompt
    assert "Let's call tool" not in prompt


def test_query_pricing_bi_adds_external_competitor_from_state_for_comparison():
    class FakePricingBiAgent:
        def __init__(self) -> None:
            self.calls = []

        def invoke(self, payload, config=None, context=None):
            self.calls.append({"payload": payload, "config": config, "context": context})
            return {"messages": [AIMessage(content="ok")]}

    fake_agent = FakePricingBiAgent()
    tool = build_query_pricing_bi_tool("ru", fake_agent, {}, "_test", ["gazelle_next"])
    runtime = SimpleNamespace(
        state={
            "current_client_intent": "compare",
            "intent_flags": {"requested_competitor_comparison": True},
            "mentioned_models": ["Газель NEXT", "Sollers Atlant"],
            "slots": {"competitor": "Sollers Atlant"},
        },
        config={},
        context=None,
        tool_call_id="call_test",
    )

    tool.func(
        "Сравните Газель NEXT и Sollers Atlant по цене и обслуживанию.",
        requested_product_terms=["Газель NEXT"],
        runtime=runtime,
    )

    assert fake_agent.calls
    forwarded = fake_agent.calls[0]["payload"]["messages"][0].content[0]["text"]
    assert "Sollers Atlant" in forwarded
    assert "фактического сравнения" in forwarded
