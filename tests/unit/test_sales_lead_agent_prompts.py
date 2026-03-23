from agents.sales_lead_agent.prompts import build_system_prompt


def test_build_system_prompt_instructs_immediate_purchase_search_for_procurement_turn():
    prompt = build_system_prompt(
        {
            "task_understanding": {
                "task_kind": "procurement_search",
                "search_url": None,
                "search_filters": {"query_text": "transport insurance"},
            },
            "turn_tool_requirements": {
                "purchase_search_required": True,
            },
            "turn_tool_usage": [],
        }
    )

    assert "call purchase_search_tool immediately" in prompt
    assert "must not block the first acquisition attempt" in prompt
