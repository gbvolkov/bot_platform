from __future__ import annotations

from simulate_privacy_flow import PrivacyFlowParams, simulate_flow


def test_simulate_privacy_flow_reports_real_palimpsest_outputs():
    params = PrivacyFlowParams(
        system_prompt="You are a CRM assistant for Иван Петров.",
        user_request="Find Иван Петров by phone +7 999 111-22-33.",
        tool_call_parameters={"query": "Иван Петров", "phone": "+7 999 111-22-33"},
        tool_calls_after_anonymization={"query": "Иван Петров", "phone": "+7 999 111-22-33"},
        raw_llm_response="Found Иван Петров.",
        user_llm_response="Found Иван Петров.",
        tool_result="CRM record: Иван Петров, phone +7 999 111-22-33.",
        palimpsest_run_entities=["RU_PERSON", "PHONE_NUMBER"],
    )

    result = simulate_flow(params)

    assert "palimpsest_table" in result
    assert "palimpsest_tool_call_parameters_anonymized" in result
    assert "palimpsest_raw_llm_response_deanonymized" in result
    assert result["model_visible_tool_call"]["args"] == result["tool_execution_parameters"]
    assert set(result["checks"]) == {
        "system_prompt_changed_by_palimpsest",
        "user_request_changed_by_palimpsest",
        "provided_tool_calls_match_palimpsest_anonymization",
        "model_tool_call_deanonymized",
        "tool_execution_parameters_deanonymized",
        "tool_result_changed_by_palimpsest_for_llm",
        "provided_user_response_matches_palimpsest_deanonymization",
        "final_response_deanonymized",
    }
