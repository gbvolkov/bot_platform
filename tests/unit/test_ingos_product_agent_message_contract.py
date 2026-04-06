from __future__ import annotations

from langchain_core.messages import HumanMessage

from agents.ingos_product_agent import agent as ingos_agent


def test_reset_or_run_accepts_string_content():
    state = {"messages": [HumanMessage(content="reset")]}

    result = ingos_agent.reset_or_run(state, {})

    assert result == "reset_memory"


def test_reset_or_run_accepts_content_parts():
    state = {
        "messages": [
            HumanMessage(content=[{"type": "reset", "text": "RESET"}]),
        ]
    }

    result = ingos_agent.reset_or_run(state, {})

    assert result == "reset_memory"


def test_content_text_accepts_string_and_content_parts():
    assert ingos_agent._content_text("plain text") == "plain text"
    assert (
        ingos_agent._content_text(
            [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        )
        == "first\nsecond"
    )
