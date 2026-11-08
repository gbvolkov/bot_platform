"""Offline LangChain request contracts for instance-scoped named models."""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from platform_capabilities.models import get_llm, get_model


def test_named_model_supports_responses_reasoning_and_function_tools(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")

    @tool
    def read_note(path: str) -> str:
        """Read a saved note by its path."""
        return path

    model = get_llm(model="base", provider="openai", model_name="gpt-6-luna",
                    reasoning="medium", temperature=None, streaming=False)
    bound = model.bind_tools([read_note])
    payload = model._get_request_payload([HumanMessage(content="Read the current note")], **bound.kwargs)
    assert model.use_responses_api is True
    assert model.use_previous_response_id is False
    assert payload["model"] == "gpt-6-luna" and payload["reasoning"] == {"effort": "medium"}
    assert "temperature" not in payload and "top_p" not in payload and "top_logprobs" not in payload
    assert [entry["name"] for entry in payload["tools"]] == ["read_note"]
    assert payload["tools"][0]["parameters"]["properties"]["path"]["type"] == "string"


def test_instance_override_preserves_registry_and_existing_callers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    base_name, summary_name = get_model("openai", "base"), get_model("openai", "mini")
    custom = get_llm(model_name="gpt-6-luna", provider="openai", reasoning="medium", temperature=None)
    base = get_llm(provider="openai", streaming=False)
    summary = get_llm(model="mini", provider="openai", streaming=False)
    assert custom.model_name == "gpt-6-luna"
    assert base.model_name == base_name and base.reasoning == {"effort": "none"}
    assert summary.model_name == summary_name and summary.reasoning == {"effort": "minimal"}
    assert get_model("openai", "base") == base_name
    assert get_model("openai", "mini") == summary_name
