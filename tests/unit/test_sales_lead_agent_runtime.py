from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langchain.agents.middleware import ToolRetryMiddleware
from langchain.tools import tool
from langchain_core.messages import ToolMessage
import pytest

from agents.sales_lead_agent import agent as sales_agent
from agents.sales_lead_agent.agent import ToolErrorJsonMiddleware, initialize_agent
from agents.sales_lead_agent.prompts import build_system_prompt
from agents.sales_lead_agent.tools import ToolUserCorrectableError
from agents.utils import ModelType


@tool
def ping_tool() -> str:
    """Return a static value."""
    return "pong"


def test_system_prompt_contains_minimal_tool_only_contract():
    prompt = build_system_prompt()

    assert "There is no hidden orchestration" in prompt
    assert "ok=false" in prompt
    assert "reusing `run_id` and `index_id`" in prompt
    assert "страхован" in prompt
    assert "страхов" in prompt
    assert "Zakupki already applies morphology" in prompt
    assert "AND semantics, not OR" in prompt
    assert "query_texts" in prompt


def test_initialize_agent_returns_compiled_agent():
    agent = initialize_agent(
        provider=ModelType.GPT,
        model_size="mini",
        tools=[ping_tool],
        streaming=False,
    )

    assert agent is not None
    assert hasattr(agent, "invoke")


def test_initialize_agent_does_not_swallow_unexpected_kwargs():
    with pytest.raises(TypeError):
        initialize_agent(provider=ModelType.GPT, unknown_argument=True)


def test_initialize_agent_preserves_explicit_empty_values(monkeypatch):
    captured = {}

    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "llm")

    class FakeAgent:
        def with_config(self, config):
            captured["config"] = config
            return self

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(sales_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(
        sales_agent,
        "build_sales_lead_tools",
        lambda: (_ for _ in ()).throw(AssertionError("default tools must not be built")),
    )

    agent = initialize_agent(
        provider=ModelType.GPT,
        model_size="mini",
        tools=[],
        system_prompt="",
        checkpoint_saver=False,
        streaming=False,
    )

    assert isinstance(agent, FakeAgent)
    assert captured["tools"] == []
    assert captured["system_prompt"] == ""
    assert len(captured["middleware"]) == 2
    assert isinstance(captured["middleware"][0], ToolErrorJsonMiddleware)
    assert isinstance(captured["middleware"][1], ToolRetryMiddleware)
    assert captured["checkpointer"] is False


def test_tool_error_middleware_returns_json_error_tool_message():
    middleware = ToolErrorJsonMiddleware()
    request = SimpleNamespace(tool_call={"id": "call-1", "name": "purchase_search_tool", "args": {"query_texts": ["страхован"]}})

    async def handler(_request):
        raise ToolUserCorrectableError(
            code="INVALID_QUERY_TEXT",
            message="query_texts must not be empty",
            suggestion="Provide a non-empty query_texts list and call the tool again.",
            input_field="query_texts",
        )

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload == {
        "ok": False,
        "error_code": "INVALID_QUERY_TEXT",
        "message": "query_texts must not be empty",
        "retryable": True,
        "suggestion": "Provide a non-empty query_texts list and call the tool again.",
        "input_field": "query_texts",
    }


def test_tool_error_middleware_does_not_mask_programmer_errors():
    middleware = ToolErrorJsonMiddleware()
    request = SimpleNamespace(tool_call={"id": "call-1", "name": "purchase_search_tool", "args": {"query_texts": ["страхован"]}})

    async def handler(_request):
        raise AssertionError("bug")

    with pytest.raises(AssertionError, match="bug"):
        asyncio.run(middleware.awrap_tool_call(request, handler))
