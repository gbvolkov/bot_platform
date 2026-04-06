from __future__ import annotations

from types import SimpleNamespace

from agents.mycroft_agent.agent import _build_callback_handlers, initialize_agent
from agents.mycroft_agent.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_WITHOUT_WEB_SEARCH,
    build_delegate_system_prompt,
    build_delegate_tool_description,
    build_gaz_mycroft_system_prompt,
)
from agents.utils import ModelType


def test_initialize_agent_passes_stateless_and_stateful_subagents(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr("agents.mycroft_agent.agent.config.LANGFUSE_URL", "")
    monkeypatch.setattr("agents.mycroft_agent.agent.config.OPENAI_API_KEY", "test-openai-key")

    monkeypatch.setattr(
        "agents.mycroft_agent.agent.get_llm",
        lambda **kwargs: {"llm": kwargs},
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or {"agent": kwargs},
    )

    web_search_tool = SimpleNamespace(name="web_search")
    stateless_subagent = {"name": "simple_agent", "description": "Simple Agent", "runnable": object()}
    stateful_subagent = {"name": "product_agent", "description": "Product Agent", "runnable": object()}

    result = initialize_agent(
        provider=ModelType.GPT,
        model_size="mini",
        temperature=0.1,
        tools=[web_search_tool],
        stateless_subagents=[stateless_subagent],
        stateful_subagents=[stateful_subagent],
        checkpoint_saver="checkpoint",
        interrupt_on={"gmail_send_message": {"allowed_decisions": ["approve", "edit", "reject"]}},
    )

    assert result["agent"]["tools"] == [web_search_tool]
    assert captured["subagents"] == [stateless_subagent]
    assert len(captured["middleware"]) == 1
    delegate_middleware = captured["middleware"][0]
    assert delegate_middleware.tools[0].name == "delegate"
    assert delegate_middleware.tools[0].description == build_delegate_tool_description(
        tool_name="delegate"
    ).format(available_agents="- product_agent: Product Agent")
    assert delegate_middleware.system_prompt == build_delegate_system_prompt(
        tool_name="delegate"
    )
    assert captured["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    assert captured["checkpointer"] == "checkpoint"
    assert captured["interrupt_on"] == {
        "gmail_send_message": {"allowed_decisions": ["approve", "edit", "reject"]}
    }


def test_initialize_agent_uses_default_prompt_without_web_search(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr("agents.mycroft_agent.agent.config.LANGFUSE_URL", "")
    monkeypatch.setattr("agents.mycroft_agent.agent.config.OPENAI_API_KEY", "test-openai-key")

    monkeypatch.setattr(
        "agents.mycroft_agent.agent.get_llm",
        lambda **kwargs: {"llm": kwargs},
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or {"agent": kwargs},
    )

    initialize_agent(provider=ModelType.GPT, tools=[])

    assert captured["system_prompt"] == DEFAULT_SYSTEM_PROMPT_WITHOUT_WEB_SEARCH
    assert captured["subagents"] == []
    assert captured["middleware"] == []


def test_initialize_agent_adds_langfuse_callbacks(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr("agents.mycroft_agent.agent.config.LANGFUSE_URL", "https://langfuse.local")
    monkeypatch.setattr("agents.mycroft_agent.agent.config.LANGFUSE_PUBLIC", "public")
    monkeypatch.setattr("agents.mycroft_agent.agent.config.LANGFUSE_SECRET", "secret")
    monkeypatch.setattr("agents.mycroft_agent.agent.config.OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        "agents.mycroft_agent.agent.Langfuse",
        lambda **kwargs: captured.update({"langfuse_kwargs": kwargs}) or object(),
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.agent.LangfuseCallbackHandler",
        lambda: "langfuse_handler",
    )
    monkeypatch.setattr(
        "agents.mycroft_agent.agent.get_llm",
        lambda **kwargs: captured.update({"llm_kwargs": kwargs}) or {"llm": kwargs},
    )

    class FakeAgent:
        def with_config(self, value):
            captured["agent_callbacks"] = value
            return {"configured": value}

    monkeypatch.setattr(
        "agents.mycroft_agent.agent.create_deep_agent",
        lambda **kwargs: FakeAgent(),
    )

    result = initialize_agent(provider=ModelType.GPT, tools=[])

    assert captured["langfuse_kwargs"] == {
        "public_key": "public",
        "secret_key": "secret",
        "host": "https://langfuse.local",
    }
    assert captured["llm_kwargs"]["callbacks"] == ["langfuse_handler"]
    assert captured["agent_callbacks"] == {"callbacks": ["langfuse_handler"]}
    assert result == {"configured": {"callbacks": ["langfuse_handler"]}}


def test_build_callback_handlers_returns_empty_without_langfuse(monkeypatch):
    monkeypatch.setattr("agents.mycroft_agent.agent.config.LANGFUSE_URL", "")

    assert _build_callback_handlers() == []


def test_build_gaz_mycroft_system_prompt_uses_bi_subagent_and_store_tool():
    prompt = build_gaz_mycroft_system_prompt(
        locale="en",
        pricing_subagent_name="gaz_pricing_bi",
        web_tool_name="web_search",
        store_tool_name="store_artifact_tool",
        enable_web_search=True,
    )

    assert "gaz_pricing_bi" in prompt
    assert "store_artifact_tool" in prompt
    assert "web_search" in prompt
    assert "query_pricing_bi" not in prompt
    assert "composite sales tools" not in prompt


def test_build_gaz_mycroft_system_prompt_mentions_mcp_tools_and_context_rules():
    prompt = build_gaz_mycroft_system_prompt(
        locale="en",
        pricing_subagent_name="gaz_pricing_bi",
        web_tool_name="web_search",
        store_tool_name="store_artifact_tool",
        maps_search_tool_name="maps_search_places",
        maps_route_tool_name="maps_compute_routes",
        vin_decode_tool_name="nhtsa_decode_vin",
        recall_lookup_tool_name="nhtsa_lookup_recalls",
        gmail_draft_tool_name="gmail_create_draft",
        gmail_send_tool_name="gmail_send_message",
        enable_web_search=True,
    )

    assert "maps_search_places" in prompt
    assert "maps_compute_routes" in prompt
    assert "nhtsa_decode_vin" in prompt
    assert "nhtsa_lookup_recalls" in prompt
    assert "gmail_create_draft" in prompt
    assert "gmail_send_message" in prompt
    assert "do not ask for the model again" in prompt
    assert "approval-gated by the runtime" in prompt


def test_delegate_prompt_and_description_have_stateful_semantics():
    system_prompt = build_delegate_system_prompt(tool_name="delegate")
    description = build_delegate_tool_description(tool_name="delegate")

    assert "`delegate`" in system_prompt
    assert "stateful team members" in system_prompt
    assert "same conversation thread" in system_prompt
    assert "Do not call the same stateful agent multiple times in parallel" in system_prompt
    assert "ephemeral" not in system_prompt.lower()
    assert "single result" not in system_prompt.lower()

    assert "`delegate`" in description
    assert "multiple user turns" in description
    assert "{available_agents}" in description
    assert "one-shot isolated task" in description
    assert "ephemeral" not in description.lower()
