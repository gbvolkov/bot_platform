from __future__ import annotations

from types import SimpleNamespace

from agents.mycroft_agent.agent import _build_callback_handlers, initialize_agent
from agents.mycroft_agent.prompts import (
    build_delegate_system_prompt,
    build_delegate_tool_description,
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

    store_tool = SimpleNamespace(name="store_artifact_tool")
    stateless_subagent = {"name": "simple_agent", "description": "Simple Agent", "runnable": object()}
    stateful_subagent = {"name": "product_agent", "description": "Product Agent", "runnable": object()}
    system_prompt = "Scenario-specific Mycroft prompt."

    result = initialize_agent(
        provider=ModelType.GPT,
        model_size="mini",
        temperature=0.1,
        system_prompt=system_prompt,
        tools=[store_tool],
        stateless_subagents=[stateless_subagent],
        stateful_subagents=[stateful_subagent],
        checkpoint_saver="checkpoint",
        interrupt_on={"gmail_send_message": {"allowed_decisions": ["approve", "edit", "reject"]}},
        skills=["/skills/example"],
        backend="backend",
    )

    assert result["agent"]["tools"] == [store_tool]
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
    assert captured["system_prompt"] == system_prompt
    assert captured["checkpointer"] == "checkpoint"
    assert captured["interrupt_on"] == {
        "gmail_send_message": {"allowed_decisions": ["approve", "edit", "reject"]}
    }
    assert captured["skills"] == ["/skills/example"]
    assert captured["backend"] == "backend"


def test_initialize_agent_requires_scenario_system_prompt(monkeypatch):
    monkeypatch.setattr("agents.mycroft_agent.agent.config.LANGFUSE_URL", "")
    monkeypatch.setattr("agents.mycroft_agent.agent.config.OPENAI_API_KEY", "test-openai-key")

    monkeypatch.setattr(
        "agents.mycroft_agent.agent.get_llm",
        lambda **kwargs: {"llm": kwargs},
    )

    try:
        initialize_agent(provider=ModelType.GPT, tools=[])
    except ValueError as exc:
        assert "system_prompt" in str(exc)
    else:
        raise AssertionError("initialize_agent should require an explicit system_prompt")


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

    result = initialize_agent(
        provider=ModelType.GPT,
        system_prompt="Scenario prompt.",
        tools=[],
    )

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
