from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

fake_think = types.ModuleType("agents.tools.think")
fake_think.ThinkTool = object
sys.modules.setdefault("agents.tools.think", fake_think)

fake_yandex_search = types.ModuleType("agents.tools.yandex_search")
fake_yandex_search.YandexSearchTool = object
sys.modules.setdefault("agents.tools.yandex_search", fake_yandex_search)

from agents.artifact_creator_agent import agent as artifact_agent
from agents.utils import ModelType


def test_create_greetings_node_prefers_initialize_system_prompt_over_runtime_context():
    greetings_node = artifact_agent.create_greetings_node("init prompt")

    state = greetings_node(
        {},
        config=None,
        runtime=SimpleNamespace(context={"system_prompt": "context prompt"}),
    )

    assert state["system_prompt"] == "init prompt"
    assert state["phase"] == "cleanup"


def test_initialize_agent_passes_tools_and_system_prompt_to_run_builder(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(artifact_agent.config, "LANGFUSE_URL", "")
    monkeypatch.setattr(artifact_agent, "JSONFileTracer", lambda path: "json_handler")

    def fake_get_llm(**kwargs):
        if kwargs["model"] == "base":
            return "base-llm"
        if kwargs["model"] == "nano":
            return "nano-llm"
        raise AssertionError(f"Unexpected model request: {kwargs}")

    monkeypatch.setattr(artifact_agent, "get_llm", fake_get_llm)
    monkeypatch.setattr(
        artifact_agent,
        "create_greetings_node",
        lambda system_prompt=None: captured.update({"greetings_system_prompt": system_prompt}) or (lambda *_args, **_kwargs: {}),
    )
    monkeypatch.setattr(
        artifact_agent,
        "_build_run_agent",
        lambda model, tools=None, system_prompt=None: captured.update(
            {"run_model": model, "run_tools": tools, "run_system_prompt": system_prompt}
        ) or (lambda *_args, **_kwargs: {}),
    )
    monkeypatch.setattr(
        artifact_agent,
        "create_confirmation_node",
        lambda model: captured.update({"confirmation_model": model}) or (lambda *_args, **_kwargs: {}),
    )

    class FakeCompiledGraph:
        def with_config(self, value):
            captured["graph_config"] = value
            return self

    class FakeStateGraph:
        def __init__(self, state_schema):
            captured["state_schema"] = state_schema

        def add_node(self, name, node):
            captured.setdefault("nodes", {})[name] = node

        def add_conditional_edges(self, *args, **kwargs):
            return None

        def add_edge(self, *args, **kwargs):
            return None

        def compile(self, checkpointer=None, debug=False):
            captured["checkpointer"] = checkpointer
            captured["debug"] = debug
            return FakeCompiledGraph()

    monkeypatch.setattr(artifact_agent, "StateGraph", FakeStateGraph)

    custom_tool = object()
    graph = artifact_agent.initialize_agent(
        provider=ModelType.GPT,
        checkpoint_saver="checkpoint",
        tools=[custom_tool],
        system_prompt="fixed prompt",
    )

    assert graph is not None
    assert captured["greetings_system_prompt"] == "fixed prompt"
    assert captured["run_model"] == "base-llm"
    assert captured["run_tools"] == [custom_tool]
    assert captured["run_system_prompt"] == "fixed prompt"
    assert captured["confirmation_model"] == "nano-llm"
    assert captured["checkpointer"] == "checkpoint"
    assert captured["graph_config"] == {"callbacks": ["json_handler"]}
