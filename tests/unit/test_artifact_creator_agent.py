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
_existing_think = sys.modules.get("agents.tools.think")
if _existing_think is None:
    sys.modules["agents.tools.think"] = fake_think


class FakeYandexSearchTool:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


fake_yandex_search = types.ModuleType("agents.tools.yandex_search")
fake_yandex_search.YandexSearchTool = FakeYandexSearchTool
fake_yandex_search.SEARCH_TOOL_POLICY_PROMPT_EN = ""
fake_yandex_search.SEARCH_TOOL_POLICY_PROMPT_RU = ""
_existing_yandex_search = sys.modules.get("agents.tools.yandex_search")
if _existing_yandex_search is None:
    sys.modules["agents.tools.yandex_search"] = fake_yandex_search

from agents.artifact_creator_agent import agent as artifact_agent
from agents.utils import ModelType

if _existing_think is None:
    sys.modules.pop("agents.tools.think", None)
if _existing_yandex_search is None:
    sys.modules.pop("agents.tools.yandex_search", None)


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
    monkeypatch.setattr(artifact_agent, "RedactingJSONFileTracer", lambda path: "json_handler")

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
        lambda model, tools=None, system_prompt=None, privacy_middleware=None: captured.update(
            {
                "run_model": model,
                "run_tools": tools,
                "run_system_prompt": system_prompt,
                "run_privacy_middleware": privacy_middleware,
            }
        ) or (lambda *_args, **_kwargs: {}),
    )
    monkeypatch.setattr(
        artifact_agent,
        "create_confirmation_node",
        lambda model, privacy_middleware=None: captured.update(
            {
                "confirmation_model": model,
                "confirmation_privacy_middleware": privacy_middleware,
            }
        ) or (lambda *_args, **_kwargs: {}),
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
    assert captured["run_privacy_middleware"] is None
    assert captured["confirmation_model"] == "nano-llm"
    assert captured["confirmation_privacy_middleware"] is None
    assert captured["checkpointer"] == "checkpoint"
    assert captured["graph_config"] == {"callbacks": ["json_handler"]}


def test_initialize_agent_wires_guardrails_when_enabled(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(artifact_agent.config, "LANGFUSE_URL", "https://langfuse.example")
    monkeypatch.setattr(artifact_agent, "RedactingJSONFileTracer", lambda path: "redacting_handler")
    monkeypatch.setattr(artifact_agent, "CallbackHandler", lambda: "langfuse_handler")

    def fake_get_llm(**kwargs):
        return f"{kwargs['model']}-llm"

    class FakePrivacyRail:
        pass

    def fake_from_palimpsest(**kwargs):
        captured["privacy_rail_kwargs"] = kwargs
        return FakePrivacyRail()

    def fake_privacy_middleware(privacy_rail, **kwargs):
        captured.setdefault("privacy_middlewares", []).append((privacy_rail, kwargs))
        return f"privacy:{kwargs['agent_name']}"

    monkeypatch.setattr(artifact_agent, "get_llm", fake_get_llm)
    monkeypatch.setattr(artifact_agent.PrivacyRail, "from_palimpsest", staticmethod(fake_from_palimpsest))
    monkeypatch.setattr(artifact_agent, "PrivacyModelRequestMiddleware", fake_privacy_middleware)
    monkeypatch.setattr(
        artifact_agent,
        "create_greetings_node",
        lambda system_prompt=None: (lambda *_args, **_kwargs: {}),
    )
    monkeypatch.setattr(
        artifact_agent,
        "_build_run_agent",
        lambda model, tools=None, system_prompt=None, privacy_middleware=None: captured.update(
            {"run_privacy_middleware": privacy_middleware}
        ) or (lambda *_args, **_kwargs: {}),
    )
    monkeypatch.setattr(
        artifact_agent,
        "create_confirmation_node",
        lambda model, privacy_middleware=None: captured.update(
            {"confirmation_privacy_middleware": privacy_middleware}
        ) or (lambda *_args, **_kwargs: {}),
    )

    class FakeCompiledGraph:
        def with_config(self, value):
            captured["graph_config"] = value
            return self

    class FakeStateGraph:
        def __init__(self, state_schema):
            return None

        def add_node(self, name, node):
            return None

        def add_conditional_edges(self, *args, **kwargs):
            return None

        def add_edge(self, *args, **kwargs):
            return None

        def compile(self, checkpointer=None, debug=False):
            return FakeCompiledGraph()

    monkeypatch.setattr(artifact_agent, "StateGraph", FakeStateGraph)

    artifact_agent.initialize_agent(
        provider=ModelType.GPT,
        checkpoint_saver="checkpoint",
        guardrails_enabled=True,
        guardrails_locale="ru-RU",
    )

    assert captured["privacy_rail_kwargs"] == {"locale": "ru-RU"}
    assert captured["run_privacy_middleware"] == "privacy:artifact_creator_agent.run"
    assert captured["confirmation_privacy_middleware"] == "privacy:artifact_creator_agent.confirm"
    assert captured["graph_config"] == {"callbacks": ["redacting_handler", "langfuse_handler"]}
    middlewares = captured["privacy_middlewares"]
    assert len(middlewares) == 2
    assert all(item[0] is middlewares[0][0] for item in middlewares)
