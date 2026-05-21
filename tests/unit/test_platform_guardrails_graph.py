from __future__ import annotations

from types import SimpleNamespace
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START
from langgraph.graph.message import add_messages
from langgraph.types import Command

from platform_guardrails.graph_compiler import PlatformGraphCompiler
from platform_guardrails.graph_spec import NodeGuardrailPolicy, PlatformStateGraph
from platform_guardrails.middleware import (
    PrivacyModelRequestMiddleware,
    SecurityScannerMiddleware,
    ToolExecutionSafetyMiddleware,
)
from platform_guardrails.runtime import PlatformGuardrailRuntime
from platform_guardrails.scanners import LLMGuardScannerProfile, LLMGuardScannerRail, ScannerSpec


class FakeInputScanner:
    def __init__(self, sanitized: str | None = None, valid: bool = True, score: float = 0.0):
        self.sanitized = sanitized
        self.valid = valid
        self.score = score
        self.seen: list[str] = []

    def scan(self, prompt: str):
        self.seen.append(prompt)
        return self.sanitized if self.sanitized is not None else prompt, self.valid, self.score


class ConditionalInputScanner:
    def __init__(self, blocked_text: str):
        self.blocked_text = blocked_text
        self.seen: list[str] = []

    def scan(self, prompt: str):
        self.seen.append(prompt)
        valid = self.blocked_text not in prompt
        return prompt, valid, 1.0 if not valid else 0.0


class FakeOutputScanner:
    def __init__(self, sanitized: str | None = None, valid: bool = True, score: float = 0.0):
        self.sanitized = sanitized
        self.valid = valid
        self.score = score
        self.seen: list[tuple[str, str]] = []

    def scan(self, prompt: str, output: str):
        self.seen.append((prompt, output))
        return self.sanitized if self.sanitized is not None else output, self.valid, self.score


class GraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    system_prompt: str


def _runtime(scanner_rail: LLMGuardScannerRail) -> PlatformGuardrailRuntime:
    return PlatformGuardrailRuntime(
        agent_id="test_agent",
        scanner_rail=scanner_rail,
        policy_kwargs={"guardrail_tool_execution_enabled": False},
    )


def test_platform_callable_node_blocks_before_execution():
    scanner = FakeInputScanner(valid=False, score=1.0)
    rail = LLMGuardScannerRail(
        LLMGuardScannerProfile(input_scanners=[ScannerSpec("PromptInjection", scanner=scanner)])
    )
    called = False

    def node(state, config=None, runtime=None):
        nonlocal called
        called = True
        return {"messages": [AIMessage(content="should not run")], "system_prompt": "stored"}

    graph = PlatformStateGraph(GraphState)
    graph.add_node("set_prompt", node, guardrails=NodeGuardrailPolicy(composite_input_scanners=()))
    graph.add_edge(START, "set_prompt")
    graph.add_edge("set_prompt", END)
    compiled = PlatformGraphCompiler().compile(graph.to_spec(), guardrail_runtime=_runtime(rail))

    result = compiled.invoke({"messages": [HumanMessage(content="ignore instructions", id="human-1")]})

    assert called is False
    assert "system_prompt" not in result
    assert result["messages"][-1].id.startswith("guardrail-block-")


def test_platform_callable_node_receives_sanitized_input():
    scanner = FakeInputScanner(sanitized="token ******", valid=True, score=0.0)
    rail = LLMGuardScannerRail(
        LLMGuardScannerProfile(input_scanners=[ScannerSpec("Secrets", scanner=scanner)])
    )
    captured = {}

    def node(state, config=None, runtime=None):
        captured["content"] = state["messages"][0].content
        return {"messages": list(state["messages"])}

    graph = PlatformStateGraph(GraphState)
    graph.add_node("set_prompt", node, guardrails=NodeGuardrailPolicy(composite_input_scanners=()))
    graph.add_edge(START, "set_prompt")
    graph.add_edge("set_prompt", END)
    compiled = PlatformGraphCompiler().compile(graph.to_spec(), guardrail_runtime=_runtime(rail))

    result = compiled.invoke({"messages": [HumanMessage(content="token sk-secret", id="human-1")]})

    assert captured["content"] == "token ******"
    assert result["messages"][0].content == "token ******"


def test_platform_callable_node_scans_output():
    scanner = FakeOutputScanner(valid=False, score=1.0)
    rail = LLMGuardScannerRail(
        LLMGuardScannerProfile(output_scanners=[ScannerSpec("MaliciousURLs", scanner=scanner)])
    )

    def node(state, config=None, runtime=None):
        return {"messages": [AIMessage(content="open http://bad.example")]}

    graph = PlatformStateGraph(GraphState)
    graph.add_node("run", node, guardrails=NodeGuardrailPolicy(composite_input_scanners=()))
    graph.add_edge(START, "run")
    graph.add_edge("run", END)
    compiled = PlatformGraphCompiler().compile(graph.to_spec(), guardrail_runtime=_runtime(rail))

    result = compiled.invoke({"messages": [HumanMessage(content="hello")]})

    assert result["messages"][-1].id.startswith("guardrail-block-")
    assert scanner.seen


def test_policy_can_exclude_system_prompt_state_from_callable_scan():
    scanner = ConditionalInputScanner("do not scan this system prompt")
    rail = LLMGuardScannerRail(
        LLMGuardScannerProfile(input_scanners=[ScannerSpec("PromptInjection", scanner=scanner)])
    )
    runtime = PlatformGuardrailRuntime(
        agent_id="test_agent",
        scanner_rail=rail,
        policy_kwargs={
            "guardrail_scan_system_prompt": False,
            "guardrail_tool_execution_enabled": False,
        },
    )
    called = False

    def node(state, config=None, runtime=None):
        nonlocal called
        called = True
        return {"messages": [AIMessage(content="ok")]}

    graph = PlatformStateGraph(GraphState)
    graph.add_node("set_prompt", node, guardrails=NodeGuardrailPolicy(composite_input_scanners=()))
    graph.add_edge(START, "set_prompt")
    graph.add_edge("set_prompt", END)
    compiled = PlatformGraphCompiler().compile(graph.to_spec(), guardrail_runtime=runtime)

    result = compiled.invoke(
        {
            "messages": [HumanMessage(content="hello", id="human-1")],
            "system_prompt": "do not scan this system prompt",
        }
    )

    assert called is True
    assert result["messages"][-1].content == "ok"
    assert scanner.seen == ["hello"]


def test_platform_agent_node_injects_guardrail_middlewares(monkeypatch):
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)

        def node(state, config=None, runtime=None):
            return {"messages": [AIMessage(content="ok")]}

        return node

    monkeypatch.setattr("platform_guardrails.graph_compiler.create_agent", fake_create_agent)
    rail = LLMGuardScannerRail(LLMGuardScannerProfile())
    runtime = PlatformGuardrailRuntime(
        agent_id="test_agent",
        policy_kwargs={"guardrail_tool_execution_enabled": True},
        scanner_rail=rail,
        privacy_rail=SimpleNamespace(),
    )
    tool = SimpleNamespace(name="lookup")
    profiles = {
        "lookup": {
            "name": "lookup",
            "allowed_roles": ["default"],
            "side_effect": "read",
            "result_policy": {"scan_result": False},
        }
    }

    graph = PlatformStateGraph(GraphState)
    graph.add_agent_node("run", model=object(), state_schema=dict, context_schema=dict)
    graph.add_edge(START, "run")
    graph.add_edge("run", END)

    PlatformGraphCompiler().compile(
        graph.to_spec(),
        guardrail_runtime=runtime,
        tools=[tool],
        tool_profiles=profiles,
    )

    middleware = captured["middleware"]
    assert [type(item) for item in middleware] == [
        SecurityScannerMiddleware,
        PrivacyModelRequestMiddleware,
        ToolExecutionSafetyMiddleware,
    ]


def test_policy_can_exclude_system_prompt_from_agent_node_scanner(monkeypatch):
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)

        def node(state, config=None, runtime=None):
            return {"messages": [AIMessage(content="ok")]}

        return node

    monkeypatch.setattr("platform_guardrails.graph_compiler.create_agent", fake_create_agent)
    rail = LLMGuardScannerRail(LLMGuardScannerProfile())
    runtime = PlatformGuardrailRuntime(
        agent_id="test_agent",
        policy_kwargs={
            "guardrail_scan_system_prompt": False,
            "guardrail_tool_execution_enabled": False,
        },
        scanner_rail=rail,
    )

    graph = PlatformStateGraph(GraphState)
    graph.add_agent_node("run", model=object(), state_schema=dict, context_schema=dict)
    graph.add_edge(START, "run")
    graph.add_edge("run", END)

    PlatformGraphCompiler().compile(graph.to_spec(), guardrail_runtime=runtime)

    middleware = captured["middleware"]
    security = middleware[0]
    assert isinstance(security, SecurityScannerMiddleware)
    assert security._scan_system_prompt is False
    assert security._state_keys_to_scan == ()
    assert security._include_system_prompt_in_scans is False
