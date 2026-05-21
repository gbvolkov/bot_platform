from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import simple_agent_cli as cli
from agents.utils import ModelType


def test_load_agent_registry_settings_reads_platform_guardrails_and_tools(tmp_path):
    config_path = tmp_path / "load.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "id": "simple_agent",
                        "tools": [{"type": "internal", "name": "web_search_tool"}],
                        "guardrails": {
                            "policy": "default_guardrails",
                            "mode": "platform",
                        },
                        "params": {"provider": "openai"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    settings = cli._load_agent_registry_settings(config_path)

    assert settings.guardrail_policy_id == "default_guardrails"
    assert settings.guardrail_mode == "platform"
    assert settings.tools_config.internal_tools[0].name == "web_search_tool"


def test_load_agent_registry_settings_rejects_mixed_guardrail_config(tmp_path):
    config_path = tmp_path / "load.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "id": "simple_agent",
                        "guardrails": {
                            "policy": "default_guardrails",
                            "mode": "platform",
                        },
                        "params": {"guardrail_policy": "legacy"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot combine"):
        cli._load_agent_registry_settings(config_path)


def test_build_cli_tools_config_uses_platform_tool_specs():
    args = SimpleNamespace(
        enable_think=True,
        enable_web_search=True,
        web_search_max_results=2,
    )

    tools_config = cli._build_cli_tools_config(args)

    assert [spec.name for spec in tools_config.internal_tools] == [
        "think_tool",
        "web_search_tool",
    ]
    assert tools_config.internal_tools[1].params["max_results"] == 2
    assert tools_config.internal_tools[1].params["summarize"] is True


def test_merge_tools_config_lets_cli_override_duplicate_internal_tool_params():
    registry_config = cli.AgentToolsConfig(
        internal_tools=(
            cli.InternalToolSpec(
                name="web_search_tool",
                params={"max_results": 5},
            ),
        )
    )
    cli_config = cli.AgentToolsConfig(
        internal_tools=(
            cli.InternalToolSpec(
                name="web_search_tool",
                params={"max_results": 3},
            ),
        )
    )

    merged = cli._merge_tools_config(registry_config, cli_config)

    assert len(merged.internal_tools) == 1
    assert merged.internal_tools[0].name == "web_search_tool"
    assert merged.internal_tools[0].params["max_results"] == 3


def test_new_config_can_allow_external_tool_access():
    _thread_id, run_config = cli._new_config(
        "thread-1",
        allow_external_tool_access=True,
    )

    assert run_config == {
        "configurable": {
            "thread_id": "thread-1",
            "allow_external_tool_access": True,
        }
    }


def test_build_guardrail_runtime_can_override_verbose_logging(monkeypatch):
    settings = cli.SimpleAgentRegistrySettings(
        tools_config=cli.AgentToolsConfig(),
        guardrail_policy_id="default_guardrails",
        guardrail_mode="platform",
    )
    monkeypatch.setattr(
        cli,
        "resolve_guardrail_policy",
        lambda policy_id: {
            "guardrail_scanners_enabled": False,
            "guardrail_verbose_logging": False,
        },
    )

    runtime = cli._build_guardrail_runtime(settings, verbose_logging=True)

    assert runtime.policy_id == "default_guardrails"
    assert runtime.verbose_logging is True


def test_build_registry_tools_uses_profiled_think_tool():
    tools_config = cli.AgentToolsConfig(
        internal_tools=(cli.InternalToolSpec(name="think_tool"),)
    )

    bundle = asyncio.run(
        cli._build_registry_tools(tools_config, require_guardrail_profiles=True)
    )

    assert [tool.name for tool in bundle.tools] == ["think_tool"]
    assert bundle.guardrail_profiles["think_tool"]["category"] == "internal_state"
    assert bundle.guardrail_profiles["think_tool"]["result_policy"]["scan_result"] is False


def test_compile_platform_graph_injects_runtime_tools_and_profiles(monkeypatch):
    captured = {}

    class FakeCompiler:
        def compile(self, spec, *, guardrail_runtime, checkpointer, tools, tool_profiles):
            captured["spec"] = spec
            captured["guardrail_runtime"] = guardrail_runtime
            captured["checkpointer"] = checkpointer
            captured["tools"] = tools
            captured["tool_profiles"] = tool_profiles
            return "compiled"

    monkeypatch.setattr(cli, "build_agent_graph", lambda **kwargs: {"kwargs": kwargs})
    monkeypatch.setattr(cli, "PlatformGraphCompiler", lambda: FakeCompiler())

    runtime = object()
    tool = SimpleNamespace(name="tool")
    bundle = cli.BuiltAgentTools(
        tools=[tool],
        guardrail_profiles={"tool": {"name": "tool"}},
    )

    result = cli._compile_platform_graph(
        provider=ModelType.GPT,
        guardrail_runtime=runtime,
        tool_bundle=bundle,
    )

    assert result == "compiled"
    assert captured["spec"] == {
        "kwargs": {"provider": ModelType.GPT, "streaming": False}
    }
    assert captured["guardrail_runtime"] is runtime
    assert captured["tools"] == [tool]
    assert captured["tool_profiles"] == {"tool": {"name": "tool"}}
    assert captured["checkpointer"] is not None
