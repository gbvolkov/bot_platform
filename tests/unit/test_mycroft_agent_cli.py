from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

import mycroft_agent_cli as cli
from agents.mycroft_agent.cli_config import (
    MCPConfig,
    InternalToolSpec,
    MCPServerSpec,
    build_internal_tools,
    load_cli_config,
    load_mcp_tools_from_config,
    required_environment_variables,
    select_mcp_tools,
    validate_required_environment,
)


def test_load_cli_config_reads_required_sections(tmp_path):
    config_path = tmp_path / "mycroft.json"
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": {
                    "type": "gaz_mycroft",
                    "locale": "en",
                    "pricing_subagent": "gaz_pricing_bi",
                    "web_tool": "web_search",
                    "store_tool": "store_artifact_tool",
                    "maps_search_tool": "maps_search_places",
                    "maps_route_tool": "maps_compute_routes",
                    "vin_decode_tool": "nhtsa_decode_vin",
                    "recall_lookup_tool": "nhtsa_search_recalls",
                    "gmail_draft_tool": "gmail_create_draft",
                    "gmail_send_tool": "gmail_send_message",
                    "enable_web_search": True
                },
                "agents": ["simple_agent", "new_ideator"],
                "internal_tools": [
                    {"name": "web_search", "max_results": 3, "summarize": False},
                    "store_artifact_tool"
                ],
                "deepagents": {
                    "interrupt_on": {
                        "gmail_send_message": {
                            "allowed_decisions": ["approve", "edit", "reject"],
                            "description": "Review outbound Gmail send before execution."
                        }
                    }
                },
                "mcp": {
                    "tool_name_prefix": True,
                    "servers": [
                        {
                            "name": "sysadmin",
                            "transport": "http",
                            "url": "https://example.com/mcp",
                            "tools": ["list_services"],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_cli_config(config_path)

    assert "gaz_pricing_bi" in config.system_prompt
    assert "store_artifact_tool" in config.system_prompt
    assert config.agents == ("simple_agent", "new_ideator")
    assert config.internal_tools[0] == InternalToolSpec(
        name="web_search",
        params={"max_results": 3, "summarize": False},
    )
    assert config.internal_tools[1] == InternalToolSpec(name="store_artifact_tool", params={})
    assert config.mcp.tool_name_prefix is True
    assert config.mcp.servers[0].name == "sysadmin"
    assert config.mcp.servers[0].connection["transport"] == "http"
    assert config.mcp.servers[0].tools == ("list_services",)
    assert config.deepagents.interrupt_on == {
        "gmail_send_message": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "description": "Review outbound Gmail send before execution.",
        }
    }
    assert "maps_search_places" in config.system_prompt
    assert "gmail_send_message" in config.system_prompt


def test_build_internal_tools_flattens_bundle_builders():
    specs = (
        InternalToolSpec(name="single", params={}),
        InternalToolSpec(name="bundle", params={}),
    )

    tools = build_internal_tools(
        specs,
        builders={
            "single": lambda: SimpleNamespace(name="single_tool"),
            "bundle": lambda: [
                SimpleNamespace(name="bundle_tool_a"),
                SimpleNamespace(name="bundle_tool_b"),
            ],
        },
    )

    assert [tool.name for tool in tools] == [
        "single_tool",
        "bundle_tool_a",
        "bundle_tool_b",
    ]


def test_select_mcp_tools_accepts_prefixed_and_raw_names():
    server = MCPServerSpec(
        name="sysadmin",
        connection={"transport": "http", "url": "http://127.0.0.1:8000/mcp"},
        tools=("list_services", "sysadmin_restart_service"),
    )
    loaded_tools = [
        SimpleNamespace(name="sysadmin_list_services"),
        SimpleNamespace(name="sysadmin_restart_service"),
    ]

    selected = select_mcp_tools(server, loaded_tools)

    assert [tool.name for tool in selected] == [
        "sysadmin_list_services",
        "sysadmin_restart_service",
    ]


def test_default_cli_config_uses_gaz_pricing_bi_and_requested_tools():
    config = load_cli_config()

    assert config.agents == ("gaz_pricing_bi",)
    assert [tool.name for tool in config.internal_tools] == [
        "web_search",
        "store_artifact_tool",
    ]
    assert tuple(server.name for server in config.mcp.servers) == ("maps", "gmail", "nhtsa")
    assert config.mcp.servers[0].connection["url"] == "https://mapstools.googleapis.com/mcp"
    assert config.mcp.servers[1].connection["command"] == "npx"
    assert config.mcp.servers[2].connection["url"] == "https://nhtsa.caseyjhand.com/mcp"
    assert config.mcp.servers[2].tool_name_prefix is False
    assert config.mcp.servers[2].tools == ("nhtsa_decode_vin", "nhtsa_search_recalls")
    assert config.deepagents.interrupt_on["gmail_send_message"] == {
        "allowed_decisions": ["approve", "edit", "reject"],
        "description": "Review outbound Gmail send before execution.",
    }


def test_required_environment_variables_include_default_runtime_requirements():
    config = load_cli_config()

    required = set(required_environment_variables(config, "openai"))

    assert required == {
        "OPENAI_API_KEY",
        "YA_API_KEY",
        "YA_FOLDER_ID",
        "GOOGLE_MAPS_GROUNDING_LITE_API_KEY",
        "GMAIL_MCP_CLIENT_ID",
        "GMAIL_MCP_CLIENT_SECRET",
        "GMAIL_MCP_REFRESH_TOKEN",
    }


def test_validate_required_environment_reports_missing_names(monkeypatch):
    config = load_cli_config()
    for variable_name in required_environment_variables(config, "openai"):
        monkeypatch.delenv(variable_name, raising=False)

    with pytest.raises(ValueError) as exc_info:
        validate_required_environment(config, "openai")

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert "GOOGLE_MAPS_GROUNDING_LITE_API_KEY" in message
    assert "GMAIL_MCP_CLIENT_ID" in message


def test_load_mcp_tools_from_config_expands_nested_environment_values(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setenv("BASE_URL", "https://example.com")
    monkeypatch.setenv("API_KEY", "secret-key")
    monkeypatch.setenv("CLIENT_SECRET", "client-secret")

    async def fake_load_mcp_tools(_session, *, connection, server_name, tool_name_prefix):
        captured["connection"] = connection
        captured["server_name"] = server_name
        captured["tool_name_prefix"] = tool_name_prefix
        return [SimpleNamespace(name="maps_search_places")]

    monkeypatch.setattr("agents.mycroft_agent.cli_config.load_mcp_tools", fake_load_mcp_tools)

    config = MCPConfig(
        tool_name_prefix=True,
        servers=(
            MCPServerSpec(
                name="maps",
                connection={
                    "transport": "http",
                    "url": "${BASE_URL}/mcp",
                    "headers": {"X-Api-Key": "${API_KEY}"},
                    "env": {"CLIENT_SECRET": "${CLIENT_SECRET}"},
                    "args": ["--endpoint=${BASE_URL}/api"],
                },
                tools=("search_places",),
            ),
        ),
    )

    tools = asyncio.run(load_mcp_tools_from_config(config))

    assert [tool.name for tool in tools] == ["maps_search_places"]
    assert captured["server_name"] == "maps"
    assert captured["tool_name_prefix"] is True
    assert captured["connection"] == {
        "transport": "http",
        "url": "https://example.com/mcp",
        "headers": {"X-Api-Key": "secret-key"},
        "env": {"CLIENT_SECRET": "client-secret"},
        "args": ["--endpoint=https://example.com/api"],
    }


def test_load_mcp_tools_from_config_fails_on_unresolved_environment_variable(monkeypatch):
    monkeypatch.delenv("MISSING_TOKEN", raising=False)

    config = MCPConfig(
        tool_name_prefix=True,
        servers=(
            MCPServerSpec(
                name="maps",
                connection={
                    "transport": "http",
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer ${MISSING_TOKEN}"},
                },
                tools=("search_places",),
            ),
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(load_mcp_tools_from_config(config))

    assert str(exc_info.value) == "MCP server 'maps' requires environment variable 'MISSING_TOKEN'."


def test_cli_invoke_turn_resumes_after_hitl_approve(monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.calls: list[object] = []

        async def ainvoke(self, payload, config):
            self.calls.append(payload)
            if len(self.calls) == 1:
                return {
                    "__interrupt__": [
                        SimpleNamespace(
                            value={
                                "action_requests": [
                                    {
                                        "name": "gmail_send_message",
                                        "args": {"to": "client@example.com"},
                                        "description": "Review send.",
                                    }
                                ],
                                "review_configs": [
                                    {
                                        "action_name": "gmail_send_message",
                                        "allowed_decisions": ["approve", "edit", "reject"],
                                    }
                                ],
                            }
                        )
                    ]
                }
            return {"messages": [AIMessage(content="Sent.")]}

    agent = FakeAgent()
    responses = iter(["approve"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    answer = asyncio.run(
        cli._invoke_turn(
            agent,
            user_text="Send the email.",
            config={"configurable": {"thread_id": "thread-1"}},
        )
    )

    assert answer == "Sent."
    assert isinstance(agent.calls[1], Command)
    assert agent.calls[1].resume == {"decisions": [{"type": "approve"}]}


def test_cli_invoke_turn_resumes_after_hitl_edit(monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.calls: list[object] = []

        async def ainvoke(self, payload, config):
            self.calls.append(payload)
            if len(self.calls) == 1:
                return {
                    "__interrupt__": [
                        SimpleNamespace(
                            value={
                                "action_requests": [
                                    {
                                        "name": "gmail_send_message",
                                        "args": {"to": "client@example.com"},
                                    }
                                ],
                                "review_configs": [
                                    {
                                        "action_name": "gmail_send_message",
                                        "allowed_decisions": ["approve", "edit", "reject"],
                                    }
                                ],
                            }
                        )
                    ]
                }
            return {"messages": [AIMessage(content="Edited.")]}

    agent = FakeAgent()
    responses = iter(["edit", "", "{\"to\": \"edited@example.com\"}"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    answer = asyncio.run(
        cli._invoke_turn(
            agent,
            user_text="Send the email.",
            config={"configurable": {"thread_id": "thread-2"}},
        )
    )

    assert answer == "Edited."
    assert isinstance(agent.calls[1], Command)
    assert agent.calls[1].resume == {
        "decisions": [
            {
                "type": "edit",
                "edited_action": {
                    "name": "gmail_send_message",
                    "args": {"to": "edited@example.com"},
                },
            }
        ]
    }


def test_cli_invoke_turn_resumes_after_hitl_reject(monkeypatch):
    class FakeAgent:
        def __init__(self):
            self.calls: list[object] = []

        async def ainvoke(self, payload, config):
            self.calls.append(payload)
            if len(self.calls) == 1:
                return {
                    "__interrupt__": [
                        SimpleNamespace(
                            value={
                                "action_requests": [
                                    {
                                        "name": "gmail_send_message",
                                        "args": {"to": "client@example.com"},
                                    }
                                ],
                                "review_configs": [
                                    {
                                        "action_name": "gmail_send_message",
                                        "allowed_decisions": ["approve", "edit", "reject"],
                                    }
                                ],
                            }
                        )
                    ]
                }
            return {"messages": [AIMessage(content="Rejected.")]}

    agent = FakeAgent()
    responses = iter(["reject", "Do not send yet"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    answer = asyncio.run(
        cli._invoke_turn(
            agent,
            user_text="Send the email.",
            config={"configurable": {"thread_id": "thread-3"}},
        )
    )

    assert answer == "Rejected."
    assert isinstance(agent.calls[1], Command)
    assert agent.calls[1].resume == {
        "decisions": [
            {
                "type": "reject",
                "message": "Do not send yet",
            }
        ]
    }
