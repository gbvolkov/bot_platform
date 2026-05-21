from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

import mycroft_agent_cli as cli
from agents.mycroft_agent.cli_config import (
    MCPConfig,
    InternalToolSpec,
    MCPServerSpec,
    SkillsConfig,
    SubagentsConfig,
    build_internal_tools,
    resolve_cli_config_path,
    load_cli_config,
    load_mcp_tools_from_config,
    required_environment_variables,
    select_mcp_tools,
    validate_required_environment,
)


@dataclass(frozen=True)
class _FakeDefinition:
    id: str
    name: str
    description: str
    is_active: bool = False


def test_load_cli_config_reads_required_sections(tmp_path):
    prompt_path = tmp_path / "scenario_prompt.txt"
    prompt_path.write_text(
        "Scenario prompt with gaz_pricing_bi, web_search_agent, store_artifact_tool, "
        "maps_search_places, gmail_send_message.",
        encoding="utf-8",
    )
    config_path = tmp_path / "mycroft.json"
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": {
                    "type": "file",
                    "path": str(prompt_path),
                },
                "subagents": {
                    "stateless": ["simple_agent", "new_ideator", "web_search_agent"],
                    "stateful": ["product_Инголаб"],
                },
                "internal_tools": [
                    "store_artifact_tool",
                ],
                "deepagents": {
                    "interrupt_on": {
                        "gmail_send_message": {
                            "allowed_decisions": ["approve", "edit", "reject"],
                            "description": "Review outbound Gmail send before execution.",
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
    assert config.subagents == SubagentsConfig(
        stateless=("simple_agent", "new_ideator", "web_search_agent"),
        stateful=("product_Инголаб",),
    )
    assert config.internal_tools == (
        InternalToolSpec(name="store_artifact_tool", params={}),
    )
    assert config.skills == SkillsConfig(paths=())
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


def test_load_cli_config_rejects_scenario_specific_system_prompt_types(tmp_path):
    config_path = tmp_path / "mycroft.json"
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": {
                    "type": "gaz_mycroft",
                    "locale": "en",
                },
                "subagents": {
                    "stateless": [],
                    "stateful": [],
                },
                "internal_tools": [],
                "mcp": {"tool_name_prefix": True, "servers": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_cli_config(config_path)

    message = str(exc_info.value)
    assert "Unsupported Mycroft CLI system prompt type 'gaz_mycroft'" in message
    assert "Supported values: file" in message


def test_load_cli_config_reads_system_prompt_from_file(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Mycroft file prompt", encoding="utf-8")
    config_path = tmp_path / "mycroft.json"
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": {
                    "type": "file",
                    "path": str(prompt_path),
                },
                "subagents": {
                    "stateless": ["simple_agent"],
                    "stateful": [],
                },
                "internal_tools": [],
                "mcp": {"tool_name_prefix": True, "servers": []},
            }
        ),
        encoding="utf-8",
    )

    config = load_cli_config(config_path)

    assert config.system_prompt == "Mycroft file prompt"
    assert config.subagents == SubagentsConfig(
        stateless=("simple_agent",),
        stateful=(),
    )


def test_default_cli_config_path_uses_working_directory_config_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MYCROFT_CONFIG_ROOT", raising=False)

    assert resolve_cli_config_path() == (
        tmp_path / "data" / "config" / "mycroft" / "gaz_config.json"
    )


def test_default_cli_config_path_allows_configurable_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MYCROFT_CONFIG_ROOT", "runtime/configs")

    assert resolve_cli_config_path() == (
        tmp_path / "runtime" / "configs" / "gaz_config.json"
    )


def test_load_cli_config_resolves_system_prompt_from_working_directory(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    prompt_path = tmp_path / "prompts" / "prompt.txt"
    prompt_path.parent.mkdir()
    prompt_path.write_text("Prompt from working directory.", encoding="utf-8")
    config_path = tmp_path / "data" / "config" / "mycroft" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": {
                    "type": "file",
                    "path": "prompts/prompt.txt",
                },
                "subagents": {"stateless": [], "stateful": []},
                "internal_tools": [],
                "mcp": {"tool_name_prefix": True, "servers": []},
            }
        ),
        encoding="utf-8",
    )

    config = load_cli_config(config_path)

    assert config.system_prompt == "Prompt from working directory."


def test_load_cli_config_rejects_legacy_agents_field(tmp_path):
    config_path = tmp_path / "mycroft.json"
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": "Prompt",
                "agents": ["legacy_agent"],
                "internal_tools": [],
                "mcp": {"tool_name_prefix": True, "servers": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_cli_config(config_path)

    assert "field 'agents' is no longer supported" in str(exc_info.value)


def test_new_config_includes_default_user_id():
    thread_id, run_config = cli._new_config("thread-123")

    assert thread_id == "thread-123"
    assert run_config == {
        "configurable": {
            "thread_id": "thread-123",
            "user_id": "mycroft-agent-cli",
        }
    }


def test_persistent_checkpoint_saver_uses_hardcoded_workspace_path(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "mycroft" / "cli_checkpoints.sqlite"
    captured: dict[str, str] = {}

    class FakeAsyncSqliteSaver:
        @staticmethod
        def from_conn_string(conn_string):
            captured["conn_string"] = conn_string
            return "checkpoint-cm"

    monkeypatch.setattr(cli, "MYCROFT_CLI_CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(cli, "AsyncSqliteSaver", FakeAsyncSqliteSaver)

    assert cli._persistent_checkpoint_saver() == "checkpoint-cm"
    assert captured["conn_string"] == str(checkpoint_path)
    assert checkpoint_path.parent.is_dir()


def test_parse_save_thread_command_supports_alias_and_optional_path():
    assert cli._parse_save_thread_command("/save-thread") == ""
    assert cli._parse_save_thread_command("/save-thread logs/thread.md") == "logs/thread.md"
    assert cli._parse_save_thread_command("/save export.json") == "export.json"
    assert cli._parse_save_thread_command("/reset") is None


def test_normalize_export_path_defaults_to_markdown_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert cli._normalize_export_path(None, thread_id="thread-123") == tmp_path / "thread-123.md"
    assert cli._normalize_export_path("logs/thread", thread_id="thread-123") == tmp_path / "logs" / "thread.md"


def test_save_thread_export_writes_markdown(tmp_path):
    path = tmp_path / "thread.md"

    saved = cli._save_thread_export(
        path=path,
        thread_id="thread-123",
        config_path="config.json",
        provider="openai",
        model_size="base",
        turns=[{"user": "Hello", "assistant": "Hi"}],
    )

    assert saved == path
    text = path.read_text(encoding="utf-8")
    assert "# Mycroft Conversation Thread" in text
    assert "Thread id: `thread-123`" in text
    assert "Hello" in text
    assert "Hi" in text


def test_save_thread_export_writes_json(tmp_path):
    path = tmp_path / "thread.json"

    cli._save_thread_export(
        path=path,
        thread_id="thread-123",
        config_path="config.json",
        provider="openai",
        model_size="base",
        turns=[{"user": "Hello", "assistant": "Hi"}],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["thread_id"] == "thread-123"
    assert payload["config_path"] == "config.json"
    assert payload["provider"] == "openai"
    assert payload["model_size"] == "base"
    assert payload["turns"] == [{"user": "Hello", "assistant": "Hi"}]


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


def test_build_internal_tools_loads_importable_bundle(monkeypatch):
    fake_module = SimpleNamespace(
        build_tools=lambda prefix: [
            SimpleNamespace(name=f"{prefix}_a"),
            SimpleNamespace(name=f"{prefix}_b"),
        ]
    )
    monkeypatch.setitem(sys.modules, "fake_mycroft_tools", fake_module)

    tools = build_internal_tools(
        (
            InternalToolSpec(
                import_path="fake_mycroft_tools:build_tools",
                params={"prefix": "imported"},
            ),
        )
    )

    assert [tool.name for tool in tools] == ["imported_a", "imported_b"]


def test_load_cli_config_reads_skills_and_importable_tool_bundle(tmp_path):
    config_path = tmp_path / "mycroft.json"
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": "Prompt",
                "skills": {"paths": ["/skills/marketing_analyst"]},
                "subagents": {"stateless": [], "stateful": []},
                "internal_tools": [
                    {
                        "import": "agents.gaz_agent.marketing_tools:build_marketing_document_tools",
                        "params": {"locale": "ru", "docs_collection": "gaz"},
                    }
                ],
                "mcp": {"tool_name_prefix": True, "servers": []},
            }
        ),
        encoding="utf-8",
    )

    config = load_cli_config(config_path)

    assert config.skills == SkillsConfig(paths=("/skills/marketing_analyst",))
    assert config.internal_tools == (
        InternalToolSpec(
            import_path="agents.gaz_agent.marketing_tools:build_marketing_document_tools",
            params={"locale": "ru", "docs_collection": "gaz"},
        ),
    )


def test_load_cli_config_rejects_malformed_import_tool_specs(tmp_path):
    config_path = tmp_path / "mycroft.json"
    config_path.write_text(
        json.dumps(
            {
                "system_prompt": "Prompt",
                "subagents": {"stateless": [], "stateful": []},
                "internal_tools": [{"import": "module:function", "name": "tool"}],
                "mcp": {"tool_name_prefix": True, "servers": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_cli_config(config_path)

    assert "exactly one of 'name' or 'import'" in str(exc_info.value)


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


def test_default_cli_config_uses_generic_config_file():
    config = load_cli_config()

    assert config.subagents == SubagentsConfig(
        stateless=("gaz_pricing_bi_int", "marketing_analyst", "web_search_agent"),
        stateful=(),
    )
    assert config.internal_tools == (
        InternalToolSpec(name="store_artifact_tool", params={}),
    )
    assert tuple(server.name for server in config.mcp.servers) == ("maps", "gmail", "nhtsa")
    assert "Skill-driven orchestration" in config.system_prompt
    assert "Keep subagent routing, source authority, and BI request shape inside the loaded Mycroft skills" in config.system_prompt
    assert "Do not embed or improvise specialist-routing policy from this prompt" in config.system_prompt
    assert "Preserve the user's requested brand and portfolio scope" in config.system_prompt
    assert "source-backed selection" in config.system_prompt
    assert "reconcile them through the synthesis skill" in config.system_prompt
    assert "Source authority:" not in config.system_prompt
    assert "BI context sufficiency" not in config.system_prompt
    assert "Critical GAZ vehicle-selection routing" not in config.system_prompt
    assert "gaz_pricing_bi_int" not in config.system_prompt
    assert "marketing_analyst" not in config.system_prompt
    assert "web_search_agent" not in config.system_prompt
    assert "latest recommended mix first" in config.system_prompt
    assert "Do not blindly accept an incorrect premise" in config.system_prompt
    assert "Do not write a procurement mix like" in config.system_prompt
    assert config.skills == SkillsConfig(paths=("skills/mycroft",))


def test_ingos_products_cli_config_loads_stateful_agents_and_idea_check():
    config = load_cli_config(Path("data/config/mycroft/ingos_products_config.json"))

    assert config.subagents == SubagentsConfig(
        stateless=("web_search_agent",),
        stateful=(
            "product_Car",
            "product_Household",
            "product_Personal",
            "product_Tick Bite",
            "product_Инголаб",
            "product_Инголаб ПДФ",
            "product_Инголаб ППТХ",
            "product_Овертайм",
            "product_Юридическая помощь",
        ),
    )
    assert config.internal_tools == ()
    assert "product_Car" in config.system_prompt
    assert "idea_check" in config.system_prompt
    assert tuple(server.name for server in config.mcp.servers) == ("idea_reality",)
    assert config.mcp.servers[0].connection["command"] == "uvx"
    assert config.mcp.servers[0].tool_name_prefix is False
    assert config.mcp.servers[0].tools == ("idea_check",)


def test_kpi_agent_cli_config_loads_kpi_bi_subagent_and_skills():
    config = load_cli_config(Path("data/config/mycroft/kpi_agent_config.json"))

    assert config.subagents == SubagentsConfig(
        stateless=("kpi_bi_int",),
        stateful=(),
    )
    assert config.internal_tools[0].import_path == (
        "agents.mycroft_agent.scenarios.kpi_agent.tools:"
        "build_kpi_staff_structure_fuzzy_search_tool"
    )
    assert config.mcp.servers == ()
    assert config.skills == SkillsConfig(
        paths=("agents/mycroft_agent/scenarios/kpi_agent/skills",)
    )
    assert "position-first" in config.system_prompt
    assert "kpi_bi_int" in config.system_prompt
    assert "kpi_method_ref" in config.system_prompt


def test_required_environment_variables_include_default_runtime_requirements():
    config = load_cli_config()

    required = set(required_environment_variables(config, "openai"))

    assert required == {
        "GMAIL_MCP_CLIENT_ID",
        "GMAIL_MCP_CLIENT_SECRET",
        "GMAIL_MCP_REFRESH_TOKEN",
        "GOOGLE_MAPS_GROUNDING_LITE_API_KEY",
        "OPENAI_API_KEY",
        "YA_API_KEY",
        "YA_FOLDER_ID",
    }


def test_validate_required_environment_reports_missing_names(monkeypatch):
    config = load_cli_config()
    for variable_name in required_environment_variables(config, "openai"):
        monkeypatch.delenv(variable_name, raising=False)

    with pytest.raises(ValueError) as exc_info:
        validate_required_environment(config, "openai")

    message = str(exc_info.value)
    assert "GMAIL_MCP_CLIENT_ID" in message
    assert "GOOGLE_MAPS_GROUNDING_LITE_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert "YA_API_KEY" in message
    assert "YA_FOLDER_ID" in message


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


def test_load_mcp_tools_from_config_surfaces_idea_reality_startup_error(monkeypatch):
    async def fake_load_mcp_tools(_session, *, connection, server_name, tool_name_prefix):
        raise RuntimeError(f"Failed to start MCP server {server_name}")

    monkeypatch.setattr("agents.mycroft_agent.cli_config.load_mcp_tools", fake_load_mcp_tools)

    config = MCPConfig(
        tool_name_prefix=True,
        servers=(
            MCPServerSpec(
                name="idea_reality",
                connection={
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["idea-reality-mcp"],
                },
                tools=("idea_check",),
                tool_name_prefix=False,
            ),
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(load_mcp_tools_from_config(config))

    assert str(exc_info.value) == "Failed to start MCP server idea_reality"


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


def test_initialize_configured_subagents_accepts_inactive_explicit_agents(monkeypatch):
    class FakeAgent:
        pass

    fake_agent = FakeAgent()

    class FakeRegistry:
        _definitions = {
            "product_Household": _FakeDefinition(
                id="product_Household",
                name="Household",
                description="Inactive product expert.",
                is_active=False,
            )
        }

        async def ensure_agent_ready(self, agent_id):
            assert agent_id == "product_Household"
            return True

        def get_agent(self, agent_id):
            assert agent_id == "product_Household"
            return fake_agent

    monkeypatch.setattr("bot_service.agent_registry.agent_registry", FakeRegistry())

    subagents = asyncio.run(cli._initialize_configured_subagents(("product_Household",)))

    assert len(subagents) == 1
    assert subagents[0]["name"] == "product_Household"
    assert subagents[0]["description"] == "Household. Inactive product expert."
    assert subagents[0]["runnable"] is fake_agent


def test_initialize_configured_subagents_builds_builtin_web_search_agent(monkeypatch):
    monkeypatch.setattr(
        "agents.mycroft_agent.subagent_loader.build_web_search_subagent",
        lambda: {"name": "web_search_agent", "description": "web", "system_prompt": "prompt", "tools": []},
    )

    class FakeRegistry:
        _definitions = {}

        async def ensure_agent_ready(self, agent_id):
            raise AssertionError("registry should not be used for built-in web_search_agent")

        def get_agent(self, agent_id):
            raise AssertionError("registry should not be used for built-in web_search_agent")

    monkeypatch.setattr("bot_service.agent_registry.agent_registry", FakeRegistry())

    subagents = asyncio.run(cli._initialize_configured_subagents(("web_search_agent",)))

    assert subagents == [
        {
            "name": "web_search_agent",
            "description": "web",
            "system_prompt": "prompt",
            "tools": [],
        }
    ]


def test_initialize_configured_subagents_fails_on_unknown_explicit_agent(monkeypatch):
    class FakeRegistry:
        _definitions = {}

        async def ensure_agent_ready(self, agent_id):
            raise AssertionError("ensure_agent_ready should not be called for unknown agents")

        def get_agent(self, agent_id):
            raise AssertionError("get_agent should not be called for unknown agents")

    monkeypatch.setattr("bot_service.agent_registry.agent_registry", FakeRegistry())

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(cli._initialize_configured_subagents(("product_Unknown",)))

    assert "Unknown registry agent 'product_Unknown'" in str(exc_info.value)
