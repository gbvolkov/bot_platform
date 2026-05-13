from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agents.utils import ModelType
from bot_service import agent_registry as agent_registry_module
from bot_service.agent_registry import AgentDefinition, AgentRegistry
from bot_service.schemas import ContentType
from bot_service.tool_registry import parse_agent_tools_config
from platform_tools.registry import (
    DEFAULT_TOOL_REGISTRY_CONFIG_PATH,
    InternalToolSpec,
    PlatformToolRegistry,
    ToolRegistryError,
    build_tool_registry_from_config,
    load_tool_registry_config,
)


def test_default_tool_registry_is_backed_by_editable_config_file():
    assert DEFAULT_TOOL_REGISTRY_CONFIG_PATH.name == "tools.json"
    config = load_tool_registry_config()

    registry = build_tool_registry_from_config(config)

    assert all("import" in entry or "factory" in entry for entry in config["internal_tools"])
    assert "web_search_tool" in registry.list_internal_tools()
    maps_template = registry.mcp_server_template("googleapis_maps")
    assert maps_template is not None
    assert maps_template.name == "maps"
    assert maps_template.connection["url"] == "https://mapstools.googleapis.com/mcp"
    assert registry.mcp_server_template("google_maps") is maps_template


def test_build_tool_registry_from_config_accepts_custom_editable_entries(monkeypatch):
    captured = {}
    module = ModuleType("demo_platform_tools")

    def build_demo_tool(**params):
        captured["params"] = params
        return SimpleNamespace(name=f"demo_runtime_tool_{params['suffix']}")

    module.build_demo_tool = build_demo_tool
    module.imported_tool = SimpleNamespace(name="demo_imported_tool")
    monkeypatch.setitem(sys.modules, "demo_platform_tools", module)

    registry = build_tool_registry_from_config(
        {
            "internal_tools": [
                {
                    "id": "demo_tool",
                    "factory": "demo_platform_tools:build_demo_tool",
                    "params": {"suffix": "default"},
                },
                {"id": "demo_imported_tool", "import": "demo_platform_tools:imported_tool"},
            ],
            "mcp_servers": [
                {
                    "id": "demo_mcp",
                    "name": "demo",
                    "aliases": ["demo_alias"],
                    "transport": "http",
                    "url": "https://example.com/mcp",
                    "tool_name_prefix": False,
                }
            ],
        }
    )

    assert [
        tool.name
        for tool in registry.build_internal_tools(
            [
                InternalToolSpec(name="demo_tool", params={"suffix": "override"}),
                InternalToolSpec(name="demo_imported_tool"),
            ]
        )
    ] == [
        "demo_runtime_tool_override",
        "demo_imported_tool",
    ]
    assert captured["params"] == {"suffix": "override"}
    assert registry.mcp_server_template("demo_alias").connection["url"] == "https://example.com/mcp"


def test_build_tool_registry_from_config_rejects_duplicate_tool_ids():
    with pytest.raises(ToolRegistryError, match="Duplicate internal tool registration: duplicate"):
        build_tool_registry_from_config(
            {
                "internal_tools": [
                    {"id": "duplicate", "import": "builtins:str"},
                    {"id": "duplicate", "factory": "builtins:dict"},
                ]
            }
        )


def test_registry_fails_closed_for_unknown_internal_tool():
    registry = PlatformToolRegistry()

    with pytest.raises(ToolRegistryError, match="Unknown internal tool 'missing'"):
        registry.build_internal_tools([InternalToolSpec(name="missing")])


def test_parse_agent_tools_config_accepts_internal_and_mcp_entries():
    config = parse_agent_tools_config(
        [
            {
                "type": "internal",
                "name": "web_search_tool",
                "params": {"max_results": 3, "summarize": False},
            },
            {
                "type": "mcp",
                "server": "googleapis_maps",
                "tools": ["search_places"],
            },
        ],
        agent_id="artifact_creator_agent",
    )

    assert config.internal_tools[0].name == "web_search_tool"
    assert config.internal_tools[0].params == {"max_results": 3, "summarize": False}
    assert config.mcp.tool_name_prefix is True
    assert config.mcp.servers[0].name == "maps"
    assert config.mcp.servers[0].connection["url"] == "https://mapstools.googleapis.com/mcp"
    assert config.mcp.servers[0].tools == ("search_places",)


def test_parse_agent_tools_config_rejects_non_array_tools():
    with pytest.raises(ValueError, match="field 'tools' must be a list"):
        parse_agent_tools_config({"type": "internal"}, agent_id="broken")


def test_build_agent_tools_combines_internal_and_mcp_tools(monkeypatch):
    captured = {}
    registry = PlatformToolRegistry()
    registry.register_internal_tool("web_search_tool", lambda: SimpleNamespace(name="web_search"))
    registry.register_mcp_server(
        "googleapis_maps",
        name="maps",
        connection={"transport": "http", "url": "https://mapstools.googleapis.com/mcp"},
        tool_name_prefix=True,
    )

    async def fake_load_mcp_tools(_session, *, connection, server_name, tool_name_prefix):
        captured["connection"] = connection
        captured["server_name"] = server_name
        captured["tool_name_prefix"] = tool_name_prefix
        return [SimpleNamespace(name="maps_search_places")]

    monkeypatch.setattr("platform_tools.registry.load_mcp_tools", fake_load_mcp_tools)

    config = registry.parse_agent_tools_config(
        [
            "web_search_tool",
            {
                "type": "mcp",
                "server": "googleapis_maps",
                "tools": ["search_places"],
            },
        ],
        agent_id="artifact_creator_agent",
    )

    tools = asyncio.run(registry.build_tools(config))

    assert [tool.name for tool in tools] == ["web_search", "maps_search_places"]
    assert captured["connection"] == {"transport": "http", "url": "https://mapstools.googleapis.com/mcp"}
    assert captured["server_name"] == "maps"
    assert captured["tool_name_prefix"] is True


def test_agent_registry_injects_configured_tools_into_any_tool_aware_agent(monkeypatch):
    captured = {}
    derived_tools = [SimpleNamespace(name="web_search"), SimpleNamespace(name="maps_search_places")]

    async def fake_build_agent_tools(tools_config):
        captured["tools_config"] = tools_config
        return derived_tools

    def factory(**params):
        captured["factory_params"] = params
        return {"agent": "ok"}

    monkeypatch.setattr(agent_registry_module, "build_agent_tools", fake_build_agent_tools)
    registry = AgentRegistry.__new__(AgentRegistry)
    registry._definitions = {
        "tool_aware": AgentDefinition(
            id="tool_aware",
            name="Tool aware",
            description="Tool aware agent",
            factory=factory,
            default_provider=ModelType.GPT,
            supported_content_types=(ContentType.TEXT_FILES,),
            init_params={},
            tools_config=parse_agent_tools_config(["web_search_tool"], agent_id="tool_aware"),
            param_names=frozenset({"provider", "tools"}),
            accepts_kwargs=False,
        )
    }
    registry._instances = {}
    registry._init_tasks = {}
    registry._init_errors = {}
    registry._checkpointer = None
    registry._checkpointer_cm = None
    registry._checkpointer_lock = asyncio.Lock()

    async def run_initialization():
        registry._start_initialization("tool_aware")
        await registry._init_tasks["tool_aware"]
        await asyncio.sleep(0)

    asyncio.run(run_initialization())

    assert registry._instances["tool_aware"] == {"agent": "ok"}
    assert captured["factory_params"]["tools"] == derived_tools
    assert captured["factory_params"]["provider"] == ModelType.GPT


def test_agent_registry_fails_if_tools_configured_for_agent_without_tools_param(monkeypatch):
    async def fake_build_agent_tools(_tools_config):
        raise AssertionError("tools should not be built when signature cannot accept them")

    monkeypatch.setattr(agent_registry_module, "build_agent_tools", fake_build_agent_tools)
    registry = AgentRegistry.__new__(AgentRegistry)
    registry._definitions = {
        "plain": AgentDefinition(
            id="plain",
            name="Plain",
            description="Plain agent",
            factory=lambda **_params: object(),
            default_provider=ModelType.GPT,
            supported_content_types=(),
            init_params={},
            tools_config=parse_agent_tools_config(["web_search_tool"], agent_id="plain"),
            param_names=frozenset({"provider"}),
            accepts_kwargs=False,
        )
    }
    registry._instances = {}
    registry._init_tasks = {}
    registry._init_errors = {}
    registry._checkpointer = None
    registry._checkpointer_cm = None
    registry._checkpointer_lock = asyncio.Lock()

    async def run_initialization():
        registry._start_initialization("plain")
        with pytest.raises(ValueError):
            await registry._init_tasks["plain"]
        await asyncio.sleep(0)

    asyncio.run(run_initialization())

    assert "plain" not in registry._instances
    assert isinstance(registry._init_errors["plain"], ValueError)
    assert "does not accept a 'tools' parameter" in str(registry._init_errors["plain"])
