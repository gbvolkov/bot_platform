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
    BuiltAgentTools,
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
    assert maps_template.connection["timeout"] == 90
    assert maps_template.connection["sse_read_timeout"] == 300
    assert maps_template.guardrail_profiles["maps_search_places"] == "external_public_no_result_anonymization"
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
                    "guardrail_profile": "demo_public",
                },
                {"id": "demo_imported_tool", "import": "demo_platform_tools:imported_tool"},
            ],
            "tool_guardrail_profiles": {
                "demo_public": {
                    "allowed_roles": ["default"],
                    "side_effect": "read",
                    "category": "external_access",
                    "result_policy": {"scan_result": False},
                }
            },
            "mcp_servers": [
                {
                    "id": "demo_mcp",
                    "name": "demo",
                    "aliases": ["demo_alias"],
                    "transport": "http",
                    "url": "https://example.com/mcp",
                    "tool_name_prefix": False,
                    "guardrail_profiles": {"search": "demo_public"},
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
    bundle = registry.build_internal_tool_bundle(
        [InternalToolSpec(name="demo_tool", params={"suffix": "profiled"})],
        require_guardrail_profiles=True,
    )
    assert [tool.name for tool in bundle.tools] == ["demo_runtime_tool_profiled"]
    assert bundle.guardrail_profiles["demo_runtime_tool_profiled"]["category"] == "external_access"
    assert bundle.guardrail_profiles["demo_runtime_tool_profiled"]["name"] == "demo_runtime_tool_profiled"


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
    registry.register_tool_guardrail_profile(
        "external_public",
        {
            "allowed_roles": ["default"],
            "side_effect": "read",
            "category": "external_access",
            "result_policy": {"scan_result": False},
        },
    )
    registry.register_internal_tool(
        "web_search_tool",
        lambda: SimpleNamespace(name="web_search"),
        guardrail_profile="external_public",
    )
    registry.register_mcp_server(
        "googleapis_maps",
        name="maps",
        connection={"transport": "http", "url": "https://mapstools.googleapis.com/mcp"},
        tool_name_prefix=True,
        guardrail_profiles={"maps_search_places": "external_public"},
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

    bundle = asyncio.run(registry.build_tool_bundle(config, require_guardrail_profiles=True))

    assert set(bundle.guardrail_profiles) == {"web_search", "maps_search_places"}
    assert bundle.guardrail_profiles["web_search"]["name"] == "web_search"


def test_build_agent_tools_wraps_mcp_connection_failure(monkeypatch):
    registry = PlatformToolRegistry()
    registry.register_mcp_server(
        "googleapis_maps",
        name="maps",
        connection={"transport": "http", "url": "https://mapstools.googleapis.com/mcp"},
        tool_name_prefix=True,
    )

    async def fake_load_mcp_tools(_session, *, connection, server_name, tool_name_prefix):
        raise ExceptionGroup("mcp task group", [TimeoutError("connect timed out")])

    monkeypatch.setattr("platform_tools.registry.load_mcp_tools", fake_load_mcp_tools)

    config = registry.parse_agent_tools_config(
        [
            {
                "type": "mcp",
                "server": "googleapis_maps",
                "tools": ["search_places"],
            },
        ],
        agent_id="artifact_creator_agent",
    )

    with pytest.raises(ToolRegistryError, match="MCP server 'maps'.*TimeoutError: connect timed out"):
        asyncio.run(registry.build_tools(config))


def test_agent_registry_injects_configured_tools_into_any_tool_aware_agent(monkeypatch):
    captured = {}
    derived_tools = [SimpleNamespace(name="web_search"), SimpleNamespace(name="maps_search_places")]

    async def fake_build_agent_tool_bundle(tools_config, *, require_guardrail_profiles=False):
        captured["tools_config"] = tools_config
        captured["require_guardrail_profiles"] = require_guardrail_profiles
        return BuiltAgentTools(tools=derived_tools, guardrail_profiles={})

    def factory(**params):
        captured["factory_params"] = params
        return {"agent": "ok"}

    monkeypatch.setattr(agent_registry_module, "build_agent_tool_bundle", fake_build_agent_tool_bundle)
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
    assert captured["require_guardrail_profiles"] is False


def test_agent_registry_injects_resolved_tool_profiles_when_tool_guardrails_enabled(monkeypatch):
    captured = {}
    derived_tools = [SimpleNamespace(name="web_search")]
    profiles = {
        "web_search": {
            "name": "web_search",
            "allowed_roles": ["default"],
            "side_effect": "read",
            "result_policy": {"scan_result": False},
        }
    }

    async def fake_build_agent_tool_bundle(tools_config, *, require_guardrail_profiles=False):
        captured["tools_config"] = tools_config
        captured["require_guardrail_profiles"] = require_guardrail_profiles
        return BuiltAgentTools(tools=derived_tools, guardrail_profiles=profiles)

    def factory(**params):
        captured["factory_params"] = params
        return {"agent": "ok"}

    monkeypatch.setattr(agent_registry_module, "build_agent_tool_bundle", fake_build_agent_tool_bundle)
    registry = AgentRegistry.__new__(AgentRegistry)
    registry._definitions = {
        "tool_guarded": AgentDefinition(
            id="tool_guarded",
            name="Tool guarded",
            description="Tool guarded agent",
            factory=factory,
            default_provider=ModelType.GPT,
            supported_content_types=(ContentType.TEXT_FILES,),
            init_params={"guardrail_tool_execution_enabled": True},
            tools_config=parse_agent_tools_config(["web_search_tool"], agent_id="tool_guarded"),
            param_names=frozenset({"provider", "tools", "guardrail_tool_execution_enabled", "guardrail_tool_profiles"}),
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
        registry._start_initialization("tool_guarded")
        await registry._init_tasks["tool_guarded"]
        await asyncio.sleep(0)

    asyncio.run(run_initialization())

    assert captured["require_guardrail_profiles"] is True
    assert captured["factory_params"]["tools"] == derived_tools
    assert captured["factory_params"]["guardrail_tool_profiles"] == profiles


def test_agent_registry_uses_platform_graph_factory_when_configured(monkeypatch):
    captured = {}

    class FakeRuntime:
        tool_execution_enabled = False

    class FakeRuntimeFactory:
        @staticmethod
        def from_policy_id(policy_id, *, agent_id):
            captured["runtime_policy_id"] = policy_id
            captured["runtime_agent_id"] = agent_id
            return FakeRuntime()

    class FakeCompiler:
        def compile(self, spec, *, guardrail_runtime=None, checkpointer=None, tools=None, tool_profiles=None):
            captured["compiled_spec"] = spec
            captured["guardrail_runtime"] = guardrail_runtime
            captured["checkpointer"] = checkpointer
            captured["tools"] = tools
            captured["tool_profiles"] = tool_profiles
            return {"compiled": spec}

    def legacy_factory(**_params):
        raise AssertionError("legacy factory should not be called")

    def graph_factory(**params):
        captured["graph_params"] = params
        return {"spec": "ok"}

    monkeypatch.setattr(agent_registry_module, "PlatformGuardrailRuntime", FakeRuntimeFactory)
    monkeypatch.setattr(agent_registry_module, "PlatformGraphCompiler", lambda: FakeCompiler())
    registry = AgentRegistry.__new__(AgentRegistry)
    registry._definitions = {
        "platform_agent": AgentDefinition(
            id="platform_agent",
            name="Platform",
            description="Platform guarded",
            factory=legacy_factory,
            default_provider=ModelType.GPT,
            supported_content_types=(ContentType.TEXT_FILES,),
            init_params={},
            graph_factory=graph_factory,
            graph_param_names=frozenset({"provider"}),
            guardrail_policy_id="default_guardrails",
            guardrail_mode="platform",
        )
    }
    registry._instances = {}
    registry._init_tasks = {}
    registry._init_errors = {}
    registry._checkpointer = None
    registry._checkpointer_cm = None
    registry._checkpointer_lock = asyncio.Lock()

    async def run_initialization():
        registry._start_initialization("platform_agent")
        await registry._init_tasks["platform_agent"]
        await asyncio.sleep(0)

    asyncio.run(run_initialization())

    assert captured["graph_params"] == {"provider": ModelType.GPT}
    assert captured["runtime_policy_id"] == "default_guardrails"
    assert captured["runtime_agent_id"] == "platform_agent"
    assert captured["compiled_spec"] == {"spec": "ok"}
    assert registry._instances["platform_agent"] == {"compiled": {"spec": "ok"}}


def test_agent_config_parses_top_level_platform_guardrails(monkeypatch):
    module = ModuleType("demo_platform_agent")

    def initialize_agent():
        return object()

    def build_agent_graph():
        return {"spec": "ok"}

    module.initialize_agent = initialize_agent
    module.build_agent_graph = build_agent_graph
    monkeypatch.setitem(sys.modules, "demo_platform_agent", module)

    definitions = agent_registry_module._build_definitions_from_config(
        {
            "agents": [
                {
                    "id": "demo",
                    "name": "Demo",
                    "description": "Demo agent",
                    "module": "demo_platform_agent",
                    "guardrails": {
                        "policy": "default_guardrails",
                        "mode": "platform",
                    },
                    "tools": [
                        {
                            "type": "internal",
                            "name": "web_search_tool",
                            "params": {
                                "max_results": 2,
                                "summarize": True,
                            },
                        }
                    ],
                    "params": {
                        "provider": "openai",
                        "allow_external_tool_access": True,
                    },
                }
            ]
        },
        ModelType.GPT,
        (),
    )

    definition = definitions["demo"]
    assert definition.guardrail_policy_id == "default_guardrails"
    assert definition.guardrail_mode == "platform"
    assert definition.graph_factory is build_agent_graph
    assert definition.factory is initialize_agent
    assert definition.tools_config.internal_tools[0].name == "web_search_tool"
    assert definition.tools_config.internal_tools[0].params == {
        "max_results": 2,
        "summarize": True,
    }
    assert definition.init_params["allow_external_tool_access"] is True


def test_agent_registry_platform_mode_requires_graph_factory():
    registry = AgentRegistry.__new__(AgentRegistry)
    registry._definitions = {
        "broken": AgentDefinition(
            id="broken",
            name="Broken",
            description="Missing graph factory",
            factory=lambda **_params: object(),
            default_provider=ModelType.GPT,
            supported_content_types=(),
            guardrail_policy_id="default_guardrails",
            guardrail_mode="platform",
        )
    }
    registry._instances = {}
    registry._init_tasks = {}
    registry._init_errors = {}
    registry._checkpointer = None
    registry._checkpointer_cm = None
    registry._checkpointer_lock = asyncio.Lock()

    async def run_initialization():
        registry._start_initialization("broken")
        with pytest.raises(ValueError, match="build_agent_graph"):
            await registry._init_tasks["broken"]
        await asyncio.sleep(0)

    asyncio.run(run_initialization())

    assert isinstance(registry._init_errors["broken"], ValueError)


def test_agent_registry_platform_mode_requires_tool_profiles(monkeypatch):
    captured = {}
    derived_tools = [SimpleNamespace(name="web_search")]
    profiles = {"web_search": {"name": "web_search"}}

    class FakeRuntime:
        tool_execution_enabled = True

    class FakeRuntimeFactory:
        @staticmethod
        def from_policy_id(_policy_id, *, agent_id):
            return FakeRuntime()

    class FakeCompiler:
        def compile(self, spec, *, guardrail_runtime=None, checkpointer=None, tools=None, tool_profiles=None):
            captured["tools"] = tools
            captured["tool_profiles"] = tool_profiles
            return {"compiled": spec}

    async def fake_build_agent_tool_bundle(tools_config, *, require_guardrail_profiles=False):
        captured["require_guardrail_profiles"] = require_guardrail_profiles
        return BuiltAgentTools(tools=derived_tools, guardrail_profiles=profiles)

    monkeypatch.setattr(agent_registry_module, "PlatformGuardrailRuntime", FakeRuntimeFactory)
    monkeypatch.setattr(agent_registry_module, "PlatformGraphCompiler", lambda: FakeCompiler())
    monkeypatch.setattr(agent_registry_module, "build_agent_tool_bundle", fake_build_agent_tool_bundle)
    registry = AgentRegistry.__new__(AgentRegistry)
    registry._definitions = {
        "platform_tools": AgentDefinition(
            id="platform_tools",
            name="Platform tools",
            description="Platform guarded tools",
            factory=lambda **_params: object(),
            default_provider=ModelType.GPT,
            supported_content_types=(ContentType.TEXT_FILES,),
            tools_config=parse_agent_tools_config(["web_search_tool"], agent_id="platform_tools"),
            graph_factory=lambda **_params: {"spec": "ok"},
            graph_param_names=frozenset({"provider"}),
            guardrail_policy_id="default_guardrails",
            guardrail_mode="platform",
        )
    }
    registry._instances = {}
    registry._init_tasks = {}
    registry._init_errors = {}
    registry._checkpointer = None
    registry._checkpointer_cm = None
    registry._checkpointer_lock = asyncio.Lock()

    async def run_initialization():
        registry._start_initialization("platform_tools")
        await registry._init_tasks["platform_tools"]
        await asyncio.sleep(0)

    asyncio.run(run_initialization())

    assert captured["require_guardrail_profiles"] is True
    assert captured["tools"] == derived_tools
    assert captured["tool_profiles"] == profiles


def test_agent_registry_fails_if_tools_configured_for_agent_without_tools_param(monkeypatch):
    async def fake_build_agent_tool_bundle(_tools_config, *, require_guardrail_profiles=False):
        raise AssertionError("tools should not be built when signature cannot accept them")

    monkeypatch.setattr(agent_registry_module, "build_agent_tool_bundle", fake_build_agent_tool_bundle)
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
