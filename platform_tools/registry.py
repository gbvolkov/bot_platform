from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain_mcp_adapters.tools import load_mcp_tools

from platform_guardrails.tool_policy import coerce_tool_security_profile
from platform_guardrails.tool_registry import get_tool_name


_BRACED_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")
_ALLOWED_MCP_TRANSPORTS = {
    "stdio",
    "sse",
    "http",
    "streamable_http",
    "streamable-http",
    "websocket",
}
DEFAULT_TOOL_REGISTRY_CONFIG_PATH = Path(__file__).with_name("tools.json")


class ToolRegistryError(ValueError):
    """Raised when platform tool config cannot be resolved safely."""


@dataclass(frozen=True)
class InternalToolSpec:
    name: str | None = None
    import_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InternalToolTemplate:
    id: str
    builder: Callable[..., Any]
    guardrail_profile: Any = None


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    connection: dict[str, Any]
    tools: tuple[str, ...] | None = None
    tool_name_prefix: bool | None = None
    guardrail_profiles: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPConfig:
    tool_name_prefix: bool
    servers: tuple[MCPServerSpec, ...]


@dataclass(frozen=True)
class MCPServerTemplate:
    id: str
    name: str
    connection: dict[str, Any]
    tools: tuple[str, ...] | None = None
    tool_name_prefix: bool | None = None
    guardrail_profiles: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentToolsConfig:
    internal_tools: tuple[InternalToolSpec, ...] = ()
    mcp: MCPConfig = MCPConfig(tool_name_prefix=True, servers=())

    @property
    def configured(self) -> bool:
        return bool(self.internal_tools or self.mcp.servers)


@dataclass(frozen=True)
class BuiltAgentTools:
    tools: list[Any]
    guardrail_profiles: dict[str, dict[str, Any]]


class PlatformToolRegistry:
    """Central registry for named internal tools and MCP server templates."""

    def __init__(self) -> None:
        self._internal_tools: dict[str, InternalToolTemplate] = {}
        self._mcp_servers: dict[str, MCPServerTemplate] = {}
        self._tool_guardrail_profiles: dict[str, dict[str, Any]] = {}

    def register_internal_tool(
        self,
        name: str,
        builder: Callable[..., Any],
        *,
        guardrail_profile: Any = None,
    ) -> None:
        tool_name = _require_str(name, "internal tool name")
        if tool_name in self._internal_tools:
            raise ToolRegistryError(f"Duplicate internal tool registration: {tool_name}")
        self._internal_tools[tool_name] = InternalToolTemplate(
            id=tool_name,
            builder=builder,
            guardrail_profile=guardrail_profile,
        )

    def register_tool_guardrail_profile(self, profile_id: str, profile: dict[str, Any]) -> None:
        resolved_id = _require_str(profile_id, "tool guardrail profile id")
        if resolved_id in self._tool_guardrail_profiles:
            raise ToolRegistryError(f"Duplicate tool guardrail profile: {resolved_id}")
        if not isinstance(profile, dict):
            raise ToolRegistryError(f"Tool guardrail profile '{resolved_id}' must be an object.")
        self._tool_guardrail_profiles[resolved_id] = dict(profile)

    def register_mcp_server(
        self,
        template_id: str,
        *,
        name: str,
        connection: dict[str, Any],
        tools: tuple[str, ...] | list[str] | None = None,
        tool_name_prefix: bool | None = None,
        guardrail_profiles: dict[str, Any] | None = None,
        aliases: tuple[str, ...] | list[str] = (),
    ) -> None:
        resolved_id = _require_str(template_id, "mcp server id")
        template = MCPServerTemplate(
            id=resolved_id,
            name=_require_str(name, "mcp server name"),
            connection=dict(connection),
            tools=None if tools is None else tuple(_require_str(item, "mcp server tool") for item in tools),
            tool_name_prefix=tool_name_prefix,
            guardrail_profiles=dict(guardrail_profiles or {}),
        )
        self._register_mcp_template(resolved_id, template)
        for alias in aliases:
            self._register_mcp_template(_require_str(alias, "mcp server alias"), template)

    def list_internal_tools(self) -> list[str]:
        return sorted(self._internal_tools)

    def mcp_server_template(self, template_id: str) -> MCPServerTemplate | None:
        return self._mcp_servers.get(template_id)

    def build_internal_tools(self, specs: list[InternalToolSpec] | tuple[InternalToolSpec, ...]) -> list[Any]:
        return self.build_internal_tool_bundle(specs).tools

    def build_internal_tool_bundle(
        self,
        specs: list[InternalToolSpec] | tuple[InternalToolSpec, ...],
        *,
        require_guardrail_profiles: bool = False,
    ) -> BuiltAgentTools:
        tools: list[Any] = []
        profiles: dict[str, dict[str, Any]] = {}
        for spec in specs:
            profile_ref = None
            if spec.import_path is not None:
                builder = _load_imported_builder(spec.import_path)
            else:
                if spec.name is None:
                    raise ToolRegistryError("Internal tool spec must define either 'name' or 'import'.")
                template = self._internal_tools.get(spec.name)
                if template is None:
                    available = ", ".join(self.list_internal_tools())
                    raise ToolRegistryError(
                        f"Unknown internal tool '{spec.name}'. Available tools: {available}"
                    )
                builder = template.builder
                profile_ref = template.guardrail_profile

            built = builder(**spec.params)
            built_tools = list(built) if isinstance(built, (list, tuple)) else [built]
            tools.extend(built_tools)
            for tool in built_tools:
                profile = self._resolve_tool_guardrail_profile(
                    profile_ref,
                    tool_name=get_tool_name(tool),
                    required=require_guardrail_profiles,
                    source=f"internal tool {spec.name or spec.import_path!r}",
                )
                if profile is not None:
                    _add_guardrail_profile(profiles, profile)
        return BuiltAgentTools(tools=tools, guardrail_profiles=profiles)

    async def load_mcp_tools_from_config(self, mcp_config: MCPConfig) -> list[Any]:
        return (await self.load_mcp_tool_bundle(mcp_config)).tools

    async def load_mcp_tool_bundle(
        self,
        mcp_config: MCPConfig,
        *,
        require_guardrail_profiles: bool = False,
    ) -> BuiltAgentTools:
        tools: list[Any] = []
        profiles: dict[str, dict[str, Any]] = {}
        for server in mcp_config.servers:
            use_prefix = (
                server.tool_name_prefix
                if server.tool_name_prefix is not None
                else mcp_config.tool_name_prefix
            )
            resolved_connection = _expand_mcp_connection_env(server.connection, server.name)
            try:
                server_tools = await load_mcp_tools(
                    None,
                    connection=resolved_connection,
                    server_name=server.name,
                    tool_name_prefix=use_prefix,
                )
            except Exception as exc:
                raise ToolRegistryError(
                    "Could not load MCP tools from "
                    f"{_mcp_server_description(server, resolved_connection)}: "
                    f"{_summarize_exception(exc)}"
                ) from exc
            selected_tools = select_mcp_tools(server, server_tools)
            tools.extend(selected_tools)
            for tool in selected_tools:
                tool_name = get_tool_name(tool)
                profile = self._resolve_tool_guardrail_profile(
                    _mcp_profile_ref(server, tool_name),
                    tool_name=tool_name,
                    required=require_guardrail_profiles,
                    source=f"MCP server {server.name!r} tool {tool_name!r}",
                )
                if profile is not None:
                    _add_guardrail_profile(profiles, profile)
        return BuiltAgentTools(tools=tools, guardrail_profiles=profiles)

    async def build_tools(self, config: AgentToolsConfig) -> list[Any]:
        return (await self.build_tool_bundle(config)).tools

    async def build_tool_bundle(
        self,
        config: AgentToolsConfig,
        *,
        require_guardrail_profiles: bool = False,
    ) -> BuiltAgentTools:
        if not config.configured:
            return BuiltAgentTools(tools=[], guardrail_profiles={})
        internal = self.build_internal_tool_bundle(
            config.internal_tools,
            require_guardrail_profiles=require_guardrail_profiles,
        )
        mcp = await self.load_mcp_tool_bundle(
            config.mcp,
            require_guardrail_profiles=require_guardrail_profiles,
        )
        profiles = dict(internal.guardrail_profiles)
        for profile in mcp.guardrail_profiles.values():
            _add_guardrail_profile(profiles, profile)
        return BuiltAgentTools(
            tools=[*internal.tools, *mcp.tools],
            guardrail_profiles=profiles,
        )

    def parse_agent_tools_config(self, value: Any, *, agent_id: str) -> AgentToolsConfig:
        if value is None:
            return AgentToolsConfig()
        if not isinstance(value, list):
            raise ToolRegistryError(f"Agent '{agent_id}' field 'tools' must be a list.")

        internal_tools: list[InternalToolSpec] = []
        mcp_servers: list[MCPServerSpec] = []
        mcp_tool_name_prefix = True

        for index, entry in enumerate(value):
            if isinstance(entry, str):
                internal_tools.append(InternalToolSpec(name=_require_str(entry, f"tools[{index}]")))
                continue
            if not isinstance(entry, dict):
                raise ToolRegistryError(f"Agent '{agent_id}' tools[{index}] must be a string or object.")

            tool_type = str(entry.get("type") or entry.get("kind") or "").strip().lower()
            if not tool_type:
                if "import" in entry or "tool" in entry:
                    tool_type = "internal"
                elif "server" in entry or "transport" in entry or "connection" in entry:
                    tool_type = "mcp"

            if tool_type == "internal":
                internal_tools.append(_parse_internal_tool_entry(entry, agent_id=agent_id, index=index))
                continue
            if tool_type == "mcp":
                server, server_prefix = self._parse_mcp_tool_entry(entry, agent_id=agent_id, index=index)
                if server_prefix is not None:
                    mcp_tool_name_prefix = server_prefix
                mcp_servers.append(server)
                continue

            raise ToolRegistryError(
                f"Agent '{agent_id}' tools[{index}] has unsupported type {tool_type!r}. "
                "Supported values: internal, mcp."
            )

        return AgentToolsConfig(
            internal_tools=tuple(internal_tools),
            mcp=MCPConfig(tool_name_prefix=mcp_tool_name_prefix, servers=tuple(mcp_servers)),
        )

    def _register_mcp_template(self, key: str, template: MCPServerTemplate) -> None:
        if key in self._mcp_servers:
            raise ToolRegistryError(f"Duplicate MCP server registration: {key}")
        self._mcp_servers[key] = template

    def _parse_mcp_tool_entry(
        self,
        entry: dict[str, Any],
        *,
        agent_id: str,
        index: int,
    ) -> tuple[MCPServerSpec, bool | None]:
        forbidden_guardrail_fields = {"guardrail_profile", "guardrail_profiles", "tool_guardrail_profile"}
        forbidden = forbidden_guardrail_fields.intersection(entry)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ToolRegistryError(
                f"Agent '{agent_id}' tools[{index}] cannot define guardrail fields: {names}. "
                "Configure tool guardrail profiles in platform_tools/tools.json."
            )

        raw_server_id = entry.get("server") or entry.get("name") or entry.get("id")
        server_id = _require_str(raw_server_id, f"tools[{index}].server")
        template = self.mcp_server_template(server_id)

        raw_connection = entry.get("connection")
        if raw_connection is not None and not isinstance(raw_connection, dict):
            raise ToolRegistryError(f"Agent '{agent_id}' tools[{index}].connection must be an object.")

        explicit_connection = (
            dict(raw_connection)
            if raw_connection is not None
            else {
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "type",
                    "kind",
                    "id",
                    "name",
                    "server",
                    "tools",
                    "tool_name_prefix",
                    "mcp_tool_name_prefix",
                    *forbidden_guardrail_fields,
                }
            }
        )

        if template is None and not explicit_connection:
            raise ToolRegistryError(f"Agent '{agent_id}' references unknown MCP server '{server_id}'.")

        if template is not None:
            connection = {**template.connection, **explicit_connection}
            server_name = entry.get("mcp_server_name") or template.name
            template_tools = template.tools
            template_prefix = template.tool_name_prefix
        else:
            connection = explicit_connection
            server_name = server_id
            template_tools = None
            template_prefix = None

        transport = _require_str(connection.get("transport"), f"tools[{index}].transport")
        if transport not in _ALLOWED_MCP_TRANSPORTS:
            allowed = ", ".join(sorted(_ALLOWED_MCP_TRANSPORTS))
            raise ToolRegistryError(
                f"Unsupported MCP transport '{transport}' for server '{server_name}'. "
                f"Allowed transports: {allowed}"
            )

        raw_tools = entry.get("tools")
        if raw_tools is None:
            selected_tools = template_tools
        elif isinstance(raw_tools, list):
            selected_tools = tuple(_require_str(item, f"tools[{index}].tools[]") for item in raw_tools)
        else:
            raise ToolRegistryError(f"Agent '{agent_id}' tools[{index}].tools must be a list.")

        raw_prefix = entry.get("tool_name_prefix")
        if raw_prefix is not None and not isinstance(raw_prefix, bool):
            raise ToolRegistryError(f"Agent '{agent_id}' tools[{index}].tool_name_prefix must be a boolean.")
        if raw_prefix is None:
            raw_prefix = template_prefix

        global_prefix = entry.get("mcp_tool_name_prefix")
        if global_prefix is not None and not isinstance(global_prefix, bool):
            raise ToolRegistryError(f"Agent '{agent_id}' tools[{index}].mcp_tool_name_prefix must be a boolean.")

        return (
            MCPServerSpec(
                name=_require_str(server_name, f"tools[{index}].name"),
                connection=connection,
                tools=selected_tools,
                tool_name_prefix=raw_prefix,
                guardrail_profiles=dict(template.guardrail_profiles if template is not None else {}),
            ),
            global_prefix,
        )

    def _resolve_tool_guardrail_profile(
        self,
        profile_ref: Any,
        *,
        tool_name: str | None,
        required: bool,
        source: str,
    ) -> dict[str, Any] | None:
        if not tool_name:
            if required:
                raise ToolRegistryError(f"{source} is missing a runtime tool name for guardrail profile binding.")
            return None
        if profile_ref is None:
            if required:
                raise ToolRegistryError(f"{source} is missing a guardrail profile.")
            return None
        if isinstance(profile_ref, str):
            profile = self._tool_guardrail_profiles.get(profile_ref)
            if profile is None:
                if required:
                    raise ToolRegistryError(f"{source} references unknown guardrail profile {profile_ref!r}.")
                return None
            return _validated_runtime_profile({**profile, "name": tool_name}, source=source)
        if isinstance(profile_ref, Mapping):
            return _validated_runtime_profile({**dict(profile_ref), "name": tool_name}, source=source)
        if required:
            raise ToolRegistryError(f"{source} guardrail profile reference must be a string or object.")
        return None


def _mcp_profile_ref(server: MCPServerSpec, tool_name: str | None) -> Any:
    if not tool_name:
        return None
    if tool_name in server.guardrail_profiles:
        return server.guardrail_profiles[tool_name]
    prefixed_name = f"{server.name}_{tool_name}"
    if prefixed_name in server.guardrail_profiles:
        return server.guardrail_profiles[prefixed_name]
    prefix = f"{server.name}_"
    if tool_name.startswith(prefix):
        unprefixed_name = tool_name[len(prefix) :]
        return server.guardrail_profiles.get(unprefixed_name)
    return None


def _validated_runtime_profile(profile: dict[str, Any], *, source: str) -> dict[str, Any]:
    try:
        coerce_tool_security_profile(profile)
    except Exception as exc:  # noqa: BLE001
        raise ToolRegistryError(f"{source} has an invalid guardrail profile: {exc}") from exc
    return profile


def _add_guardrail_profile(
    profiles: dict[str, dict[str, Any]],
    profile: dict[str, Any],
) -> None:
    profile_name = _require_str(profile.get("name"), "tool guardrail profile name")
    existing = profiles.get(profile_name)
    if existing is not None and existing != profile:
        raise ToolRegistryError(
            f"Conflicting guardrail profiles resolved for runtime tool {profile_name!r}."
        )
    profiles[profile_name] = profile


def _mcp_server_description(server: MCPServerSpec, connection: Mapping[str, Any]) -> str:
    url = connection.get("url")
    if isinstance(url, str) and url:
        return f"MCP server {server.name!r} at {url!r}"
    command = connection.get("command")
    if isinstance(command, str) and command:
        return f"MCP server {server.name!r} command {command!r}"
    return f"MCP server {server.name!r}"


def _summarize_exception(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        leaves = _exception_group_leaves(exc)
        if leaves:
            return "; ".join(_single_exception_summary(item) for item in leaves[:3])
    return _single_exception_summary(exc)


def _exception_group_leaves(exc: BaseExceptionGroup) -> list[BaseException]:
    leaves: list[BaseException] = []
    for item in exc.exceptions:
        if isinstance(item, BaseExceptionGroup):
            leaves.extend(_exception_group_leaves(item))
        else:
            leaves.append(item)
    return leaves


def _single_exception_summary(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return type(exc).__name__


def load_tool_registry_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = DEFAULT_TOOL_REGISTRY_CONFIG_PATH if path is None else Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ToolRegistryError("Tool registry config must be a JSON object.")
    return data


def build_tool_registry_from_config(config: dict[str, Any]) -> PlatformToolRegistry:
    registry = PlatformToolRegistry()

    raw_tool_profiles = config.get("tool_guardrail_profiles") or {}
    if not isinstance(raw_tool_profiles, dict):
        raise ToolRegistryError("Tool registry config field 'tool_guardrail_profiles' must be an object.")
    for profile_id, profile in raw_tool_profiles.items():
        registry.register_tool_guardrail_profile(str(profile_id), profile)

    raw_internal_tools = config.get("internal_tools") or []
    if not isinstance(raw_internal_tools, list):
        raise ToolRegistryError("Tool registry config field 'internal_tools' must be a list.")
    for index, entry in enumerate(raw_internal_tools):
        tool_id, builder, guardrail_profile = _parse_registry_internal_tool(entry, index=index)
        registry.register_internal_tool(tool_id, builder, guardrail_profile=guardrail_profile)

    raw_mcp_servers = config.get("mcp_servers")
    if raw_mcp_servers is None:
        raw_mcp_servers = (config.get("mcp") or {}).get("servers", [])
    if not isinstance(raw_mcp_servers, list):
        raise ToolRegistryError("Tool registry config field 'mcp_servers' must be a list.")
    for index, entry in enumerate(raw_mcp_servers):
        (
            template_id,
            name,
            connection,
            tools,
            tool_name_prefix,
            guardrail_profiles,
            aliases,
        ) = _parse_registry_mcp_server(entry, index=index)
        registry.register_mcp_server(
            template_id,
            name=name,
            connection=connection,
            tools=tools,
            tool_name_prefix=tool_name_prefix,
            guardrail_profiles=guardrail_profiles,
            aliases=aliases,
        )

    return registry


def build_default_tool_registry(path: str | Path | None = None) -> PlatformToolRegistry:
    return build_tool_registry_from_config(load_tool_registry_config(path))


def list_available_internal_tools() -> list[str]:
    return default_tool_registry.list_internal_tools()


def build_internal_tools(specs: list[InternalToolSpec] | tuple[InternalToolSpec, ...]) -> list[Any]:
    return default_tool_registry.build_internal_tools(specs)


async def load_mcp_tools_from_config(mcp_config: MCPConfig) -> list[Any]:
    return await default_tool_registry.load_mcp_tools_from_config(mcp_config)


async def build_agent_tools(config: AgentToolsConfig) -> list[Any]:
    return await default_tool_registry.build_tools(config)


async def build_agent_tool_bundle(
    config: AgentToolsConfig,
    *,
    require_guardrail_profiles: bool = False,
) -> BuiltAgentTools:
    return await default_tool_registry.build_tool_bundle(
        config,
        require_guardrail_profiles=require_guardrail_profiles,
    )


def parse_agent_tools_config(value: Any, *, agent_id: str) -> AgentToolsConfig:
    return default_tool_registry.parse_agent_tools_config(value, agent_id=agent_id)


def select_mcp_tools(server: MCPServerSpec, loaded_tools: list[Any]) -> list[Any]:
    if server.tools is None:
        return list(loaded_tools)

    selected: list[Any] = []
    missing: list[str] = []
    for requested_name in server.tools:
        match = _find_mcp_tool(server.name, requested_name, loaded_tools)
        if match is None:
            missing.append(requested_name)
            continue
        if match not in selected:
            selected.append(match)

    if missing:
        available = ", ".join(tool.name for tool in loaded_tools)
        missing_text = ", ".join(missing)
        raise ToolRegistryError(
            f"MCP server '{server.name}' does not expose requested tools: {missing_text}. "
            f"Available tools: {available}"
        )
    return selected


def required_environment_variables(config: AgentToolsConfig) -> tuple[str, ...]:
    required: list[str] = []
    for server in config.mcp.servers:
        required.extend(_collect_env_placeholders(server.connection))
    return tuple(sorted(dict.fromkeys(required)))


def _find_mcp_tool(server_name: str, requested_name: str, loaded_tools: list[Any]) -> Any | None:
    prefixed_name = f"{server_name}_{requested_name}"
    for tool in loaded_tools:
        if tool.name in {requested_name, prefixed_name}:
            return tool
    return None


def _parse_internal_tool_entry(
    entry: dict[str, Any],
    *,
    agent_id: str,
    index: int,
) -> InternalToolSpec:
    forbidden_guardrail_fields = {"guardrail_profile", "guardrail_profiles", "tool_guardrail_profile"}
    forbidden = forbidden_guardrail_fields.intersection(entry)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise ToolRegistryError(
            f"Agent '{agent_id}' tools[{index}] cannot define guardrail fields: {names}. "
            "Configure tool guardrail profiles in platform_tools/tools.json."
        )

    has_import = "import" in entry
    raw_name = entry.get("name") or entry.get("tool")
    has_name = raw_name is not None
    if has_import == has_name:
        raise ToolRegistryError(
            f"Agent '{agent_id}' tools[{index}] internal tool must contain exactly one of "
            "'name', 'tool', or 'import'."
        )

    params = entry.get("params") or {}
    if not isinstance(params, dict):
        raise ToolRegistryError(f"Agent '{agent_id}' tools[{index}].params must be an object.")

    if has_import:
        unexpected = set(entry) - {"type", "kind", "import", "params"}
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ToolRegistryError(f"Agent '{agent_id}' tools[{index}] has unsupported internal fields: {names}")
        return InternalToolSpec(
            import_path=_require_str(entry.get("import"), f"tools[{index}].import"),
            params=dict(params),
        )

    unexpected = set(entry) - {"type", "kind", "name", "tool", "params"}
    if unexpected:
        extra_params = {key: entry[key] for key in unexpected}
        params = {**extra_params, **params}
    return InternalToolSpec(name=_require_str(raw_name, f"tools[{index}].name"), params=dict(params))


def _parse_registry_internal_tool(
    entry: Any,
    *,
    index: int,
) -> tuple[str, Callable[..., Any], Any]:
    if not isinstance(entry, dict):
        raise ToolRegistryError("Each internal_tools entry must be an object.")

    tool_id = _require_str(entry.get("id") or entry.get("name"), f"internal_tools[{index}].id")
    import_ref = entry.get("import")
    factory_ref = entry.get("factory")
    configured_refs = [value for value in (import_ref, factory_ref) if value is not None]
    if len(configured_refs) != 1:
        raise ToolRegistryError(
            f"internal_tools[{index}] must contain exactly one of 'import' or 'factory'."
        )

    params = entry.get("params") or {}
    if not isinstance(params, dict):
        raise ToolRegistryError(f"Tool registry config field 'internal_tools[{index}].params' must be an object.")

    guardrail_profile = entry.get("guardrail_profile")
    unexpected = set(entry) - {"id", "name", "import", "factory", "params", "description", "guardrail_profile"}
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ToolRegistryError(f"Unsupported internal_tools[{index}] fields: {names}")
    if import_ref is not None:
        if params:
            raise ToolRegistryError(f"internal_tools[{index}] cannot define params with 'import'. Use 'factory'.")
        return (
            tool_id,
            _make_import_tool_builder(tool_id, _require_str(import_ref, f"internal_tools[{index}].import")),
            guardrail_profile,
        )
    if factory_ref is not None:
        return (
            tool_id,
            _make_factory_tool_builder(
                tool_id,
                _require_str(factory_ref, f"internal_tools[{index}].factory"),
                params,
            ),
            guardrail_profile,
        )
    raise ToolRegistryError(f"internal_tools[{index}] is missing an importable tool definition.")


def _parse_registry_mcp_server(
    entry: Any,
    *,
    index: int,
) -> tuple[str, str, dict[str, Any], tuple[str, ...] | None, bool | None, dict[str, Any], tuple[str, ...]]:
    if not isinstance(entry, dict):
        raise ToolRegistryError("Each mcp_servers entry must be an object.")

    template_id = _require_str(entry.get("id") or entry.get("server"), f"mcp_servers[{index}].id")
    name = _require_str(entry.get("name") or template_id, f"mcp_servers[{index}].name")
    aliases_value = entry.get("aliases") or ()
    if not isinstance(aliases_value, (list, tuple)):
        raise ToolRegistryError(f"Tool registry config field 'mcp_servers[{index}].aliases' must be a list.")
    aliases = tuple(_require_str(alias, f"mcp_servers[{index}].aliases[]") for alias in aliases_value)

    raw_tools = entry.get("tools")
    if raw_tools is None:
        tools = None
    elif isinstance(raw_tools, list):
        tools = tuple(_require_str(item, f"mcp_servers[{index}].tools[]") for item in raw_tools)
    else:
        raise ToolRegistryError(f"Tool registry config field 'mcp_servers[{index}].tools' must be a list.")

    raw_prefix = entry.get("tool_name_prefix")
    if raw_prefix is not None and not isinstance(raw_prefix, bool):
        raise ToolRegistryError(f"Tool registry config field 'mcp_servers[{index}].tool_name_prefix' must be a boolean.")

    raw_guardrail_profiles = entry.get("guardrail_profiles") or {}
    if not isinstance(raw_guardrail_profiles, dict):
        raise ToolRegistryError(f"Tool registry config field 'mcp_servers[{index}].guardrail_profiles' must be an object.")

    raw_connection = entry.get("connection")
    if raw_connection is not None:
        if not isinstance(raw_connection, dict):
            raise ToolRegistryError(f"Tool registry config field 'mcp_servers[{index}].connection' must be an object.")
        connection = dict(raw_connection)
    else:
        connection = {
            key: value
            for key, value in entry.items()
            if key not in {
                "id",
                "server",
                "name",
                "aliases",
                "tools",
                "tool_name_prefix",
                "description",
                "guardrail_profiles",
            }
        }

    transport = _require_str(connection.get("transport"), f"mcp_servers[{index}].transport")
    if transport not in _ALLOWED_MCP_TRANSPORTS:
        allowed = ", ".join(sorted(_ALLOWED_MCP_TRANSPORTS))
        raise ToolRegistryError(
            f"Unsupported MCP transport '{transport}' for server '{name}'. "
            f"Allowed transports: {allowed}"
        )
    return template_id, name, connection, tools, raw_prefix, dict(raw_guardrail_profiles), aliases


def _make_import_tool_builder(tool_id: str, import_path: str) -> Callable[..., Any]:
    def builder(**params: Any) -> Any:
        if params:
            raise ToolRegistryError(f"Internal tool '{tool_id}' is an imported object and does not accept params.")
        return _load_imported_value(import_path)

    return builder


def _make_factory_tool_builder(
    tool_id: str,
    factory_path: str,
    default_params: dict[str, Any],
) -> Callable[..., Any]:
    def builder(**params: Any) -> Any:
        factory = _load_imported_builder(factory_path)
        merged_params = {**default_params, **params}
        return factory(**_expand_internal_tool_params(merged_params, tool_id))

    return builder


def _load_imported_builder(import_path: str) -> Callable[..., Any]:
    builder = _load_imported_value(import_path)
    if not callable(builder):
        raise ToolRegistryError(f"Imported tool builder '{import_path}' is not callable.")
    return builder


def _load_imported_value(import_path: str) -> Any:
    module_path, sep, attr_name = import_path.partition(":")
    if not sep or not module_path.strip() or not attr_name.strip():
        raise ToolRegistryError("Importable tool bundle must use 'module:function' format.")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ToolRegistryError(f"Could not import tool module '{module_path}'.") from exc
    value = getattr(module, attr_name, None)
    if value is None:
        raise ToolRegistryError(f"Imported tool attribute '{import_path}' does not exist.")
    return value


def _collect_env_placeholders(value: Any) -> list[str]:
    if isinstance(value, str):
        return [match.group(1) for match in _BRACED_ENV_VAR_RE.finditer(value)]
    if isinstance(value, dict):
        found: list[str] = []
        for nested in value.values():
            found.extend(_collect_env_placeholders(nested))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for nested in value:
            found.extend(_collect_env_placeholders(nested))
        return found
    return []


def _expand_internal_tool_params(value: Any, tool_id: str) -> Any:
    if isinstance(value, str):
        return _BRACED_ENV_VAR_RE.sub(
            lambda match: _resolve_tool_env_reference(match.group(1), tool_id),
            value,
        )
    if isinstance(value, dict):
        return {
            key: _expand_internal_tool_params(nested, tool_id)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_internal_tool_params(nested, tool_id)
            for nested in value
        ]
    if isinstance(value, tuple):
        return tuple(_expand_internal_tool_params(nested, tool_id) for nested in value)
    return value


def _expand_mcp_connection_env(value: Any, server_name: str) -> Any:
    if isinstance(value, str):
        return _BRACED_ENV_VAR_RE.sub(
            lambda match: _resolve_env_reference(match.group(1), server_name),
            value,
        )
    if isinstance(value, dict):
        return {
            key: _expand_mcp_connection_env(nested, server_name)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_mcp_connection_env(nested, server_name)
            for nested in value
        ]
    if isinstance(value, tuple):
        return tuple(_expand_mcp_connection_env(nested, server_name) for nested in value)
    return value


def _resolve_env_reference(variable_name: str, server_name: str) -> str:
    resolved = os.environ.get(variable_name)
    if not str(resolved or "").strip():
        raise ToolRegistryError(
            f"MCP server '{server_name}' requires environment variable '{variable_name}'."
        )
    return resolved


def _resolve_tool_env_reference(variable_name: str, tool_id: str) -> str:
    resolved = os.environ.get(variable_name)
    if not str(resolved or "").strip():
        raise ToolRegistryError(
            f"Internal tool '{tool_id}' requires environment variable '{variable_name}'."
        )
    return resolved


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolRegistryError(f"Tool config field '{field_name}' must be a non-empty string.")
    return value.strip()


default_tool_registry = build_default_tool_registry()


__all__ = [
    "AgentToolsConfig",
    "BuiltAgentTools",
    "DEFAULT_TOOL_REGISTRY_CONFIG_PATH",
    "InternalToolSpec",
    "MCPConfig",
    "MCPServerSpec",
    "MCPServerTemplate",
    "PlatformToolRegistry",
    "ToolRegistryError",
    "build_agent_tool_bundle",
    "build_agent_tools",
    "build_internal_tools",
    "build_tool_registry_from_config",
    "default_tool_registry",
    "list_available_internal_tools",
    "load_tool_registry_config",
    "load_mcp_tools_from_config",
    "parse_agent_tools_config",
    "required_environment_variables",
    "select_mcp_tools",
]
