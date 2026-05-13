from __future__ import annotations

import importlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_mcp_adapters.tools import load_mcp_tools


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
class MCPServerSpec:
    name: str
    connection: dict[str, Any]
    tools: tuple[str, ...] | None = None
    tool_name_prefix: bool | None = None


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


@dataclass(frozen=True)
class AgentToolsConfig:
    internal_tools: tuple[InternalToolSpec, ...] = ()
    mcp: MCPConfig = MCPConfig(tool_name_prefix=True, servers=())

    @property
    def configured(self) -> bool:
        return bool(self.internal_tools or self.mcp.servers)


class PlatformToolRegistry:
    """Central registry for named internal tools and MCP server templates."""

    def __init__(self) -> None:
        self._internal_builders: dict[str, Callable[..., Any]] = {}
        self._mcp_servers: dict[str, MCPServerTemplate] = {}

    def register_internal_tool(self, name: str, builder: Callable[..., Any]) -> None:
        tool_name = _require_str(name, "internal tool name")
        if tool_name in self._internal_builders:
            raise ToolRegistryError(f"Duplicate internal tool registration: {tool_name}")
        self._internal_builders[tool_name] = builder

    def register_mcp_server(
        self,
        template_id: str,
        *,
        name: str,
        connection: dict[str, Any],
        tools: tuple[str, ...] | list[str] | None = None,
        tool_name_prefix: bool | None = None,
        aliases: tuple[str, ...] | list[str] = (),
    ) -> None:
        resolved_id = _require_str(template_id, "mcp server id")
        template = MCPServerTemplate(
            id=resolved_id,
            name=_require_str(name, "mcp server name"),
            connection=dict(connection),
            tools=None if tools is None else tuple(_require_str(item, "mcp server tool") for item in tools),
            tool_name_prefix=tool_name_prefix,
        )
        self._register_mcp_template(resolved_id, template)
        for alias in aliases:
            self._register_mcp_template(_require_str(alias, "mcp server alias"), template)

    def list_internal_tools(self) -> list[str]:
        return sorted(self._internal_builders)

    def mcp_server_template(self, template_id: str) -> MCPServerTemplate | None:
        return self._mcp_servers.get(template_id)

    def build_internal_tools(self, specs: list[InternalToolSpec] | tuple[InternalToolSpec, ...]) -> list[Any]:
        tools: list[Any] = []
        for spec in specs:
            if spec.import_path is not None:
                builder = _load_imported_builder(spec.import_path)
            else:
                if spec.name is None:
                    raise ToolRegistryError("Internal tool spec must define either 'name' or 'import'.")
                builder = self._internal_builders.get(spec.name)
                if builder is None:
                    available = ", ".join(self.list_internal_tools())
                    raise ToolRegistryError(
                        f"Unknown internal tool '{spec.name}'. Available tools: {available}"
                    )

            built = builder(**spec.params)
            if isinstance(built, (list, tuple)):
                tools.extend(built)
            else:
                tools.append(built)
        return tools

    async def load_mcp_tools_from_config(self, mcp_config: MCPConfig) -> list[Any]:
        tools: list[Any] = []
        for server in mcp_config.servers:
            use_prefix = (
                server.tool_name_prefix
                if server.tool_name_prefix is not None
                else mcp_config.tool_name_prefix
            )
            resolved_connection = _expand_mcp_connection_env(server.connection, server.name)
            server_tools = await load_mcp_tools(
                None,
                connection=resolved_connection,
                server_name=server.name,
                tool_name_prefix=use_prefix,
            )
            tools.extend(select_mcp_tools(server, server_tools))
        return tools

    async def build_tools(self, config: AgentToolsConfig) -> list[Any]:
        if not config.configured:
            return []
        internal_tools = self.build_internal_tools(config.internal_tools)
        mcp_tools = await self.load_mcp_tools_from_config(config.mcp)
        return [*internal_tools, *mcp_tools]

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
            ),
            global_prefix,
        )


def load_tool_registry_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = DEFAULT_TOOL_REGISTRY_CONFIG_PATH if path is None else Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ToolRegistryError("Tool registry config must be a JSON object.")
    return data


def build_tool_registry_from_config(config: dict[str, Any]) -> PlatformToolRegistry:
    registry = PlatformToolRegistry()

    raw_internal_tools = config.get("internal_tools") or []
    if not isinstance(raw_internal_tools, list):
        raise ToolRegistryError("Tool registry config field 'internal_tools' must be a list.")
    for index, entry in enumerate(raw_internal_tools):
        tool_id, builder = _parse_registry_internal_tool(entry, index=index)
        registry.register_internal_tool(tool_id, builder)

    raw_mcp_servers = config.get("mcp_servers")
    if raw_mcp_servers is None:
        raw_mcp_servers = (config.get("mcp") or {}).get("servers", [])
    if not isinstance(raw_mcp_servers, list):
        raise ToolRegistryError("Tool registry config field 'mcp_servers' must be a list.")
    for index, entry in enumerate(raw_mcp_servers):
        template_id, name, connection, tools, tool_name_prefix, aliases = _parse_registry_mcp_server(entry, index=index)
        registry.register_mcp_server(
            template_id,
            name=name,
            connection=connection,
            tools=tools,
            tool_name_prefix=tool_name_prefix,
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
) -> tuple[str, Callable[..., Any]]:
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

    unexpected = set(entry) - {"id", "name", "import", "factory", "params", "description"}
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ToolRegistryError(f"Unsupported internal_tools[{index}] fields: {names}")
    if import_ref is not None:
        if params:
            raise ToolRegistryError(f"internal_tools[{index}] cannot define params with 'import'. Use 'factory'.")
        return tool_id, _make_import_tool_builder(tool_id, _require_str(import_ref, f"internal_tools[{index}].import"))
    if factory_ref is not None:
        return tool_id, _make_factory_tool_builder(
            tool_id,
            _require_str(factory_ref, f"internal_tools[{index}].factory"),
            params,
        )
    raise ToolRegistryError(f"internal_tools[{index}] is missing an importable tool definition.")


def _parse_registry_mcp_server(
    entry: Any,
    *,
    index: int,
) -> tuple[str, str, dict[str, Any], tuple[str, ...] | None, bool | None, tuple[str, ...]]:
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

    raw_connection = entry.get("connection")
    if raw_connection is not None:
        if not isinstance(raw_connection, dict):
            raise ToolRegistryError(f"Tool registry config field 'mcp_servers[{index}].connection' must be an object.")
        connection = dict(raw_connection)
    else:
        connection = {
            key: value
            for key, value in entry.items()
            if key not in {"id", "server", "name", "aliases", "tools", "tool_name_prefix", "description"}
        }

    transport = _require_str(connection.get("transport"), f"mcp_servers[{index}].transport")
    if transport not in _ALLOWED_MCP_TRANSPORTS:
        allowed = ", ".join(sorted(_ALLOWED_MCP_TRANSPORTS))
        raise ToolRegistryError(
            f"Unsupported MCP transport '{transport}' for server '{name}'. "
            f"Allowed transports: {allowed}"
        )
    return template_id, name, connection, tools, raw_prefix, aliases


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
    "DEFAULT_TOOL_REGISTRY_CONFIG_PATH",
    "InternalToolSpec",
    "MCPConfig",
    "MCPServerSpec",
    "MCPServerTemplate",
    "PlatformToolRegistry",
    "ToolRegistryError",
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
