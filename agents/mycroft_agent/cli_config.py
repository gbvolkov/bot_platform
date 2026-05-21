from __future__ import annotations

import json
import os
import re
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_mcp_adapters.tools import load_mcp_tools

MYCROFT_CONFIG_ROOT_ENV = "MYCROFT_CONFIG_ROOT"
DEFAULT_MYCROFT_CONFIG_ROOT = Path("data/config/mycroft")
DEFAULT_CLI_CONFIG_NAME = "gaz_config.json"
DEFAULT_CLI_CONFIG_PATH = DEFAULT_MYCROFT_CONFIG_ROOT / DEFAULT_CLI_CONFIG_NAME
_BRACED_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")
_ALLOWED_MCP_TRANSPORTS = {
    "stdio",
    "sse",
    "http",
    "streamable_http",
    "streamable-http",
    "websocket",
}
_ALLOWED_INTERRUPT_DECISIONS = {"approve", "edit", "reject"}
_OPENAI_PROVIDER_VALUES = {
    "openai",
    "openai_4",
    "openai_pers",
    "openai_think",
}


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
class DeepAgentsConfig:
    interrupt_on: dict[str, bool | dict[str, Any]]


@dataclass(frozen=True)
class SubagentsConfig:
    stateless: tuple[str, ...]
    stateful: tuple[str, ...]


@dataclass(frozen=True)
class SkillsConfig:
    paths: tuple[str, ...]


@dataclass(frozen=True)
class MycroftCliConfig:
    system_prompt: str
    skills: SkillsConfig
    subagents: SubagentsConfig
    internal_tools: tuple[InternalToolSpec, ...]
    mcp: MCPConfig
    deepagents: DeepAgentsConfig


def resolve_cli_config_path(raw_path: str | Path | None = None) -> Path:
    if raw_path is None:
        return default_cli_config_path()
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def mycroft_config_root() -> Path:
    raw_root = os.environ.get(MYCROFT_CONFIG_ROOT_ENV, "").strip()
    root = Path(raw_root) if raw_root else DEFAULT_MYCROFT_CONFIG_ROOT
    if root.is_absolute():
        return root
    return (Path.cwd() / root).resolve()


def default_cli_config_path() -> Path:
    return mycroft_config_root() / DEFAULT_CLI_CONFIG_NAME


def load_cli_config(raw_path: str | Path | None = None) -> MycroftCliConfig:
    path = resolve_cli_config_path(raw_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Mycroft CLI config must be a JSON object.")
    if "agents" in data:
        raise ValueError(
            "Mycroft CLI config field 'agents' is no longer supported. "
            "Use 'subagents.stateless' and 'subagents.stateful'."
        )

    system_prompt = _parse_system_prompt(data.get("system_prompt"))
    skills = _parse_skills(data.get("skills"))
    subagents = _parse_subagents(data.get("subagents"))
    internal_tools = _parse_internal_tools(data.get("internal_tools") or [])
    mcp = _parse_mcp_config(data.get("mcp") or {})
    deepagents = _parse_deepagents_config(data.get("deepagents") or {})
    return MycroftCliConfig(
        system_prompt=system_prompt,
        skills=skills,
        subagents=subagents,
        internal_tools=internal_tools,
        mcp=mcp,
        deepagents=deepagents,
    )


def list_available_internal_tools() -> list[str]:
    return sorted(_internal_tool_builders().keys())


def build_internal_tools(
    specs: list[InternalToolSpec] | tuple[InternalToolSpec, ...],
    *,
    builders: dict[str, Callable[..., Any]] | None = None,
) -> list[Any]:
    resolved_builders = builders or _internal_tool_builders()
    tools: list[Any] = []
    for spec in specs:
        if spec.import_path is not None:
            builder = _load_imported_builder(spec.import_path)
        else:
            if spec.name is None:
                raise ValueError("Internal tool spec must define either 'name' or 'import'.")
            builder = resolved_builders.get(spec.name)
            if builder is None:
                available = ", ".join(sorted(resolved_builders))
                raise ValueError(
                    f"Unknown internal tool '{spec.name}'. Available tools: {available}"
                )

        built = builder(**spec.params)
        if isinstance(built, (list, tuple)):
            tools.extend(built)
        else:
            tools.append(built)
    return tools


def _load_imported_builder(import_path: str) -> Callable[..., Any]:
    module_path, sep, attr_name = import_path.partition(":")
    if not sep or not module_path.strip() or not attr_name.strip():
        raise ValueError(
            "Importable internal tool bundle must use 'module:function' format."
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"Could not import internal tool module '{module_path}'.") from exc

    builder = getattr(module, attr_name, None)
    if not callable(builder):
        raise ValueError(
            f"Imported internal tool builder '{import_path}' is not callable."
        )
    return builder


async def load_mcp_tools_from_config(mcp_config: MCPConfig) -> list[Any]:
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


def required_environment_variables(
    cli_config: MycroftCliConfig,
    provider_name: str,
) -> tuple[str, ...]:
    required: list[str] = []
    if provider_name in _OPENAI_PROVIDER_VALUES:
        required.append("OPENAI_API_KEY")

    all_subagents = (*cli_config.subagents.stateless, *cli_config.subagents.stateful)
    if "web_search_agent" in all_subagents:
        required.extend(["YA_API_KEY", "YA_FOLDER_ID"])

    for server in cli_config.mcp.servers:
        required.extend(_collect_env_placeholders(server.connection))

    return tuple(sorted(dict.fromkeys(required)))


def validate_required_environment(
    cli_config: MycroftCliConfig,
    provider_name: str,
) -> None:
    missing = [
        name
        for name in required_environment_variables(cli_config, provider_name)
        if not str(os.environ.get(name, "")).strip()
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Mycroft CLI missing required environment variables: "
            f"{missing_text}"
        )


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
        raise ValueError(
            f"MCP server '{server.name}' does not expose requested tools: {missing_text}. "
            f"Available tools: {available}"
        )
    return selected


def _find_mcp_tool(server_name: str, requested_name: str, loaded_tools: list[Any]) -> Any | None:
    prefixed_name = f"{server_name}_{requested_name}"
    for tool in loaded_tools:
        if tool.name in {requested_name, prefixed_name}:
            return tool
    return None


def _parse_subagents(value: Any) -> SubagentsConfig:
    if not isinstance(value, dict):
        raise ValueError("Mycroft CLI config field 'subagents' must be an object.")

    stateless_value = value.get("stateless", [])
    stateful_value = value.get("stateful", [])
    if not isinstance(stateless_value, list):
        raise ValueError("Mycroft CLI config field 'subagents.stateless' must be a list.")
    if not isinstance(stateful_value, list):
        raise ValueError("Mycroft CLI config field 'subagents.stateful' must be a list.")

    return SubagentsConfig(
        stateless=tuple(
            _require_str(item, "subagents.stateless[]") for item in stateless_value
        ),
        stateful=tuple(
            _require_str(item, "subagents.stateful[]") for item in stateful_value
        ),
    )


def _parse_skills(value: Any) -> SkillsConfig:
    if value is None:
        return SkillsConfig(paths=())
    if not isinstance(value, dict):
        raise ValueError("Mycroft CLI config field 'skills' must be an object.")

    raw_paths = value.get("paths", [])
    if not isinstance(raw_paths, list):
        raise ValueError("Mycroft CLI config field 'skills.paths' must be a list.")

    return SkillsConfig(paths=tuple(_require_str(item, "skills.paths[]") for item in raw_paths))


def _parse_system_prompt(value: Any) -> str:
    if isinstance(value, str):
        return _require_str(value, "system_prompt")
    if not isinstance(value, dict):
        raise ValueError(
            "Mycroft CLI config field 'system_prompt' must be a string or an object."
        )

    prompt_type = _require_str(value.get("type"), "system_prompt.type")
    if prompt_type == "file":
        prompt_path = _require_str(value.get("path"), "system_prompt.path")
        resolved_prompt_path = Path(prompt_path)
        if not resolved_prompt_path.is_absolute():
            resolved_prompt_path = (Path.cwd() / resolved_prompt_path).resolve()
        if not resolved_prompt_path.is_file():
            raise FileNotFoundError(
                f"Mycroft CLI system prompt file not found: {resolved_prompt_path}"
            )
        return resolved_prompt_path.read_text(encoding="utf-8")

    raise ValueError(
        "Unsupported Mycroft CLI system prompt type "
        f"'{prompt_type}'. Supported values: file"
    )


def _parse_internal_tools(value: Any) -> tuple[InternalToolSpec, ...]:
    if not isinstance(value, list):
        raise ValueError("Mycroft CLI config field 'internal_tools' must be a list.")

    parsed: list[InternalToolSpec] = []
    for entry in value:
        if isinstance(entry, str):
            parsed.append(InternalToolSpec(name=_require_str(entry, "internal_tools[]"), params={}))
            continue
        if not isinstance(entry, dict):
            raise ValueError("Each internal_tools entry must be a string or an object.")
        has_name = "name" in entry
        has_import = "import" in entry
        if has_name == has_import:
            raise ValueError("Each internal_tools object must contain exactly one of 'name' or 'import'.")

        if has_import:
            import_path = _require_str(entry.get("import"), "internal_tools[].import")
            params = entry.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("Mycroft CLI config field 'internal_tools[].params' must be an object.")
            unexpected = set(entry) - {"import", "params"}
            if unexpected:
                names = ", ".join(sorted(unexpected))
                raise ValueError(f"Unsupported internal_tools import fields: {names}")
            parsed.append(InternalToolSpec(import_path=import_path, params=dict(params)))
            continue

        name = _require_str(entry.get("name"), "internal_tools[].name")
        params = {key: entry[key] for key in entry if key != "name"}
        parsed.append(InternalToolSpec(name=name, params=params))
    return tuple(parsed)


def _parse_mcp_config(value: Any) -> MCPConfig:
    if not isinstance(value, dict):
        raise ValueError("Mycroft CLI config field 'mcp' must be an object.")

    tool_name_prefix = _require_bool(value.get("tool_name_prefix", True), "mcp.tool_name_prefix")
    raw_servers = value.get("servers") or []
    if not isinstance(raw_servers, list):
        raise ValueError("Mycroft CLI config field 'mcp.servers' must be a list.")

    servers = tuple(_parse_mcp_server(entry) for entry in raw_servers)
    return MCPConfig(tool_name_prefix=tool_name_prefix, servers=servers)


def _parse_deepagents_config(value: Any) -> DeepAgentsConfig:
    if not isinstance(value, dict):
        raise ValueError("Mycroft CLI config field 'deepagents' must be an object.")

    raw_interrupt_on = value.get("interrupt_on") or {}
    if not isinstance(raw_interrupt_on, dict):
        raise ValueError("Mycroft CLI config field 'deepagents.interrupt_on' must be an object.")

    interrupt_on: dict[str, bool | dict[str, Any]] = {}
    for raw_tool_name, raw_tool_config in raw_interrupt_on.items():
        tool_name = _require_str(raw_tool_name, "deepagents.interrupt_on.<tool_name>")
        if isinstance(raw_tool_config, bool):
            interrupt_on[tool_name] = raw_tool_config
            continue
        if not isinstance(raw_tool_config, dict):
            raise ValueError(
                f"Mycroft CLI config field 'deepagents.interrupt_on.{tool_name}' must be a boolean or an object."
            )

        allowed_decisions = raw_tool_config.get("allowed_decisions")
        if not isinstance(allowed_decisions, list) or not allowed_decisions:
            raise ValueError(
                f"Mycroft CLI config field 'deepagents.interrupt_on.{tool_name}.allowed_decisions' "
                "must be a non-empty list."
            )

        parsed_decisions = []
        for index, decision in enumerate(allowed_decisions):
            resolved = _require_str(
                decision,
                f"deepagents.interrupt_on.{tool_name}.allowed_decisions[{index}]",
            )
            if resolved not in _ALLOWED_INTERRUPT_DECISIONS:
                allowed = ", ".join(sorted(_ALLOWED_INTERRUPT_DECISIONS))
                raise ValueError(
                    f"Unsupported interrupt decision '{resolved}' for tool '{tool_name}'. "
                    f"Allowed values: {allowed}"
                )
            parsed_decisions.append(resolved)

        parsed_config: dict[str, Any] = {"allowed_decisions": parsed_decisions}

        description = raw_tool_config.get("description")
        if description is not None:
            parsed_config["description"] = _require_str(
                description,
                f"deepagents.interrupt_on.{tool_name}.description",
            )

        args_schema = raw_tool_config.get("args_schema")
        if args_schema is not None:
            if not isinstance(args_schema, dict):
                raise ValueError(
                    f"Mycroft CLI config field 'deepagents.interrupt_on.{tool_name}.args_schema' "
                    "must be an object."
                )
            parsed_config["args_schema"] = args_schema

        interrupt_on[tool_name] = parsed_config

    return DeepAgentsConfig(interrupt_on=interrupt_on)


def _parse_mcp_server(entry: Any) -> MCPServerSpec:
    if not isinstance(entry, dict):
        raise ValueError("Each mcp.servers entry must be an object.")

    name = _require_str(entry.get("name"), "mcp.servers[].name")
    connection = {key: value for key, value in entry.items() if key not in {"name", "tools", "tool_name_prefix"}}
    transport = _require_str(connection.get("transport"), f"mcp.servers[{name}].transport")
    if transport not in _ALLOWED_MCP_TRANSPORTS:
        allowed = ", ".join(sorted(_ALLOWED_MCP_TRANSPORTS))
        raise ValueError(
            f"Unsupported MCP transport '{transport}' for server '{name}'. Allowed transports: {allowed}"
        )

    raw_tools = entry.get("tools")
    tools: tuple[str, ...] | None
    if raw_tools is None:
        tools = None
    else:
        if not isinstance(raw_tools, list):
            raise ValueError(f"MCP server '{name}' field 'tools' must be a list.")
        tools = tuple(_require_str(item, f"mcp.servers[{name}].tools[]") for item in raw_tools)

    raw_prefix = entry.get("tool_name_prefix")
    tool_name_prefix = None if raw_prefix is None else _require_bool(raw_prefix, f"mcp.servers[{name}].tool_name_prefix")
    return MCPServerSpec(
        name=name,
        connection=connection,
        tools=tools,
        tool_name_prefix=tool_name_prefix,
    )


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Mycroft CLI config field '{field_name}' must be a non-empty string.")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Mycroft CLI config field '{field_name}' must be a boolean.")
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
        raise ValueError(
            f"MCP server '{server_name}' requires environment variable '{variable_name}'."
        )
    return resolved


def _internal_tool_builders() -> dict[str, Callable[..., Any]]:
    return {
        "fetch_user_info": _build_fetch_user_info_tool,
        "lookup_term": _build_lookup_term_tool,
        "search_kb": _build_search_kb_tool,
        "search_tickets": _build_search_tickets_tool,
        "store_artifact_tool": _build_store_artifact_tool,
        "think_tool": _build_think_tool,
    }


def _build_fetch_user_info_tool() -> Any:
    from agents.user_info import fetch_user_info

    return fetch_user_info


def _build_lookup_term_tool() -> Any:
    from agents.retrievers.retriever import get_term_and_defition_tools

    return get_term_and_defition_tools()


def _build_search_kb_tool() -> Any:
    from agents.retrievers.retriever import get_search_tool

    return get_search_tool()


def _build_search_tickets_tool() -> Any:
    from agents.retrievers.retriever import get_tickets_search_tool

    return get_tickets_search_tool()


def _build_store_artifact_tool() -> Any:
    from langchain.tools import tool

    from agents.store_artifacts import store_chapters

    @tool("store_artifact_tool")
    def mycroft_store_artifact_tool(title: str, artifact: str) -> str:
        """
        Use the tool to store an artifact when the user explicitly asks to save or export it.

        Args:
            title: Artifact title.
            artifact: Artifact body in Markdown.

        Returns:
            A Markdown link to the saved file.
        """
        url = store_chapters([{"title": title, "body": artifact}], artifact)
        return f"[You can now download the file.]({url})"

    return mycroft_store_artifact_tool


def _build_think_tool() -> Any:
    from agents.tools.think import ThinkTool

    return ThinkTool()
