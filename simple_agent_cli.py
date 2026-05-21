from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from agents.simple_agent.agent import build_agent_graph
from agents.utils import ModelType, extract_text
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from platform_guardrails.config import resolve_guardrail_policy
from platform_guardrails.graph_compiler import PlatformGraphCompiler
from platform_guardrails.runtime import PlatformGuardrailRuntime
from platform_tools.registry import (
    AgentToolsConfig,
    BuiltAgentTools,
    InternalToolSpec,
    MCPConfig,
    build_agent_tool_bundle,
    parse_agent_tools_config,
)


EXIT_COMMANDS = {"exit", "/exit", "quit", "/quit"}
RESET_COMMANDS = {"reset", "/reset"}
HELP_COMMANDS = {"help", "/help"}
MULTILINE_START_COMMANDS = {"/multi", "/multiline", "<<<"}
MULTILINE_END_COMMANDS = {"/send", "/end", ">>>"}
MULTILINE_CANCEL_COMMANDS = {"/cancel", "/abort"}
DEFAULT_PROMPTS_DIR = Path("./prompts")
SIMPLE_AGENT_ID = "simple_agent"
DEFAULT_AGENT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "data" / "config" / "bot_service" / "load.json"
)


@dataclass(frozen=True)
class SimpleAgentRegistrySettings:
    tools_config: AgentToolsConfig
    guardrail_policy_id: str | None = None
    guardrail_mode: Literal["none", "platform", "agent"] = "none"


def _parse_provider(raw_value: str) -> ModelType:
    normalized = raw_value.strip().lower()
    for provider in ModelType:
        if normalized in {provider.value.lower(), provider.name.lower()}:
            return provider
    available = ", ".join(sorted(provider.value for provider in ModelType))
    raise ValueError(f"Unknown provider '{raw_value}'. Available values: {available}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal interactive CLI for agents.simple_agent."
    )
    system_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--provider",
        default=ModelType.GPT.value,
        help="Model provider to use for simple_agent.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional thread id. If omitted, a random one is generated.",
    )
    parser.add_argument(
        "--agent-config",
        default=str(DEFAULT_AGENT_CONFIG_PATH),
        help="Agent load config used to resolve simple_agent guardrails and configured tools.",
    )
    system_group.add_argument(
        "--system",
        default=None,
        help="Optional system prompt passed through runtime context.",
    )
    system_group.add_argument(
        "--system-file",
        default="./prompts/webass.txt",
        help="Read system prompt from a file. Relative paths are resolved under ./prompts.",
    )
    parser.add_argument(
        "--enable-think",
        action="store_true",
        help="Enable ThinkTool through the platform tool factory.",
    )
    parser.add_argument(
        "--enable-web-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Yandex web search through the platform tool factory.",
    )
    parser.add_argument(
        "--web-search-max-results",
        type=int,
        default=3,
        help="Maximum Yandex search results when --enable-web-search is active.",
    )
    parser.add_argument(
        "--allow-external-tool-access",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow tools with external_access guardrail profiles. Defaults to true when such tools are enabled.",
    )
    parser.add_argument(
        "--guardrail-verbose-logging",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include confirmed prompt-injection text in guardrail logs. Defaults to registry config.",
    )
    return parser.parse_args()


def _new_config(
    thread_id: Optional[str] = None,
    *,
    allow_external_tool_access: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved_thread_id = thread_id or f"simple-agent-cli-{uuid.uuid4().hex}"
    configurable: dict[str, Any] = {"thread_id": resolved_thread_id}
    if allow_external_tool_access is not None:
        configurable["allow_external_tool_access"] = allow_external_tool_access
    return resolved_thread_id, {"configurable": configurable}


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Agent config field '{field_name}' must be a non-empty string.")
    return value.strip()


def _parse_guardrails_config(
    value: Any,
    *,
    agent_id: str,
) -> tuple[str | None, Literal["none", "platform", "agent"]]:
    if value is None:
        return None, "none"
    if not isinstance(value, dict):
        raise ValueError(f"Agent '{agent_id}' guardrails must be a mapping.")
    policy = value.get("policy")
    if policy is not None:
        policy = _require_str(policy, "guardrails.policy")
    mode = str(value.get("mode") or ("platform" if policy else "none")).strip().lower()
    if mode not in {"none", "platform", "agent"}:
        raise ValueError(
            f"Agent '{agent_id}' guardrails.mode must be one of: none, platform, agent."
        )
    if mode != "none" and not policy:
        raise ValueError(
            f"Agent '{agent_id}' guardrails.policy is required when guardrails are enabled."
        )
    return policy, mode  # type: ignore[return-value]


def _load_agent_registry_settings(
    raw_path: str | Path,
    *,
    agent_id: str = SIMPLE_AGENT_ID,
) -> SimpleAgentRegistrySettings:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Agent config must be a JSON object: {path}")

    entries = data.get("agents") or data.get("modules")
    if not isinstance(entries, list):
        raise ValueError(f"Agent config must contain an agents list: {path}")

    agent_entry = None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == agent_id:
            agent_entry = entry
            break
    if agent_entry is None:
        raise ValueError(f"Agent config '{path}' does not define agent '{agent_id}'.")

    params = agent_entry.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"Agent '{agent_id}' params must be an object.")

    policy_id, guardrail_mode = _parse_guardrails_config(
        agent_entry.get("guardrails"),
        agent_id=agent_id,
    )
    legacy_policy_id = params.get("guardrail_policy")
    if legacy_policy_id is not None and policy_id is not None:
        raise ValueError(
            f"Agent '{agent_id}' cannot combine top-level guardrails with params.guardrail_policy."
        )
    if legacy_policy_id is not None:
        policy_id = _require_str(legacy_policy_id, "params.guardrail_policy")
        guardrail_mode = "platform"

    return SimpleAgentRegistrySettings(
        tools_config=parse_agent_tools_config(agent_entry.get("tools"), agent_id=agent_id),
        guardrail_policy_id=policy_id,
        guardrail_mode=guardrail_mode,
    )


def _build_cli_tools_config(args: argparse.Namespace) -> AgentToolsConfig:
    internal_tools: list[InternalToolSpec] = []

    if args.enable_think:
        internal_tools.append(InternalToolSpec(name="think_tool"))

    if args.enable_web_search:
        internal_tools.append(
            InternalToolSpec(
                name="web_search_tool",
                params={
                    "max_results": max(1, int(args.web_search_max_results)),
                    "summarize": True,
                },
            )
        )

    return AgentToolsConfig(internal_tools=tuple(internal_tools))


def _merge_tools_config(*configs: AgentToolsConfig) -> AgentToolsConfig:
    internal_tools_by_key: dict[tuple[str | None, str | None], InternalToolSpec] = {}
    mcp_servers = []
    mcp_tool_name_prefix = True

    for tools_config in configs:
        for spec in tools_config.internal_tools:
            internal_tools_by_key[(spec.name, spec.import_path)] = spec
        if tools_config.mcp.servers:
            mcp_tool_name_prefix = tools_config.mcp.tool_name_prefix
            mcp_servers.extend(tools_config.mcp.servers)

    return AgentToolsConfig(
        internal_tools=tuple(internal_tools_by_key.values()),
        mcp=MCPConfig(
            tool_name_prefix=mcp_tool_name_prefix,
            servers=tuple(mcp_servers),
        ),
    )


async def _build_registry_tools(
    tools_config: AgentToolsConfig,
    *,
    require_guardrail_profiles: bool = False,
) -> BuiltAgentTools:
    return await build_agent_tool_bundle(
        tools_config,
        require_guardrail_profiles=require_guardrail_profiles,
    )


def _build_guardrail_runtime(
    settings: SimpleAgentRegistrySettings,
    *,
    verbose_logging: bool | None = None,
) -> PlatformGuardrailRuntime:
    if settings.guardrail_mode == "agent":
        raise ValueError(
            "simple_agent_cli supports guardrails.mode 'platform' or 'none'; "
            "agent-local guardrails are not used for the Stage 1 graph path."
        )
    if settings.guardrail_mode != "platform" or not settings.guardrail_policy_id:
        return PlatformGuardrailRuntime.disabled(agent_id=SIMPLE_AGENT_ID)
    if verbose_logging is None:
        return PlatformGuardrailRuntime.from_policy_id(
            settings.guardrail_policy_id,
            agent_id=SIMPLE_AGENT_ID,
        )
    policy_kwargs = resolve_guardrail_policy(settings.guardrail_policy_id)
    policy_kwargs["guardrail_verbose_logging"] = verbose_logging
    return PlatformGuardrailRuntime.from_policy_kwargs(
        policy_kwargs,
        agent_id=SIMPLE_AGENT_ID,
        policy_id=settings.guardrail_policy_id,
    )


def _tool_bundle_has_external_access(tool_bundle: BuiltAgentTools) -> bool:
    for profile in tool_bundle.guardrail_profiles.values():
        if isinstance(profile, dict) and profile.get("category") == "external_access":
            return True
    return False


def _compile_platform_graph(
    *,
    provider: ModelType,
    guardrail_runtime: PlatformGuardrailRuntime,
    tool_bundle: BuiltAgentTools,
) -> Any:
    spec = build_agent_graph(provider=provider, streaming=False)
    return PlatformGraphCompiler().compile(
        spec,
        guardrail_runtime=guardrail_runtime,
        checkpointer=MemorySaver(),
        tools=tool_bundle.tools,
        tool_profiles=tool_bundle.guardrail_profiles,
    )


def _resolve_system_prompt_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        prompt_candidate = DEFAULT_PROMPTS_DIR / candidate
        if prompt_candidate.is_file():
            return prompt_candidate
    raise FileNotFoundError(
        f"System prompt file not found: {raw_path}. "
        f"Checked '{candidate}' and '{DEFAULT_PROMPTS_DIR / candidate}'."
    )


def _load_system_prompt(args: argparse.Namespace) -> tuple[Optional[str], Optional[Path]]:
    if args.system is not None:
        return args.system, None
    if not args.system_file:
        return None, None

    prompt_path = _resolve_system_prompt_path(args.system_file)
    return prompt_path.read_text(encoding="utf-8-sig"), prompt_path


def _build_context(*, system_prompt: Optional[str]) -> dict[str, Any] | None:
    context: dict[str, Any] = {}
    if system_prompt:
        context["system_prompt"] = system_prompt
    return context or None


def _extract_last_ai_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return extract_text(message)
    return ""


def _invoke_turn(
    graph: Any,
    *,
    user_text: Optional[str],
    config: dict[str, Any],
    context: dict[str, Any] | None,
) -> str:
    messages = [HumanMessage(content=user_text)] if user_text is not None else []
    result = graph.invoke({"messages": messages}, config=config, context=context)
    if not isinstance(result, dict):
        return ""
    return _extract_last_ai_text(result)


def _print_help() -> None:
    print("Commands:")
    print("  /help   show commands")
    print("  /multi  enter multiline mode")
    print("  /reset  start a fresh thread")
    print("  /exit   leave the CLI")


def _collect_multiline_input() -> Optional[str]:
    print("Multiline mode: finish with /send, cancel with /cancel.")
    lines: list[str] = []
    while True:
        line = input("...> ")
        command = line.strip().lower()
        if command in EXIT_COMMANDS:
            return None
        if command in MULTILINE_CANCEL_COMMANDS:
            return ""
        if command in MULTILINE_END_COMMANDS:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> int:
    args = _parse_args()

    try:
        provider = _parse_provider(args.provider)
        system_prompt, system_file_path = _load_system_prompt(args)
        registry_settings = _load_agent_registry_settings(args.agent_config)
        guardrail_runtime = _build_guardrail_runtime(
            registry_settings,
            verbose_logging=args.guardrail_verbose_logging,
        )
        tools_config = _merge_tools_config(
            registry_settings.tools_config,
            _build_cli_tools_config(args),
        )
        tool_bundle = asyncio.run(
            _build_registry_tools(
                tools_config,
                require_guardrail_profiles=guardrail_runtime.tool_execution_enabled,
            )
        )
        graph = _compile_platform_graph(
            provider=provider,
            guardrail_runtime=guardrail_runtime,
            tool_bundle=tool_bundle,
        )
    except Exception as exc:
        print(f"Failed to start CLI: {exc}", file=sys.stderr)
        return 1

    allow_external_tool_access = args.allow_external_tool_access
    if allow_external_tool_access is None:
        allow_external_tool_access = _tool_bundle_has_external_access(tool_bundle)
    thread_id, run_config = _new_config(
        args.thread_id,
        allow_external_tool_access=allow_external_tool_access,
    )
    context = _build_context(system_prompt=system_prompt)

    print(f"simple_agent CLI started (thread_id={thread_id}, provider={provider.value}).")
    print(
        "Guardrails: "
        f"{registry_settings.guardrail_mode}"
        + (
            f" ({registry_settings.guardrail_policy_id})"
            if registry_settings.guardrail_policy_id
            else ""
        )
    )
    print(
        "Active tools: "
        + (
            ", ".join(getattr(tool, "name", tool.__class__.__name__) for tool in tool_bundle.tools)
            if tool_bundle.tools
            else "none"
        )
    )
    print(f"External tool access allowed: {allow_external_tool_access}")

    greeting = _invoke_turn(graph, user_text=None, config=run_config, context=context)
    if greeting:
        print(f"\nAssistant: {greeting}\n")

    if system_file_path is not None:
        print(f"System prompt was loaded from file: {system_file_path}")
    elif system_prompt:
        print("System prompt was provided through runtime context.")
    else:
        print("No system prompt was provided. Your next message will be used as the system prompt.\n")

    _print_help()
    print("")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if not user_input:
            continue

        user_input_lower = user_input.lower()
        if user_input_lower in EXIT_COMMANDS:
            print("Goodbye!")
            return 0
        if user_input_lower in HELP_COMMANDS:
            _print_help()
            print("")
            continue
        if user_input_lower in MULTILINE_START_COMMANDS:
            multiline_input = _collect_multiline_input()
            if multiline_input is None:
                print("Goodbye!")
                return 0
            if not multiline_input:
                print("Canceled.\n")
                continue
            user_input = multiline_input
            user_input_lower = user_input.lower()
        if user_input_lower in RESET_COMMANDS:
            thread_id, run_config = _new_config(
                allow_external_tool_access=allow_external_tool_access,
            )
            print(f"Started fresh thread: {thread_id}")
            greeting = _invoke_turn(graph, user_text=None, config=run_config, context=context)
            if greeting:
                print(f"\nAssistant: {greeting}\n")
            if system_file_path is not None:
                print(f"System prompt was loaded from file: {system_file_path}\n")
            elif system_prompt:
                print("System prompt was provided through runtime context.\n")
            else:
                print("No system prompt was provided. Your next message will be used as the system prompt.\n")
            continue

        response = _invoke_turn(graph, user_text=user_input, config=run_config, context=context)
        print(f"\nAssistant: {response or '(empty response)'}\n")


if __name__ == "__main__":
    raise SystemExit(main())
