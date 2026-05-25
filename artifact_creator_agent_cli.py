from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents.artifact_creator_agent.agent import initialize_agent
from agents.utils import ModelType, extract_text
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from platform_guardrails.config import inline_guardrail_config_keys, resolve_guardrail_policy
from platform_guardrails.url_policy import UrlPolicyConfig
from platform_tools.registry import AgentToolsConfig, BuiltAgentTools, build_agent_tool_bundle, parse_agent_tools_config


EXIT_COMMANDS = {"exit", "/exit", "quit", "/quit"}
RESET_COMMANDS = {"reset", "/reset"}
HELP_COMMANDS = {"help", "/help"}
MULTILINE_START_COMMANDS = {"/multi", "/multiline", "<<<"}
MULTILINE_END_COMMANDS = {"/send", "/end", ">>>"}
MULTILINE_CANCEL_COMMANDS = {"/cancel", "/abort"}
SAVE_THREAD_COMMANDS = {"/save-thread", "/save"}
DEFAULT_PROMPTS_DIR = Path("./prompts")
DEFAULT_CLI_USER_ID = "artifact-creator-cli"
DEFAULT_CLI_USER_ROLE = "default"
DEFAULT_CLI_TENANT_ID = "cli"
ARTIFACT_AGENT_ID = "artifact_creator_agent"
DEFAULT_AGENT_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "config" / "bot_service" / "load.json"
ARTIFACT_CLI_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "artifact_creator_agent"
    / "cli_checkpoints.sqlite"
)


@dataclass(frozen=True)
class ArtifactAgentRegistrySettings:
    tools_config: AgentToolsConfig
    guardrails_locale: str = "ru-RU"
    guardrail_privacy_enabled: bool = False
    guardrail_scanners_enabled: bool = False
    guardrail_tool_execution_enabled: bool = False
    guardrail_scanner_failure_policy: str = "fail_closed"
    guardrail_banned_topics: list[str] | None = None
    guardrail_prompt_injection_model: str | dict[str, Any] | None = None
    guardrail_prompt_injection_model_revision: str | None = None
    guardrail_prompt_injection_threshold: float | None = None
    guardrail_tool_result_prompt_injection_threshold: float | None = None
    guardrail_url_policy: UrlPolicyConfig | None = None
    guardrail_scan_system_prompt: bool = True
    guardrail_verbose_logging: bool = False
    guardrail_composite_input_scanners: tuple[str, ...] | None = None
    guardrail_composite_recent_message_limit: int = 20
    guardrail_palimpsest_run_entities: list[str] | None = None
    guardrail_palimpsest_entity_replacements: Any | None = None
    guardrail_palimpsest_options: dict[str, Any] | None = None
    guardrail_palimpsest_session_options: dict[str, Any] | None = None


def _persistent_checkpoint_saver():
    ARTIFACT_CLI_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(str(ARTIFACT_CLI_CHECKPOINT_PATH))


def _parse_provider(raw_value: str) -> ModelType:
    normalized = raw_value.strip().lower()
    for provider in ModelType:
        if normalized in {provider.value.lower(), provider.name.lower()}:
            return provider
    available = ", ".join(sorted(provider.value for provider in ModelType))
    raise ValueError(f"Unknown provider '{raw_value}'. Available values: {available}")


def _parse_scanner_failure_policy(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    if normalized not in {"fail_closed", "fail_open"}:
        raise ValueError("Scanner failure policy must be fail_closed or fail_open.")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for agents.artifact_creator_agent."
    )
    system_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional prompt for a single non-interactive turn.",
    )
    parser.add_argument(
        "--provider",
        default=ModelType.GPT.value,
        help="Model provider to use for artifact_creator_agent.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional thread id. If omitted, a random one is generated.",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_CLI_USER_ID,
        help="User id passed through RunnableConfig.",
    )
    parser.add_argument(
        "--user-role",
        default=DEFAULT_CLI_USER_ROLE,
        help="User role passed through RunnableConfig.",
    )
    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_CLI_TENANT_ID,
        help="Tenant id passed through RunnableConfig for guardrail scope.",
    )
    parser.add_argument(
        "--locale",
        default="ru",
        help="Agent locale passed to initialize_agent.",
    )
    parser.add_argument(
        "--agent-config",
        default=str(DEFAULT_AGENT_CONFIG_PATH),
        help="Agent load config used to resolve artifact_creator_agent tools.",
    )
    system_group.add_argument(
        "--system",
        default=None,
        help="Optional fixed artifact system prompt.",
    )
    system_group.add_argument(
        "--system-file",
        default=None,
        help="Read the fixed artifact system prompt from a file.",
    )
    parser.add_argument(
        "--privacy-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Palimpsest privacy for user/model text. Defaults to registry config.",
    )
    parser.add_argument(
        "--guardrails-locale",
        default=None,
        help="Locale passed to the privacy guardrail rail.",
    )
    parser.add_argument(
        "--allow-external-tool-access",
        "--allow-external-search",
        dest="allow_external_tool_access",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow external tools such as web_search and maps_search_places in guardrail context.",
    )
    parser.add_argument(
        "--scanners-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable LLM Guard scanner enforcement. Defaults to registry config.",
    )
    parser.add_argument(
        "--tool-execution-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable tool execution guardrails. Defaults to registry config.",
    )
    parser.add_argument(
        "--scanner-failure-policy",
        default=None,
        help="Scanner failure mode: fail_closed or fail_open.",
    )
    parser.add_argument(
        "--banned-topic",
        action="append",
        default=None,
        help="Additional banned topic for the scanner profile. Can be repeated.",
    )
    parser.add_argument(
        "--guardrail-prompt-injection-model",
        "--prompt-injection-model",
        dest="guardrail_prompt_injection_model",
        default=None,
        help="Optional Hugging Face model path for LLM Guard PromptInjection.",
    )
    parser.add_argument(
        "--guardrail-prompt-injection-model-revision",
        "--prompt-injection-model-revision",
        dest="guardrail_prompt_injection_model_revision",
        default=None,
        help="Optional model revision for the PromptInjection model.",
    )
    parser.add_argument(
        "--guardrail-prompt-injection-threshold",
        "--prompt-injection-threshold",
        dest="guardrail_prompt_injection_threshold",
        type=float,
        default=None,
        help="Optional PromptInjection score threshold.",
    )
    parser.add_argument(
        "--guardrail-tool-result-prompt-injection-threshold",
        dest="guardrail_tool_result_prompt_injection_threshold",
        type=float,
        default=None,
        help="Optional PromptInjection score threshold for tool results.",
    )
    parser.add_argument(
        "--guardrail-verbose-logging",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include confirmed prompt-injection text in guardrail logs. Defaults to registry config.",
    )
    parser.add_argument(
        "--save-thread",
        default=None,
        help="Optional path to save the current conversation thread after the run.",
    )
    return parser.parse_args()


def _new_config(
    thread_id: Optional[str] = None,
    *,
    user_id: str = DEFAULT_CLI_USER_ID,
    user_role: str = DEFAULT_CLI_USER_ROLE,
    tenant_id: str = DEFAULT_CLI_TENANT_ID,
    allow_external_tool_access: bool | None = None,
    allow_external_search: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved_thread_id = thread_id or f"artifact-creator-cli-{uuid.uuid4().hex}"
    configurable: dict[str, Any] = {
        "thread_id": resolved_thread_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "user_role": user_role,
    }
    if allow_external_tool_access is None:
        allow_external_tool_access = allow_external_search
    if allow_external_tool_access is not None:
        configurable["allow_external_tool_access"] = allow_external_tool_access
    return resolved_thread_id, {
        "configurable": {
            **configurable,
        }
    }


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


def _load_agent_registry_settings(
    raw_path: str | Path,
    *,
    agent_id: str = ARTIFACT_AGENT_ID,
) -> ArtifactAgentRegistrySettings:
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
    params = dict(params)
    policy_id = params.pop("guardrail_policy", None)
    inline_keys = inline_guardrail_config_keys(params)
    if policy_id is None:
        if inline_keys:
            names = ", ".join(sorted(inline_keys))
            raise ValueError(
                f"Agent '{agent_id}' uses inline guardrail params without guardrail_policy: {names}."
            )
        guardrail_kwargs: dict[str, Any] = {}
    else:
        if inline_keys:
            names = ", ".join(sorted(inline_keys))
            raise ValueError(
                f"Agent '{agent_id}' mixes guardrail_policy with inline guardrail params: {names}."
            )
        guardrail_kwargs = resolve_guardrail_policy(policy_id)

    guardrails_locale = guardrail_kwargs.get("guardrails_locale", "ru-RU")
    if not isinstance(guardrails_locale, str):
        raise ValueError(f"Agent '{agent_id}' guardrails_locale must be a string.")
    privacy_enabled = guardrail_kwargs.get("guardrail_privacy_enabled", False)
    if not isinstance(privacy_enabled, bool):
        raise ValueError(f"Agent '{agent_id}' guardrail_privacy_enabled must be a boolean.")
    scanners_enabled = guardrail_kwargs.get("guardrail_scanners_enabled", False)
    if not isinstance(scanners_enabled, bool):
        raise ValueError(f"Agent '{agent_id}' guardrail_scanners_enabled must be a boolean.")
    tool_execution_enabled = guardrail_kwargs.get("guardrail_tool_execution_enabled", False)
    if not isinstance(tool_execution_enabled, bool):
        raise ValueError(f"Agent '{agent_id}' guardrail_tool_execution_enabled must be a boolean.")
    scanner_failure_policy = guardrail_kwargs.get("guardrail_scanner_failure_policy", "fail_closed")
    scanner_failure_policy = _parse_scanner_failure_policy(str(scanner_failure_policy))
    banned_topics = guardrail_kwargs.get("guardrail_banned_topics")
    if banned_topics is not None and not isinstance(banned_topics, list):
        raise ValueError(f"Agent '{agent_id}' guardrail_banned_topics must be a list.")
    prompt_injection_model = guardrail_kwargs.get("guardrail_prompt_injection_model")
    if prompt_injection_model is not None and not isinstance(prompt_injection_model, (str, dict)):
        raise ValueError(
            f"Agent '{agent_id}' guardrail_prompt_injection_model must be a string or object."
        )
    prompt_injection_model_revision = guardrail_kwargs.get("guardrail_prompt_injection_model_revision")
    if prompt_injection_model_revision is not None and not isinstance(prompt_injection_model_revision, str):
        raise ValueError(
            f"Agent '{agent_id}' guardrail_prompt_injection_model_revision must be a string."
        )
    prompt_injection_threshold = guardrail_kwargs.get("guardrail_prompt_injection_threshold")
    if prompt_injection_threshold is not None:
        prompt_injection_threshold = float(prompt_injection_threshold)
    tool_result_prompt_injection_threshold = guardrail_kwargs.get(
        "guardrail_tool_result_prompt_injection_threshold"
    )
    if tool_result_prompt_injection_threshold is not None:
        tool_result_prompt_injection_threshold = float(tool_result_prompt_injection_threshold)
    url_policy = guardrail_kwargs.get("guardrail_url_policy")
    if url_policy is not None and not isinstance(url_policy, UrlPolicyConfig):
        raise ValueError(f"Agent '{agent_id}' guardrail_url_policy must be a UrlPolicyConfig.")
    scan_system_prompt = guardrail_kwargs.get("guardrail_scan_system_prompt", True)
    if not isinstance(scan_system_prompt, bool):
        raise ValueError(f"Agent '{agent_id}' guardrail_scan_system_prompt must be a boolean.")
    verbose_logging = guardrail_kwargs.get("guardrail_verbose_logging", False)
    if not isinstance(verbose_logging, bool):
        raise ValueError(f"Agent '{agent_id}' guardrail_verbose_logging must be a boolean.")
    composite_input_scanners = guardrail_kwargs.get("guardrail_composite_input_scanners")
    if composite_input_scanners is not None and not isinstance(composite_input_scanners, (list, tuple)):
        raise ValueError(f"Agent '{agent_id}' guardrail_composite_input_scanners must be a list.")
    composite_recent_message_limit = int(guardrail_kwargs.get("guardrail_composite_recent_message_limit", 20))
    palimpsest_run_entities = guardrail_kwargs.get("guardrail_palimpsest_run_entities")
    if palimpsest_run_entities is not None and not isinstance(palimpsest_run_entities, list):
        raise ValueError(
            f"Agent '{agent_id}' guardrail_palimpsest_run_entities must be a list."
        )
    palimpsest_entity_replacements = guardrail_kwargs.get("guardrail_palimpsest_entity_replacements")
    if (
        palimpsest_entity_replacements is not None
        and not isinstance(palimpsest_entity_replacements, (dict, list))
    ):
        raise ValueError(
            f"Agent '{agent_id}' guardrail_palimpsest_entity_replacements must be an object or list."
        )
    palimpsest_options = guardrail_kwargs.get("guardrail_palimpsest_options")
    if palimpsest_options is not None and not isinstance(palimpsest_options, dict):
        raise ValueError(f"Agent '{agent_id}' guardrail_palimpsest_options must be an object.")
    palimpsest_session_options = guardrail_kwargs.get("guardrail_palimpsest_session_options")
    if palimpsest_session_options is not None and not isinstance(palimpsest_session_options, dict):
        raise ValueError(
            f"Agent '{agent_id}' guardrail_palimpsest_session_options must be an object."
        )

    return ArtifactAgentRegistrySettings(
        tools_config=parse_agent_tools_config(agent_entry.get("tools"), agent_id=agent_id),
        guardrails_locale=guardrails_locale,
        guardrail_privacy_enabled=privacy_enabled,
        guardrail_scanners_enabled=scanners_enabled,
        guardrail_tool_execution_enabled=tool_execution_enabled,
        guardrail_scanner_failure_policy=scanner_failure_policy,
        guardrail_banned_topics=list(banned_topics) if banned_topics is not None else None,
        guardrail_prompt_injection_model=prompt_injection_model,
        guardrail_prompt_injection_model_revision=prompt_injection_model_revision,
        guardrail_prompt_injection_threshold=prompt_injection_threshold,
        guardrail_tool_result_prompt_injection_threshold=tool_result_prompt_injection_threshold,
        guardrail_url_policy=url_policy,
        guardrail_scan_system_prompt=scan_system_prompt,
        guardrail_verbose_logging=verbose_logging,
        guardrail_composite_input_scanners=(
            tuple(str(item) for item in composite_input_scanners)
            if composite_input_scanners is not None
            else None
        ),
        guardrail_composite_recent_message_limit=composite_recent_message_limit,
        guardrail_palimpsest_run_entities=(
            list(palimpsest_run_entities) if palimpsest_run_entities is not None else None
        ),
        guardrail_palimpsest_entity_replacements=palimpsest_entity_replacements,
        guardrail_palimpsest_options=(
            dict(palimpsest_options) if palimpsest_options is not None else None
        ),
        guardrail_palimpsest_session_options=(
            dict(palimpsest_session_options) if palimpsest_session_options is not None else None
        ),
    )


async def _build_registry_tools(
    settings: ArtifactAgentRegistrySettings,
    *,
    require_guardrail_profiles: bool = False,
) -> BuiltAgentTools:
    return await build_agent_tool_bundle(
        settings.tools_config,
        require_guardrail_profiles=require_guardrail_profiles,
    )


def _build_human_message(user_text: str) -> HumanMessage:
    return HumanMessage(
        content=[{"type": "text", "text": user_text}],
        id=f"human-{uuid.uuid4().hex}",
    )


def _extract_last_ai_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return extract_text(message)
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
    return ""


async def _invoke_turn(
    agent: Any,
    *,
    user_text: Optional[str],
    config: dict[str, Any],
) -> str:
    messages = [_build_human_message(user_text)] if user_text is not None else []
    result = await agent.ainvoke({"messages": messages}, config=config)
    if not isinstance(result, dict):
        return ""
    return _extract_last_ai_text(result)


def _normalize_export_path(raw_path: str | None, *, thread_id: str) -> Path:
    if raw_path and raw_path.strip():
        candidate = Path(raw_path.strip()).expanduser()
        if not candidate.suffix:
            candidate = candidate.with_suffix(".md")
    else:
        candidate = Path.cwd() / f"{thread_id}.md"
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def _thread_export_payload(
    *,
    thread_id: str,
    provider: str,
    privacy_enabled: bool,
    scanners_enabled: bool,
    tool_execution_enabled: bool,
    turns: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "privacy_enabled": privacy_enabled,
        "scanners_enabled": scanners_enabled,
        "tool_execution_enabled": tool_execution_enabled,
        "turns": turns,
    }


def _render_thread_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Artifact Creator Conversation Thread",
        "",
        f"- Thread id: `{payload['thread_id']}`",
        f"- Saved at (UTC): `{payload['saved_at']}`",
        f"- Provider: `{payload['provider']}`",
        f"- Privacy enabled: `{payload['privacy_enabled']}`",
        f"- Scanners enabled: `{payload['scanners_enabled']}`",
        f"- Tool execution guardrails enabled: `{payload['tool_execution_enabled']}`",
        "",
    ]
    turns = payload.get("turns") or []
    if not turns:
        lines.extend(["_No conversation turns recorded._", ""])
        return "\n".join(lines)

    for index, turn in enumerate(turns, start=1):
        lines.extend(
            [
                f"## Turn {index}",
                "",
                "User:",
                "",
                "```text",
                str(turn.get("user", "")),
                "```",
                "",
                "Artifact Creator:",
                "",
                "```text",
                str(turn.get("assistant", "")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _save_thread_export(
    *,
    path: Path,
    thread_id: str,
    provider: str,
    privacy_enabled: bool,
    scanners_enabled: bool,
    tool_execution_enabled: bool,
    turns: list[dict[str, str]],
) -> Path:
    payload = _thread_export_payload(
        thread_id=thread_id,
        provider=provider,
        privacy_enabled=privacy_enabled,
        scanners_enabled=scanners_enabled,
        tool_execution_enabled=tool_execution_enabled,
        turns=turns,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(_render_thread_markdown(payload), encoding="utf-8")
    return path


def _parse_save_thread_command(user_input: str) -> str | None:
    stripped = user_input.strip()
    if not stripped:
        return None
    parts = stripped.split(maxsplit=1)
    if parts[0].lower() not in SAVE_THREAD_COMMANDS:
        return None
    if len(parts) == 1:
        return ""
    return parts[1].strip()


def _print_help() -> None:
    print("Commands:")
    print("  /help   show commands")
    print("  /multi  enter multiline mode")
    print("  /save-thread [path]  save the current thread to a file (.md by default, .json also supported)")
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

        with asyncio.Runner() as runner:
            checkpoint_cm = _persistent_checkpoint_saver()
            checkpoint_saver = runner.run(checkpoint_cm.__aenter__())
            try:
                runner.run(checkpoint_saver.setup())
                prompt_injection_model = (
                    args.guardrail_prompt_injection_model
                    if args.guardrail_prompt_injection_model is not None
                    else registry_settings.guardrail_prompt_injection_model
                )
                prompt_injection_model_revision = (
                    args.guardrail_prompt_injection_model_revision
                    if args.guardrail_prompt_injection_model_revision is not None
                    else registry_settings.guardrail_prompt_injection_model_revision
                )
                prompt_injection_threshold = (
                    args.guardrail_prompt_injection_threshold
                    if args.guardrail_prompt_injection_threshold is not None
                    else registry_settings.guardrail_prompt_injection_threshold
                )
                tool_result_prompt_injection_threshold = (
                    args.guardrail_tool_result_prompt_injection_threshold
                    if args.guardrail_tool_result_prompt_injection_threshold is not None
                    else registry_settings.guardrail_tool_result_prompt_injection_threshold
                )
                verbose_logging = (
                    args.guardrail_verbose_logging
                    if args.guardrail_verbose_logging is not None
                    else registry_settings.guardrail_verbose_logging
                )
                privacy_enabled = (
                    args.privacy_enabled
                    if args.privacy_enabled is not None
                    else registry_settings.guardrail_privacy_enabled
                )
                scanners_enabled = (
                    args.scanners_enabled
                    if args.scanners_enabled is not None
                    else registry_settings.guardrail_scanners_enabled
                )
                tool_execution_enabled = (
                    args.tool_execution_enabled
                    if args.tool_execution_enabled is not None
                    else registry_settings.guardrail_tool_execution_enabled
                )
                scanner_failure_policy = (
                    _parse_scanner_failure_policy(args.scanner_failure_policy)
                    if args.scanner_failure_policy is not None
                    else registry_settings.guardrail_scanner_failure_policy
                )
                guardrails_locale = (
                    args.guardrails_locale
                    if args.guardrails_locale is not None
                    else registry_settings.guardrails_locale
                )
                banned_topics = (
                    args.banned_topic
                    if args.banned_topic is not None
                    else registry_settings.guardrail_banned_topics
                )
                tool_bundle = runner.run(
                    _build_registry_tools(
                        registry_settings,
                        require_guardrail_profiles=tool_execution_enabled,
                    )
                )
                registry_tools = tool_bundle.tools
                agent = initialize_agent(
                    provider=provider,
                    locale=args.locale,
                    checkpoint_saver=checkpoint_saver,
                    tools=registry_tools,
                    system_prompt=system_prompt,
                    guardrails_locale=guardrails_locale,
                    guardrail_privacy_enabled=privacy_enabled,
                    guardrail_scanners_enabled=scanners_enabled,
                    guardrail_tool_execution_enabled=tool_execution_enabled,
                    guardrail_scanner_failure_policy=scanner_failure_policy,
                    guardrail_banned_topics=banned_topics,
                    guardrail_prompt_injection_model=prompt_injection_model,
                    guardrail_prompt_injection_model_revision=prompt_injection_model_revision,
                    guardrail_prompt_injection_threshold=prompt_injection_threshold,
                    guardrail_tool_result_prompt_injection_threshold=tool_result_prompt_injection_threshold,
                    guardrail_url_policy=registry_settings.guardrail_url_policy,
                    guardrail_scan_system_prompt=registry_settings.guardrail_scan_system_prompt,
                    guardrail_verbose_logging=verbose_logging,
                    guardrail_composite_input_scanners=registry_settings.guardrail_composite_input_scanners,
                    guardrail_composite_recent_message_limit=registry_settings.guardrail_composite_recent_message_limit,
                    guardrail_palimpsest_run_entities=registry_settings.guardrail_palimpsest_run_entities,
                    guardrail_palimpsest_entity_replacements=registry_settings.guardrail_palimpsest_entity_replacements,
                    guardrail_palimpsest_options=registry_settings.guardrail_palimpsest_options,
                    guardrail_palimpsest_session_options=registry_settings.guardrail_palimpsest_session_options,
                    guardrail_tool_profiles=tool_bundle.guardrail_profiles,
                    guardrail_unprofiled_tools="block",
                )
                thread_id, run_config = _new_config(
                    args.thread_id,
                    user_id=args.user_id,
                    user_role=args.user_role,
                    tenant_id=args.tenant_id,
                    allow_external_tool_access=args.allow_external_tool_access,
                )
                prompt = " ".join(args.prompt).strip()
                current_turns: list[dict[str, str]] = []

                greeting = runner.run(_invoke_turn(agent, user_text=None, config=run_config))
                if greeting and not prompt:
                    print(f"\nArtifact Creator: {greeting}\n")

                if prompt:
                    answer = runner.run(_invoke_turn(agent, user_text=prompt, config=run_config))
                    if answer:
                        current_turns.append({"user": prompt, "assistant": answer})
                        print(answer)
                    if args.save_thread:
                        saved_path = _save_thread_export(
                            path=_normalize_export_path(args.save_thread, thread_id=thread_id),
                            thread_id=thread_id,
                            provider=provider.value,
                            privacy_enabled=privacy_enabled,
                            scanners_enabled=scanners_enabled,
                            tool_execution_enabled=tool_execution_enabled,
                            turns=current_turns,
                        )
                        print(f"Thread saved to: {saved_path}")
                    return 0

                print(
                    "artifact_creator_agent CLI started "
                    f"(thread_id={thread_id}, provider={provider.value})."
                )
                print(f"Checkpoint store: {ARTIFACT_CLI_CHECKPOINT_PATH}")
                tool_names = ", ".join(getattr(tool, "name", type(tool).__name__) for tool in registry_tools)
                print(f"Registry tools: {tool_names or 'none'}")
                print(f"Privacy enabled: {privacy_enabled}")
                print(f"Scanners enabled: {scanners_enabled}")
                print(f"Tool execution guardrails enabled: {tool_execution_enabled}")
                print(f"Prompt injection model: {prompt_injection_model or 'LLM Guard default'}")
                print(f"External tool access allowed: {args.allow_external_tool_access}")

                if system_file_path is not None:
                    print(f"System prompt was loaded from file: {system_file_path}")
                elif system_prompt:
                    print("System prompt was provided through the command line.")
                else:
                    print("No fixed system prompt was provided. The agent will ask for one.")

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
                    save_thread_path = _parse_save_thread_command(user_input)
                    if save_thread_path is not None:
                        saved_path = _save_thread_export(
                            path=_normalize_export_path(save_thread_path, thread_id=thread_id),
                            thread_id=thread_id,
                            provider=provider.value,
                            privacy_enabled=privacy_enabled,
                            scanners_enabled=scanners_enabled,
                            tool_execution_enabled=tool_execution_enabled,
                            turns=current_turns,
                        )
                        print(f"Thread saved to: {saved_path}\n")
                        continue
                    if user_input_lower in MULTILINE_START_COMMANDS:
                        multiline_input = _collect_multiline_input()
                        if multiline_input is None:
                            print("Goodbye!")
                            return 0
                        if not multiline_input:
                            print("Multiline input cancelled.\n")
                            continue
                        user_input = multiline_input
                    elif user_input_lower in RESET_COMMANDS:
                        thread_id, run_config = _new_config(
                            user_id=args.user_id,
                            user_role=args.user_role,
                            tenant_id=args.tenant_id,
                            allow_external_tool_access=args.allow_external_tool_access,
                        )
                        current_turns = []
                        print(f"Started a fresh thread: {thread_id}")
                        greeting = runner.run(_invoke_turn(agent, user_text=None, config=run_config))
                        if greeting:
                            print(f"\nArtifact Creator: {greeting}\n")
                        continue

                    try:
                        answer = runner.run(_invoke_turn(agent, user_text=user_input, config=run_config))
                    except Exception as exc:
                        print(f"\nArtifact Creator error: {exc}\n", file=sys.stderr)
                        continue

                    if answer:
                        current_turns.append({"user": user_input, "assistant": answer})
                        print(f"\nArtifact Creator: {answer}\n")
            finally:
                runner.run(checkpoint_cm.__aexit__(None, None, None))
    except Exception as exc:
        print(f"Failed to start CLI: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
