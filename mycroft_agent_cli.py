from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from agents.mycroft_agent.agent import VALID_MODEL_SIZES, initialize_agent
from agents.mycroft_agent.cli_config import (
    DEFAULT_CLI_CONFIG_PATH,
    build_internal_tools,
    list_available_internal_tools,
    load_cli_config,
    load_mcp_tools_from_config,
    validate_required_environment,
)
from agents.mycroft_agent.configured_agent import (
    build_skills_backend,
    normalize_skill_source,
)
from agents.mycroft_agent.subagent_loader import initialize_configured_subagents
from agents.utils import ModelType, extract_text
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command


EXIT_COMMANDS = {"exit", "/exit", "quit", "/quit"}
RESET_COMMANDS = {"reset", "/reset"}
HELP_COMMANDS = {"help", "/help"}
MULTILINE_START_COMMANDS = {"/multi", "/multiline", "<<<"}
MULTILINE_END_COMMANDS = {"/send", "/end", ">>>"}
MULTILINE_CANCEL_COMMANDS = {"/cancel", "/abort"}
SAVE_THREAD_COMMANDS = {"/save-thread", "/save"}
DEFAULT_PROMPTS_DIR = Path("./prompts")
DEFAULT_AGENT_LOCALE = {
    "save_confirmation": "[You can now download the file.]({url})"
}
DEFAULT_CLI_USER_ID = "mycroft-agent-cli"
MYCROFT_CLI_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "mycroft_agent"
    / "cli_checkpoints.sqlite"
)


def _persistent_checkpoint_saver():
    MYCROFT_CLI_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(str(MYCROFT_CLI_CHECKPOINT_PATH))


def _parse_provider(raw_value: str) -> ModelType:
    normalized = raw_value.strip().lower()
    for provider in ModelType:
        if normalized in {provider.value.lower(), provider.name.lower()}:
            return provider
    available = ", ".join(sorted(provider.value for provider in ModelType))
    raise ValueError(f"Unknown provider '{raw_value}'. Available values: {available}")


def _parse_model_size(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    if normalized not in VALID_MODEL_SIZES:
        choices = ", ".join(sorted(VALID_MODEL_SIZES))
        raise ValueError(f"Unknown model size '{raw_value}'. Available values: {choices}")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for agents.mycroft_agent."
    )
    system_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional prompt for a single non-interactive run.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CLI_CONFIG_PATH),
        help="Path to the Mycroft CLI JSON config.",
    )
    parser.add_argument(
        "--provider",
        default=ModelType.GPT.value,
        help="Model provider to use for Mycroft itself.",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        help="Model size resolved through get_llm: base, mini, or nano.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Optional thread id. If omitted, a random one is generated.",
    )
    system_group.add_argument(
        "--system",
        default=None,
        help="Optional system prompt override.",
    )
    system_group.add_argument(
        "--system-file",
        default=None,
        help="Read the system prompt override from a file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature passed to get_llm.",
    )
    parser.add_argument(
        "--save-thread",
        default=None,
        help="Optional path to save the current conversation thread after the run (.md by default, .json also supported).",
    )
    return parser.parse_args()


def _new_config(thread_id: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    resolved_thread_id = thread_id or f"mycroft-agent-cli-{uuid.uuid4().hex}"
    return resolved_thread_id, {
        "configurable": {
            "thread_id": resolved_thread_id,
            "user_id": DEFAULT_CLI_USER_ID,
        }
    }


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
    config_path: str,
    provider: str,
    model_size: str,
    turns: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "provider": provider,
        "model_size": model_size,
        "turns": turns,
    }


def _render_thread_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Mycroft Conversation Thread",
        "",
        f"- Thread id: `{payload['thread_id']}`",
        f"- Saved at (UTC): `{payload['saved_at']}`",
        f"- Config: `{payload['config_path']}`",
        f"- Provider: `{payload['provider']}`",
        f"- Model size: `{payload['model_size']}`",
        "",
    ]
    turns = payload.get("turns") or []
    if not turns:
        lines.append("_No conversation turns recorded._")
        lines.append("")
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
                "Mycroft:",
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
    config_path: str,
    provider: str,
    model_size: str,
    turns: list[dict[str, str]],
) -> Path:
    payload = _thread_export_payload(
        thread_id=thread_id,
        config_path=config_path,
        provider=provider,
        model_size=model_size,
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


def _extract_last_ai_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return extract_text(message)
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = str(item.get("text", "")).strip()
                        if text:
                            parts.append(text)
                return "\n".join(parts).strip()
    return ""


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return None
    payload = getattr(interrupts[-1], "value", interrupts[-1])
    if isinstance(payload, dict):
        return payload
    return {"content": str(payload)}


def _read_interrupt_input(prompt: str) -> str:
    value = input(prompt)
    if value.strip().lower() in EXIT_COMMANDS:
        raise KeyboardInterrupt
    return value


def _print_interrupt_header(title: str) -> None:
    print(f"\n{title}")


def _format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _interrupt_fields(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_fields = payload.get("fields")
    if isinstance(raw_fields, list):
        return [field for field in raw_fields if isinstance(field, dict)]

    requested_schema = payload.get("requestedSchema")
    if not isinstance(requested_schema, dict):
        return []

    properties = requested_schema.get("properties")
    if not isinstance(properties, dict):
        return []

    raw_required = requested_schema.get("required")
    required = set(raw_required) if isinstance(raw_required, list) else set()
    fields: list[dict[str, Any]] = []
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue
        fields.append(
            {
                "name": str(field_name),
                "title": field_schema.get("title") or str(field_name),
                "required": str(field_name) in required,
                "type": field_schema.get("type"),
            }
        )
    return fields


def _prompt_interrupt_decision(allowed_decisions: list[str]) -> str:
    allowed = [decision.strip().lower() for decision in allowed_decisions if decision]
    if not allowed:
        raise ValueError("Interrupt request does not define any allowed decisions.")

    prompt = f"Decision ({'/'.join(allowed)})> "
    while True:
        decision = _read_interrupt_input(prompt).strip().lower()
        if decision in allowed:
            return decision
        print(f"Please choose one of: {', '.join(allowed)}")


def _prompt_json_args(current_args: dict[str, Any]) -> dict[str, Any]:
    while True:
        edited_args_text = _read_interrupt_input(
            "Edited action args as JSON (blank keeps current)> "
        ).strip()
        if not edited_args_text:
            return current_args
        try:
            parsed = json.loads(edited_args_text)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON: {exc}")
            continue
        if not isinstance(parsed, dict):
            print("Edited action args must be a JSON object.")
            continue
        return parsed


def _collect_hitl_resume_value(payload: dict[str, Any]) -> dict[str, Any] | None:
    action_requests = payload.get("action_requests")
    review_configs = payload.get("review_configs")
    if not isinstance(action_requests, list) or not isinstance(review_configs, list):
        return None
    if len(action_requests) != len(review_configs):
        raise ValueError("Interrupt payload has mismatched action_requests and review_configs.")

    decisions: list[dict[str, Any]] = []
    for index, raw_action in enumerate(action_requests):
        if not isinstance(raw_action, dict):
            raise ValueError("Interrupt payload contains a non-object action request.")

        raw_review = review_configs[index]
        if not isinstance(raw_review, dict):
            raise ValueError("Interrupt payload contains a non-object review config.")

        action_name = str(raw_action.get("name") or "").strip()
        if not action_name:
            raise ValueError("Interrupt payload action request is missing a name.")

        action_args = raw_action.get("args")
        if not isinstance(action_args, dict):
            raise ValueError(f"Interrupt action '{action_name}' is missing object args.")

        allowed_decisions = raw_review.get("allowed_decisions")
        if not isinstance(allowed_decisions, list) or not allowed_decisions:
            raise ValueError(
                f"Interrupt action '{action_name}' is missing allowed_decisions."
            )

        description = str(raw_action.get("description") or "").strip()
        _print_interrupt_header(f"[Approval required] {action_name}")
        if description:
            print(description)
        print("Action args:")
        print(_format_json(action_args))

        decision = _prompt_interrupt_decision(
            [str(item) for item in allowed_decisions if str(item).strip()]
        )
        if decision == "approve":
            decisions.append({"type": "approve"})
            continue
        if decision == "reject":
            message = _read_interrupt_input("Reject reason (optional)> ").strip()
            item: dict[str, Any] = {"type": "reject"}
            if message:
                item["message"] = message
            decisions.append(item)
            continue

        edited_name = _read_interrupt_input(
            f"Edited action name [{action_name}]> "
        ).strip() or action_name
        edited_args = _prompt_json_args(action_args)
        decisions.append(
            {
                "type": "edit",
                "edited_action": {
                    "name": edited_name,
                    "args": edited_args,
                },
            }
        )

    return {"decisions": decisions}


def _collect_generic_interrupt_resume_value(payload: dict[str, Any]) -> Any:
    fields = _interrupt_fields(payload)
    question = (
        str(payload.get("question") or "").strip()
        or str(payload.get("content") or "").strip()
        or "Agent requested additional input."
    )
    _print_interrupt_header("[Additional input required]")
    print(question)

    if not fields:
        return _read_interrupt_input("Resume> ").strip()

    if payload.get("responseMode") == "text" and len(fields) == 1:
        field = fields[0]
        label = str(field.get("title") or field.get("name") or "Value")
        return {"value": _read_interrupt_input(f"{label}> ")}

    response: dict[str, Any] = {}
    for field in fields:
        field_name = str(field.get("name") or "").strip()
        if not field_name:
            continue
        label = str(field.get("title") or field_name)
        field_type = str(field.get("type") or "").strip().lower()
        raw_value = _read_interrupt_input(f"{label}> ")
        if not raw_value and not field.get("required"):
            continue
        if field_type == "boolean":
            response[field_name] = raw_value.strip().lower() in {"y", "yes", "true", "1"}
        else:
            response[field_name] = raw_value
    return response


def _collect_interrupt_resume_value(payload: dict[str, Any]) -> Any:
    hitl_resume = _collect_hitl_resume_value(payload)
    if hitl_resume is not None:
        return hitl_resume
    return _collect_generic_interrupt_resume_value(payload)


async def _invoke_turn(
    agent: Any,
    *,
    user_text: str,
    config: dict[str, Any],
) -> str:
    pending_input: Any = {
        "messages": [HumanMessage(content=user_text)],
        "locale": DEFAULT_AGENT_LOCALE,
    }
    while True:
        result = await agent.ainvoke(pending_input, config=config)
        if not isinstance(result, dict):
            return ""
        interrupt_payload = _interrupt_payload(result)
        if interrupt_payload is None:
            return _extract_last_ai_text(result)
        pending_input = Command(resume=_collect_interrupt_resume_value(interrupt_payload))


def _print_help() -> None:
    print("Commands:")
    print("  /help   show commands")
    print("  /list-tools  show available internal tool names for the config")
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


_initialize_configured_subagents = initialize_configured_subagents


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", repr(tool)))


def _print_runtime_summary(
    *,
    config_path: str,
    stateless_subagents: list[SubAgent | CompiledSubAgent],
    stateful_subagents: list[SubAgent | CompiledSubAgent],
    internal_tools: list[Any],
    mcp_tools: list[Any],
) -> None:
    print(f"Mycroft config: {config_path}")
    if stateless_subagents:
        print(
            "Configured stateless subagents: "
            + ", ".join(agent["name"] for agent in stateless_subagents)
        )
    else:
        print("Configured stateless subagents: none")

    if stateful_subagents:
        print(
            "Configured stateful subagents: "
            + ", ".join(agent["name"] for agent in stateful_subagents)
        )
    else:
        print("Configured stateful subagents: none")

    all_tool_names = [_tool_name(tool) for tool in [*internal_tools, *mcp_tools]]
    if all_tool_names:
        print("Configured tools: " + ", ".join(all_tool_names))
    else:
        print("Configured tools: none")

    if mcp_tools:
        print("MCP tools: " + ", ".join(_tool_name(tool) for tool in mcp_tools))
    else:
        print("MCP tools: none")


def main() -> int:
    args = _parse_args()

    try:
        provider = _parse_provider(args.provider)
        model_size = _parse_model_size(args.model_size)
        cli_config = load_cli_config(args.config)
        validate_required_environment(cli_config, provider.value)
        system_prompt_override, system_file_path = _load_system_prompt(args)
        system_prompt = system_prompt_override or cli_config.system_prompt

        with asyncio.Runner() as runner:
            checkpoint_cm = _persistent_checkpoint_saver()
            checkpoint_saver = runner.run(checkpoint_cm.__aenter__())
            try:
                runner.run(checkpoint_saver.setup())
                stateless_subagents = runner.run(
                    _initialize_configured_subagents(cli_config.subagents.stateless)
                )
                stateful_subagents = runner.run(
                    _initialize_configured_subagents(cli_config.subagents.stateful)
                )
                internal_tools = build_internal_tools(cli_config.internal_tools)
                mcp_tools = runner.run(load_mcp_tools_from_config(cli_config.mcp))
                skills = tuple(normalize_skill_source(skill) for skill in cli_config.skills.paths)
                agent = initialize_agent(
                    provider=provider,
                    model_size=model_size,
                    temperature=args.temperature,
                    system_prompt=system_prompt,
                    tools=[*internal_tools, *mcp_tools],
                    stateless_subagents=stateless_subagents,
                    stateful_subagents=stateful_subagents,
                    streaming=False,
                    checkpoint_saver=checkpoint_saver,
                    interrupt_on=cli_config.deepagents.interrupt_on or None,
                    skills=skills,
                    backend=build_skills_backend(skills),
                )

                thread_id, run_config = _new_config(args.thread_id)
                prompt = " ".join(args.prompt).strip()
                current_turns: list[dict[str, str]] = []

                if prompt:
                    try:
                        answer = runner.run(_invoke_turn(agent, user_text=prompt, config=run_config))
                    except KeyboardInterrupt:
                        print("Goodbye!")
                        return 0
                    if answer:
                        current_turns.append({"user": prompt, "assistant": answer})
                    if answer:
                        print(answer)
                    if args.save_thread:
                        saved_path = _save_thread_export(
                            path=_normalize_export_path(args.save_thread, thread_id=thread_id),
                            thread_id=thread_id,
                            config_path=args.config,
                            provider=provider.value,
                            model_size=model_size,
                            turns=current_turns,
                        )
                        print(f"Thread saved to: {saved_path}")
                    return 0

                print(
                    f"mycroft_agent CLI started (thread_id={thread_id}, provider={provider.value}, model_size={model_size})."
                )
                print(f"Checkpoint store: {MYCROFT_CLI_CHECKPOINT_PATH}")
                _print_runtime_summary(
                    config_path=args.config,
                    stateless_subagents=stateless_subagents,
                    stateful_subagents=stateful_subagents,
                    internal_tools=internal_tools,
                    mcp_tools=mcp_tools,
                )

                if system_file_path is not None:
                    print(f"System prompt override was loaded from file: {system_file_path}")
                elif system_prompt_override:
                    print("System prompt override was provided through the command line.")
                else:
                    print("Using the system prompt from the Mycroft CLI config.")

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
                            config_path=args.config,
                            provider=provider.value,
                            model_size=model_size,
                            turns=current_turns,
                        )
                        print(f"Thread saved to: {saved_path}\n")
                        continue
                    if user_input_lower == "/list-tools":
                        names = ", ".join(list_available_internal_tools())
                        print(f"Available internal tools: {names}\n")
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
                        thread_id, run_config = _new_config()
                        current_turns = []
                        print(f"Started a fresh thread: {thread_id}\n")
                        continue

                    try:
                        answer = runner.run(_invoke_turn(agent, user_text=user_input, config=run_config))
                    except KeyboardInterrupt:
                        print("\nGoodbye!")
                        return 0
                    except Exception as exc:
                        print(f"\nAssistant error: {exc}\n", file=sys.stderr)
                        continue

                    if answer:
                        current_turns.append({"user": user_input, "assistant": answer})
                        print(f"\nMycroft: {answer}\n")
            finally:
                from bot_service.agent_registry import agent_registry

                runner.run(agent_registry.aclose())
                runner.run(checkpoint_cm.__aexit__(None, None, None))
    except Exception as exc:
        print(f"Failed to start CLI: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
