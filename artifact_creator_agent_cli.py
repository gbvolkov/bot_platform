from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents.artifact_creator_agent.agent import initialize_agent
from agents.utils import ModelType, extract_text
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


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
ARTIFACT_CLI_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "artifact_creator_agent"
    / "cli_checkpoints.sqlite"
)


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
        "--guardrails-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the platform guardrail middleware.",
    )
    parser.add_argument(
        "--guardrails-locale",
        default="ru-RU",
        help="Locale passed to the privacy guardrail rail.",
    )
    parser.add_argument(
        "--scanners-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable LLM Guard scanner enforcement. Defaults to guardrails setting.",
    )
    parser.add_argument(
        "--scanner-failure-policy",
        default="fail_closed",
        help="Scanner failure mode: fail_closed or fail_open.",
    )
    parser.add_argument(
        "--banned-topic",
        action="append",
        default=None,
        help="Additional banned topic for the scanner profile. Can be repeated.",
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
) -> tuple[str, dict[str, Any]]:
    resolved_thread_id = thread_id or f"artifact-creator-cli-{uuid.uuid4().hex}"
    return resolved_thread_id, {
        "configurable": {
            "thread_id": resolved_thread_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "user_role": user_role,
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
    guardrails_enabled: bool,
    scanners_enabled: bool | None,
    turns: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "guardrails_enabled": guardrails_enabled,
        "scanners_enabled": scanners_enabled,
        "turns": turns,
    }


def _render_thread_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Artifact Creator Conversation Thread",
        "",
        f"- Thread id: `{payload['thread_id']}`",
        f"- Saved at (UTC): `{payload['saved_at']}`",
        f"- Provider: `{payload['provider']}`",
        f"- Guardrails enabled: `{payload['guardrails_enabled']}`",
        f"- Scanners enabled: `{payload['scanners_enabled']}`",
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
    guardrails_enabled: bool,
    scanners_enabled: bool | None,
    turns: list[dict[str, str]],
) -> Path:
    payload = _thread_export_payload(
        thread_id=thread_id,
        provider=provider,
        guardrails_enabled=guardrails_enabled,
        scanners_enabled=scanners_enabled,
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
        scanner_failure_policy = _parse_scanner_failure_policy(args.scanner_failure_policy)
        system_prompt, system_file_path = _load_system_prompt(args)

        with asyncio.Runner() as runner:
            checkpoint_cm = _persistent_checkpoint_saver()
            checkpoint_saver = runner.run(checkpoint_cm.__aenter__())
            try:
                runner.run(checkpoint_saver.setup())
                agent = initialize_agent(
                    provider=provider,
                    locale=args.locale,
                    checkpoint_saver=checkpoint_saver,
                    system_prompt=system_prompt,
                    guardrails_enabled=args.guardrails_enabled,
                    guardrails_locale=args.guardrails_locale,
                    guardrail_scanners_enabled=args.scanners_enabled,
                    guardrail_scanner_failure_policy=scanner_failure_policy,
                    guardrail_banned_topics=args.banned_topic,
                )
                thread_id, run_config = _new_config(
                    args.thread_id,
                    user_id=args.user_id,
                    user_role=args.user_role,
                    tenant_id=args.tenant_id,
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
                            guardrails_enabled=args.guardrails_enabled,
                            scanners_enabled=args.scanners_enabled,
                            turns=current_turns,
                        )
                        print(f"Thread saved to: {saved_path}")
                    return 0

                print(
                    "artifact_creator_agent CLI started "
                    f"(thread_id={thread_id}, provider={provider.value})."
                )
                print(f"Checkpoint store: {ARTIFACT_CLI_CHECKPOINT_PATH}")
                print(f"Guardrails enabled: {args.guardrails_enabled}")
                print(f"Scanners enabled: {args.scanners_enabled}")

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
                            guardrails_enabled=args.guardrails_enabled,
                            scanners_enabled=args.scanners_enabled,
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
