from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

import config
from agents.simple_agent.agent import initialize_agent
from agents.utils import ModelType, extract_text
from langchain_core.messages import AIMessage, HumanMessage


EXIT_COMMANDS = {"exit", "/exit", "quit", "/quit"}
RESET_COMMANDS = {"reset", "/reset"}
HELP_COMMANDS = {"help", "/help"}
MULTILINE_START_COMMANDS = {"/multi", "/multiline", "<<<"}
MULTILINE_END_COMMANDS = {"/send", "/end", ">>>"}
MULTILINE_CANCEL_COMMANDS = {"/cancel", "/abort"}
DEFAULT_PROMPTS_DIR = Path("./prompts")


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
        help="Enable ThinkTool and pass it in runtime context.",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        default=True,
        help="Enable Yandex web search tool and pass it in runtime context.",
    )
    return parser.parse_args()


def _new_config(thread_id: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    resolved_thread_id = thread_id or f"simple-agent-cli-{uuid.uuid4().hex}"
    return resolved_thread_id, {"configurable": {"thread_id": resolved_thread_id}}


def _build_tools(args: argparse.Namespace) -> list[Any]:
    tools: list[Any] = []

    if args.enable_think:
        from agents.tools.think import ThinkTool

        tools.append(ThinkTool())

    if args.enable_web_search:
        if not config.YA_API_KEY or not config.YA_FOLDER_ID:
            raise RuntimeError(
                "Web search requires YA_API_KEY and YA_FOLDER_ID in the environment."
            )

        from agents.tools.yandex_search import YandexSearchTool

        tools.append(
            YandexSearchTool(
                api_key=config.YA_API_KEY,
                folder_id=config.YA_FOLDER_ID,
                max_results=3,
                summarize=True,
            )
        )

    return tools


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


def _build_context(*, system_prompt: Optional[str], tools: list[Any]) -> dict[str, Any] | None:
    context: dict[str, Any] = {}
    if system_prompt:
        context["system_prompt"] = system_prompt
    if tools:
        context["tools"] = tools
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
        tools = _build_tools(args)
        system_prompt, system_file_path = _load_system_prompt(args)
    except Exception as exc:
        print(f"Failed to start CLI: {exc}", file=sys.stderr)
        return 1

    graph = initialize_agent(provider=provider, streaming=False)
    thread_id, run_config = _new_config(args.thread_id)
    context = _build_context(system_prompt=system_prompt, tools=tools)

    print(f"simple_agent CLI started (thread_id={thread_id}, provider={provider.value}).")
    print(
        "Active tools: "
        + (", ".join(getattr(tool, "name", tool.__class__.__name__) for tool in tools) if tools else "none")
    )

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
            thread_id, run_config = _new_config()
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
