from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from agents.mycroft_agent.agent import VALID_MODEL_SIZES, initialize_agent
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


def _parse_model_size(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    if normalized not in VALID_MODEL_SIZES:
        choices = ", ".join(sorted(VALID_MODEL_SIZES))
        raise ValueError(f"Unknown model size '{raw_value}'. Available values: {choices}")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal interactive CLI for agents.mycroft_agent."
    )
    system_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional prompt for a single non-interactive run.",
    )
    parser.add_argument(
        "--provider",
        default=ModelType.GPT.value,
        help="Model provider to use for mycroft_agent.",
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
        default="./prompts/ideator.txt",
        help="Read the system prompt override from a file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature passed to get_llm.",
    )
    parser.add_argument(
        "--disable-web-search",
        action="store_true",
        help="Disable the default Yandex web_search tool.",
    )
    parser.add_argument(
        "--max-search-results",
        type=int,
        default=5,
        help="Maximum number of documents requested from Yandex search.",
    )
    parser.add_argument(
        "--no-search-summary",
        action="store_true",
        help="Pass raw fetched pages to the agent instead of summarized search results.",
    )
    return parser.parse_args()


def _new_config(thread_id: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    resolved_thread_id = thread_id or f"mycroft-agent-cli-{uuid.uuid4().hex}"
    return resolved_thread_id, {"configurable": {"thread_id": resolved_thread_id}}


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


def _invoke_turn(
    agent: Any,
    *,
    user_text: str,
    config: dict[str, Any],
) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)
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
        model_size = _parse_model_size(args.model_size)
        system_prompt, system_file_path = _load_system_prompt(args)
        agent = initialize_agent(
            provider=provider,
            model_size=model_size,
            temperature=args.temperature,
            system_prompt=system_prompt,
            enable_web_search=not args.disable_web_search,
            max_search_results=args.max_search_results,
            summarize_search=not args.no_search_summary,
            streaming=False,
        )
    except Exception as exc:
        print(f"Failed to start CLI: {exc}", file=sys.stderr)
        return 1

    thread_id, run_config = _new_config(args.thread_id)
    prompt = " ".join(args.prompt).strip()

    if prompt:
        answer = _invoke_turn(agent, user_text=prompt, config=run_config)
        if answer:
            print(answer)
        return 0

    print(
        f"mycroft_agent CLI started (thread_id={thread_id}, provider={provider.value}, model_size={model_size})."
    )
    print(
        "Active tools: "
        + ("web_search" if not args.disable_web_search else "none")
    )

    if system_file_path is not None:
        print(f"System prompt override was loaded from file: {system_file_path}")
    elif system_prompt:
        print("System prompt override was provided through the command line.")
    else:
        print("Using the default Mycroft system prompt.")

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
                print("Multiline input cancelled.\n")
                continue
            user_input = multiline_input
        elif user_input_lower in RESET_COMMANDS:
            thread_id, run_config = _new_config()
            print(f"Started a fresh thread: {thread_id}\n")
            continue

        try:
            answer = _invoke_turn(agent, user_text=user_input, config=run_config)
        except Exception as exc:
            print(f"\nAssistant error: {exc}\n", file=sys.stderr)
            continue

        if answer:
            print(f"\nMycroft: {answer}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
