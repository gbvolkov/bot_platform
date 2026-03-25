from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GAZ_PRICING_DB_URL = f"sqlite:///{(REPO_ROOT / 'data' / 'gaz-pricing' / 'gaz_pricing.sqlite').as_posix()}"
DEFAULT_PROMPT_CONTEXT_PATH = REPO_ROOT / "agents" / "gaz_agent" / "pricing_bi_prompt_context.txt"
MULTILINE_TERMINATOR = "."


@dataclass
class SessionSettings:
    database_url: str
    prompt_context: str
    return_files: bool
    return_images: bool
    default_database_url: str
    default_prompt_context: str
    default_return_files: bool
    default_return_images: bool


def _normalize_input_line(line: str) -> str:
    return line.lstrip("\ufeff")


def _load_default_prompt_context() -> str:
    return DEFAULT_PROMPT_CONTEXT_PATH.read_text(encoding="utf-8")


def _resolve_question(question_arg: Optional[str], question_option: Optional[str]) -> str:
    question = question_option or question_arg
    return (question or "").strip()


def _resolve_prompt_context(prompt_context_text: Optional[str], prompt_context_file: Optional[Path]) -> str:
    if prompt_context_text:
        return prompt_context_text
    if prompt_context_file:
        return prompt_context_file.read_text(encoding="utf-8")
    return _load_default_prompt_context()


def _build_init_context(settings: SessionSettings) -> Dict[str, Any]:
    return {
        "database_url": settings.database_url,
        "database_prompt_context": settings.prompt_context,
        "return_files": settings.return_files,
        "return_images": settings.return_images,
    }


def _create_bi_agent(settings: SessionSettings) -> Any:
    from agents.bi_agent import initialize_agent

    return initialize_agent(notify_on_reload=False, init_context=_build_init_context(settings))


def _find_last_ai_message(messages: List[Any]) -> AIMessage:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    raise RuntimeError("BI agent did not return an AI message.")


def _extract_parts(message: AIMessage) -> List[Dict[str, Any]]:
    content = getattr(message, "content", [])
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [{"type": "text", "text": str(content)}]


def _decode_attachments(parts: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    saved_paths: List[Path] = []
    for index, part in enumerate(parts, start=1):
        data = part.get("data")
        if not isinstance(data, str) or not data:
            continue
        filename = part.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            filename = f"attachment_{index}"
        target_path = output_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(base64.b64decode(data))
        saved_paths.append(target_path)
    return saved_paths


def _build_config(thread_id: str) -> Dict[str, Any]:
    return {
        "configurable": {
            "user_id": "bi-cli",
            "user_role": "default",
            "thread_id": thread_id,
        }
    }


def _build_human_message(question: str) -> Dict[str, Any]:
    return {
        "messages": [
            HumanMessage(content=[{"type": "text", "text": question}]),
        ]
    }


def _build_reset_message() -> Dict[str, Any]:
    return {
        "messages": [
            HumanMessage(content=[{"type": "reset", "text": "RESET"}]),
        ]
    }


async def _invoke_bi_agent(
    agent: Any,
    thread_id: str,
    question: Optional[str] = None,
    *,
    reset: bool = False,
) -> Dict[str, Any]:
    payload = _build_reset_message() if reset else _build_human_message(question or "")
    response = await agent.ainvoke(payload, config=_build_config(thread_id))
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected BI agent response type.")
    return response


def _extract_answer_and_attachments(response: Dict[str, Any], output_dir: Path) -> None:
    messages = response.get("messages") or []
    ai_message = _find_last_ai_message(messages)
    parts = _extract_parts(ai_message)
    text_parts = [part.get("text", "") for part in parts if part.get("type") == "text"]
    answer = "\n\n".join(text.strip() for text in text_parts if isinstance(text, str) and text.strip())
    print(answer or "[empty answer]")

    attachment_parts = [part for part in parts if part.get("type") in {"file", "image"}]
    if not attachment_parts:
        return

    run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_paths = _decode_attachments(attachment_parts, run_dir)
    if saved_paths:
        print("")
        print("Saved attachments:")
        for path in saved_paths:
            print(str(path))


def _print_session_help() -> None:
    print("Interactive commands:")
    print("  /help           Show this help")
    print("  /show           Show current database URL, prompt context, and attachment flags")
    print("  /url <value>    Set SQLAlchemy database URL")
    print("  /url default    Restore default database URL")
    print("  /files on|off   Enable or disable file attachments")
    print("  /images on|off  Enable or disable image attachments")
    print("  /prompt         Edit prompt context in multiline mode")
    print("  /prompt <text>  Set prompt context from one line")
    print("  /prompt default Restore default prompt context")
    print("  /prompt-file <path>  Load prompt context from UTF-8 text file")
    print("  /defaults       Restore both default URL and default prompt context")
    print("  /reset          Clear BI agent memory in current session")
    print("  /exit           Exit interactive mode")
    print(f"  End multiline input with a single {MULTILINE_TERMINATOR!r} line")


def _show_settings(settings: SessionSettings) -> None:
    print(f"database_url: {settings.database_url}")
    print(f"return_files: {settings.return_files}")
    print(f"return_images: {settings.return_images}")
    print("")
    print("prompt_context:")
    print(settings.prompt_context)


def _read_multiline_block(first_line: Optional[str], prompt_label: str) -> str:
    lines: List[str] = []
    if first_line is not None:
        lines.append(_normalize_input_line(first_line))

    while True:
        prompt = f"{prompt_label}> " if not lines else "... "
        try:
            line = input(prompt)
        except EOFError:
            return "\n".join(lines).strip()
        line = _normalize_input_line(line)
        if line == MULTILINE_TERMINATOR:
            return "\n".join(lines).strip()
        lines.append(line)


def _handle_command(
    command_line: str,
    settings: SessionSettings,
    agent: Any,
    thread_id: str,
    output_dir: Path,
) -> tuple[bool, Any]:
    command, _, raw_arg = command_line.partition(" ")
    arg = raw_arg.strip()

    if command in {"/exit", "/quit"}:
        return False, agent
    if command == "/help":
        _print_session_help()
        return True, agent
    if command == "/show":
        _show_settings(settings)
        return True, agent
    if command == "/url":
        if not arg:
            print("Usage: /url <sqlalchemy-url> or /url default")
            return True, agent
        settings.database_url = settings.default_database_url if arg == "default" else arg
        print(f"database_url updated: {settings.database_url}")
        return True, None
    if command == "/files":
        normalized = arg.lower()
        if normalized not in {"on", "off"}:
            print("Usage: /files on|off")
            return True, agent
        settings.return_files = normalized == "on"
        print(f"return_files updated: {settings.return_files}")
        return True, None
    if command == "/images":
        normalized = arg.lower()
        if normalized not in {"on", "off"}:
            print("Usage: /images on|off")
            return True, agent
        settings.return_images = normalized == "on"
        print(f"return_images updated: {settings.return_images}")
        return True, None
    if command == "/prompt":
        if arg == "default":
            settings.prompt_context = settings.default_prompt_context
            print("prompt_context restored to default")
            return True, None
        if arg:
            settings.prompt_context = arg
            print("prompt_context updated from inline text")
            return True, None
        print(f"Enter prompt context. Finish with {MULTILINE_TERMINATOR!r} on its own line.")
        settings.prompt_context = _read_multiline_block(None, "prompt")
        print("prompt_context updated")
        return True, None
    if command == "/prompt-file":
        if not arg:
            print("Usage: /prompt-file <path>")
            return True, agent
        path = Path(arg)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        settings.prompt_context = path.read_text(encoding="utf-8")
        print(f"prompt_context loaded from {path}")
        return True, None
    if command == "/defaults":
        settings.database_url = settings.default_database_url
        settings.prompt_context = settings.default_prompt_context
        settings.return_files = settings.default_return_files
        settings.return_images = settings.default_return_images
        print("Defaults restored")
        return True, None
    if command == "/reset":
        if agent is None:
            agent = _create_bi_agent(settings)
        response = asyncio.run(_invoke_bi_agent(agent, thread_id, reset=True))
        _extract_answer_and_attachments(response, output_dir)
        return True, agent

    print(f"Unknown command: {command}. Use /help.")
    return True, agent


def _run_single_question(question: str, settings: SessionSettings, output_dir: Path) -> None:
    agent = _create_bi_agent(settings)
    thread_id = f"bi-cli-{datetime.now().strftime('%Y%m%d%H%M%S_%f')}"
    response = asyncio.run(_invoke_bi_agent(agent, thread_id, question=question))
    _extract_answer_and_attachments(response, output_dir)


def _run_interactive_session(settings: SessionSettings, output_dir: Path) -> None:
    print("Interactive ai_bi session")
    print(f"Current database_url: {settings.database_url}")
    print(f"Finish multiline input with {MULTILINE_TERMINATOR!r} on its own line.")
    print("Use /help for commands.")
    print("")

    thread_id = f"bi-cli-{datetime.now().strftime('%Y%m%d%H%M%S_%f')}"
    agent = None

    while True:
        try:
            first_line = _normalize_input_line(input("You> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            print("Session closed.")
            return

        if not first_line:
            continue

        if first_line.startswith("/"):
            keep_running, agent = _handle_command(first_line, settings, agent, thread_id, output_dir)
            if not keep_running:
                print("Session closed.")
                return
            print("")
            continue

        question = _read_multiline_block(first_line, "You")
        if not question:
            continue

        if agent is None:
            agent = _create_bi_agent(settings)

        response = asyncio.run(_invoke_bi_agent(agent, thread_id, question=question))
        _extract_answer_and_attachments(response, output_dir)
        print("")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ai_bi directly without bot_service.")
    parser.add_argument(
        "question_arg",
        nargs="?",
        help="Question to ask ai_bi. If omitted, starts interactive mode.",
    )
    parser.add_argument("--question", help="Question to ask ai_bi.")
    parser.add_argument(
        "--database-url",
        default=DEFAULT_GAZ_PRICING_DB_URL,
        help="SQLAlchemy URL for the source database.",
    )
    parser.add_argument(
        "--prompt-context-file",
        type=Path,
        help="UTF-8 text file with additional prompt context for SQL generation.",
    )
    parser.add_argument(
        "--prompt-context-text",
        help="Inline prompt context text for SQL generation.",
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="Disable file attachment generation.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image attachment generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "bi_cli_outputs",
        help="Directory where decoded BI attachments will be stored.",
    )
    return parser


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    prompt_context_file = args.prompt_context_file
    if prompt_context_file is not None:
        if not prompt_context_file.exists():
            parser.error(f"--prompt-context-file does not exist: {prompt_context_file}")
        if not prompt_context_file.is_file():
            parser.error(f"--prompt-context-file is not a file: {prompt_context_file}")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    default_prompt_context = _load_default_prompt_context()
    settings = SessionSettings(
        database_url=args.database_url,
        prompt_context=_resolve_prompt_context(args.prompt_context_text, args.prompt_context_file),
        return_files=not args.no_files,
        return_images=not args.no_images,
        default_database_url=DEFAULT_GAZ_PRICING_DB_URL,
        default_prompt_context=default_prompt_context,
        default_return_files=True,
        default_return_images=True,
    )

    resolved_question = _resolve_question(args.question_arg, args.question)
    if resolved_question:
        _run_single_question(resolved_question, settings, args.output_dir)
        return 0

    _run_interactive_session(settings, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
