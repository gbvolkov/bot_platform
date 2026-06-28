from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agents.utils import ModelType

from .agent import initialize_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ismart-generator-agent")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to task JSON, tasks JSON, or full course JSON.")
    source.add_argument("--input-url", help="URL to task JSON, tasks JSON, or full course JSON.")
    parser.add_argument("--output", required=True, help="Output root directory.")
    parser.add_argument("--task-id")
    parser.add_argument("--lesson-number")
    parser.add_argument("--generation-target")
    parser.add_argument("--max-generation-iterations", type=int, default=3)
    parser.add_argument("--max-package-repair-iterations", type=int, default=2)
    parser.add_argument("--max-reference-chars", type=int, default=0)
    parser.add_argument("--provider", default=ModelType.GPT.value, help="Model provider value or enum name.")
    parser.add_argument("--model-mode", choices=("base", "mini", "nano"), default="base")
    parser.add_argument("--verbose", action="store_true", help="Print deterministic generation trace to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        provider = _parse_provider(args.provider)
        graph = initialize_agent(
            provider=provider,
            use_platform_store=False,
            streaming=False,
            model_mode=args.model_mode,
        )
        configurable = _build_configurable(args)
        state = graph.invoke(
            {"messages": [HumanMessage(content="Run iSMART material generation from CLI.")]},
            config={
                "configurable": configurable,
                "recursion_limit": 50,
            },
        )
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failures.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_text = _last_ai_text(state.get("messages") or [])
    if output_text:
        print(output_text)
    elif state.get("error"):
        print(f"error: {state['error']}", file=sys.stderr)
    else:
        print("iSMART generation finished.")

    if state.get("error"):
        return 1
    results = state.get("results") or []
    if not results or not all(isinstance(item, dict) for item in results):
        return 1
    return 0 if all(item.get("status") == "approved" for item in results) else 1


def _build_configurable(args: argparse.Namespace) -> dict[str, Any]:
    configurable: dict[str, Any] = {
        "thread_id": f"ismart-generator-cli-{_timestamp()}",
        "output": args.output,
        "max_generation_iterations": args.max_generation_iterations,
        "max_package_repair_iterations": args.max_package_repair_iterations,
        "max_reference_chars": args.max_reference_chars,
        "verbose": bool(args.verbose),
    }
    if args.input:
        configurable["input"] = args.input
    if args.input_url:
        configurable["input_url"] = args.input_url
    if args.task_id:
        configurable["task_id"] = args.task_id
    if args.lesson_number:
        configurable["lesson_number"] = str(args.lesson_number)
    if args.generation_target:
        configurable["generation_target"] = args.generation_target
    return configurable


def _parse_provider(value: str) -> ModelType:
    text = str(value or "").strip()
    for candidate in ModelType:
        if text.lower() == candidate.value.lower() or text.upper() == candidate.name:
            return candidate
    known = ", ".join(f"{item.name}/{item.value}" for item in ModelType)
    raise ValueError(f"Unknown provider {value!r}. Known providers: {known}")


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) or getattr(message, "type", "") == "ai":
            return str(getattr(message, "content", "") or "")
    return ""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
