from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

from agents.utils import ModelType

from .agent import initialize_agent, load_payload_from_path_or_text, load_payload_from_url, tasks_from_payload
from .context import task_identity
from .profiles import resolve_course_level
from .task_skip import practice_task_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ismart-generator-sequential",
        description="Generate iSMART artifacts one task at a time from task/course JSON.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to task JSON, tasks JSON, or full course JSON.")
    source.add_argument("--input-url", help="URL to task JSON, tasks JSON, or full course JSON.")
    parser.add_argument("--output", required=True, help="Output root directory.")
    parser.add_argument("--lesson-number", action="append", help="Lesson number to include. May be repeated.")
    parser.add_argument("--task-id", action="append", help="Task id to include. May be repeated.")
    parser.add_argument("--from-lesson", type=int, help="First lesson number to include.")
    parser.add_argument("--to-lesson", type=int, help="Last lesson number to include.")
    parser.add_argument("--limit", type=int, help="Maximum selected tasks to run.")
    parser.add_argument("--generation-target")
    parser.add_argument("--max-generation-iterations", type=int, default=3)
    parser.add_argument("--max-package-repair-iterations", type=int, default=2)
    parser.add_argument("--max-reference-chars", type=int, default=0)
    parser.add_argument("--provider", default=ModelType.GPT.value, help="Model provider value or enum name.")
    parser.add_argument("--model-mode", choices=("base", "mini", "nano"), default="base")
    parser.add_argument("--prompts-dir", help="Prompt/skill directory. Defaults to agents/ismart_generator_agent/prompts_skills/basic.")
    parser.add_argument(
        "--resume-missing-from",
        help="Existing lesson folder or root with lesson folders. Runs only missing/unusable materials in-place.",
    )
    parser.add_argument("--run-name", help="Name of the run directory under --output.")
    parser.add_argument(
        "--preserve-source-index",
        action="store_true",
        help="Use the task index from the full input JSON for output folder names after filtering.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed generation trace.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selected tasks; do not call the LLM.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first exception.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop after the first non-approved task result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    if args.dry_run:
        payload = load_payload_from_url(args.input_url) if args.input_url else load_payload_from_path_or_text(args.input)
        all_tasks = tasks_from_payload(payload)
        tasks = select_tasks(all_tasks, args)
        if args.limit is not None:
            tasks = tasks[: args.limit]
        print_selected_tasks(tasks)
        return 0

    provider = parse_provider(args.provider)
    request = build_graph_request(args)
    graph = initialize_agent(
        provider=provider,
        use_platform_store=False,
        streaming=False,
        model_mode=args.model_mode,
    )
    state = graph.invoke(
        {"messages": []},
        config={
            "configurable": {"thread_id": f"ismart-generator-sequential-{timestamp()}"},
            "recursion_limit": 100,
        },
        context=request,
    )
    if state.get("error"):
        print(f"error: {state['error']}", file=sys.stderr)
        return 1
    task_entries = state.get("task_entries") or []
    status = overall_status(task_entries)
    return 0 if successful_overall_status(status) else 1


def build_graph_request(args: argparse.Namespace) -> dict[str, Any]:
    request: dict[str, Any] = {
        "output": args.output,
        "lesson_numbers": [str(item) for item in (args.lesson_number or [])],
        "task_ids": [str(item) for item in (args.task_id or [])],
        "max_generation_iterations": args.max_generation_iterations,
        "max_package_repair_iterations": args.max_package_repair_iterations,
        "max_reference_chars": args.max_reference_chars,
        "run_name": args.run_name or f"sequential_{timestamp()}",
        "preserve_source_index": bool(args.preserve_source_index),
        "stop_on_error": bool(args.stop_on_error),
        "stop_on_failure": bool(args.stop_on_failure),
        "verbose": bool(args.verbose),
    }
    if args.input:
        request["input"] = args.input
    if args.input_url:
        request["input_url"] = args.input_url
    if args.from_lesson is not None:
        request["from_lesson"] = args.from_lesson
    if args.to_lesson is not None:
        request["to_lesson"] = args.to_lesson
    if args.limit is not None:
        request["limit"] = args.limit
    if args.generation_target:
        request["generation_target"] = args.generation_target
    if args.prompts_dir:
        request["prompts_dir"] = args.prompts_dir
    if args.resume_missing_from:
        request["resume_mode"] = "missing_only"
        request["existing_output_root"] = args.resume_missing_from
    return request


def select_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    lesson_numbers = {str(value) for value in (args.lesson_number or [])}
    task_ids = {str(value) for value in (args.task_id or [])}
    selected: list[dict[str, Any]] = []

    for task in tasks:
        task_id, lesson_number, _ = task_identity(task)
        if lesson_numbers and lesson_number not in lesson_numbers:
            continue
        if task_ids and task_id not in task_ids:
            continue
        lesson_as_int = parse_int(lesson_number)
        if args.from_lesson is not None and (lesson_as_int is None or lesson_as_int < args.from_lesson):
            continue
        if args.to_lesson is not None and (lesson_as_int is None or lesson_as_int > args.to_lesson):
            continue
        selected.append(task)

    if not selected:
        raise ValueError("No tasks matched selector.")
    return selected


def overall_status(entries: list[dict[str, Any]]) -> str:
    if any(entry.get("status") == "error" for entry in entries):
        return "has_errors"
    if any(entry.get("status") not in {"approved", "skipped", "completed_with_skips"} for entry in entries):
        return "has_failures"
    if any(entry.get("status") in {"skipped", "completed_with_skips"} for entry in entries):
        return "completed_with_skips"
    return "approved"


def successful_overall_status(status: str) -> bool:
    return status in {"approved", "completed_with_skips"}


def print_selected_tasks(tasks: list[dict[str, Any]]) -> None:
    print(
        json.dumps(
            [
                {
                    "task_id": task_identity(task)[0],
                    "lesson_number": task_identity(task)[1],
                    "lesson_title": task_identity(task)[2],
                    "course_level": resolve_course_level(task),
                    "resolved_profile": resolve_course_level(task),
                    "practice_task_count": practice_task_count(task),
                }
                for task in tasks
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_provider(value: str) -> ModelType:
    text = str(value or "").strip()
    for candidate in ModelType:
        if text.lower() == candidate.value.lower() or text.upper() == candidate.name:
            return candidate
    known = ", ".join(f"{item.name}/{item.value}" for item in ModelType)
    raise ValueError(f"Unknown provider {value!r}. Known providers: {known}")


def parse_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
