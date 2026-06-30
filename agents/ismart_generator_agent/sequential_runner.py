from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.utils import ModelType, get_llm

from .agent import load_payload_from_path_or_text, load_payload_from_url, tasks_from_payload
from .context import material_result_summary, task_identity
from .contracts import IsmartGenerationConfig, IsmartGenerationResult
from .observability import build_callback_handlers
from .profiles import resolve_course_level
from .runtime import run_ismart_task
from .subagents import build_subagent_registry
from .task_skip import build_skipped_result, practice_task_count, skip_reason_for_task
from .writer import safe_slug, write_json, write_task_output


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
    parser.add_argument("--prompts-dir", help="Prompt/skill directory. Defaults to iSMART workspace prompts_skills.")
    parser.add_argument("--run-name", help="Name of the run directory under --output.")
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
    payload = load_payload_from_url(args.input_url) if args.input_url else load_payload_from_path_or_text(args.input)
    tasks = select_tasks(tasks_from_payload(payload), args)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    if args.dry_run:
        print_selected_tasks(tasks)
        return 0

    provider = parse_provider(args.provider)
    callback_handlers = build_callback_handlers(f"ismart_generator_agent_{time.strftime('%Y%m%d%H%M')}")

    output_root = Path(args.output)
    batch_dir = output_root / (args.run_name or f"sequential_{timestamp()}")
    batch_dir.mkdir(parents=True, exist_ok=True)

    config = IsmartGenerationConfig(
        prompts_dir=Path(args.prompts_dir) if args.prompts_dir else IsmartGenerationConfig().prompts_dir,
        output_root=batch_dir,
        max_generation_iterations=args.max_generation_iterations,
        max_package_repair_iterations=args.max_package_repair_iterations,
        max_reference_chars=args.max_reference_chars,
        generation_target=args.generation_target,
        verbose=bool(args.verbose),
        langchain_config={"callbacks": callback_handlers},
    )

    print(
        json.dumps(
            {
                "event": "sequential.start",
                "output_dir": str(batch_dir),
                "task_count": len(tasks),
                "provider": provider.value,
                "model_mode": args.model_mode,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(batch_dir),
        "input": args.input,
        "input_url": args.input_url,
        "provider": provider.value,
        "model_mode": args.model_mode,
        "task_count": len(tasks),
        "tasks": [],
    }
    write_runner_manifest(batch_dir, manifest)

    module_summaries: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, task in enumerate(tasks, start=1):
        task_id, lesson_number, lesson_title = task_identity(task)
        course_level = resolve_course_level(task)
        run_dir = batch_dir / safe_slug(f"{index:03d}-{lesson_number}-{task_id}")
        module_key = str((task.get("module") or {}).get("title") or (task.get("lesson") or {}).get("module") or "")
        summaries = module_summaries.setdefault(module_key, {})
        skip_reason = skip_reason_for_task(task)
        if skip_reason:
            result = build_skipped_result(task=task, output_dir=run_dir, reason=skip_reason)
            write_task_output(
                result=result,
                output_dir=run_dir,
                validation_reports={"package": result.package_validation},
            )
            manifest["tasks"].append(manifest_entry_from_result(index, result))
            print(
                json.dumps(
                    {
                        "event": "task.skipped",
                        "index": index,
                        "task_id": task_id,
                        "lesson_number": lesson_number,
                        "lesson_title": lesson_title,
                        "course_level": course_level,
                        "resolved_profile": course_level,
                        "output_dir": str(run_dir),
                        "practice_task_count": practice_task_count(task),
                        "reason": skip_reason,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            write_runner_manifest(batch_dir, manifest)
            continue

        print(
            json.dumps(
                {
                    "event": "task.start",
                    "index": index,
                    "task_id": task_id,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "course_level": course_level,
                    "resolved_profile": course_level,
                    "output_dir": str(run_dir),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        try:
            task_llm = get_llm(
                model=args.model_mode,
                provider=provider.value,
                temperature=0.2,
                streaming=False,
            )
            subagents = build_subagent_registry(task_llm)
            if args.verbose:
                print(
                    json.dumps(
                        {
                            "event": "task.subagents.reset",
                            "index": index,
                            "task_id": task_id,
                            "lesson_number": lesson_number,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            result = run_ismart_task(
                task,
                config,
                subagents=subagents,
                run_dir=run_dir,
                module_material_summaries=summaries,
            )
            summaries[lesson_number] = [material_result_summary(material) for material in result.materials]
            entry = manifest_entry_from_result(index, result)
            manifest["tasks"].append(entry)
            print(
                json.dumps(
                    {
                        "event": "task.done",
                        "index": index,
                        "task_id": task_id,
                        "lesson_number": lesson_number,
                        "course_level": result.course_level,
                        "resolved_profile": result.course_level,
                        "status": result.status,
                        "output_dir": result.output_dir,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.stop_on_failure and result.status != "approved":
                manifest["status"] = "stopped_on_failure"
                write_runner_manifest(batch_dir, manifest)
                return 1
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
            write_runner_manifest(batch_dir, manifest)
            raise
        except Exception as exc:  # noqa: BLE001 - runner should isolate per-task failures.
            error_entry = {
                "index": index,
                "task_id": task_id,
                "lesson_number": lesson_number,
                "lesson_title": lesson_title,
                "course_level": course_level,
                "resolved_profile": course_level,
                "status": "error",
                "output_dir": str(run_dir),
                "error": str(exc),
            }
            manifest["tasks"].append(error_entry)
            write_json(run_dir / "error.json", error_entry)
            print(
                json.dumps(
                    {
                        "event": "task.error",
                        "index": index,
                        "task_id": task_id,
                        "lesson_number": lesson_number,
                        "course_level": course_level,
                        "resolved_profile": course_level,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )
            if args.stop_on_error:
                manifest["status"] = "stopped_on_error"
                write_runner_manifest(batch_dir, manifest)
                return 1
        finally:
            write_runner_manifest(batch_dir, manifest)

    manifest["status"] = overall_status(manifest["tasks"])
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_runner_manifest(batch_dir, manifest)
    print(
        json.dumps(
            {
                "event": "sequential.done",
                "status": manifest["status"],
                "output_dir": str(batch_dir),
                "manifest": str(batch_dir / "sequential_manifest.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if successful_overall_status(str(manifest["status"])) else 1


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


def manifest_entry_from_result(index: int, result: IsmartGenerationResult) -> dict[str, Any]:
    entry = {
        "index": index,
        "task_id": result.task_id,
        "lesson_number": result.lesson_number,
        "lesson_title": result.lesson_title,
        "course_level": result.course_level,
        "resolved_profile": result.course_level,
        "status": result.status,
        "output_dir": result.output_dir,
        "materials": [
            {
                "kind": material.kind,
                "status": material.status,
                "iterations": material.iterations,
                "validation_issues": list(material.validation_issues),
            }
            for material in result.materials
        ],
        "package_validation": {
            "approved": result.package_validation.approved,
            "issues": result.package_validation.issues,
        },
    }
    if result.skip_reason:
        entry["skip_reason"] = result.skip_reason
        entry["practice_task_count"] = 0
    return entry


def write_runner_manifest(batch_dir: Path, manifest: dict[str, Any]) -> None:
    update_runner_manifest_counts(manifest)
    write_json(batch_dir / "sequential_manifest.json", manifest)


def overall_status(entries: list[dict[str, Any]]) -> str:
    if any(entry.get("status") == "error" for entry in entries):
        return "has_errors"
    if any(entry.get("status") not in {"approved", "skipped"} for entry in entries):
        return "has_failures"
    if any(entry.get("status") == "skipped" for entry in entries):
        return "completed_with_skips"
    return "approved"


def update_runner_manifest_counts(manifest: dict[str, Any]) -> None:
    entries = manifest.get("tasks") or []
    manifest["generated_count"] = sum(1 for entry in entries if entry.get("status") not in {"skipped", "error"})
    manifest["approved_count"] = sum(1 for entry in entries if entry.get("status") == "approved")
    manifest["skipped_count"] = sum(1 for entry in entries if entry.get("status") == "skipped")
    manifest["error_count"] = sum(1 for entry in entries if entry.get("status") == "error")
    manifest["failed_count"] = sum(1 for entry in entries if entry.get("status") not in {"approved", "skipped", "error"})


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
                    "skip_reason": skip_reason_for_task(task),
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
