from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent import run_ismart_task
from .context import material_result_summary, task_identity
from .contracts import IsmartGenerationConfig, IsmartGenerationResult, JsonLLMClient
from .writer import safe_slug, write_batch_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ismart-materials-agent")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to task JSON, tasks JSON, or full course JSON.")
    source.add_argument("--input-url", help="URL to task JSON, tasks JSON, or full course JSON.")
    parser.add_argument("--output", required=True, help="Output root directory.")
    parser.add_argument("--task-id")
    parser.add_argument("--lesson-number")
    parser.add_argument("--model", default=os.getenv("UMK_LLM_MODEL", "gpt-5.2"))
    parser.add_argument("--base-url", default=os.getenv("UMK_LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max-generation-iterations", type=int, default=3)
    parser.add_argument("--max-package-repair-iterations", type=int, default=2)
    parser.add_argument("--max-reference-chars", type=int, default=0)
    parser.add_argument("--no-llm-validator", action="store_true")
    parser.add_argument("--generation-target")
    return parser


def load_payload_from_path(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_payload_from_url(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def tasks_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_ensure_task(item) for item in payload]
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object, an array, or {'tasks': [...]}.")
    if isinstance(payload.get("tasks"), list):
        return [_ensure_task(item) for item in payload["tasks"]]
    if _is_single_task(payload):
        return [_ensure_task(payload)]
    if "course" in payload and isinstance(payload.get("modules"), list):
        return _tasks_from_course(payload)
    raise ValueError("Could not recognize input JSON shape.")


def filter_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_id: str | None = None,
    lesson_number: str | None = None,
) -> list[dict[str, Any]]:
    result = []
    for task in tasks:
        current_task_id, current_lesson_number, _ = task_identity(task)
        if task_id is not None and current_task_id != task_id:
            continue
        if lesson_number is not None and current_lesson_number != str(lesson_number):
            continue
        result.append(task)
    if (task_id or lesson_number) and not result:
        raise ValueError("No tasks matched selector.")
    return result


def run_tasks(
    tasks: list[dict[str, Any]],
    *,
    config: IsmartGenerationConfig,
    client: JsonLLMClient | None = None,
) -> list[IsmartGenerationResult]:
    output_root = config.output_root
    if len(tasks) == 1:
        run_dir = output_root / f"run_{_timestamp()}_{safe_slug(task_identity(tasks[0])[0])}"
        return [run_ismart_task(tasks[0], config, client=client, run_dir=run_dir)]

    batch_dir = output_root / f"batch_{_timestamp()}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    results: list[IsmartGenerationResult] = []
    module_summaries: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task in tasks:
        task_id, lesson_number, _ = task_identity(task)
        module_key = str((task.get("module") or {}).get("title") or (task.get("lesson") or {}).get("module") or "")
        summaries = module_summaries.setdefault(module_key, {})
        run_dir = batch_dir / safe_slug(f"{lesson_number}-{task_id}")
        result = run_ismart_task(
            task,
            config,
            client=client,
            run_dir=run_dir,
            module_material_summaries=summaries,
        )
        results.append(result)
        summaries[lesson_number] = [material_result_summary(material) for material in result.materials]
    write_batch_manifest(batch_dir, results)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = load_payload_from_url(args.input_url) if args.input_url else load_payload_from_path(args.input)
        tasks = filter_tasks(
            tasks_from_payload(payload),
            task_id=args.task_id,
            lesson_number=args.lesson_number,
        )
        config = IsmartGenerationConfig(
            output_root=Path(args.output),
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_generation_iterations=args.max_generation_iterations,
            max_package_repair_iterations=args.max_package_repair_iterations,
            max_reference_chars=args.max_reference_chars,
            use_llm_validator=not args.no_llm_validator,
            generation_target=args.generation_target,
        )
        results = run_tasks(tasks, config=config)
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failures.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(result.output_dir)
    return 0 if all(result.status == "approved" for result in results) else 1


def _ensure_task(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not _is_single_task(value):
        raise ValueError("Each task must be an object with course, module, and lesson.")
    return value


def _is_single_task(value: dict[str, Any]) -> bool:
    return all(key in value for key in ("course", "module", "lesson"))


def _tasks_from_course(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    course = payload["course"]
    modules = payload.get("modules") or []
    for module in modules:
        for lesson in module.get("lessons") or []:
            tasks.append(
                {
                    "task_id": f"lesson-{lesson.get('lesson_number', len(tasks) + 1)}",
                    "course": course,
                    "module": module,
                    "lesson": lesson,
                    "modules": modules,
                    "markdown_references_base": payload.get("markdown_references_base"),
                }
            )
    return tasks


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
