from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .context import task_identity
from .contracts import IsmartGenerationResult, ValidationResult
from .profiles import resolve_course_level


NO_PRACTICE_TASKS_SKIP_REASON = (
    "practice generation requested, but lesson.practice_tasks.l1/l2/l3 contain no task items"
)


def practice_task_count(task: dict[str, Any]) -> int:
    lesson = task.get("lesson") or {}
    practice_tasks = lesson.get("practice_tasks") or {}
    if not isinstance(practice_tasks, dict):
        return 0

    count = 0
    for level in ("l1", "l2", "l3"):
        items = practice_tasks.get(level) or []
        if isinstance(items, list):
            count += len(items)
    return count


def skip_reason_for_task(task: dict[str, Any]) -> str | None:
    if not _practice_generation_requested(task):
        return None
    if practice_task_count(task) > 0:
        return None
    return NO_PRACTICE_TASKS_SKIP_REASON


def build_skipped_result(
    *,
    task: dict[str, Any],
    output_dir: Path,
    reason: str,
) -> IsmartGenerationResult:
    task_id, lesson_number, lesson_title = task_identity(task)
    course_level = resolve_course_level(task)
    return IsmartGenerationResult(
        task_id=task_id,
        lesson_number=lesson_number,
        lesson_title=lesson_title,
        course_level=course_level,
        status="skipped",
        output_dir=str(output_dir),
        materials=[],
        package_validation=ValidationResult.ok(),
        reference_summary={},
        agents_called=[],
        prompt_files_used=[],
        skip_reason=reason,
    )


def _practice_generation_requested(task: dict[str, Any]) -> bool:
    lesson = task.get("lesson") or {}
    flags = lesson.get("content_flags") or {}
    if _truthy(flags.get("practice")):
        return True
    hours = lesson.get("hours") or {}
    return _positive_number(hours.get("practice"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "да", "истина"}


def _positive_number(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return value > 0
    text = str(value or "").strip().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return False
    try:
        return float(match.group(0)) > 0
    except ValueError:
        return False
