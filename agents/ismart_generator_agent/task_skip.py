from __future__ import annotations

import re
from typing import Any

from .contracts import MaterialResult, MaterialSpec, MaterialStatus


NO_PRACTICE_TASKS_SKIP_REASON = (
    "practice generation requested, but lesson.practice_tasks.l1/l2/l3 contain no task items"
)
PRACTICE_DEPENDENCY_SKIPPED_REASON = "practice guidance skipped because practice material was skipped"
SKIPPED_MATERIAL_STATUSES = {"skipped", "skipped_dependency"}
PROJECT_CONTENT_FIELDS = (
    "general",
    "audience_specific",
    "for_grades_8_9",
    "for_grades_10_11",
    "project",
    "project_description",
    "intro",
    "description",
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


def practice_material_skip_reason(task: dict[str, Any], spec: MaterialSpec) -> str | None:
    if spec.kind != "practice":
        return None
    if not _practice_generation_requested(task):
        return None
    if practice_task_count(task) > 0:
        return None
    if project_practice_source_text(task):
        return None
    return NO_PRACTICE_TASKS_SKIP_REASON


def project_practice_source_text(task: dict[str, Any]) -> str:
    if not _project_generation_requested(task):
        return ""
    lesson = task.get("lesson") or {}
    content = lesson.get("content") or {}
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, dict):
        return ""

    sections: list[str] = []
    seen: set[str] = set()
    for field in PROJECT_CONTENT_FIELDS:
        text = str(content.get(field) or "").strip()
        if text and text not in seen:
            sections.append(f"{field}:\n{text}")
            seen.add(text)
    if not sections:
        for field, value in content.items():
            if field == "audience":
                continue
            text = str(value or "").strip()
            if text and text not in seen:
                sections.append(f"{field}:\n{text}")
                seen.add(text)
    return "\n\n".join(sections)


def dependency_skip_reason(spec: MaterialSpec, dependency_results: list[MaterialResult]) -> str | None:
    if spec.kind != "mr_practice":
        return None
    for dependency in dependency_results:
        if dependency.kind == "practice" and dependency.status in SKIPPED_MATERIAL_STATUSES:
            return PRACTICE_DEPENDENCY_SKIPPED_REASON
    return None


def build_skipped_material(
    *,
    spec: MaterialSpec,
    status: MaterialStatus,
    reason: str,
    dependency_results: list[MaterialResult] | None = None,
) -> MaterialResult:
    dependencies = [
        {"kind": item.kind, "status": item.status}
        for item in (dependency_results or [])
        if item.status in SKIPPED_MATERIAL_STATUSES
    ]
    artifacts: dict[str, Any] = {"skip_reason": reason}
    if dependencies:
        artifacts["skipped_dependencies"] = dependencies
    return MaterialResult(
        kind=spec.kind,
        material_type=spec.material_type,
        agent_type=spec.agent_type,
        status=status,
        iterations=0,
        content="",
        prompt_files=spec.prompt_files,
        agent_notes=[reason],
        generation_artifacts=artifacts,
    )


def _practice_generation_requested(task: dict[str, Any]) -> bool:
    lesson = task.get("lesson") or {}
    flags = lesson.get("content_flags") or {}
    if _truthy(flags.get("practice")):
        return True
    hours = lesson.get("hours") or {}
    return _positive_number(hours.get("practice"))


def _project_generation_requested(task: dict[str, Any]) -> bool:
    lesson = task.get("lesson") or {}
    flags = lesson.get("content_flags") or {}
    if not _truthy(flags.get("project")):
        return False
    hours = lesson.get("hours") or {}
    return _positive_number(hours.get("practice")) or _truthy(flags.get("practice"))


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
