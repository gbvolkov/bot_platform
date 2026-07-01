from __future__ import annotations

from typing import Any

from .contracts import IsmartGenerationConfig, MaterialSpec
from .registry import get_material_spec


TEACHER_MATERIAL_FALSE_VALUES = {"", "н/п", "n/a", "нет", "false", "-"}


def teacher_material_required(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized not in TEACHER_MATERIAL_FALSE_VALUES


def positive_hours(lesson: dict[str, Any], key: str) -> bool:
    hours = lesson.get("hours") or {}
    value = hours.get(key, 0)
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def build_material_plan(
    task: dict[str, Any],
    config: IsmartGenerationConfig | None = None,
) -> list[MaterialSpec]:
    lesson = task.get("lesson") or {}
    flags = lesson.get("content_flags") or {}
    kinds: list[str] = []
    course_level = config.course_level if config else "basic"

    target = task.get("generation_target") or (config.generation_target if config else None)
    if target == "final_project":
        kinds.append("final_project")
        kinds.append("specification_qa")
        return [get_material_spec(kind, course_level=course_level) for kind in kinds]

    if flags.get("attestation"):
        kinds.extend(["intermediate", "mr_intermediate", "specification_qa"])
        return [get_material_spec(kind, course_level=course_level) for kind in kinds]

    if flags.get("theory") and positive_hours(lesson, "theory"):
        kinds.append("theory")
    if (flags.get("practice") or flags.get("project")) and positive_hours(lesson, "practice"):
        kinds.append("practice")

    teacher_materials = lesson.get("teacher_materials") or {}
    if "theory" in kinds and teacher_material_required(teacher_materials.get("theory")):
        kinds.append("mr_theory")
    if "practice" in kinds and teacher_material_required(teacher_materials.get("practice")):
        kinds.append("mr_practice")

    if flags.get("self_work") and positive_hours(lesson, "self_study"):
        kinds.append("self_work")
    if flags.get("current_control"):
        kinds.append("current_control")

    kinds.append("specification_qa")
    return [get_material_spec(kind, course_level=course_level) for kind in kinds]
