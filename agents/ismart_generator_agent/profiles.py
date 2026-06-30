from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from .contracts import IsmartGenerationConfig, default_workspace_dir


CourseLevel = Literal["basic", "advanced"]
DEFAULT_COURSE_LEVEL: CourseLevel = "basic"
ADVANCED_PROMPTS_DIRNAME = "prompts_skills_advanced"


def normalize_course_level(value: Any) -> CourseLevel | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = text.replace("ё", "е")
    if text in {"basic", "base", "базовый", "база", "python basic"}:
        return "basic"
    if text in {"advanced", "adv", "продвинутый", "продвинутыи", "python advanced"}:
        return "advanced"
    if "продвинут" in text or "advanced" in text:
        return "advanced"
    if "базов" in text or "basic" in text or "base" in text:
        return "basic"
    return None


def resolve_course_level(task: dict[str, Any]) -> CourseLevel:
    lesson = task.get("lesson") or {}
    course = task.get("course") or {}
    return (
        normalize_course_level(lesson.get("course_level"))
        or normalize_course_level(course.get("level"))
        or DEFAULT_COURSE_LEVEL
    )


def resolve_course_class(task: dict[str, Any]) -> str:
    lesson = task.get("lesson") or {}
    course = task.get("course") or {}
    module = task.get("module") or {}
    content = lesson.get("content") or {}
    candidates = [
        lesson.get("course_class"),
        lesson.get("class"),
        lesson.get("grade"),
        content.get("audience"),
        content.get("audience_specific"),
        content.get("for_grades_8_9"),
        course.get("class"),
        course.get("grade"),
        course.get("audience"),
        course.get("grades"),
        module.get("audience"),
    ]
    text = " ".join(str(value or "") for value in candidates).lower()
    if re.search(r"\b(10|11)\b", text):
        return "10"
    if re.search(r"\b(8|9)\b", text):
        return "8"
    return "unknown"


def resolve_lesson_number(task: dict[str, Any]) -> str:
    lesson = task.get("lesson") or {}
    value = str(lesson.get("lesson_number") or task.get("lesson_number") or "").strip()
    if not value:
        return "unknown"
    return re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", value).strip("_") or "unknown"


def langfuse_agent_name_for_task(task: dict[str, Any]) -> str:
    return (
        f"ismart_generator_{resolve_course_level(task)}"
        f"_{resolve_course_class(task)}"
        f"_lesson_{resolve_lesson_number(task)}"
    )


def langchain_config_for_task(base_config: dict[str, Any] | None, task: dict[str, Any]) -> dict[str, Any]:
    config = dict(base_config or {})
    course_level = resolve_course_level(task)
    course_class = resolve_course_class(task)
    agent_name = langfuse_agent_name_for_task(task)
    task_id = str(task.get("task_id") or "")
    lesson = task.get("lesson") or {}
    lesson_number = str(lesson.get("lesson_number") or "")
    tags = list(
        dict.fromkeys(
            [
                *(config.get("tags") or []),
                "ismart_generator",
                f"profile:{course_level}",
                f"class:{course_class}",
                f"lesson:{lesson_number or 'unknown'}",
            ]
        )
    )
    metadata = {
        **(config.get("metadata") or {}),
        "agent": agent_name,
        "resolved_profile": course_level,
        "course_level": course_level,
        "course_class": course_class,
        "task_id": task_id,
        "lesson_number": lesson_number,
    }
    config["run_name"] = agent_name
    config["tags"] = tags
    config["metadata"] = metadata
    return config


def prompts_dir_for_level(level: CourseLevel, *, base_prompts_dir: Path | None = None) -> Path:
    base = base_prompts_dir or default_workspace_dir() / "prompts_skills"
    if base.name == ADVANCED_PROMPTS_DIRNAME:
        base = base.parent / "prompts_skills"
    if level == "advanced":
        return base.parent / ADVANCED_PROMPTS_DIRNAME
    return base


def config_for_task_profile(
    config: IsmartGenerationConfig,
    task: dict[str, Any],
) -> IsmartGenerationConfig:
    level = resolve_course_level(task)
    prompts_dir = prompts_dir_for_level(level, base_prompts_dir=config.prompts_dir)
    return replace(
        config,
        course_level=level,
        prompts_dir=prompts_dir,
        langchain_config=langchain_config_for_task(config.langchain_config, task),
    )
