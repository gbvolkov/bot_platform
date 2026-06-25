from __future__ import annotations

from typing import Any

from .contracts import MaterialResult, MaterialSpec, ReferenceBundle, ValidationResult
from .sources import compact_json


def select_references(spec: MaterialSpec, references: ReferenceBundle) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for field in spec.reference_fields:
        selected[field] = [
            document.to_public_json(include_content=True)
            for document in references.get(field, [])
        ]
    return selected


def task_identity(task: dict[str, Any]) -> tuple[str, str, str]:
    lesson = task.get("lesson") or {}
    lesson_number = str(lesson.get("lesson_number") or task.get("task_id") or "task")
    title = str(lesson.get("title") or lesson.get("topic") or lesson_number)
    task_id = str(task.get("task_id") or f"lesson-{lesson_number}")
    return task_id, lesson_number, title


def module_context_for_spec(task: dict[str, Any], spec: MaterialSpec) -> dict[str, Any]:
    module = task.get("module") or {}
    if spec.kind in {"intermediate", "final_project"}:
        return module
    return {key: value for key, value in module.items() if key != "lessons"}


def json_context_for_spec(task: dict[str, Any], spec: MaterialSpec) -> dict[str, Any]:
    lesson = task.get("lesson") or {}
    context: dict[str, Any] = {
        "course": task.get("course") or {},
        "module": module_context_for_spec(task, spec),
        "lesson": lesson,
        "json_field_labels": list(spec.json_field_labels),
    }
    if spec.kind == "final_project":
        context["modules"] = task.get("modules") or []
    return context


def material_result_summary(material: MaterialResult, *, max_content_chars: int = 12000) -> dict[str, Any]:
    return {
        "kind": material.kind,
        "type": material.material_type,
        "agent": material.agent_type,
        "status": material.status,
        "iterations": material.iterations,
        "validation_issues": list(material.validation_issues),
        "content": material.content[:max_content_chars],
    }


def build_generation_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    previous_content: str,
    previous_issues: list[str],
    module_material_summaries: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    return f"""
Сгенерируй материал УМК.

AGENT TYPE:
{spec.agent_type}

MATERIAL KIND:
{spec.kind}

MATERIAL TYPE:
{spec.material_type}

VALIDATOR KIND:
{spec.validator_kind}

PROMPT/SKILL FILES READ FROM prompts_skills:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item) for item in dependencies])}

MODULE MATERIAL SUMMARIES:
{compact_json(module_material_summaries or {})}

MATERIAL-SPECIFIC INSTRUCTION:
{spec.prompt_addendum}

PREVIOUS FAILED CONTENT, IF ANY:
{previous_content}

VALIDATOR ISSUES TO FIX, IF ANY:
{compact_json(previous_issues)}

STRICT OUTPUT CONTRACT:
Return only JSON:
{{
  "content": "<style>...</style><div class=\\"cc-lesson\\">...</div>",
  "agent_notes": ["short notes about used inputs"]
}}
""".strip()


def build_generator_system_prompt(spec: MaterialSpec) -> str:
    return (
        f"Ты специализированный агент {spec.agent_type}. "
        "Используй только переданные JSON-поля, существующие prompt/skill-файлы и Markdown references. "
        "Не используй внешние источники. Верни строго JSON."
    )


def build_validation_prompt(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    prompt_contents: dict[str, str],
    references: ReferenceBundle,
    dependencies: list[MaterialResult],
    content: str,
    rule_result: ValidationResult,
) -> str:
    return f"""
Проверь один материал. Не исправляй и не перегенерируй контент.

AGENT TYPE:
MaterialValidatorAgent

CHECKED MATERIAL KIND:
{spec.kind}

CHECKED MATERIAL TYPE:
{spec.material_type}

PROMPT/SKILL FILES FOR CHECKED MATERIAL:
{compact_json(prompt_contents)}

JSON CONTEXT:
{compact_json(json_context_for_spec(task, spec))}

MARKDOWN REFERENCES CONTENT:
{compact_json(select_references(spec, references))}

DEPENDENCY MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item) for item in dependencies])}

RULE VALIDATOR ISSUES:
{compact_json(rule_result.issues)}

MATERIAL HTML:
{content}

Return only JSON:
{{
  "approved": true,
  "issues": [],
  "fix_instructions": []
}}
If RULE VALIDATOR ISSUES is not empty, approved must be false.
""".strip()


def build_validator_system_prompt() -> str:
    return (
        "Ты MaterialValidatorAgent. Проверяешь материал по переданным prompt/skill-файлам, "
        "JSON и Markdown references. Не вызывай генерацию и не исправляй текст. Верни строго JSON."
    )


def build_package_validation_prompt(
    *,
    task: dict[str, Any],
    specs: list[MaterialSpec],
    materials: list[MaterialResult],
    rule_result: ValidationResult,
) -> str:
    return f"""
Проверь полный пакет материалов. Не перегенерируй контент.

TASK JSON:
{compact_json(task)}

EXPECTED MATERIAL SPECS:
{compact_json([{"kind": spec.kind, "type": spec.material_type, "agent": spec.agent_type} for spec in specs])}

MATERIALRESULT OBJECTS:
{compact_json([material_result_summary(item, max_content_chars=8000) for item in materials])}

RULE VALIDATOR ISSUES:
{compact_json(rule_result.issues)}

Return only JSON:
{{
  "approved": true,
  "issues": [],
  "fix_instructions": []
}}
""".strip()
