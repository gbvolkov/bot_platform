from __future__ import annotations

import html
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from .attempts import attempt_timestamp
from .context import build_validation_prompt, validation_result_summary
from .contracts import IsmartGenerationConfig, MaterialSpec, ReferenceDocument, ReferenceBundle, ValidationResult
from .profiles import prompts_dir_for_level
from .registry import get_material_spec
from .schemas import MaterialValidationDecision, PracticeGuidanceArtifact
from .sources import read_prompt_files
from .trace import TraceLogger
from .validators import RuleValidator
from .workers import StructuredSubagentInvoker, load_html_format_template
from .writer import safe_slug, write_json


PRACTICE_GUIDANCE_INPUT_VERSION = "practice_guidance_input_v1"
PRACTICE_GUIDANCE_KIND = "practice_guidance"
PRACTICE_GUIDANCE_AGENT = "PracticeGuidanceArtifactAgent"
REFERENCE_FIELDS = (
    "requirements",
    "reference_examples",
    "goals_and_tasks",
    "donor_materials",
    "template_descriptions",
)
STUDENT_PRACTICE_TASK_FIELDS = (
    "id",
    "template_id",
    "level",
    "task_type",
    "scenario",
    "student_condition",
    "starter_code",
    "faulty_code_display",
    "display_note",
    "input_requirements",
    "output_requirements",
    "runtime_tests",
    "tests",
    "manual_checks",
    "run_mode",
    "subtasks",
)
SOURCE_TEMPLATE_FIELDS = (
    "source_text",
    "skill_target",
    "invariants",
    "constraints",
    "test_policy",
)


def run_practice_guidance_postprocess(
    *,
    lesson_output_dir: Path,
    config: IsmartGenerationConfig,
    subagents: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = lesson_output_dir / "result.json"
    manifest_path = lesson_output_dir / "manifest.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found: {result_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    course_level = _course_level(result)
    task_config = replace(
        config,
        course_level=course_level,
        prompts_dir=prompts_dir_for_level(course_level, base_prompts_dir=config.prompts_dir),
    )
    spec = get_material_spec(PRACTICE_GUIDANCE_KIND, course_level=course_level)
    prompt_contents = read_prompt_files(task_config, spec.prompt_files)
    trace = TraceLogger(enabled=config.verbose)
    invoker = StructuredSubagentInvoker(
        subagents,
        trace=trace,
        langchain_config=config.langchain_config,
    )
    rule_validator = RuleValidator()
    attempts_dir = lesson_output_dir / "tmp" / PRACTICE_GUIDANCE_KIND
    attempts_dir.mkdir(parents=True, exist_ok=True)

    previous_artifact: dict[str, Any] | None = None
    previous_validation: ValidationResult | None = None
    previous_issues: list[str] = []
    latest_payload: dict[str, Any] | None = None

    for attempt in range(1, config.max_generation_iterations + 1):
        input_payload = build_practice_guidance_input(
            lesson_output_dir=lesson_output_dir,
            result=result,
            attempt=attempt,
            previous_artifact=previous_artifact,
            previous_validation=previous_validation,
        )
        latest_payload = input_payload
        prefix = f"{attempt_timestamp()}__attempt_{attempt:02d}__practice_guidance"
        write_json(attempts_dir / f"{prefix}.input.json", input_payload)

        artifact_model = invoker.invoke(
            PRACTICE_GUIDANCE_AGENT,
            system=build_practice_guidance_system_prompt(),
            prompt=build_practice_guidance_prompt(
                prompt_contents=prompt_contents,
                input_payload=input_payload,
                previous_issues=previous_issues,
            ),
            schema=PracticeGuidanceArtifact,
        )
        if not isinstance(artifact_model, PracticeGuidanceArtifact):
            raise TypeError(
                f"{PRACTICE_GUIDANCE_AGENT} returned {type(artifact_model)!r}, expected PracticeGuidanceArtifact"
            )
        artifact = _model_to_dict(artifact_model)
        write_json(attempts_dir / f"{prefix}.artifact.json", {"attempt": attempt, "practice_guidance_artifact": artifact})

        content = render_practice_guidance_html(artifact)
        artifacts = {
            "practice_guidance_input": input_payload,
            "practice_guidance_artifact": artifact,
        }
        task = _task_for_validation(result)
        rule_result = rule_validator.validate_material(content, spec, task)
        llm_result = _validate_practice_guidance(
            invoker=invoker,
            config=config,
            spec=spec,
            task=task,
            prompt_contents=prompt_contents,
            input_payload=input_payload,
            content=content,
            artifacts=artifacts,
            rule_result=rule_result,
        )
        validation = rule_result.merge(llm_result)
        write_json(
            attempts_dir / f"{prefix}.validation.json",
            {
                "attempt": attempt,
                "rule_validation": _validation_to_json(rule_result),
                "llm_validation": _validation_to_json(llm_result),
                "merged_validation": _validation_to_json(validation),
            },
        )

        if validation.approved:
            material = _practice_guidance_material_json(
                spec=spec,
                iterations=attempt,
                content=content,
                validation=validation,
                artifact=artifact,
                input_payload=input_payload,
            )
            filename = _practice_guidance_filename(lesson_output_dir, manifest)
            (lesson_output_dir / filename).write_text(content, encoding="utf-8")
            _update_result_json(result, material)
            _update_manifest_json(manifest, material, filename)
            write_json(result_path, result)
            write_json(manifest_path, manifest)
            write_json(
                lesson_output_dir / "validation_reports" / "practice-guidance.json",
                _validation_to_json(validation),
            )
            return {
                "status": "approved",
                "output_dir": str(lesson_output_dir),
                "file": filename,
                "attempts": attempt,
                "material": material,
            }

        previous_artifact = artifact
        previous_validation = validation
        previous_issues = list(validation.issues)

    failure_context = {
        "status": "failed",
        "output_dir": str(lesson_output_dir),
        "attempts": config.max_generation_iterations,
        "last_input": latest_payload,
        "issues": previous_issues,
    }
    write_json(lesson_output_dir / "tmp" / PRACTICE_GUIDANCE_KIND / "practice_guidance.failed.json", failure_context)
    raise RuntimeError(f"practice_guidance failed after {config.max_generation_iterations} attempts: {previous_issues}")


def build_practice_guidance_input(
    *,
    lesson_output_dir: Path,
    result: dict[str, Any],
    attempt: int,
    previous_artifact: dict[str, Any] | None,
    previous_validation: ValidationResult | None,
) -> dict[str, Any]:
    practice = _material_by_kind(result, "practice")
    if not practice or practice.get("status") != "approved":
        raise ValueError("approved practice material is required for practice_guidance")
    practice_artifacts = practice.get("generation_artifacts")
    if not isinstance(practice_artifacts, dict):
        raise ValueError("practice generation_artifacts are required for practice_guidance")
    practice_instances = practice_artifacts.get("practice_instances")
    if not isinstance(practice_instances, dict) or not isinstance(practice_instances.get("tasks"), list):
        raise ValueError("practice generation_artifacts.practice_instances.tasks are required for practice_guidance")

    source_warnings: list[dict[str, Any]] = []
    practice_templates = practice_artifacts.get("practice_templates") if isinstance(practice_artifacts, dict) else {}
    normalized_tasks = _normalize_practice_tasks_for_guidance(
        practice_instances=practice_instances,
        practice_templates=practice_templates if isinstance(practice_templates, dict) else {},
        source_warnings=source_warnings,
    )
    theory = _material_by_kind(result, "theory")
    raw_theory_status = str((theory or {}).get("status") or "").strip()
    theory_status = "approved" if raw_theory_status == "approved" else ("not_approved" if theory else "missing")
    theory_sections = _theory_public_sections(theory.get("content", "") if theory and theory_status == "approved" else "")
    references = _load_reference_contents(result, source_warnings=source_warnings)

    return {
        "input_version": PRACTICE_GUIDANCE_INPUT_VERSION,
        "source_output_dir": str(lesson_output_dir),
        "task_meta": {
            "task_id": str(result.get("task_id") or ""),
            "lesson_number": str(result.get("lesson_number") or ""),
            "lesson_title": str(result.get("lesson_title") or ""),
            "course_level": _course_level(result),
            "resolved_profile": _course_level(result),
            "status": str(result.get("status") or ""),
        },
        "approved_materials": {
            "practice": {
                "status": "approved",
                "material_type": str(practice.get("type") or ""),
                "prompt_files": list(practice.get("prompt_files") or []),
                "practice_templates": practice_templates if isinstance(practice_templates, dict) else {"tasks": []},
                "practice_instances": _filter_practice_instances_for_guidance(practice_instances),
            },
            "theory": {
                "status": theory_status if theory else "missing",
                "material_type": str((theory or {}).get("type") or "Материалы занятия — теория"),
                "public_sections": theory_sections,
            },
        },
        "practice_tasks": normalized_tasks,
        "theory_brief_source": {
            "available": bool(theory_sections),
            "source": "approved_theory_html_to_text" if theory_sections else "missing",
            "sections": theory_sections,
        },
        "references": references,
        "source_warnings": source_warnings,
        "retry_context": {
            "attempt": attempt,
            "previous_artifact": previous_artifact,
            "previous_validation_issues": list(previous_validation.issues) if previous_validation else [],
            "previous_passed_blocks": list(previous_validation.passed_blocks) if previous_validation else [],
        },
    }


def build_practice_guidance_system_prompt() -> str:
    return (
        "You are PracticeGuidanceArtifactAgent. Build structured practice guidance from PracticeGuidanceInput. "
        "Use only the provided structured input, prompt/skill files, and reference content. "
        "Do not use raw HTML or external sources. Return only the configured PracticeGuidanceArtifact structured output."
    )


def build_practice_guidance_prompt(
    *,
    prompt_contents: dict[str, str],
    input_payload: dict[str, Any],
    previous_issues: list[str],
) -> str:
    return f"""
Build "Указания к практической работе" as a separate structured artifact.

PROMPT/SKILL FILES FOR THE RESOLVED PROFILE:
{json.dumps(prompt_contents, ensure_ascii=False, indent=2)}

PRACTICE_GUIDANCE_INPUT JSON:
{json.dumps(input_payload, ensure_ascii=False, indent=2)}

PREVIOUS VALIDATION ISSUES:
{json.dumps(previous_issues, ensure_ascii=False, indent=2)}

REQUIREMENTS:
- Return PracticeGuidanceArtifact structured output only.
- Use practice_tasks as the source of module tasks.
- Do not change, replace, merge, split, or reconstruct module tasks.
- Build stages by level and methodical similarity; source_task_ids must reference practice_tasks ids.
- Create a worked analogous example for each stage. It must be similar by method but different from the module tasks.
- Do not reveal keys, corrected code, internal answer/explanation fields, raw field names, JSON/process wording, SHA, or local paths.
- If exact reference values are unavailable, add a requires_check item instead of inventing them.
""".strip()


def render_practice_guidance_html(artifact: dict[str, Any]) -> str:
    template = load_html_format_template()
    header = artifact.get("header") if isinstance(artifact.get("header"), dict) else {}
    goals = artifact.get("goals") if isinstance(artifact.get("goals"), dict) else {}
    theory = artifact.get("theory_brief") if isinstance(artifact.get("theory_brief"), dict) else {}
    guidance = artifact.get("methodical_guidance") if isinstance(artifact.get("methodical_guidance"), dict) else {}
    result_requirements = (
        artifact.get("result_requirements") if isinstance(artifact.get("result_requirements"), dict) else {}
    )

    parts = [
        f"<h1>{_e(header.get('work_title') or 'Указания к практической работе')}</h1>",
        f"<p><strong>Тема:</strong> {_e(header.get('topic') or '')}</p>",
    ]
    if header.get("audience"):
        parts.append(f"<p><strong>Целевая аудитория:</strong> {_e(header.get('audience'))}</p>")

    parts.append("<h2>1. Цели и задачи выполнения практического задания</h2>")
    if goals.get("goal"):
        parts.append(f"<p><strong>Цель работы:</strong> {_e(goals.get('goal'))}</p>")
    parts.append(_list_html("Задачи работы", goals.get("objectives")))

    parts.append("<h2>2. Краткое изложение теоретических материалов</h2>")
    if theory.get("intro"):
        parts.append(f"<p>{_e(theory.get('intro'))}</p>")
    for section in _list_of_dicts(theory.get("sections")):
        parts.append(f"<h3>{_e(section.get('title'))}</h3>")
        parts.append(f"<p>{_e(section.get('content'))}</p>")

    parts.append("<h2>3. Подробные методические указания по выполнению</h2>")
    if guidance.get("problem_statement"):
        parts.append(f"<p>{_e(guidance.get('problem_statement'))}</p>")
    if guidance.get("environment"):
        parts.append(f"<p><strong>Рабочая среда:</strong> {_e(guidance.get('environment'))}</p>")
    before = guidance.get("before_start") if isinstance(guidance.get("before_start"), dict) else {}
    if before:
        parts.append("<h3>Перед началом работы</h3>")
        parts.append(_list_html("", before.get("steps")))
        if before.get("checkpoint"):
            parts.append(f"<p><strong>Контрольный вывод:</strong> {_e(before.get('checkpoint'))}</p>")
    for index, stage in enumerate(_list_of_dicts(guidance.get("stages")), start=1):
        parts.append(f"<h3>Этап {index}: {_e(stage.get('title') or stage.get('id') or index)}</h3>")
        if stage.get("stage_goal"):
            parts.append(f"<p><strong>Задача этапа:</strong> {_e(stage.get('stage_goal'))}</p>")
        parts.append(_numbered_list_html("Алгоритм выполнения", stage.get("algorithm_steps")))
        example = stage.get("worked_example") if isinstance(stage.get("worked_example"), dict) else {}
        if example:
            parts.append("<h4>Разбор аналогичного задания</h4>")
            if example.get("task_statement"):
                parts.append(f"<p>{_e(example.get('task_statement'))}</p>")
            if example.get("code_cell"):
                parts.append(
                    "<p><strong>Ячейка для выполнения</strong></p>"
                    f"<pre><code>{_code(example.get('code_cell'))}</code></pre>"
                )
            for label, key in (
                ("Комментарий", "commentary"),
                ("Анализ вывода", "output_analysis"),
                ("Интерпретация", "interpretation"),
            ):
                if example.get(key):
                    parts.append(f"<p><strong>{label}:</strong> {_e(example.get(key))}</p>")
            parts.append(_list_html("Реперные значения", example.get("reference_values")))
        module_tasks = _list_of_dicts(stage.get("module_tasks"))
        if module_tasks:
            parts.append("<h4>Задания модуля</h4>")
            for task in module_tasks:
                parts.append(f"<section><h5>{_e(task.get('task_id'))}</h5>")
                if task.get("student_condition"):
                    parts.append(f"<p>{_e(task.get('student_condition'))}</p>")
                if task.get("code_cell"):
                    parts.append(
                        "<p><strong>Ячейка для вашего кода</strong></p>"
                        f"<pre><code>{_code(task.get('code_cell'))}</code></pre>"
                    )
                if task.get("input_requirements"):
                    parts.append(f"<p><strong>Входные данные:</strong> {_e(task.get('input_requirements'))}</p>")
                if task.get("output_requirements"):
                    parts.append(f"<p><strong>Требование к результату:</strong> {_e(task.get('output_requirements'))}</p>")
                parts.append(_list_html("Проверка", task.get("checks")))
                parts.append(_list_html("Ручная проверка", task.get("manual_checks")))
                parts.append("</section>")
    parts.append("<p><strong>Проверка:</strong> сопоставьте выполненные задания с требованиями к результату.</p>")

    parts.append("<h2>4. Требования к оформлению результата</h2>")
    if result_requirements.get("deliverable"):
        parts.append(f"<p>{_e(result_requirements.get('deliverable'))}</p>")
    parts.append(_list_html("Критерии оценки", result_requirements.get("criteria")))

    parts.append("<h2>5. Контрольные вопросы для самопроверки</h2>")
    parts.append(_numbered_list_html("", artifact.get("self_check_questions")))

    requires_check = artifact.get("requires_check") if isinstance(artifact.get("requires_check"), list) else []
    if requires_check:
        parts.append("<h2>Требует проверки / уточнения</h2>")
        parts.append(_list_html("", requires_check))

    return template.render("".join(parts))


def _validate_practice_guidance(
    *,
    invoker: StructuredSubagentInvoker,
    config: IsmartGenerationConfig,
    spec: MaterialSpec,
    task: dict[str, Any],
    prompt_contents: dict[str, str],
    input_payload: dict[str, Any],
    content: str,
    artifacts: dict[str, Any],
    rule_result: ValidationResult,
) -> ValidationResult:
    if not config.use_llm_validator:
        return ValidationResult.ok()
    references = _reference_bundle_from_input(input_payload)
    prompt = build_validation_prompt(
        task=task,
        spec=spec,
        prompt_contents=prompt_contents,
        references=references,
        dependencies=[],
        content=content,
        rule_result=rule_result,
        generation_artifacts=artifacts,
    )
    decision = invoker.invoke(
        "MaterialValidatorAgent",
        system=(
            "You are MaterialValidatorAgent. Validate the structured practice_guidance artifact. "
            "Do not validate raw HTML and do not ask to inspect raw HTML."
        ),
        prompt=prompt,
        schema=MaterialValidationDecision,
    )
    if not isinstance(decision, MaterialValidationDecision):
        raise TypeError(f"MaterialValidatorAgent returned {type(decision)!r}, expected MaterialValidationDecision")
    return ValidationResult(
        approved=bool(decision.approved),
        issues=[str(item) for item in decision.issues],
        fix_instructions=[str(item) for item in decision.fix_instructions],
        issues_by_block=[_model_to_dict(item) for item in decision.issues_by_block],
        passed_blocks=[_model_to_dict(item) for item in decision.passed_blocks],
    )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _course_level(result: dict[str, Any]) -> str:
    value = str(result.get("resolved_profile") or result.get("course_level") or "basic").strip().lower()
    return "advanced" if value == "advanced" else "basic"


def _material_by_kind(result: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for material in result.get("materials") or []:
        if isinstance(material, dict) and material.get("kind") == kind:
            return material
    return None


def _filter_practice_instances_for_guidance(instances: dict[str, Any]) -> dict[str, Any]:
    return {
        "lesson_goal": instances.get("lesson_goal") or "",
        "lesson_objectives": list(instances.get("lesson_objectives") or []),
        "tasks": [
            {key: task.get(key) for key in STUDENT_PRACTICE_TASK_FIELDS if key in task}
            for task in _list_of_dicts(instances.get("tasks"))
        ],
    }


def _normalize_practice_tasks_for_guidance(
    *,
    practice_instances: dict[str, Any],
    practice_templates: dict[str, Any],
    source_warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    templates_by_id = {
        str(item.get("id")): item
        for item in _list_of_dicts(practice_templates.get("tasks"))
        if item.get("id")
    }
    tasks: list[dict[str, Any]] = []
    for item in _list_of_dicts(practice_instances.get("tasks")):
        task = {key: item.get(key) for key in STUDENT_PRACTICE_TASK_FIELDS if key in item}
        if item.get("faulty_code") and not item.get("faulty_code_display"):
            task.pop("faulty_code", None)
            source_warnings.append(
                {
                    "code": "raw_faulty_code_omitted",
                    "task_id": str(item.get("id") or ""),
                    "message": "raw faulty_code was not passed to practice_guidance input",
                }
            )
        template = templates_by_id.get(str(item.get("template_id") or item.get("id") or ""))
        if template:
            task["source_template"] = {
                key: template.get(key)
                for key in SOURCE_TEMPLATE_FIELDS
                if key in template
            }
        tasks.append(task)
    return tasks


def _theory_public_sections(content: str) -> list[dict[str, str]]:
    if not content:
        return []
    text = re.sub(r"<style\b[^>]*>.*?</style>", "", content, flags=re.I | re.S)
    text = re.sub(r"<script\b[^>]*>.*?</script>", "", text, flags=re.I | re.S)
    matches = list(re.finditer(r"<h([1-3])\b[^>]*>(.*?)</h\1>", text, flags=re.I | re.S))
    if not matches:
        plain = _html_to_text(text)
        return [{"heading": "Теория", "text": plain, "source_material_kind": "theory"}] if plain else []
    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        heading = _html_to_text(match.group(2))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = _html_to_text(text[start:end])
        if heading and body:
            sections.append({"heading": heading, "text": body, "source_material_kind": "theory"})
    return sections


def _load_reference_contents(result: dict[str, Any], *, source_warnings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {field: [] for field in REFERENCE_FIELDS}
    references = result.get("references") if isinstance(result.get("references"), dict) else {}
    for field in REFERENCE_FIELDS:
        for item in references.get(field) or []:
            if not isinstance(item, dict):
                continue
            source_name = _reference_source_name(item)
            resolved_path = str(item.get("resolved_path") or "").strip()
            path = Path(resolved_path) if resolved_path else None
            if path and path.exists() and path.is_file():
                output[field].append(
                    {
                        "field": field,
                        "source_name": source_name,
                        "resolved": True,
                        "truncated": bool(item.get("truncated")),
                        "content": path.read_text(encoding="utf-8"),
                    }
                )
            else:
                output[field].append(
                    {
                        "field": field,
                        "source_name": source_name,
                        "resolved": False,
                        "truncated": bool(item.get("truncated")),
                        "content": "",
                    }
                )
                source_warnings.append({"code": "missing_reference", "field": field, "source_name": source_name})
    return output


def _reference_source_name(item: dict[str, Any]) -> str:
    for key in ("path", "resolved_path"):
        value = str(item.get(key) or "").strip()
        if value:
            return Path(value).stem
    return "reference"


def _reference_bundle_from_input(input_payload: dict[str, Any]) -> ReferenceBundle:
    bundle: ReferenceBundle = {}
    references = input_payload.get("references") if isinstance(input_payload.get("references"), dict) else {}
    for field in REFERENCE_FIELDS:
        documents: list[ReferenceDocument] = []
        for index, item in enumerate(references.get(field) or [], start=1):
            if not isinstance(item, dict) or not item.get("content"):
                continue
            name = str(item.get("source_name") or f"{field}_{index}")
            documents.append(
                ReferenceDocument(
                    field=field,
                    path=name,
                    resolved_path="",
                    sha="",
                    truncated=bool(item.get("truncated")),
                    content=str(item.get("content") or ""),
                )
            )
        bundle[field] = documents
    return bundle


def _task_for_validation(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(result.get("task_id") or ""),
        "course": {"level": _course_level(result)},
        "module": {},
        "lesson": {
            "lesson_number": str(result.get("lesson_number") or ""),
            "title": str(result.get("lesson_title") or ""),
            "course_level": _course_level(result),
            "content_flags": {"practice": True},
        },
    }


def _practice_guidance_material_json(
    *,
    spec: MaterialSpec,
    iterations: int,
    content: str,
    validation: ValidationResult,
    artifact: dict[str, Any],
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": spec.kind,
        "type": spec.material_type,
        "agent": spec.agent_type,
        "status": "approved",
        "iterations": iterations,
        "prompt_files": list(spec.prompt_files),
        "validation_issues": [],
        "validation_issues_by_block": validation.issues_by_block,
        "validation_passed_blocks": validation.passed_blocks,
        "agent_notes": list(artifact.get("agent_notes") or []),
        "controller_called": False,
        "generation_artifacts": {
            "practice_guidance_input": input_payload,
            "practice_guidance_artifact": artifact,
        },
        "content": content,
    }


def _practice_guidance_filename(lesson_output_dir: Path, manifest: dict[str, Any]) -> str:
    for material in manifest.get("materials") or []:
        if isinstance(material, dict) and material.get("kind") == PRACTICE_GUIDANCE_KIND and material.get("file"):
            return str(material["file"])
    max_index = 0
    for path in lesson_output_dir.glob("*.html"):
        match = re.match(r"^(\d+)_", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"{max_index + 1:02d}_{safe_slug(PRACTICE_GUIDANCE_KIND)}.html"


def _update_result_json(result: dict[str, Any], material: dict[str, Any]) -> None:
    materials = [item for item in result.get("materials") or [] if not (isinstance(item, dict) and item.get("kind") == PRACTICE_GUIDANCE_KIND)]
    materials.append(material)
    result["materials"] = materials
    agents = list(result.get("agents_called") or [])
    for agent in (PRACTICE_GUIDANCE_AGENT, "PracticeGuidanceAgent"):
        if agent not in agents:
            agents.append(agent)
    result["agents_called"] = agents
    prompt_files = list(result.get("prompt_files_used") or [])
    for item in material.get("prompt_files") or []:
        if item not in prompt_files:
            prompt_files.append(item)
    result["prompt_files_used"] = prompt_files


def _update_manifest_json(manifest: dict[str, Any], material: dict[str, Any], filename: str) -> None:
    materials = [
        item
        for item in manifest.get("materials") or []
        if not (isinstance(item, dict) and item.get("kind") == PRACTICE_GUIDANCE_KIND)
    ]
    manifest_material = {key: value for key, value in material.items() if key != "content"}
    manifest_material["file"] = filename
    materials.append(manifest_material)
    manifest["materials"] = materials
    agents = list(manifest.get("agents_called") or [])
    for agent in (PRACTICE_GUIDANCE_AGENT, "PracticeGuidanceAgent"):
        if agent not in agents:
            agents.append(agent)
    manifest["agents_called"] = agents
    prompt_files = list(manifest.get("prompt_files_used") or [])
    for item in material.get("prompt_files") or []:
        if item not in prompt_files:
            prompt_files.append(item)
    manifest["prompt_files_used"] = prompt_files


def _model_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _validation_to_json(validation: ValidationResult | None) -> dict[str, Any] | None:
    if validation is None:
        return None
    return validation_result_summary(validation)


def _html_to_text(value: str) -> str:
    text = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", value, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _list_html(title: str, values: Any) -> str:
    items = [str(item).strip() for item in values or [] if str(item).strip()] if isinstance(values, list) else []
    if not items:
        return ""
    title_html = f"<p><strong>{_e(title)}:</strong></p>" if title else ""
    return title_html + "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ul>"


def _numbered_list_html(title: str, values: Any) -> str:
    items = [str(item).strip() for item in values or [] if str(item).strip()] if isinstance(values, list) else []
    if not items:
        return ""
    title_html = f"<p><strong>{_e(title)}:</strong></p>" if title else ""
    return title_html + "<ol>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ol>"


def _e(value: Any) -> str:
    return html.escape(str(value or ""))


def _code(value: Any) -> str:
    return html.escape(str(value or ""))
