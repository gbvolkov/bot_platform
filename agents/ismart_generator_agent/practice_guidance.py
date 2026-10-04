from __future__ import annotations

import html
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from .attempts import attempt_timestamp
from .context import (
    build_validation_controller_prompt,
    build_validation_controller_system_prompt,
    build_validation_prompt,
    validation_result_summary,
)
from .context import task_identity
from .contracts import (
    IsmartGenerationConfig,
    MaterialResult,
    MaterialSpec,
    ReferenceDocument,
    ReferenceBundle,
    ValidationResult,
)
from .profiles import prompts_dir_for_level, resolve_course_class
from .python_sandbox import DisabledPythonSandbox, PythonSandbox
from .registry import get_material_spec
from .schemas import MaterialValidationDecision, PracticeGuidanceArtifact, ValidationControllerDecision
from .sources import reference_summary
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
    python_sandbox = PythonSandbox(task_config) if task_config.use_python_sandbox else DisabledPythonSandbox()

    for attempt in range(1, config.max_generation_iterations + 1):
        input_payload = build_practice_guidance_input(
            lesson_output_dir=lesson_output_dir,
            result=result,
            attempt=attempt,
            previous_artifact=previous_artifact,
            previous_validation=previous_validation,
            previous_lessons_context=[],
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
        execution_evidence = _build_practice_guidance_execution_evidence(
            input_payload=input_payload,
            artifact=artifact,
            config=task_config,
            python_sandbox=python_sandbox,
        )
        write_json(
            attempts_dir / f"{prefix}.execution_evidence.json",
            {"attempt": attempt, "technical_evidence": execution_evidence},
        )
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
            technical_evidence=execution_evidence,
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
            python_sandbox.close()
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
    python_sandbox.close()
    raise RuntimeError(f"practice_guidance failed after {config.max_generation_iterations} attempts: {previous_issues}")


def run_practice_guidance_material(
    *,
    task: dict[str, Any],
    spec: MaterialSpec,
    config: IsmartGenerationConfig,
    subagents: Mapping[str, Any],
    references: ReferenceBundle,
    materials: list[MaterialResult],
    output_dir: Path,
    attempts_dir: Path,
    trace: TraceLogger | None = None,
    rule_validator: RuleValidator | None = None,
    python_sandbox: PythonSandbox | DisabledPythonSandbox | None = None,
) -> MaterialResult:
    trace = trace or TraceLogger()
    trace.log(
        "worker.start",
        kind=spec.kind,
        agent=PRACTICE_GUIDANCE_AGENT,
        dependency_kinds=list(spec.dependency_kinds),
        dependencies=[{"kind": item.kind, "status": item.status} for item in materials if item.kind in spec.dependency_kinds],
    )
    prompt_contents = read_prompt_files(config, spec.prompt_files)
    trace.log("worker.prompt_files_loaded", kind=spec.kind, prompt_files=list(spec.prompt_files))

    invoker = StructuredSubagentInvoker(
        subagents,
        trace=trace,
        langchain_config=config.langchain_config,
    )
    validator = rule_validator or RuleValidator()
    python_sandbox = python_sandbox or DisabledPythonSandbox()
    material_attempts_dir = attempts_dir / PRACTICE_GUIDANCE_KIND
    material_attempts_dir.mkdir(parents=True, exist_ok=True)

    previous_artifact: dict[str, Any] | None = None
    previous_validation: ValidationResult | None = None
    previous_issues: list[str] = []
    last_content = ""
    last_artifact: dict[str, Any] = {}
    last_input: dict[str, Any] = {}
    last_validation: ValidationResult | None = None
    last_rule_result: ValidationResult | None = None
    last_llm_result: ValidationResult | None = None

    for attempt in range(1, config.max_generation_iterations + 1):
        trace.log(
            "worker.attempt.start",
            kind=spec.kind,
            attempt=attempt,
            max_attempts=config.max_generation_iterations,
            previous_content_chars=len(last_content),
            previous_issues_count=len(previous_issues),
        )
        result_payload = _in_memory_result_payload(
            task=task,
            course_level=config.course_level,
            output_dir=output_dir,
            materials=materials,
            references=references,
        )
        input_payload = build_practice_guidance_input(
            lesson_output_dir=output_dir,
            result=result_payload,
            attempt=attempt,
            previous_artifact=previous_artifact,
            previous_validation=previous_validation,
            previous_lessons_context=config.previous_lessons_context,
        )
        last_input = input_payload
        prefix = f"{attempt_timestamp()}__attempt_{attempt:02d}__practice_guidance"
        write_json(material_attempts_dir / f"{prefix}.input.json", input_payload)

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
        last_artifact = artifact
        write_json(
            material_attempts_dir / f"{prefix}.artifact.json",
            {"attempt": attempt, "practice_guidance_artifact": artifact},
        )

        content = render_practice_guidance_html(artifact)
        last_content = content
        artifacts = {
            "practice_guidance_input": input_payload,
            "practice_guidance_artifact": artifact,
        }
        execution_evidence = _build_practice_guidance_execution_evidence(
            input_payload=input_payload,
            artifact=artifact,
            config=config,
            python_sandbox=python_sandbox,
        )
        write_json(
            material_attempts_dir / f"{prefix}.execution_evidence.json",
            {"attempt": attempt, "technical_evidence": execution_evidence},
        )
        trace.log(
            "worker.execution_evidence.done",
            kind=spec.kind,
            attempt=attempt,
            enabled=execution_evidence.get("enabled"),
            task_runs=len(execution_evidence.get("practice_task_runs") or []),
            artifact_runs=len(execution_evidence.get("artifact_module_task_runs") or []),
        )
        rule_result = validator.validate_material(content, spec, task)
        last_rule_result = rule_result
        trace.log(
            "worker.rule_validation.done",
            kind=spec.kind,
            attempt=attempt,
            approved=rule_result.approved,
            issues=rule_result.issues,
        )
        llm_result = _validate_practice_guidance(
            invoker=invoker,
            config=config,
            spec=spec,
            task=task,
            prompt_contents=prompt_contents,
            input_payload=input_payload,
            content=content,
            artifacts=artifacts,
            technical_evidence=execution_evidence,
            rule_result=rule_result,
        )
        last_llm_result = llm_result
        validation = rule_result.merge(llm_result)
        last_validation = validation
        trace.log(
            "worker.validation.merged",
            kind=spec.kind,
            attempt=attempt,
            approved=validation.approved,
            issues=validation.issues,
        )
        write_json(
            material_attempts_dir / f"{prefix}.validation.json",
            {
                "attempt": attempt,
                "rule_validation": _validation_to_json(rule_result),
                "llm_validation": _validation_to_json(llm_result),
                "merged_validation": _validation_to_json(validation),
            },
        )

        if validation.approved:
            trace.log("worker.approved", kind=spec.kind, attempt=attempt, content_chars=len(content))
            return MaterialResult(
                kind=spec.kind,
                material_type=spec.material_type,
                agent_type=PRACTICE_GUIDANCE_AGENT,
                status="approved",
                iterations=attempt,
                content=content,
                prompt_files=spec.prompt_files,
                validation_issues_by_block=validation.issues_by_block,
                validation_passed_blocks=validation.passed_blocks,
                agent_notes=[str(item) for item in artifact.get("agent_notes") or []],
                generation_artifacts=artifacts,
            )

        previous_artifact = artifact
        previous_validation = validation
        previous_issues = list(validation.issues)
        if attempt < config.max_generation_iterations:
            trace.log("worker.retry", kind=spec.kind, next_attempt=attempt + 1, issues=previous_issues)

    controller_decision = _review_practice_guidance_validation_failure(
        invoker=invoker,
        config=config,
        spec=spec,
        task=task,
        prompt_contents=prompt_contents,
        input_payload=last_input,
        content=last_content,
        artifacts={
            "practice_guidance_input": last_input,
            "practice_guidance_artifact": last_artifact,
        },
        technical_evidence=_build_practice_guidance_execution_evidence(
            input_payload=last_input,
            artifact=last_artifact,
            config=config,
            python_sandbox=python_sandbox,
        ),
        rule_result=last_rule_result,
        llm_result=last_llm_result,
        validation=last_validation,
        trace=trace,
    )
    controller_score = float(controller_decision.get("quality_score", 0.0) or 0.0)
    if controller_score >= config.validation_controller_accept_score:
        rationale = str(controller_decision.get("rationale") or "validator rejection was not blocking")
        trace.log(
            "worker.controller.accepted_by_score",
            kind=spec.kind,
            quality_score=controller_score,
            accept_score=config.validation_controller_accept_score,
            controller_approved=controller_decision.get("approved"),
            rationale=rationale,
        )
        write_json(
            material_attempts_dir / "practice_guidance.controller_review.json",
            {"controller_decision": controller_decision},
        )
        return MaterialResult(
            kind=spec.kind,
            material_type=spec.material_type,
            agent_type=PRACTICE_GUIDANCE_AGENT,
            status="approved",
            iterations=config.max_generation_iterations,
            content=last_content,
            prompt_files=spec.prompt_files,
            validation_issues=[],
            validation_issues_by_block=last_validation.issues_by_block if last_validation else [],
            validation_passed_blocks=last_validation.passed_blocks if last_validation else [],
            agent_notes=[
                *[str(item) for item in last_artifact.get("agent_notes") or []],
                (
                    "ValidationControllerAgent accepted after validator review "
                    f"with quality_score={controller_score:g}: {rationale}"
                ),
            ],
            controller_called=True,
            controller_decision=controller_decision,
            generation_artifacts={
                "practice_guidance_input": last_input,
                "practice_guidance_artifact": last_artifact,
            },
        )
    if controller_decision:
        previous_issues = [str(item) for item in controller_decision.get("blocking_issues") or previous_issues]
        write_json(
            material_attempts_dir / "practice_guidance.controller_review.json",
            {"controller_decision": controller_decision},
        )
        trace.log("worker.controller.kept_failed", kind=spec.kind, issues=previous_issues)

    failure_context = {
        "status": "failed",
        "output_dir": str(output_dir),
        "attempts": config.max_generation_iterations,
        "last_input": last_input,
        "issues": previous_issues,
    }
    write_json(material_attempts_dir / "practice_guidance.failed.json", failure_context)
    trace.log("worker.failed", kind=spec.kind, issues=previous_issues)
    return MaterialResult(
        kind=spec.kind,
        material_type=spec.material_type,
        agent_type=PRACTICE_GUIDANCE_AGENT,
        status="failed",
        iterations=config.max_generation_iterations,
        content=last_content,
        prompt_files=spec.prompt_files,
        validation_issues=previous_issues,
        validation_issues_by_block=last_validation.issues_by_block if last_validation else [],
        validation_passed_blocks=last_validation.passed_blocks if last_validation else [],
        generation_artifacts={
            "practice_guidance_input": last_input,
            "practice_guidance_artifact": last_artifact,
        },
    )


def _in_memory_result_payload(
    *,
    task: dict[str, Any],
    course_level: str,
    output_dir: Path,
    materials: list[MaterialResult],
    references: ReferenceBundle,
) -> dict[str, Any]:
    task_id, lesson_number, lesson_title = task_identity(task)
    course_class = resolve_course_class(task)
    return {
        "task_id": task_id,
        "lesson_number": lesson_number,
        "lesson_title": lesson_title,
        "course_class": course_class,
        "audience": _audience_for_task(task, course_class=course_class),
        "course_level": course_level,
        "resolved_profile": course_level,
        "status": "approved",
        "output_dir": str(output_dir),
        "agents_called": [],
        "prompt_files_used": [],
        "materials": [item.to_public_json() for item in materials],
        "references": reference_summary(references),
    }


def build_practice_guidance_input(
    *,
    lesson_output_dir: Path,
    result: dict[str, Any],
    attempt: int,
    previous_artifact: dict[str, Any] | None,
    previous_validation: ValidationResult | None,
    previous_lessons_context: list[dict[str, Any]] | None = None,
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
    previous_context = _normalize_previous_lessons_context(previous_lessons_context)
    audience = _audience_from_result(result, references)

    return {
        "input_version": PRACTICE_GUIDANCE_INPUT_VERSION,
        "source_output_dir": str(lesson_output_dir),
        "task_meta": {
            "task_id": str(result.get("task_id") or ""),
            "lesson_number": str(result.get("lesson_number") or ""),
            "lesson_title": str(result.get("lesson_title") or ""),
            "course_class": str(result.get("course_class") or ""),
            "audience": audience,
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
        "previous_lessons_context": previous_context,
        "previous_lessons_context_policy": {
            "available": bool(previous_context),
            "usage": (
                "Use previous_lessons_context for explicit links to earlier lessons only when the array is non-empty. "
                "If it is empty, write the guidance without references to previous lessons."
            ),
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


def _normalize_previous_lessons_context(value: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not value:
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        lesson_number = str(item.get("lesson_number") or item.get("number") or "").strip()
        title = str(item.get("lesson_title") or item.get("title") or "").strip()
        summary = str(item.get("summary") or item.get("content") or item.get("text") or "").strip()
        materials = item.get("materials") if isinstance(item.get("materials"), list) else []
        material_summaries = []
        for material in materials:
            if not isinstance(material, dict):
                continue
            material_summaries.append(
                {
                    "kind": str(material.get("kind") or "").strip(),
                    "status": str(material.get("status") or "").strip(),
                    "summary": str(material.get("summary") or material.get("content") or "").strip(),
                }
            )
        normalized.append(
            {
                "lesson_number": lesson_number,
                "lesson_title": title,
                "summary": summary,
                "materials": material_summaries,
            }
        )
    return normalized


def _audience_for_task(task: dict[str, Any], *, course_class: str | None = None) -> str:
    lesson = task.get("lesson") if isinstance(task.get("lesson"), dict) else {}
    course = task.get("course") if isinstance(task.get("course"), dict) else {}
    module = task.get("module") if isinstance(task.get("module"), dict) else {}
    content = lesson.get("content") if isinstance(lesson.get("content"), dict) else {}
    candidates = (
        lesson.get("audience"),
        lesson.get("course_class"),
        lesson.get("class"),
        lesson.get("grade"),
        content.get("audience"),
        course.get("audience"),
        course.get("grades"),
        course.get("class"),
        course.get("grade"),
        module.get("audience"),
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return _audience_label_for_course_class(course_class or resolve_course_class(task))


def _audience_from_result(result: dict[str, Any], references: dict[str, list[dict[str, Any]]]) -> str:
    for key in ("audience", "course_class"):
        text = str(result.get(key) or "").strip()
        if text:
            if key == "course_class":
                return _audience_label_for_course_class(text)
            return text
    joined_sources = " ".join(
        str(item.get("source_name") or item.get("path") or item.get("content") or "")
        for documents in references.values()
        for item in documents
        if isinstance(item, dict)
    )
    if re.search(r"\b(10|11)\b", joined_sources):
        return "Обучающиеся 10–11 классов"
    if re.search(r"\b(8|9)\b", joined_sources):
        return "Обучающиеся 8–9 классов"
    return _audience_label_for_course_class("")


def _audience_label_for_course_class(course_class: str) -> str:
    text = str(course_class or "").strip().lower()
    if text == "10" or re.search(r"\b(10|11)\b", text):
        return "Обучающиеся 10–11 классов"
    if text == "8" or re.search(r"\b(8|9)\b", text):
        return "Обучающиеся 8–9 классов"
    return "Обучающиеся курса Python"


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
- Fill every required section of PracticeGuidanceArtifact. Empty strings/lists are not allowed for header fields, goals.objectives, result_requirements.deliverable, result_requirements.criteria, or self_check_questions.
- header.audience must use PracticeGuidanceInput.task_meta.audience.
- result_requirements.criteria must contain at least 2 learner-facing success criteria.
- self_check_questions must contain at least 3 learner-facing questions without answers or keys.
- Use practice_tasks as the source of module tasks.
- Do not change, replace, merge, split, or reconstruct module tasks.
- For each PracticeGuidanceModuleTask, fill module_tasks[].code_cell from the corresponding practice_tasks item:
  - if practice_tasks[].faulty_code_display is non-empty, copy it to module_tasks[].code_cell exactly;
  - otherwise, if practice_tasks[].starter_code is non-empty, copy starter_code to module_tasks[].code_cell exactly;
  - leaving module_tasks[].code_cell empty is valid only when both faulty_code_display and starter_code are empty.
- module_tasks[].code_cell is the learner-facing code display field for starter or intentionally faulty code; it is not a solution field.
- Do not add requires_check saying there is no field for faulty_code_display or starter_code; use code_cell.
- Do not put raw faulty_code into code_cell when faulty_code_display or starter_code exists.
- Build stages by level and methodical similarity; source_task_ids must reference practice_tasks ids.
- Create a worked analogous example for each stage. It must be similar by method but different from the module tasks.
- A worked example is a scaffold, not a module task key. It may be solved for its own analogous input, but it must not reuse exact task values, exact error tokens, exact outputs, or exact faulty/starter code from practice_tasks.
- For debugging/error-fixing stages, do not show corrected code in worked_example. You may show an analogous faulty fragment and the reasoning/checking process, or leave worked_example.code_cell empty and explain the process in commentary.
- For error-reading stages, an analogous error token/value in worked_example is allowed only when it is different from all practice_tasks tokens and is clearly part of the example, not an answer to P tasks.
- If stage/module tasks prohibit a technique, worked_example must obey the same prohibition. For example, if tasks forbid string concatenation or f-strings, worked_example must not use + for strings or f-strings.
- Do not reveal keys, corrected code, internal answer/explanation fields, raw field names, JSON/process wording, SHA, or local paths.
- Use previous_lessons_context for explicit links to previous lessons only when it is non-empty.
- If previous_lessons_context is empty, do not invent previous-lesson references and write the guidance without them.
- Do not use requires_check for missing theory_brief_source.sections, missing exact UI/interface labels, absent previous_lessons_context, or other normal source gaps. Write self-contained guidance from practice_tasks and available references without unsupported specifics.
- Keep requires_check empty unless the runtime explicitly asks for a non-publishable internal blocker. Never write learner-facing phrases such as "Требует проверки", "требует уточнения", "отсутствует утверждённый источник", or "при необходимости уточнить".
- If exact reference values are unavailable, omit the exact value and use a neutral accurate formulation. Put non-publishable source limitations in consistency_notes or agent_notes, not in learner-facing content.
- On retry, start from retry_context.previous_artifact.
- Preserve every block listed in retry_context.previous_passed_blocks unchanged unless that exact block is named in previous_validation_issues.
- If previous_passed_blocks contains methodical_guidance.stages/module_tasks and current issues target only theory_brief, do not modify module_tasks, source_task_ids, student_condition, code_cell, checks, or manual_checks.
- Repair only fields mentioned in previous_validation_issues unless a passed block directly depends on a repaired field.
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
    technical_evidence: dict[str, Any] | None = None,
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
        technical_evidence=technical_evidence,
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


def _build_practice_guidance_execution_evidence(
    *,
    input_payload: dict[str, Any],
    artifact: dict[str, Any],
    config: IsmartGenerationConfig,
    python_sandbox: PythonSandbox | DisabledPythonSandbox,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "evidence_type": "python_sandbox_execution",
        "checked_material": False,
        "student_facing": False,
        "publishable": False,
        "instruction": (
            "Use this only to ground runtime claims about learner-facing code snippets. "
            "Do not validate this evidence as material content and do not target it in fix_instructions."
        ),
        "enabled": bool(config.use_python_sandbox),
        "python_version": "",
        "practice_task_runs": [],
        "artifact_module_task_runs": [],
        "errors": [],
    }
    if not config.use_python_sandbox:
        evidence["skipped_reason"] = "python sandbox disabled"
        return evidence
    if isinstance(python_sandbox, DisabledPythonSandbox):
        evidence["enabled"] = False
        evidence["skipped_reason"] = "python sandbox was not provided to practice_guidance"
        return evidence
    version = python_sandbox.python_version()
    evidence["python_version"] = version
    if not version:
        evidence["errors"].append("python sandbox is unavailable")
        return evidence

    for task_item in _list_of_dicts(input_payload.get("practice_tasks")):
        code_source, code = _practice_guidance_task_code(task_item)
        if not code.strip():
            continue
        run = python_sandbox.run_code(code, "")
        evidence["practice_task_runs"].append(
            {
                "task_id": str(task_item.get("id") or "?"),
                "code_source": code_source,
                "code_char_count": len(code),
                "run": _public_python_run_result(run),
            }
        )

    for task_item in _artifact_module_tasks(artifact):
        code = str(task_item.get("code_cell") or "")
        if not code.strip():
            continue
        run = python_sandbox.run_code(code, "")
        evidence["artifact_module_task_runs"].append(
            {
                "task_id": str(task_item.get("task_id") or "?"),
                "stage_id": str(task_item.get("_stage_id") or ""),
                "code_source": "module_tasks[].code_cell",
                "code_char_count": len(code),
                "run": _public_python_run_result(run),
            }
        )
    return evidence


def _practice_guidance_task_code(task_item: dict[str, Any]) -> tuple[str, str]:
    faulty = str(task_item.get("faulty_code_display") or "")
    if faulty.strip():
        return "faulty_code_display", faulty
    starter = str(task_item.get("starter_code") or "")
    if starter.strip():
        return "starter_code", starter
    return "", ""


def _artifact_module_tasks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    guidance = artifact.get("methodical_guidance") if isinstance(artifact.get("methodical_guidance"), dict) else {}
    tasks: list[dict[str, Any]] = []
    for stage in _list_of_dicts(guidance.get("stages")):
        stage_id = str(stage.get("id") or "")
        for task_item in _list_of_dicts(stage.get("module_tasks")):
            item = dict(task_item)
            item["_stage_id"] = stage_id
            tasks.append(item)
    return tasks


def _public_python_run_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "exit_code": result.get("exit_code"),
        "stdout": str(result.get("stdout") or ""),
        "stderr": str(result.get("stderr") or ""),
        "exception_type": str(result.get("exception_type") or ""),
        "exception_message": str(result.get("exception_message") or ""),
        "last_error_line": str(result.get("last_error_line") or ""),
    }


def _review_practice_guidance_validation_failure(
    *,
    invoker: StructuredSubagentInvoker,
    config: IsmartGenerationConfig,
    spec: MaterialSpec,
    task: dict[str, Any],
    prompt_contents: dict[str, str],
    input_payload: dict[str, Any],
    content: str,
    artifacts: dict[str, Any],
    technical_evidence: dict[str, Any] | None,
    rule_result: ValidationResult | None,
    llm_result: ValidationResult | None,
    validation: ValidationResult | None,
    trace: TraceLogger,
) -> dict[str, Any]:
    if not config.use_llm_validator or not config.use_validation_controller:
        return {}
    if not content or rule_result is None or llm_result is None or validation is None:
        return {}
    if not rule_result.approved:
        trace.log("worker.controller.skipped_rule_failed", kind=spec.kind, issues=rule_result.issues)
        return {}

    trace.log("worker.controller.start", kind=spec.kind, issues=validation.issues)
    prompt = build_validation_controller_prompt(
        task=task,
        spec=spec,
        prompt_contents=prompt_contents,
        references=_reference_bundle_from_input(input_payload),
        dependencies=[],
        content=content,
        rule_result=rule_result,
        llm_result=llm_result,
        merged_validation=validation,
        generation_artifacts=artifacts,
        technical_evidence=technical_evidence,
    )
    data_model = invoker.invoke(
        "ValidationControllerAgent",
        system=build_validation_controller_system_prompt(),
        prompt=prompt,
        schema=ValidationControllerDecision,
    )
    if not isinstance(data_model, ValidationControllerDecision):
        raise TypeError(f"ValidationControllerAgent returned {type(data_model)!r}, expected ValidationControllerDecision")
    data = _model_to_dict(data_model)
    decision = {
        "approved": bool(data.get("approved")),
        "decision": str(data.get("decision") or ("approve_material" if data.get("approved") else "keep_failed")),
        "quality_score": _controller_quality_score(data),
        "score_rationale": str(data.get("score_rationale") or ""),
        "rationale": str(data.get("rationale") or ""),
        "blocking_issues": [str(item) for item in data.get("blocking_issues") or []],
        "non_blocking_issues": [str(item) for item in data.get("non_blocking_issues") or []],
        "overruled_validator_issues": [str(item) for item in data.get("overruled_validator_issues") or []],
        "residual_risks": [str(item) for item in data.get("residual_risks") or []],
        "fix_instructions": [str(item) for item in data.get("fix_instructions") or []],
    }
    trace.log(
        "worker.controller.done",
        kind=spec.kind,
        approved=decision["approved"],
        quality_score=decision["quality_score"],
        accept_score=config.validation_controller_accept_score,
        accepted_by_score=decision["quality_score"] >= config.validation_controller_accept_score,
        blocking_issues=decision["blocking_issues"],
        non_blocking_issues=decision["non_blocking_issues"],
    )
    return decision


def _controller_quality_score(data: dict[str, Any]) -> float:
    raw_score = data.get("quality_score", data.get("score"))
    if raw_score is None:
        return 5.0 if data.get("approved") else 0.0
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    return min(5.0, max(0.0, score))


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
