from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from .attempts import AttemptArtifactStore
from .context import (
    build_generation_prompt,
    build_generator_system_prompt,
    build_intermediate_assessment_artifact_prompt,
    build_intermediate_assessment_artifact_system_prompt,
    build_package_validation_prompt,
    build_practice_template_prompt,
    build_practice_template_system_prompt,
    build_practice_variant_prompt,
    build_practice_variant_system_prompt,
    build_self_work_autocheck_prompt,
    build_self_work_autocheck_system_prompt,
    build_validation_controller_prompt,
    build_validation_controller_system_prompt,
    build_validation_prompt,
    build_validator_system_prompt,
    source_contract_for_spec,
)
from .contracts import (
    IsmartGenerationConfig,
    MaterialResult,
    MaterialSpec,
    ReferenceBundle,
    ValidationResult,
)
from .schemas import (
    GeneratedMaterial,
    IntermediateAssessmentArtifact,
    MaterialValidationDecision,
    PackageValidationDecision,
    PracticeTaskInstanceSet,
    PracticeTaskTemplateSet,
    SelfWorkAutocheckSet,
    ValidationControllerDecision,
)
from .sources import read_prompt_files
from .trace import TraceLogger
from .validators import RuleValidator


@dataclass(frozen=True)
class ContentBoundary:
    raw_content: str
    content: str
    issues: list[str]
    prefix: str
    tail: str


@dataclass(frozen=True)
class GeneratedAttempt:
    raw_content: str
    content: str
    boundary_issues: list[str]
    agent_notes: list[str]
    generation_artifacts: dict[str, Any]
    structural_validation: ValidationResult


def isolate_material_html(raw_content: str) -> ContentBoundary:
    raw = raw_content.strip()
    issues: list[str] = []
    prefix = ""
    tail = ""
    content = raw

    lower = content.lower()
    style_index = lower.find("<style>")
    if style_index > 0:
        prefix = content[:style_index].strip()
        issues.append("generated content had non-HTML text before <style>; prefix was stripped before validation")
        content = content[style_index:].strip()
        lower = content.lower()

    div_end_index = lower.rfind("</div>")
    if div_end_index >= 0:
        html_end = div_end_index + len("</div>")
        tail = content[html_end:].strip()
        if tail:
            snippet = tail[:240].replace("\n", "\\n")
            issues.append(f"generated content had non-HTML tail after final </div>; tail was stripped before validation: {snippet}")
            content = content[:html_end].strip()

    return ContentBoundary(raw_content=raw, content=content, issues=issues, prefix=prefix, tail=tail)


def _contains_internal_source_locator(value: str) -> bool:
    normalized = value.lower().replace("\\", "/")
    if "docs/ismart/" in normalized:
        return True
    if "референсы/" in normalized:
        return True
    if "рабочая область агента" in normalized:
        return True
    if "материалы для ии-агентов" in normalized:
        return True
    if ".md" in normalized and ("референс" in normalized or "источник" in normalized):
        return True
    return False


def sanitize_internal_source_locators(content: str, spec: MaterialSpec) -> tuple[str, list[str]]:
    if spec.validator_kind == "qa" or not _contains_internal_source_locator(content):
        return content, []

    sanitized = content
    notes: list[str] = []
    neutral_sources = (
        '<h2 id="sources">Источники</h2>\n'
        "<p>Материал подготовлен на основе содержания занятия и учебных требований курса.</p>\n"
    )

    def replace_sources_section(match: re.Match[str]) -> str:
        block = match.group(0)
        if _contains_internal_source_locator(block):
            notes.append("internal source locator section was replaced before validation")
            return neutral_sources
        return block

    sanitized = re.sub(
        r"(?is)<h2\b[^>]*>\s*Источники\s*</h2>.*?(?=<h2\b|</div>\s*$)",
        replace_sources_section,
        sanitized,
    )

    if _contains_internal_source_locator(sanitized):
        before = sanitized
        sanitized = re.sub(
            r"(?is)<code\b[^>]*>[^<]*(?:docs[/\\]ismart|референсы|рабочая область агента|материалы для ии-агентов|\.md)[^<]*</code>",
            "материалы курса",
            sanitized,
        )
        sanitized = re.sub(
            r"(?is)<a\b[^>]*(?:docs[/\\]ismart|референсы|рабочая область агента|материалы для ии-агентов|\.md)[^>]*>.*?</a>",
            "материалы курса",
            sanitized,
        )
        sanitized = re.sub(
            r"(?i)(?:docs[/\\]ismart|референсы[/\\]|рабочая область агента|материалы для ии-агентов)[^\s<,;)]*",
            "материалы курса",
            sanitized,
        )
        if sanitized != before:
            notes.append("inline internal source locators were replaced before validation")

    return sanitized, list(dict.fromkeys(notes))


def sanitize_visible_selfcheck_answers(content: str, spec: MaterialSpec) -> tuple[str, list[str]]:
    if spec.kind != "self_work":
        return content, []

    sanitized = content
    notes: list[str] = []

    replacements = [
        (
            r"(?is)<div\s+class=[\"']cc-note[\"']>\s*<div\s+class=[\"']cc-note-title[\"']>\s*Ключ[^<]*</div>.*?</div>",
            "",
            "visible self-check key blocks were removed before validation",
        ),
        (
            r"(?is)<p>\s*Правильн(?:ый|ые)\s+(?:вариант|варианты|ответ|ответы)[^<]*</p>",
            "",
            "visible self-check answer paragraphs were removed before validation",
        ),
        (
            r"(?is)\{%\s*answer\s*%\}.*?\{%\s*/answer\s*%\}",
            "",
            "visible self-check template answer blocks were removed before validation",
        ),
        (
            r"(?is)\{\{\s*input-text\s*:[^}]*\}\}",
            "{{input-text}}",
            "visible self-check filled input-text answers were blanked before validation",
        ),
    ]

    for pattern, replacement, note in replacements:
        updated = re.sub(pattern, replacement, sanitized)
        if updated != sanitized:
            notes.append(note)
            sanitized = updated

    return sanitized, list(dict.fromkeys(notes))


def _normalize_copy_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", without_tags).strip().lower()


def _copy_check_snippets(value: str, *, min_chars: int = 40) -> list[str]:
    snippets: list[str] = []
    normalized = _normalize_copy_text(value)
    if len(normalized) >= min_chars:
        snippets.append(normalized)
    for raw_line in value.splitlines():
        line = _normalize_copy_text(raw_line)
        if len(line) >= min_chars:
            snippets.append(line)
    return list(dict.fromkeys(snippets))


def _normalize_test_items(tests: Any) -> list[Any]:
    if not isinstance(tests, list):
        return []
    normalized_tests: list[Any] = []
    for test in tests:
        if not isinstance(test, dict):
            normalized_tests.append(test)
            continue
        normalized = {str(key): str(value) for key, value in test.items()}
        if "input" not in normalized:
            for alias in ("stdin", "in", "вход"):
                if alias in normalized:
                    normalized["input"] = normalized[alias]
                    break
        if "expected_output" not in normalized:
            for alias in ("output", "stdout", "expected", "result", "ожидаемый_вывод"):
                if alias in normalized:
                    normalized["expected_output"] = normalized[alias]
                    break
        normalized_tests.append(normalized)
    return normalized_tests


def _normalize_practice_instance_tests(instances: dict[str, Any]) -> dict[str, Any]:
    tasks = instances.get("tasks")
    if not isinstance(tasks, list):
        return instances
    for task_item in tasks:
        if not isinstance(task_item, dict):
            continue
        tests = _normalize_test_items(task_item.get("tests"))
        runtime_tests = _normalize_test_items(task_item.get("runtime_tests"))
        if not runtime_tests and tests:
            runtime_tests = list(tests)
        if not tests and runtime_tests:
            tests = list(runtime_tests)
        task_item["tests"] = tests
        task_item["runtime_tests"] = runtime_tests
    return instances


def _model_to_dict(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value.dict()


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, BaseModel):
            result.append(_model_to_dict(item))
    return result


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _controller_quality_score(data: dict[str, Any]) -> float:
    raw_score = data.get("quality_score", data.get("score"))
    if raw_score is None:
        return 5.0 if data.get("approved") else 0.0
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    return min(5.0, max(0.0, score))


def _declares_multiple_valid_answers(*values: Any) -> bool:
    text = _normalize_copy_text("\n".join(str(value or "") for value in values))
    if not text:
        return False
    negative_markers = (
        "не несколько",
        "не допускает несколько",
        "not multiple",
        "not several",
    )
    if any(marker in text for marker in negative_markers):
        return False
    multiple_markers = (
        "любое из",
        "любой из",
        "любая из",
        "любые из",
        "каждое из",
        "каждый из",
        "каждая из",
        "несколько коррект",
        "несколько правиль",
        "несколько допустим",
        "более одного",
        "оба варианта",
        "оба ответа",
        "multiple valid",
        "more than one",
        "several correct",
        "any of",
    )
    return any(marker in text for marker in multiple_markers)


class StructuredSubagentInvoker:
    def __init__(self, subagents: Mapping[str, Any]) -> None:
        self.subagents = subagents

    def invoke(
        self,
        agent_type: str,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        graph = self.subagents.get(agent_type)
        if graph is None:
            raise KeyError(f"Subagent is not registered: {agent_type}")
        state = graph.invoke({"system_prompt": system, "prompt": prompt})
        result = state.get("result") if isinstance(state, dict) else None
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            if hasattr(schema, "model_validate"):
                return schema.model_validate(result)
            return schema.parse_obj(result)
        raise TypeError(f"Subagent {agent_type} returned unsupported structured result: {type(result)!r}")


class MaterialWorker:
    def __init__(
        self,
        *,
        subagents: Mapping[str, Any],
        config: IsmartGenerationConfig,
        rule_validator: RuleValidator | None = None,
        trace: TraceLogger | None = None,
    ) -> None:
        self.config = config
        self.rule_validator = rule_validator or RuleValidator()
        self.trace = trace or TraceLogger()
        self.invoker = StructuredSubagentInvoker(subagents)

    def run(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None = None,
        initial_previous_issues: list[str] | None = None,
        attempts_dir: Path | None = None,
    ) -> MaterialResult:
        attempt_store = AttemptArtifactStore(attempts_dir)
        self.trace.log(
            "worker.start",
            kind=spec.kind,
            agent=spec.agent_type,
            dependency_kinds=list(spec.dependency_kinds),
            dependencies=[{"kind": item.kind, "status": item.status} for item in dependency_results],
        )
        blocked = [item for item in dependency_results if item.status != "approved"]
        if blocked:
            issues = [f"dependency {item.kind} has status {item.status}" for item in blocked]
            self.trace.log("worker.blocked_dependency", kind=spec.kind, issues=issues)
            return MaterialResult(
                kind=spec.kind,
                material_type=spec.material_type,
                agent_type=spec.agent_type,
                status="blocked_dependency",
                iterations=0,
                content="",
                prompt_files=spec.prompt_files,
                validation_issues=issues,
            )

        prompt_contents = read_prompt_files(self.config, spec.prompt_files)
        self.trace.log("worker.prompt_files_loaded", kind=spec.kind, prompt_files=list(spec.prompt_files))
        previous_content = ""
        previous_issues = list(initial_previous_issues or [])
        previous_validation: ValidationResult | None = None
        last_agent_notes: list[str] = []
        last_rule_result: ValidationResult | None = None
        last_llm_result: ValidationResult | None = None
        last_validation: ValidationResult | None = None
        previous_artifacts: dict[str, Any] = {}
        last_generation_artifacts: dict[str, Any] = {}
        if previous_issues:
            self.trace.log("worker.initial_issues", kind=spec.kind, issues=previous_issues)

        for attempt in range(1, self.config.max_generation_iterations + 1):
            self.trace.log(
                "worker.attempt.start",
                kind=spec.kind,
                attempt=attempt,
                max_attempts=self.config.max_generation_iterations,
                previous_content_chars=len(previous_content),
                previous_issues_count=len(previous_issues),
            )
            if spec.kind == "practice":
                generated_attempt = self._generate_practice_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                    previous_artifacts=previous_artifacts,
                    attempt=attempt,
                    attempt_store=attempt_store,
                )
            elif spec.kind == "self_work":
                generated_attempt = self._generate_self_work_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                    previous_artifacts=previous_artifacts,
                    attempt=attempt,
                    attempt_store=attempt_store,
                )
            elif spec.kind == "intermediate":
                generated_attempt = self._generate_intermediate_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                    previous_artifacts=previous_artifacts,
                    attempt=attempt,
                    attempt_store=attempt_store,
                )
            else:
                generated_attempt = self._generate_material_attempt(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    module_material_summaries=module_material_summaries,
                    previous_content=previous_content,
                    previous_issues=previous_issues,
                    previous_validation=previous_validation,
                )
            raw_content = generated_attempt.raw_content
            content = generated_attempt.content
            agent_notes = generated_attempt.agent_notes
            content, source_sanitizer_notes = sanitize_internal_source_locators(content, spec)
            content, selfcheck_sanitizer_notes = sanitize_visible_selfcheck_answers(content, spec)
            sanitizer_notes = [*source_sanitizer_notes, *selfcheck_sanitizer_notes]
            if sanitizer_notes:
                agent_notes = [*agent_notes, *sanitizer_notes]
            generation_artifacts = generated_attempt.generation_artifacts
            last_generation_artifacts = generation_artifacts
            self.trace.log(
                "worker.generator.done",
                kind=spec.kind,
                attempt=attempt,
                content_chars=len(content),
                raw_content_chars=len(raw_content),
                agent_notes_count=len(agent_notes),
            )
            self.trace.log(
                "worker.content_boundary",
                kind=spec.kind,
                attempt=attempt,
                starts_with=content[:80],
                ends_with=content[-80:] if content else "",
                boundary_issues=generated_attempt.boundary_issues,
            )
            rule_result: ValidationResult | None = None
            llm_result: ValidationResult | None = None
            if not content:
                validation = generated_attempt.structural_validation.merge(
                    ValidationResult.fail([f"{spec.agent_type} returned empty content"])
                )
                self.trace.log("worker.validation.empty_content", kind=spec.kind, attempt=attempt)
            else:
                rule_result = self.rule_validator.validate_material(content, spec, task)
                if generated_attempt.boundary_issues:
                    rule_result = ValidationResult.fail(generated_attempt.boundary_issues).merge(rule_result)
                rule_result = generated_attempt.structural_validation.merge(rule_result)
                self.trace.log(
                    "worker.rule_validation.done",
                    kind=spec.kind,
                    attempt=attempt,
                    approved=rule_result.approved,
                    issues=rule_result.issues,
                )
                llm_result = self._validate_with_llm(
                    task=task,
                    spec=spec,
                    prompt_contents=prompt_contents,
                    references=references,
                    dependency_results=dependency_results,
                    content=content,
                    rule_result=rule_result,
                    generation_artifacts=generation_artifacts,
                )
                validation = rule_result.merge(llm_result)
                self.trace.log(
                    "worker.validation.merged",
                    kind=spec.kind,
                    attempt=attempt,
                    approved=validation.approved,
                    issues=validation.issues,
                )
            last_agent_notes = agent_notes
            last_rule_result = rule_result
            last_llm_result = llm_result
            last_validation = validation
            attempt_store.write_material_attempt(
                kind=spec.kind,
                attempt=attempt,
                raw_content=raw_content,
                content=content,
                rule_result=rule_result,
                llm_result=llm_result,
                validation=validation,
                boundary_issues=generated_attempt.boundary_issues,
                agent_notes=agent_notes,
                metadata={
                    "agent_type": spec.agent_type,
                    "material_type": spec.material_type,
                    "raw_content_chars": len(raw_content),
                    "content_chars": len(content),
                    "generation_artifacts": generation_artifacts,
                },
            )

            if validation.approved:
                self.trace.log("worker.approved", kind=spec.kind, attempt=attempt, content_chars=len(content))
                return MaterialResult(
                    kind=spec.kind,
                    material_type=spec.material_type,
                    agent_type=spec.agent_type,
                    status="approved",
                    iterations=attempt,
                    content=content,
                    prompt_files=spec.prompt_files,
                    validation_issues_by_block=validation.issues_by_block,
                    validation_passed_blocks=validation.passed_blocks,
                    agent_notes=agent_notes,
                    generation_artifacts=generation_artifacts,
                )

            previous_content = content
            previous_issues = validation.issues
            previous_validation = validation
            previous_artifacts = generation_artifacts
            if attempt < self.config.max_generation_iterations:
                self.trace.log("worker.retry", kind=spec.kind, next_attempt=attempt + 1, issues=previous_issues)

        controller_decision = self._review_validation_failure(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependency_results=dependency_results,
            content=previous_content,
            rule_result=last_rule_result,
            llm_result=last_llm_result,
            validation=last_validation,
            generation_artifacts=last_generation_artifacts,
            attempt_store=attempt_store,
        )
        controller_score = float(controller_decision.get("quality_score", 0.0) or 0.0)
        if controller_score >= self.config.validation_controller_accept_score:
            rationale = str(controller_decision.get("rationale") or "validator rejection was not blocking")
            self.trace.log(
                "worker.controller.accepted_by_score",
                kind=spec.kind,
                quality_score=controller_score,
                accept_score=self.config.validation_controller_accept_score,
                controller_approved=controller_decision.get("approved"),
                rationale=rationale,
            )
            return MaterialResult(
                kind=spec.kind,
                material_type=spec.material_type,
                agent_type=spec.agent_type,
                status="approved",
                iterations=self.config.max_generation_iterations,
                content=previous_content,
                prompt_files=spec.prompt_files,
                validation_issues=[],
                validation_issues_by_block=last_validation.issues_by_block if last_validation else [],
                validation_passed_blocks=last_validation.passed_blocks if last_validation else [],
                agent_notes=[
                    *last_agent_notes,
                    (
                        "ValidationControllerAgent accepted after validator review "
                        f"with quality_score={controller_score:g}: {rationale}"
                    ),
                ],
                controller_called=True,
                controller_decision=controller_decision,
                generation_artifacts=last_generation_artifacts,
            )

        if controller_decision:
            previous_issues = [str(item) for item in controller_decision.get("blocking_issues") or previous_issues]
            self.trace.log("worker.controller.kept_failed", kind=spec.kind, issues=previous_issues)

        self.trace.log("worker.failed", kind=spec.kind, issues=previous_issues)
        return MaterialResult(
            kind=spec.kind,
            material_type=spec.material_type,
            agent_type=spec.agent_type,
            status="failed",
            iterations=self.config.max_generation_iterations,
            content=previous_content,
            prompt_files=spec.prompt_files,
            validation_issues=previous_issues,
            validation_issues_by_block=last_validation.issues_by_block if last_validation else [],
            validation_passed_blocks=last_validation.passed_blocks if last_validation else [],
            controller_called=bool(controller_decision),
            controller_decision=controller_decision,
            generation_artifacts=last_generation_artifacts,
        )

    def _generate_material_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
    ) -> GeneratedAttempt:
        generation_prompt = build_generation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_content=previous_content,
            previous_issues=previous_issues,
            previous_validation=previous_validation,
            module_material_summaries=module_material_summaries,
        )
        generated_model = self.invoker.invoke(
            spec.agent_type,
            system=build_generator_system_prompt(spec),
            prompt=generation_prompt,
            schema=GeneratedMaterial,
        )
        if not isinstance(generated_model, GeneratedMaterial):
            raise TypeError(f"{spec.agent_type} returned {type(generated_model)!r}, expected GeneratedMaterial")
        raw_content = str(generated_model.content or "").strip()
        boundary = isolate_material_html(raw_content)
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[str(item) for item in generated_model.agent_notes],
            generation_artifacts={},
            structural_validation=ValidationResult.ok(),
        )

    def _generate_practice_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
        previous_artifacts: dict[str, Any],
        attempt: int,
        attempt_store: AttemptArtifactStore,
    ) -> GeneratedAttempt:
        template_prompt = build_practice_template_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_artifacts=previous_artifacts,
            previous_issues=previous_issues,
        )
        template_model = self.invoker.invoke(
            "PracticeTaskTemplateAgent",
            system=build_practice_template_system_prompt(),
            prompt=template_prompt,
            schema=PracticeTaskTemplateSet,
        )
        if not isinstance(template_model, PracticeTaskTemplateSet):
            raise TypeError(
                f"PracticeTaskTemplateAgent returned {type(template_model)!r}, expected PracticeTaskTemplateSet"
            )
        templates = _model_to_dict(template_model)
        template_validation = self._validate_practice_templates(task=task, spec=spec, templates=templates)
        self.trace.log(
            "worker.practice_templates.done",
            attempt=attempt,
            approved=template_validation.approved,
            issues=template_validation.issues,
        )

        if not template_validation.approved:
            artifacts = {
                "practice_templates": templates,
                "practice_instances": {},
                "practice_duplicate_check": {"approved": True, "issues": [], "matches": []},
            }
            attempt_store.write_practice_generation_artifacts(
                attempt=attempt,
                templates=templates,
                instances=None,
                duplicate_check=artifacts["practice_duplicate_check"],
                metadata={"stage": "templates"},
            )
            return GeneratedAttempt(
                raw_content="",
                content="",
                boundary_issues=[],
                agent_notes=[*templates.get("agent_notes", []), "Practice template structural validation failed."],
                generation_artifacts=artifacts,
                structural_validation=template_validation,
            )

        variant_prompt = build_practice_variant_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            templates=templates,
            previous_artifacts=previous_artifacts,
            previous_issues=previous_issues,
        )
        instance_model = self.invoker.invoke(
            "PracticeTaskVariantAgent",
            system=build_practice_variant_system_prompt(),
            prompt=variant_prompt,
            schema=PracticeTaskInstanceSet,
        )
        if not isinstance(instance_model, PracticeTaskInstanceSet):
            raise TypeError(
                f"PracticeTaskVariantAgent returned {type(instance_model)!r}, expected PracticeTaskInstanceSet"
            )
        instances = _model_to_dict(instance_model)
        instances = _normalize_practice_instance_tests(instances)
        duplicate_check = self._practice_duplicate_check(instances, dependency_results, references)
        instance_validation = self._validate_practice_instances(
            task=task,
            spec=spec,
            templates=templates,
            instances=instances,
        )
        if duplicate_check["issues"]:
            instance_validation = instance_validation.merge(ValidationResult.fail(duplicate_check["issues"]))
        self.trace.log(
            "worker.practice_instances.done",
            attempt=attempt,
            approved=instance_validation.approved,
            issues=instance_validation.issues,
            duplicate_issues=duplicate_check["issues"],
        )

        artifacts = {
            "practice_templates": templates,
            "practice_instances": instances,
            "practice_duplicate_check": duplicate_check,
        }
        attempt_store.write_practice_generation_artifacts(
            attempt=attempt,
            templates=templates,
            instances=instances,
            duplicate_check=duplicate_check,
            metadata={"stage": "instances"},
        )
        if not instance_validation.approved:
            return GeneratedAttempt(
                raw_content="",
                content="",
                boundary_issues=[],
                agent_notes=[
                    *templates.get("agent_notes", []),
                    *instances.get("agent_notes", []),
                    "Practice instance structural validation failed.",
                ],
                generation_artifacts=artifacts,
                structural_validation=instance_validation,
            )

        generation_prompt = build_generation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_content=previous_content,
            previous_issues=previous_issues,
            previous_validation=previous_validation,
            module_material_summaries=module_material_summaries,
            generation_artifacts=artifacts,
        )
        generated_model = self.invoker.invoke(
            spec.agent_type,
            system=build_generator_system_prompt(spec),
            prompt=generation_prompt,
            schema=GeneratedMaterial,
        )
        if not isinstance(generated_model, GeneratedMaterial):
            raise TypeError(f"{spec.agent_type} returned {type(generated_model)!r}, expected GeneratedMaterial")
        raw_content = str(generated_model.content or "").strip()
        boundary = isolate_material_html(raw_content)
        leak_issues = self._practice_student_leak_issues(boundary.content, instances)
        artifacts["practice_student_leak_check"] = {
            "approved": not leak_issues,
            "issues": leak_issues,
        }
        structural_validation = ValidationResult.fail(leak_issues) if leak_issues else ValidationResult.ok()
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[
                *templates.get("agent_notes", []),
                *instances.get("agent_notes", []),
                *[str(item) for item in generated_model.agent_notes],
            ],
            generation_artifacts=artifacts,
            structural_validation=structural_validation,
        )

    def _generate_self_work_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
        previous_artifacts: dict[str, Any],
        attempt: int,
        attempt_store: AttemptArtifactStore,
    ) -> GeneratedAttempt:
        autocheck_prompt = build_self_work_autocheck_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_artifacts=previous_artifacts,
            previous_issues=previous_issues,
        )
        autocheck_model = self.invoker.invoke(
            "SelfWorkAutocheckAgent",
            system=build_self_work_autocheck_system_prompt(),
            prompt=autocheck_prompt,
            schema=SelfWorkAutocheckSet,
        )
        if not isinstance(autocheck_model, SelfWorkAutocheckSet):
            raise TypeError(
                f"SelfWorkAutocheckAgent returned {type(autocheck_model)!r}, expected SelfWorkAutocheckSet"
            )

        autocheck = _model_to_dict(autocheck_model)
        artifact_validation = self._validate_self_work_autocheck(autocheck)
        artifacts = {
            "self_work_autocheck": autocheck,
            "self_work_autocheck_check": {
                "approved": artifact_validation.approved,
                "issues": artifact_validation.issues,
            },
        }
        attempt_store.write_self_work_generation_artifacts(
            attempt=attempt,
            autocheck=autocheck,
            structural_check=artifacts["self_work_autocheck_check"],
            metadata={"stage": "autocheck"},
        )
        self.trace.log(
            "worker.self_work_autocheck.done",
            attempt=attempt,
            approved=artifact_validation.approved,
            issues=artifact_validation.issues,
        )

        if not artifact_validation.approved:
            return GeneratedAttempt(
                raw_content="",
                content="",
                boundary_issues=[],
                agent_notes=[
                    *autocheck.get("agent_notes", []),
                    "Self-work autocheck artifact structural validation failed.",
                ],
                generation_artifacts=artifacts,
                structural_validation=artifact_validation,
            )

        generation_prompt = build_generation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_content=previous_content,
            previous_issues=previous_issues,
            previous_validation=previous_validation,
            module_material_summaries=module_material_summaries,
            generation_artifacts=artifacts,
        )
        generated_model = self.invoker.invoke(
            spec.agent_type,
            system=build_generator_system_prompt(spec),
            prompt=generation_prompt,
            schema=GeneratedMaterial,
        )
        if not isinstance(generated_model, GeneratedMaterial):
            raise TypeError(f"{spec.agent_type} returned {type(generated_model)!r}, expected GeneratedMaterial")
        raw_content = str(generated_model.content or "").strip()
        boundary = isolate_material_html(raw_content)
        leak_issues = self._self_work_student_leak_issues(boundary.content)
        structural_validation = ValidationResult.fail(leak_issues) if leak_issues else ValidationResult.ok()
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[
                *autocheck.get("agent_notes", []),
                *[str(item) for item in generated_model.agent_notes],
            ],
            generation_artifacts=artifacts,
            structural_validation=structural_validation,
        )

    def _validate_self_work_autocheck(self, autocheck: dict[str, Any]) -> ValidationResult:
        issues: list[str] = []
        independent_tasks = autocheck.get("independent_tasks") if isinstance(autocheck, dict) else None
        selfcheck_questions = autocheck.get("selfcheck_questions") if isinstance(autocheck, dict) else None
        if not isinstance(independent_tasks, list):
            issues.append("self_work_autocheck.independent_tasks must be a list")
            independent_tasks = []
        if not isinstance(selfcheck_questions, list):
            issues.append("self_work_autocheck.selfcheck_questions must be a list")
            selfcheck_questions = []

        if len(independent_tasks) != 8:
            issues.append(f"self_work_autocheck independent_tasks count mismatch: expected 8, got {len(independent_tasks)}")
        if len(selfcheck_questions) != 10:
            issues.append(
                f"self_work_autocheck selfcheck_questions count mismatch: expected 10, got {len(selfcheck_questions)}"
            )

        task_ids: list[str] = []
        for index, item in enumerate(independent_tasks, start=1):
            if not isinstance(item, dict):
                issues.append(f"self_work_autocheck.independent_tasks[{index}] must be an object")
                continue
            task_id = str(item.get("id") or f"?{index}")
            task_ids.append(task_id)
            for field in ("id", "student_task_title", "checked_skill", "checking_mode"):
                if not str(item.get(field) or "").strip():
                    issues.append(f"self_work_autocheck.independent_tasks.{task_id} missing required field {field}")
            runtime_tests = item.get("runtime_tests")
            manual_rules = item.get("manual_check_rules")
            has_check = bool(str(item.get("correct_answer") or "").strip())
            has_check = has_check or (isinstance(runtime_tests, list) and bool(runtime_tests))
            has_check = has_check or (isinstance(manual_rules, list) and bool(manual_rules))
            if not has_check:
                issues.append(
                    f"self_work_autocheck.independent_tasks.{task_id} needs correct_answer, runtime_tests, or manual_check_rules"
                )
            if not isinstance(runtime_tests, list):
                issues.append(f"self_work_autocheck.independent_tasks.{task_id} runtime_tests must be a list")
            if not isinstance(manual_rules, list):
                issues.append(f"self_work_autocheck.independent_tasks.{task_id} manual_check_rules must be a list")

        question_ids: list[str] = []
        for index, item in enumerate(selfcheck_questions, start=1):
            if not isinstance(item, dict):
                issues.append(f"self_work_autocheck.selfcheck_questions[{index}] must be an object")
                continue
            question_id = str(item.get("id") or f"?{index}")
            question_ids.append(question_id)
            for field in ("id", "template_code", "question_type", "skill_target", "student_prompt"):
                if not str(item.get(field) or "").strip():
                    issues.append(f"self_work_autocheck.selfcheck_questions.{question_id} missing required field {field}")
            correct_answers = item.get("correct_answers")
            if not isinstance(correct_answers, list) or not any(str(answer).strip() for answer in correct_answers):
                issues.append(
                    f"self_work_autocheck.selfcheck_questions.{question_id} needs at least one correct answer"
                )
            if not isinstance(item.get("options"), list):
                issues.append(f"self_work_autocheck.selfcheck_questions.{question_id} options must be a list")
            if not isinstance(item.get("autocheck_config"), dict):
                issues.append(f"self_work_autocheck.selfcheck_questions.{question_id} autocheck_config must be an object")

        if len(task_ids) != len(set(task_ids)):
            issues.append("self_work_autocheck.independent_tasks ids must be unique")
        if len(question_ids) != len(set(question_ids)):
            issues.append("self_work_autocheck.selfcheck_questions ids must be unique")

        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _self_work_student_leak_issues(self, content: str) -> list[str]:
        normalized = _normalize_copy_text(content)
        issues: list[str] = []
        for marker in (
            "self_work_autocheck",
            "correct_answers",
            "autocheck_config",
            "internal_explanation",
            "generation artifacts",
            "generation_artifacts",
        ):
            if marker in normalized:
                issues.append(f"self_work HTML contains internal artifact marker: {marker}")
        return list(dict.fromkeys(issues))

    def _generate_intermediate_attempt(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
        previous_content: str,
        previous_issues: list[str],
        previous_validation: ValidationResult | None,
        previous_artifacts: dict[str, Any],
        attempt: int,
        attempt_store: AttemptArtifactStore,
    ) -> GeneratedAttempt:
        artifact_prompt = build_intermediate_assessment_artifact_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_artifacts=previous_artifacts,
            previous_issues=previous_issues,
        )
        artifact_model = self.invoker.invoke(
            "IntermediateAssessmentArtifactAgent",
            system=build_intermediate_assessment_artifact_system_prompt(),
            prompt=artifact_prompt,
            schema=IntermediateAssessmentArtifact,
        )
        if not isinstance(artifact_model, IntermediateAssessmentArtifact):
            raise TypeError(
                f"IntermediateAssessmentArtifactAgent returned {type(artifact_model)!r}, "
                "expected IntermediateAssessmentArtifact"
            )

        assessment = _model_to_dict(artifact_model)
        artifact_validation = self._validate_intermediate_assessment_artifact(assessment)
        artifacts = {
            "intermediate_assessment": assessment,
            "intermediate_assessment_check": {
                "approved": artifact_validation.approved,
                "issues": artifact_validation.issues,
            },
        }
        attempt_store.write_intermediate_assessment_artifacts(
            attempt=attempt,
            artifact=assessment,
            structural_check=artifacts["intermediate_assessment_check"],
            metadata={"stage": "assessment_artifact"},
        )
        self.trace.log(
            "worker.intermediate_assessment.done",
            attempt=attempt,
            approved=artifact_validation.approved,
            issues=artifact_validation.issues,
        )

        if not artifact_validation.approved:
            return GeneratedAttempt(
                raw_content="",
                content="",
                boundary_issues=[],
                agent_notes=[
                    *assessment.get("agent_notes", []),
                    "Intermediate assessment artifact structural validation failed.",
                ],
                generation_artifacts=artifacts,
                structural_validation=artifact_validation,
            )

        generation_prompt = build_generation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            previous_content=previous_content,
            previous_issues=previous_issues,
            previous_validation=previous_validation,
            module_material_summaries=module_material_summaries,
            generation_artifacts=artifacts,
        )
        generated_model = self.invoker.invoke(
            spec.agent_type,
            system=build_generator_system_prompt(spec),
            prompt=generation_prompt,
            schema=GeneratedMaterial,
        )
        if not isinstance(generated_model, GeneratedMaterial):
            raise TypeError(f"{spec.agent_type} returned {type(generated_model)!r}, expected GeneratedMaterial")
        raw_content = str(generated_model.content or "").strip()
        boundary = isolate_material_html(raw_content)
        leak_issues = self._intermediate_student_leak_issues(boundary.content)
        structural_validation = ValidationResult.fail(leak_issues) if leak_issues else ValidationResult.ok()
        return GeneratedAttempt(
            raw_content=raw_content,
            content=boundary.content,
            boundary_issues=boundary.issues,
            agent_notes=[
                *assessment.get("agent_notes", []),
                *[str(item) for item in generated_model.agent_notes],
            ],
            generation_artifacts=artifacts,
            structural_validation=structural_validation,
        )

    def _validate_intermediate_assessment_artifact(self, assessment: dict[str, Any]) -> ValidationResult:
        issues: list[str] = []
        variants = assessment.get("variants") if isinstance(assessment, dict) else None
        if not isinstance(variants, list):
            return ValidationResult.fail(["intermediate_assessment.variants must be a list"])
        if len(variants) != 4:
            issues.append(f"intermediate_assessment variants count mismatch: expected 4, got {len(variants)}")

        variant_ids: list[str] = []
        for variant_index, variant in enumerate(variants, start=1):
            if not isinstance(variant, dict):
                issues.append(f"intermediate_assessment.variants[{variant_index}] must be an object")
                continue
            variant_id = str(variant.get("id") or f"?{variant_index}")
            variant_ids.append(variant_id)
            for field in ("id", "title"):
                if not str(variant.get(field) or "").strip():
                    issues.append(f"intermediate_assessment.{variant_id} missing required field {field}")

            closed = variant.get("closed_questions")
            opened = variant.get("open_questions")
            code_tasks = variant.get("code_tasks")
            if not isinstance(closed, list):
                issues.append(f"intermediate_assessment.{variant_id}.closed_questions must be a list")
                closed = []
            if not isinstance(opened, list):
                issues.append(f"intermediate_assessment.{variant_id}.open_questions must be a list")
                opened = []
            if not isinstance(code_tasks, list):
                issues.append(f"intermediate_assessment.{variant_id}.code_tasks must be a list")
                code_tasks = []

            if len(closed) != 16:
                issues.append(
                    f"intermediate_assessment.{variant_id} closed_questions count mismatch: expected 16, got {len(closed)}"
                )
            if len(opened) != 4:
                issues.append(
                    f"intermediate_assessment.{variant_id} open_questions count mismatch: expected 4, got {len(opened)}"
                )
            if len(code_tasks) != 3:
                issues.append(
                    f"intermediate_assessment.{variant_id} code_tasks count mismatch: expected 3, got {len(code_tasks)}"
                )

            item_ids: list[str] = []
            for item in closed:
                if not isinstance(item, dict):
                    issues.append(f"intermediate_assessment.{variant_id}.closed_questions contains non-object item")
                    continue
                item_id = str(item.get("id") or "?")
                item_ids.append(item_id)
                for field in ("id", "template_code", "skill_target", "student_prompt"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"intermediate_assessment.{variant_id}.closed_questions.{item_id} missing {field}")
                answers = item.get("correct_answers")
                if not isinstance(answers, list) or not any(str(answer).strip() for answer in answers):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.closed_questions.{item_id} needs at least one correct answer"
                    )
                else:
                    answer_values = [str(answer).strip() for answer in answers if str(answer).strip()]
                    if len(answer_values) == 1 and _declares_multiple_valid_answers(
                        item.get("student_prompt"),
                        item.get("internal_explanation"),
                        item.get("autocheck_config"),
                    ):
                        issues.append(
                            f"intermediate_assessment.{variant_id}.closed_questions.{item_id} declares multiple "
                            "valid answers but provides exactly one correct answer; make the criterion unique or "
                            "include every correct answer with a compatible template/autocheck_config"
                        )
                if not isinstance(item.get("options"), list):
                    issues.append(f"intermediate_assessment.{variant_id}.closed_questions.{item_id} options must be a list")
                if not isinstance(item.get("autocheck_config"), dict):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.closed_questions.{item_id} autocheck_config must be an object"
                    )
            coded_templates = {
                str(item.get("template_code") or "").strip().upper()
                for item in closed
                if isinstance(item, dict)
            }
            required_coded_templates = {"6A", "6D", "6G", "8D", "10D"}
            if len(coded_templates & required_coded_templates) < 4:
                issues.append(
                    f"intermediate_assessment.{variant_id} must include at least 4 coded template types "
                    "from 6A/6D/6G/8D/10D"
                )

            for item in opened:
                if not isinstance(item, dict):
                    issues.append(f"intermediate_assessment.{variant_id}.open_questions contains non-object item")
                    continue
                item_id = str(item.get("id") or "?")
                item_ids.append(item_id)
                for field in ("id", "skill_target", "student_prompt", "reference_answer"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"intermediate_assessment.{variant_id}.open_questions.{item_id} missing {field}")
                if not isinstance(item.get("rubric"), list) or not item.get("rubric"):
                    issues.append(f"intermediate_assessment.{variant_id}.open_questions.{item_id} rubric must be a non-empty list")

            for item in code_tasks:
                if not isinstance(item, dict):
                    issues.append(f"intermediate_assessment.{variant_id}.code_tasks contains non-object item")
                    continue
                item_id = str(item.get("id") or "?")
                item_ids.append(item_id)
                for field in ("id", "skill_target", "student_condition", "hidden_solution"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"intermediate_assessment.{variant_id}.code_tasks.{item_id} missing {field}")
                runtime_tests = item.get("runtime_tests")
                manual_rules = item.get("manual_check_rules")
                if not isinstance(runtime_tests, list):
                    issues.append(f"intermediate_assessment.{variant_id}.code_tasks.{item_id} runtime_tests must be a list")
                if not isinstance(manual_rules, list):
                    issues.append(f"intermediate_assessment.{variant_id}.code_tasks.{item_id} manual_check_rules must be a list")
                if not (isinstance(runtime_tests, list) and runtime_tests) and not (
                    isinstance(manual_rules, list) and manual_rules
                ):
                    issues.append(
                        f"intermediate_assessment.{variant_id}.code_tasks.{item_id} needs runtime_tests or manual_check_rules"
                    )

            if len(item_ids) != len(set(item_ids)):
                issues.append(f"intermediate_assessment.{variant_id} item ids must be unique")

        if len(variant_ids) != len(set(variant_ids)):
            issues.append("intermediate_assessment variant ids must be unique")
        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _intermediate_student_leak_issues(self, content: str) -> list[str]:
        normalized = _normalize_copy_text(content)
        issues: list[str] = []
        for marker in (
            "intermediate_assessment",
            "correct_answers",
            "autocheck_config",
            "reference_answer",
            "hidden_solution",
            "teacher_explanation",
            "internal_explanation",
            "generation artifacts",
            "generation_artifacts",
        ):
            if marker in normalized:
                issues.append(f"intermediate HTML contains internal artifact marker: {marker}")
        for marker in ("ключи", "ключ правильного", "эталон ответа", "эталоны ответов", "правильный ответ"):
            if marker in normalized:
                issues.append(f"intermediate HTML appears to disclose answer-key section: {marker}")
        return list(dict.fromkeys(issues))

    def _validate_practice_templates(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        templates: dict[str, Any],
    ) -> ValidationResult:
        expected = source_contract_for_spec(task, spec).get("tasks") or []
        actual = templates.get("tasks") if isinstance(templates, dict) else None
        issues = self._practice_task_order_issues(expected, actual, label="practice_templates")
        if isinstance(actual, list):
            for item in actual:
                if not isinstance(item, dict):
                    issues.append("practice_templates contains a non-object task")
                    continue
                for field in ("id", "level", "source_text", "task_type", "skill_target", "test_policy"):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"practice_templates.{item.get('id') or '?'} missing required field {field}")
                for field in ("invariants", "slots_to_fill", "constraints"):
                    if not isinstance(item.get(field), list):
                        issues.append(f"practice_templates.{item.get('id') or '?'} field {field} must be a list")
        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _validate_practice_instances(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        templates: dict[str, Any],
        instances: dict[str, Any],
    ) -> ValidationResult:
        expected = source_contract_for_spec(task, spec).get("tasks") or []
        actual = instances.get("tasks") if isinstance(instances, dict) else None
        issues = self._practice_task_order_issues(expected, actual, label="practice_instances")
        template_by_id = {
            str(item.get("id")): item
            for item in templates.get("tasks", [])
            if isinstance(item, dict) and item.get("id")
        }
        if isinstance(actual, list):
            for item in actual:
                if not isinstance(item, dict):
                    issues.append("practice_instances contains a non-object task")
                    continue
                task_id = str(item.get("id") or "?")
                for field in (
                    "id",
                    "template_id",
                    "level",
                    "task_type",
                    "scenario",
                    "student_condition",
                    "hidden_solution",
                    "teacher_explanation",
                ):
                    if not str(item.get(field) or "").strip():
                        issues.append(f"practice_instances.{task_id} missing required field {field}")
                if item.get("template_id") != item.get("id"):
                    issues.append(f"practice_instances.{task_id} template_id must match id")
                template = template_by_id.get(task_id)
                if template is not None:
                    if item.get("level") != template.get("level"):
                        issues.append(f"practice_instances.{task_id} level does not match template")
                    if item.get("task_type") != template.get("task_type"):
                        issues.append(f"practice_instances.{task_id} task_type does not match template")
                if not isinstance(item.get("tests"), list):
                    issues.append(f"practice_instances.{task_id} tests must be a list")
                if not isinstance(item.get("runtime_tests"), list):
                    issues.append(f"practice_instances.{task_id} runtime_tests must be a list")
                if not isinstance(item.get("manual_checks"), list):
                    issues.append(f"practice_instances.{task_id} manual_checks must be a list")
                if not isinstance(item.get("subtasks"), list):
                    issues.append(f"practice_instances.{task_id} subtasks must be a list")
                if not isinstance(item.get("uniqueness_notes"), list):
                    issues.append(f"practice_instances.{task_id} uniqueness_notes must be a list")
                issues.extend(self._practice_internal_marker_issues(item))
                issues.extend(self._practice_solution_hint_issues(item))
        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _practice_internal_marker_issues(self, task_item: dict[str, Any]) -> list[str]:
        task_id = str(task_item.get("id") or "?")
        student_text = _normalize_copy_text(
            "\n".join(
                str(task_item.get(field) or "")
                for field in ("scenario", "student_condition", "input_requirements", "output_requirements")
            )
        )
        if not student_text:
            return []

        internal_markers = (
            "source_text",
            "source task",
            "source contract",
            "source_contract",
            "generation artifacts",
            "practice_instances",
            "practice_templates",
            "pipeline",
            "как в задании урока",
            "из json",
            "from json",
        )
        for marker in internal_markers:
            if marker in student_text:
                return [
                    f"practice_instances.{task_id} student-facing fields contain internal source/pipeline marker; "
                    "rewrite learner text without source_text/source contract wording"
                ]
        return []

    def _practice_solution_hint_issues(self, task_item: dict[str, Any]) -> list[str]:
        task_id = str(task_item.get("id") or "?")
        task_type = _normalize_copy_text(str(task_item.get("task_type") or ""))
        source_text = _normalize_copy_text(str(task_item.get("source_text") or ""))
        is_fix_task = any(marker in task_type for marker in ("fix", "debug", "исправ", "отлад")) or any(
            marker in source_text for marker in ("исправ", "ошиб", "error", "syntaxerror", "nameerror")
        )
        if not is_fix_task:
            return []

        student_text = _normalize_copy_text(
            "\n".join(
                str(task_item.get(field) or "")
                for field in ("scenario", "student_condition", "input_requirements", "output_requirements")
            )
        )
        if not student_text:
            return []

        issues: list[str] = []
        forbidden_patterns = [
            r"замен(и|ить|ите|а)\s+\S+\s+на\s+\S+",
            r"добав(ь|ить|ьте|ить)\s+[^.]{0,60}(кавыч|скоб|print|принт)",
            r"(пропущен|пропущена|не хватает|отсутству\w+)\s+[^.]{0,60}(кавыч|скоб)",
            r"(незакрыт|не закрыт)[^.]{0,60}(кавыч|строков)",
            r"(опечатк|неверно написан|неправильно написан)[^.]{0,80}(функц|print|prnt|имя)",
            r"(без кавычек|вокруг текста|строковым литералом|сделать аргумент строк)",
            r"(имя функции вывода|названи[ея] функции вывода)",
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, student_text, flags=re.I):
                issues.append(
                    f"practice_instances.{task_id} student-facing fields reveal the exact fix; move the hint to hidden_solution/teacher_explanation"
                )
                break

        for snippet in _copy_check_snippets(str(task_item.get("hidden_solution") or ""), min_chars=24):
            if snippet and snippet in student_text:
                issues.append(
                    f"practice_instances.{task_id} student-facing fields copy hidden_solution; move the answer out of learner text"
                )
                break

        return list(dict.fromkeys(issues))

    def _practice_task_order_issues(
        self,
        expected: Any,
        actual: Any,
        *,
        label: str,
    ) -> list[str]:
        issues: list[str] = []
        if not isinstance(expected, list):
            expected = []
        if not isinstance(actual, list):
            return [f"{label}.tasks must be a list"]
        expected_ids = [str(item.get("id")) for item in expected if isinstance(item, dict)]
        actual_ids = [str(item.get("id")) for item in actual if isinstance(item, dict)]
        if actual_ids != expected_ids:
            issues.append(f"{label} task ids/order mismatch: expected {expected_ids}, got {actual_ids}")
        expected_levels = {
            str(item.get("id")): item.get("level")
            for item in expected
            if isinstance(item, dict) and item.get("id")
        }
        for item in actual:
            if isinstance(item, dict) and item.get("id") in expected_levels and item.get("level") != expected_levels[item["id"]]:
                issues.append(f"{label}.{item['id']} level mismatch: expected {expected_levels[item['id']]}, got {item.get('level')}")
        return issues

    def _practice_duplicate_check(
        self,
        instances: dict[str, Any],
        dependency_results: list[MaterialResult],
        references: ReferenceBundle,
    ) -> dict[str, Any]:
        anti_copy_sources: list[dict[str, str]] = []
        for dependency in dependency_results:
            anti_copy_sources.append(
                {
                    "source": f"dependency:{dependency.kind}",
                    "text": dependency.content,
                }
            )
        for field, documents in references.items():
            for index, document in enumerate(documents, start=1):
                anti_copy_sources.append(
                    {
                        "source": f"reference:{field}:{index}",
                        "text": document.content,
                    }
                )

        source_texts = [
            (item["source"], _normalize_copy_text(item["text"]))
            for item in anti_copy_sources
            if _normalize_copy_text(item["text"])
        ]
        matches: list[dict[str, str]] = []
        issues: list[str] = []
        for task_item in instances.get("tasks", []):
            if not isinstance(task_item, dict):
                continue
            task_id = str(task_item.get("id") or "?")
            for field in ("scenario", "student_condition", "starter_code", "input_requirements", "output_requirements"):
                value = str(task_item.get(field) or "")
                for snippet in _copy_check_snippets(value):
                    for source, source_text in source_texts:
                        if snippet in source_text:
                            matches.append({"task_id": task_id, "field": field, "source": source, "snippet": snippet[:160]})
                            issues.append(
                                f"practice_instances.{task_id}.{field} directly copies text/code from {source}; generate a new variant"
                            )
                            break
                    if any(match["task_id"] == task_id and match["field"] == field for match in matches):
                        break
        issues = list(dict.fromkeys(issues))
        return {"approved": not issues, "issues": issues, "matches": matches}

    def _practice_student_leak_issues(self, content: str, instances: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        lower = content.lower()
        for marker in ("hidden_solution", "teacher_explanation"):
            if marker in lower:
                issues.append(f"student practice HTML contains internal marker {marker}")
        for marker in (
            "исходный паттерн",
            "source_text",
            "source contract",
            "generation artifacts",
            "practice_instances",
            "practice_templates",
        ):
            if marker in lower:
                issues.append(f"student practice HTML contains internal source/pipeline marker: {marker}")
        normalized_content = _normalize_copy_text(content)
        for task_item in instances.get("tasks", []):
            if not isinstance(task_item, dict):
                continue
            task_id = str(task_item.get("id") or "?")
            allowed_learner_text = _normalize_copy_text(
                "\n".join(
                    str(task_item.get(field) or "")
                    for field in (
                        "scenario",
                        "student_condition",
                        "starter_code",
                        "input_requirements",
                        "output_requirements",
                        "tests",
                        "runtime_tests",
                        "manual_checks",
                        "subtasks",
                    )
                )
            )
            for field in ("hidden_solution", "teacher_explanation"):
                value = str(task_item.get(field) or "")
                for snippet in _copy_check_snippets(value, min_chars=16):
                    if snippet in allowed_learner_text:
                        continue
                    if snippet in normalized_content:
                        issues.append(f"student practice HTML leaks {field} for {task_id}")
                        break
        return list(dict.fromkeys(issues))

    def _validate_with_llm(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        content: str,
        rule_result: ValidationResult,
        generation_artifacts: dict[str, Any] | None = None,
    ) -> ValidationResult:
        if not self.config.use_llm_validator:
            self.trace.log("worker.llm_validation.skipped", kind=spec.kind)
            return ValidationResult.ok()
        self.trace.log("worker.llm_validation.start", kind=spec.kind)
        validation_prompt = build_validation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            content=content,
            rule_result=rule_result,
            generation_artifacts=generation_artifacts,
        )
        data_model = self.invoker.invoke(
            "MaterialValidatorAgent",
            system=build_validator_system_prompt(),
            prompt=validation_prompt,
            schema=MaterialValidationDecision,
        )
        data = _model_to_dict(data_model)
        result = ValidationResult(
            approved=bool(data.get("approved")) and not rule_result.issues,
            issues=_string_list(data.get("issues", [])),
            fix_instructions=_string_list(data.get("fix_instructions", data.get("issues", []))),
            issues_by_block=_list_of_dicts(data.get("issues_by_block", [])),
            passed_blocks=_list_of_dicts(data.get("passed_blocks", [])),
        )
        self.trace.log("worker.llm_validation.done", kind=spec.kind, approved=result.approved, issues=result.issues)
        return result

    def _review_validation_failure(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        prompt_contents: dict[str, str],
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        content: str,
        rule_result: ValidationResult | None,
        llm_result: ValidationResult | None,
        validation: ValidationResult | None,
        generation_artifacts: dict[str, Any] | None,
        attempt_store: AttemptArtifactStore,
    ) -> dict[str, Any]:
        if not self.config.use_llm_validator or not self.config.use_validation_controller:
            return {}
        if not content or rule_result is None or llm_result is None or validation is None:
            return {}
        if not rule_result.approved:
            self.trace.log("worker.controller.skipped_rule_failed", kind=spec.kind, issues=rule_result.issues)
            return {}

        self.trace.log("worker.controller.start", kind=spec.kind, issues=validation.issues)
        prompt = build_validation_controller_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            content=content,
            rule_result=rule_result,
            llm_result=llm_result,
            merged_validation=validation,
            generation_artifacts=generation_artifacts,
        )
        data_model = self.invoker.invoke(
            "ValidationControllerAgent",
            system=build_validation_controller_system_prompt(),
            prompt=prompt,
            schema=ValidationControllerDecision,
        )
        data = _model_to_dict(data_model)
        decision = {
            "approved": bool(data.get("approved")),
            "decision": str(data.get("decision") or ("approve_material" if data.get("approved") else "keep_failed")),
            "quality_score": _controller_quality_score(data),
            "score_rationale": str(data.get("score_rationale") or ""),
            "rationale": str(data.get("rationale") or ""),
            "blocking_issues": _string_list(data.get("blocking_issues", [])),
            "non_blocking_issues": _string_list(data.get("non_blocking_issues", [])),
            "overruled_validator_issues": _string_list(data.get("overruled_validator_issues", [])),
            "residual_risks": _string_list(data.get("residual_risks", [])),
            "fix_instructions": _string_list(data.get("fix_instructions", [])),
        }
        decision = self._apply_intermediate_appellate_policy(
            spec=spec,
            content=content,
            rule_result=rule_result,
            validation=validation,
            generation_artifacts=generation_artifacts,
            decision=decision,
        )
        self.trace.log(
            "worker.controller.done",
            kind=spec.kind,
            approved=decision["approved"],
            quality_score=decision["quality_score"],
            accept_score=self.config.validation_controller_accept_score,
            accepted_by_score=decision["quality_score"] >= self.config.validation_controller_accept_score,
            blocking_issues=decision["blocking_issues"],
            non_blocking_issues=decision["non_blocking_issues"],
        )
        attempt_store.write_material_controller_review(
            kind=spec.kind,
            content=content,
            controller_decision=decision,
            metadata={
                "agent_type": spec.agent_type,
                "material_type": spec.material_type,
                "validator_issues": validation.issues,
            },
        )
        return decision

    def _apply_intermediate_appellate_policy(
        self,
        *,
        spec: MaterialSpec,
        content: str,
        rule_result: ValidationResult,
        validation: ValidationResult,
        generation_artifacts: dict[str, Any] | None,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        if spec.kind != "intermediate" or not rule_result.approved:
            return decision
        if not self._intermediate_artifact_approved(generation_artifacts):
            return decision
        if self._intermediate_visible_key_markers(content):
            return decision

        blocking_issues = [str(item) for item in (decision.get("blocking_issues") or validation.issues or [])]
        if not blocking_issues:
            return decision

        overruled: list[str] = []
        remaining: list[str] = []
        for issue in blocking_issues:
            if self._is_overstrict_intermediate_issue(issue):
                overruled.append(issue)
            else:
                remaining.append(issue)

        if remaining or not overruled:
            return decision

        adjusted = dict(decision)
        adjusted["approved"] = True
        adjusted["decision"] = "approve_material"
        adjusted["quality_score"] = max(
            float(adjusted.get("quality_score", 0.0) or 0.0),
            self.config.validation_controller_accept_score,
        )
        note = (
            "Deterministic appellate policy overruled intermediate validator objections that confused "
            "candidate answer options, publishable HTML, or error-fixing prompts with visible answer-key leaks."
        )
        adjusted["score_rationale"] = " ".join(part for part in [str(adjusted.get("score_rationale") or ""), note] if part)
        adjusted["rationale"] = " ".join(part for part in [str(adjusted.get("rationale") or ""), note] if part)
        adjusted["blocking_issues"] = []
        adjusted["non_blocking_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("non_blocking_issues", [])), *overruled])
        )
        adjusted["overruled_validator_issues"] = list(
            dict.fromkeys([*_string_list(adjusted.get("overruled_validator_issues", [])), *overruled])
        )
        adjusted["fix_instructions"] = [
            item
            for item in _string_list(adjusted.get("fix_instructions", []))
            if item not in set(overruled)
        ]
        return adjusted

    def _intermediate_artifact_approved(self, generation_artifacts: dict[str, Any] | None) -> bool:
        if not isinstance(generation_artifacts, dict):
            return False
        check = generation_artifacts.get("intermediate_assessment_check")
        return isinstance(check, dict) and bool(check.get("approved"))

    def _intermediate_visible_key_markers(self, content: str) -> list[str]:
        normalized = _normalize_copy_text(content)
        markers = (
            "correct_answers",
            "reference_answer",
            "autocheck_config",
            "hidden_solution",
            "teacher_explanation",
            "internal_explanation",
            "intermediate_assessment",
            "generation_artifacts",
            "generation artifacts",
            "правильный ответ",
            "правильные ответы",
            "ключи",
            "ключ правильного",
            "эталон",
        )
        return [marker for marker in markers if marker in normalized]

    def _is_overstrict_intermediate_issue(self, issue: str) -> bool:
        normalized = _normalize_copy_text(issue)
        if any(
            marker in normalized
            for marker in (
                "correct_answers",
                "reference_answer",
                "autocheck_config",
                "hidden_solution",
                "teacher_explanation",
                "internal_explanation",
            )
        ):
            return False

        coded_template_tokens = ("10d", "6a", "6d", "6g", "8d")
        if "10d" in normalized and any(token in normalized for token in ("утеч", "ключ", "ответ", "answer", "key")):
            return True
        if any(token in normalized for token in coded_template_tokens) and any(
            token in normalized for token in ("шаблон", "template", "размет", "markup", "html")
        ):
            return True
        if any(token in normalized for token in ("исправ", "fix")) and any(
            token in normalized for token in ("решен", "solution", "эталон", "ключ", "key")
        ):
            return True
        return False


class PackageValidator:
    def __init__(
        self,
        *,
        subagents: Mapping[str, Any],
        config: IsmartGenerationConfig,
        rule_validator: RuleValidator | None = None,
        trace: TraceLogger | None = None,
    ) -> None:
        self.config = config
        self.rule_validator = rule_validator or RuleValidator()
        self.trace = trace or TraceLogger()
        self.invoker = StructuredSubagentInvoker(subagents)

    def validate(
        self,
        *,
        task: dict[str, Any],
        specs: list[MaterialSpec],
        materials: list[MaterialResult],
        attempts_dir: Path | None = None,
    ) -> ValidationResult:
        attempt_store = AttemptArtifactStore(attempts_dir)
        self.trace.log(
            "package_validation.start",
            material_count=len(materials),
            material_statuses=[{"kind": item.kind, "status": item.status} for item in materials],
        )
        rule_result = self.rule_validator.validate_package(specs, materials)
        self.trace.log("package_validation.rule.done", approved=rule_result.approved, issues=rule_result.issues)
        llm_result: ValidationResult | None = None
        if not self.config.use_llm_validator:
            self.trace.log("package_validation.llm.skipped")
            advisory_result = ValidationResult(
                approved=True,
                issues=list(rule_result.issues),
                fix_instructions=list(rule_result.fix_instructions),
                issues_by_block=list(rule_result.issues_by_block),
                passed_blocks=list(rule_result.passed_blocks),
            )
            attempt_store.write_package_validation(
                rule_result=rule_result,
                llm_result=None,
                validation=advisory_result,
                metadata={"material_count": len(materials), "advisory": True},
            )
            return advisory_result

        prompt = build_package_validation_prompt(
            task=task,
            specs=specs,
            materials=materials,
            rule_result=rule_result,
        )
        package_system_prompt = (
            "You are PackageValidatorAgent. Check the package only. "
            "Do not generate or repair materials. Return structured validation fields."
        )
        data_model = self.invoker.invoke(
            "PackageValidatorAgent",
            system=package_system_prompt,
            prompt=prompt,
            schema=PackageValidationDecision,
        )
        data = _model_to_dict(data_model)
        llm_result = ValidationResult(
            approved=bool(data.get("approved")) and not rule_result.issues,
            issues=_string_list(data.get("issues", [])),
            fix_instructions=_string_list(data.get("fix_instructions", data.get("issues", []))),
        )
        merged = rule_result.merge(llm_result)
        result = ValidationResult(
            approved=True,
            issues=list(merged.issues),
            fix_instructions=list(merged.fix_instructions),
            issues_by_block=list(merged.issues_by_block),
            passed_blocks=list(merged.passed_blocks),
        )
        self.trace.log("package_validation.done", approved=result.approved, issues=result.issues, advisory=True)
        attempt_store.write_package_validation(
            rule_result=rule_result,
            llm_result=llm_result,
            validation=result,
            metadata={"material_count": len(materials), "advisory": True},
        )
        return result
