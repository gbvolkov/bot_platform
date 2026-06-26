from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from .attempts import AttemptArtifactStore
from .context import (
    build_generation_prompt,
    build_generator_system_prompt,
    build_package_validation_prompt,
    build_validation_controller_prompt,
    build_validation_controller_system_prompt,
    build_validation_prompt,
    build_validator_system_prompt,
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
    MaterialValidationDecision,
    PackageValidationDecision,
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
            generated = generated_model
            raw_content = str(generated.content or "").strip()
            boundary = isolate_material_html(raw_content)
            content = boundary.content
            agent_notes = [str(item) for item in generated.agent_notes]
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
                prefix_detected=bool(boundary.prefix),
                post_html_tail_detected=bool(boundary.tail),
                tail_chars=len(boundary.tail),
            )
            rule_result: ValidationResult | None = None
            llm_result: ValidationResult | None = None
            if not content:
                validation = ValidationResult.fail([f"{spec.agent_type} returned empty content"])
                self.trace.log("worker.validation.empty_content", kind=spec.kind, attempt=attempt)
            else:
                rule_result = self.rule_validator.validate_material(content, spec, task)
                if boundary.issues:
                    rule_result = ValidationResult.fail(boundary.issues).merge(rule_result)
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
                boundary_issues=boundary.issues,
                agent_notes=agent_notes,
                metadata={
                    "agent_type": spec.agent_type,
                    "material_type": spec.material_type,
                    "raw_content_chars": len(raw_content),
                    "content_chars": len(content),
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
                )

            previous_content = content
            previous_issues = validation.issues
            previous_validation = validation
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
        )

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
