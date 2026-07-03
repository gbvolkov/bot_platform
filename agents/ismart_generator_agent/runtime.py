from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from .context import task_identity
from .contracts import (
    IsmartGenerationConfig,
    IsmartGenerationResult,
    MaterialResult,
    ValidationResult,
)
from .planner import build_material_plan
from .practice_guidance import run_practice_guidance_material
from .profiles import config_for_task_profile, resolve_course_level
from .sources import ReferenceLoader, reference_summary
from .task_skip import (
    SKIPPED_MATERIAL_STATUSES,
    build_skipped_material,
    dependency_skip_reason,
    practice_material_skip_reason,
)
from .trace import TraceLogger
from .validators import RuleValidator
from .workers import MaterialWorker, PackageValidator
from .writer import default_run_name, write_task_output


class LessonTaskGraphState(TypedDict, total=False):
    task: dict[str, Any]
    task_id: str
    lesson_number: str
    lesson_title: str
    course_level: str
    output_dir: str
    current_material_index: int
    package_validator_called: bool
    result: dict[str, Any]
    stop_generation: NotRequired[bool]


def run_ismart_task(
    task: dict[str, Any],
    config: IsmartGenerationConfig,
    *,
    subagents: Mapping[str, Any],
    run_dir: str | Path | None = None,
) -> IsmartGenerationResult:
    runtime = IsmartGeneratorRuntime(config=config, subagents=subagents)
    return runtime.run_task(
        task,
        run_dir=Path(run_dir) if run_dir is not None else None,
    )


class IsmartGeneratorRuntime:
    def __init__(
        self,
        *,
        config: IsmartGenerationConfig,
        subagents: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.subagents = subagents
        self.trace = TraceLogger(enabled=config.verbose)

    def run_task(
        self,
        task: dict[str, Any],
        *,
        run_dir: Path | None = None,
    ) -> IsmartGenerationResult:
        result_box: dict[str, IsmartGenerationResult] = {}
        graph = self._build_lesson_task_graph(result_box)
        graph.invoke({"task": task, "output_dir": str(run_dir) if run_dir is not None else None})
        result = result_box.get("result")
        if not isinstance(result, IsmartGenerationResult):
            raise RuntimeError("LessonTaskGraph finished without IsmartGenerationResult.")
        return result

    def _build_lesson_task_graph(self, result_box: dict[str, IsmartGenerationResult]):
        rule_validator = RuleValidator()
        runtime_data: dict[str, Any] = {}

        def init_task_node(state: LessonTaskGraphState) -> dict[str, Any]:
            task = state["task"]
            task_id, lesson_number, lesson_title = task_identity(task)
            course_level = resolve_course_level(task)
            task_config = config_for_task_profile(self.config, task)
            output_dir_value = state.get("output_dir")
            output_dir = Path(output_dir_value) if output_dir_value is not None else self._new_run_dir(task)
            attempts_dir = output_dir / "tmp"
            attempts_dir.mkdir(parents=True, exist_ok=True)
            runtime_data.update(
                {
                    "task_config": task_config,
                    "output_dir": output_dir,
                    "attempts_dir": attempts_dir,
                    "materials": [],
                    "validation_reports": {},
                    "package_validator_called": False,
                }
            )
            self.trace.log(
                "task.start",
                task_id=task_id,
                lesson_number=lesson_number,
                lesson_title=lesson_title,
                course_level=course_level,
                prompts_dir=str(task_config.prompts_dir),
            )
            return {
                "task_id": task_id,
                "lesson_number": lesson_number,
                "lesson_title": lesson_title,
                "course_level": course_level,
                "output_dir": str(output_dir),
                "current_material_index": 0,
                "package_validator_called": False,
                "stop_generation": False,
            }

        def plan_node(state: LessonTaskGraphState) -> dict[str, Any]:
            task = state["task"]
            task_config = runtime_data["task_config"]
            specs = build_material_plan(task, task_config)
            runtime_data["specs"] = specs
            self.trace.log(
                "planner.done",
                course_level=state["course_level"],
                material_plan=[
                    {"kind": spec.kind, "agent": spec.agent_type, "prompt_files": list(spec.prompt_files)}
                    for spec in specs
                ],
            )
            return {}

        def load_references_node(state: LessonTaskGraphState) -> dict[str, Any]:
            references = ReferenceLoader(runtime_data["task_config"], trace=self.trace).load(state["task"])
            runtime_data["references"] = references
            return {}

        def route_material_node(state: LessonTaskGraphState) -> dict[str, Any]:
            return {}

        def route_next_material(state: LessonTaskGraphState) -> str:
            if int(state.get("current_material_index") or 0) < len(runtime_data.get("specs") or []):
                return "run_material"
            return "package_validation"

        def run_material_node(state: LessonTaskGraphState) -> dict[str, Any]:
            task = state["task"]
            specs = runtime_data["specs"]
            spec = specs[int(state.get("current_material_index") or 0)]
            task_config = runtime_data["task_config"]
            attempts_dir = runtime_data["attempts_dir"]
            materials = list(runtime_data.get("materials") or [])
            validation_reports = dict(runtime_data.get("validation_reports") or {})
            dependencies = self._dependency_results(spec.dependency_kinds, specs, materials)
            skip_reason = practice_material_skip_reason(task, spec)
            skip_status = "skipped"
            if skip_reason is None:
                skip_reason = dependency_skip_reason(spec, dependencies)
                skip_status = "skipped_dependency"
            if skip_reason is not None:
                material = build_skipped_material(
                    spec=spec,
                    status=skip_status,
                    reason=skip_reason,
                    dependency_results=dependencies,
                )
                validation = ValidationResult(
                    approved=True,
                    passed_blocks=[
                        {
                            "block_id": spec.kind,
                            "block_heading": spec.material_type,
                            "reason": skip_reason,
                        }
                    ],
                )
                self.trace.log(
                    "material.skipped",
                    kind=spec.kind,
                    status=material.status,
                    reason=skip_reason,
                    dependencies=[{"kind": item.kind, "status": item.status} for item in dependencies],
                )
            else:
                self.trace.log(
                    "material.start",
                    kind=spec.kind,
                    agent=spec.agent_type,
                    dependencies=[{"kind": item.kind, "status": item.status} for item in dependencies],
                )
                if spec.kind == "practice_guidance":
                    material = run_practice_guidance_material(
                        task=task,
                        spec=spec,
                        config=task_config,
                        subagents=self.subagents,
                        references=runtime_data["references"],
                        materials=materials,
                        output_dir=runtime_data["output_dir"],
                        attempts_dir=attempts_dir,
                        trace=self.trace,
                        rule_validator=rule_validator,
                    )
                else:
                    worker = MaterialWorker(
                        subagents=self.subagents,
                        config=task_config,
                        rule_validator=rule_validator,
                        trace=self.trace,
                    )
                    material = worker.run(
                        task=task,
                        spec=spec,
                        references=runtime_data["references"],
                        dependency_results=dependencies,
                        attempts_dir=attempts_dir,
                    )
                self.trace.log(
                    "material.done",
                    kind=material.kind,
                    status=material.status,
                    iterations=material.iterations,
                    content_chars=len(material.content),
                    issues=material.validation_issues,
                )
                validation = ValidationResult(
                    approved=material.status == "approved",
                    issues=list(material.validation_issues),
                    fix_instructions=list(material.validation_issues),
                    issues_by_block=list(material.validation_issues_by_block),
                    passed_blocks=list(material.validation_passed_blocks),
                )

            materials.append(material)
            validation_reports[spec.kind] = validation
            runtime_data["materials"] = materials
            runtime_data["validation_reports"] = validation_reports
            result_update: dict[str, Any] = {
                "current_material_index": int(state.get("current_material_index") or 0) + 1,
            }
            if material.status == "failed":
                package_validation = ValidationResult.fail(
                    [
                        f"material {material.kind} failed after {material.iterations} generation/validation attempts; execution stopped"
                    ]
                )
                validation_reports["package"] = package_validation
                runtime_data["package_validation"] = package_validation
                runtime_data["validation_reports"] = validation_reports
                runtime_data["package_validator_called"] = False
                self.trace.log(
                    "task.fail_fast",
                    kind=material.kind,
                    iterations=material.iterations,
                    issues=material.validation_issues,
                )
                result_update.update(
                    {
                        "package_validator_called": False,
                        "stop_generation": True,
                    }
                )
            return result_update

        def route_after_material(state: LessonTaskGraphState) -> str:
            return "finish_task" if state.get("stop_generation") else "route_material"

        def package_validation_node(state: LessonTaskGraphState) -> dict[str, Any]:
            materials = list(runtime_data.get("materials") or [])
            validation_reports = dict(runtime_data.get("validation_reports") or {})
            if any(item.status in SKIPPED_MATERIAL_STATUSES for item in materials):
                package_validation = ValidationResult(
                    approved=True,
                    passed_blocks=[
                        {
                            "block_id": "package",
                            "block_heading": "Package",
                            "reason": "package validation skipped because one or more materials were intentionally skipped",
                        }
                    ],
                )
                self.trace.log(
                    "package.skipped_due_to_material_skips",
                    skipped=[
                        {"kind": item.kind, "status": item.status}
                        for item in materials
                        if item.status in SKIPPED_MATERIAL_STATUSES
                    ],
                )
                validation_reports["package"] = package_validation
                runtime_data["package_validation"] = package_validation
                runtime_data["validation_reports"] = validation_reports
                runtime_data["package_validator_called"] = False
                return {
                    "package_validator_called": False,
                }

            self.trace.log("package.start", material_count=len(materials))
            package_validator = PackageValidator(
                subagents=self.subagents,
                config=runtime_data["task_config"],
                rule_validator=rule_validator,
                trace=self.trace,
            )
            package_validation = package_validator.validate(
                task=state["task"],
                specs=runtime_data["specs"],
                materials=materials,
                attempts_dir=runtime_data["attempts_dir"],
            )
            if not package_validation.approved:
                self.trace.log("package.advisory_not_blocking", issues=package_validation.issues)
            validation_reports["package"] = package_validation
            runtime_data["package_validation"] = package_validation
            runtime_data["validation_reports"] = validation_reports
            runtime_data["package_validator_called"] = True
            return {
                "package_validator_called": True,
            }

        def finish_task_node(state: LessonTaskGraphState) -> dict[str, Any]:
            result = self._finish_task(
                task_id=state["task_id"],
                lesson_number=state["lesson_number"],
                lesson_title=state["lesson_title"],
                course_level=state["course_level"],
                output_dir=runtime_data["output_dir"],
                materials=list(runtime_data.get("materials") or []),
                references=runtime_data["references"],
                package_validation=runtime_data["package_validation"],
                validation_reports=dict(runtime_data.get("validation_reports") or {}),
                package_validator_called=bool(runtime_data.get("package_validator_called")),
            )
            result_box["result"] = result
            return {
                "result": result.to_public_json(),
                "output_dir": result.output_dir,
            }

        builder = StateGraph(LessonTaskGraphState)
        builder.add_node("init_task", init_task_node)
        builder.add_node("plan", plan_node)
        builder.add_node("load_references", load_references_node)
        builder.add_node("route_material", route_material_node)
        builder.add_node("run_material", run_material_node)
        builder.add_node("package_validation", package_validation_node)
        builder.add_node("finish_task", finish_task_node)
        builder.add_edge(START, "init_task")
        builder.add_edge("init_task", "plan")
        builder.add_edge("plan", "load_references")
        builder.add_edge("load_references", "route_material")
        builder.add_conditional_edges(
            "route_material",
            route_next_material,
            {
                "run_material": "run_material",
                "package_validation": "package_validation",
            },
        )
        builder.add_conditional_edges(
            "run_material",
            route_after_material,
            {
                "route_material": "route_material",
                "finish_task": "finish_task",
            },
        )
        builder.add_edge("package_validation", "finish_task")
        builder.add_edge("finish_task", END)
        return builder.compile(name="ismart_lesson_task_graph")

    def _finish_task(
        self,
        *,
        task_id: str,
        lesson_number: str,
        lesson_title: str,
        course_level: str,
        output_dir: Path,
        materials: list[MaterialResult],
        references: Any,
        package_validation: ValidationResult,
        validation_reports: dict[str, ValidationResult],
        package_validator_called: bool,
    ) -> IsmartGenerationResult:
        result = IsmartGenerationResult(
            task_id=task_id,
            lesson_number=lesson_number,
            lesson_title=lesson_title,
            course_level=course_level,
            status=self._result_status(materials, package_validation),
            output_dir=str(output_dir),
            materials=materials,
            package_validation=package_validation,
            reference_summary=reference_summary(references),
            agents_called=self._agents_called(materials, package_validator_called=package_validator_called),
            prompt_files_used=self._prompt_files_used(materials),
        )
        validation_reports["package"] = package_validation
        self.trace.log("output.write.start", output_dir=str(output_dir), material_count=len(materials))
        write_task_output(result=result, output_dir=output_dir, validation_reports=validation_reports)
        self.trace.log("output.write.done", output_dir=str(output_dir), status=result.status)
        self.trace.log("task.done", task_id=task_id, status=result.status, output_dir=str(output_dir))
        return result

    def _dependency_results(
        self,
        dependency_kinds: tuple[str, ...],
        specs: list[Any],
        materials: list[MaterialResult],
    ) -> list[MaterialResult]:
        planned_kinds = {spec.kind for spec in specs}
        material_by_kind = {item.kind: item for item in materials}
        return [
            material_by_kind[kind]
            for kind in dependency_kinds
            if kind in planned_kinds and kind in material_by_kind
        ]

    def _repair_package(
        self,
        *,
        task: dict[str, Any],
        specs: list[Any],
        references: Any,
        materials: list[MaterialResult],
        package_validation: ValidationResult,
        validation_reports: dict[str, ValidationResult],
        attempts_dir: Path,
    ) -> ValidationResult:
        current_validation = package_validation
        for iteration in range(1, self.config.max_package_repair_iterations + 1):
            affected = self._affected_specs(specs, current_validation.issues)
            if not affected:
                self.trace.log("package.repair.no_affected_materials", iteration=iteration)
                break
            self.trace.log(
                "package.repair.iteration",
                iteration=iteration,
                affected=[spec.kind for spec in affected],
                issues=current_validation.issues,
            )
            for spec in affected:
                dependencies = self._dependency_results(spec.dependency_kinds, specs, materials)
                revised = self.worker.run(
                    task=task,
                    spec=spec,
                    references=references,
                    dependency_results=dependencies,
                    initial_previous_issues=current_validation.issues,
                    attempts_dir=attempts_dir,
                )
                for index, material in enumerate(materials):
                    if material.kind == spec.kind:
                        materials[index] = revised
                        break
                self.trace.log(
                    "package.repair.material_done",
                    iteration=iteration,
                    kind=spec.kind,
                    status=revised.status,
                    issues=revised.validation_issues,
                )
                validation_reports[spec.kind] = ValidationResult(
                    approved=revised.status == "approved",
                    issues=list(revised.validation_issues),
                    fix_instructions=list(revised.validation_issues),
                    issues_by_block=list(revised.validation_issues_by_block),
                    passed_blocks=list(revised.validation_passed_blocks),
                )
                if revised.status == "failed":
                    self.trace.log(
                        "package.repair.fail_fast",
                        iteration=iteration,
                        kind=spec.kind,
                        issues=revised.validation_issues,
                    )
                    return ValidationResult.fail(
                        [f"material {spec.kind} failed during package repair after {revised.iterations} generation/validation attempts; execution stopped"]
                    )
            current_validation = self.package_validator.validate(task=task, specs=specs, materials=materials, attempts_dir=attempts_dir)
            if current_validation.approved:
                self.trace.log("package.repair.approved", iteration=iteration)
                return current_validation
        self.trace.log("package.repair.done", approved=current_validation.approved, issues=current_validation.issues)
        return current_validation

    def _affected_specs(self, specs: list[Any], issues: list[str]) -> list[Any]:
        affected = []
        for spec in specs:
            for issue in issues:
                if spec.kind in issue or spec.material_type in issue or spec.validator_kind in issue:
                    affected.append(spec)
                    break
        return affected

    def _new_run_dir(self, task: dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.output_root / f"run_{timestamp}_{default_run_name(task)}"

    def _result_status(self, materials: list[MaterialResult], package_validation: ValidationResult) -> str:
        if any(item.status == "failed" for item in materials):
            return "failed"
        if any(item.status == "blocked_dependency" for item in materials):
            return "failed"
        if any(item.status in SKIPPED_MATERIAL_STATUSES for item in materials):
            return "completed_with_skips"
        return "approved"

    def _agents_called(self, materials: list[MaterialResult], *, package_validator_called: bool) -> list[str]:
        agents: list[str] = []
        for item in materials:
            if item.kind == "practice" and item.generation_artifacts:
                agents.extend(["PracticeTaskTemplateAgent", "PracticeTaskVariantAgent"])
            if item.kind == "self_work" and item.generation_artifacts:
                agents.append("SelfWorkAutocheckAgent")
            if item.kind == "current_control" and item.generation_artifacts:
                agents.append("CurrentControlAutocheckAgent")
            if item.kind == "intermediate" and item.generation_artifacts:
                agents.append("IntermediateAssessmentArtifactAgent")
            agents.append(item.agent_type)
        if self.config.use_llm_validator and materials:
            agents.append("MaterialValidatorAgent")
        if any(item.controller_called for item in materials):
            agents.append("ValidationControllerAgent")
        if package_validator_called:
            agents.append("PackageValidatorAgent")
        return list(dict.fromkeys(agents))

    def _prompt_files_used(self, materials: list[MaterialResult]) -> list[str]:
        files: list[str] = []
        for material in materials:
            files.extend(material.prompt_files)
        return list(dict.fromkeys(files))
