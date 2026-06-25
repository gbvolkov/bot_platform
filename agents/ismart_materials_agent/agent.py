from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .context import material_result_summary, task_identity
from .contracts import (
    IsmartGenerationConfig,
    IsmartGenerationResult,
    JsonLLMClient,
    MaterialResult,
    ValidationResult,
)
from .llm import OpenAICompatibleJsonClient
from .planner import build_material_plan
from .sources import ReferenceLoader, reference_summary
from .validators import RuleValidator
from .workers import MaterialWorker, PackageValidator
from .writer import default_run_name, write_task_output


def initialize_agent(
    config: IsmartGenerationConfig | None = None,
    *,
    client: JsonLLMClient | None = None,
) -> "IsmartMaterialsRuntime":
    return IsmartMaterialsRuntime(config=config or IsmartGenerationConfig(), client=client)


def run_ismart_task(
    task: dict[str, Any],
    config: IsmartGenerationConfig | None = None,
    *,
    client: JsonLLMClient | None = None,
    run_dir: str | Path | None = None,
    module_material_summaries: dict[str, list[dict[str, Any]]] | None = None,
) -> IsmartGenerationResult:
    runtime = initialize_agent(config=config, client=client)
    return runtime.run_task(
        task,
        run_dir=Path(run_dir) if run_dir is not None else None,
        module_material_summaries=module_material_summaries,
    )


class IsmartMaterialsRuntime:
    def __init__(
        self,
        *,
        config: IsmartGenerationConfig,
        client: JsonLLMClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAICompatibleJsonClient(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
        )
        self.rule_validator = RuleValidator()
        self.worker = MaterialWorker(
            client=self.client,
            config=self.config,
            rule_validator=self.rule_validator,
        )
        self.package_validator = PackageValidator(
            client=self.client,
            config=self.config,
            rule_validator=self.rule_validator,
        )

    def run_task(
        self,
        task: dict[str, Any],
        *,
        run_dir: Path | None = None,
        module_material_summaries: dict[str, list[dict[str, Any]]] | None = None,
    ) -> IsmartGenerationResult:
        specs = build_material_plan(task, self.config)
        references = ReferenceLoader(self.config).load(task)
        materials: list[MaterialResult] = []
        validation_reports: dict[str, ValidationResult] = {}

        for spec in specs:
            dependencies = self._dependency_results(spec.dependency_kinds, specs, materials)
            material = self.worker.run(
                task=task,
                spec=spec,
                references=references,
                dependency_results=dependencies,
                module_material_summaries=module_material_summaries,
            )
            materials.append(material)
            validation_reports[spec.kind] = ValidationResult(
                approved=material.status == "approved",
                issues=list(material.validation_issues),
                fix_instructions=list(material.validation_issues),
            )

        package_validation = self.package_validator.validate(
            task=task,
            specs=specs,
            materials=materials,
        )
        if not package_validation.approved:
            package_validation = self._repair_package(
                task=task,
                specs=specs,
                references=references,
                materials=materials,
                package_validation=package_validation,
                validation_reports=validation_reports,
                module_material_summaries=module_material_summaries,
            )

        task_id, lesson_number, lesson_title = task_identity(task)
        output_dir = run_dir or self._new_run_dir(task)
        result = IsmartGenerationResult(
            task_id=task_id,
            lesson_number=lesson_number,
            lesson_title=lesson_title,
            status=self._result_status(materials, package_validation),
            output_dir=str(output_dir),
            materials=materials,
            package_validation=package_validation,
            reference_summary=reference_summary(references),
            agents_called=self._agents_called(materials),
            prompt_files_used=self._prompt_files_used(materials),
        )
        validation_reports["package"] = package_validation
        write_task_output(result=result, output_dir=output_dir, validation_reports=validation_reports)
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
        module_material_summaries: dict[str, list[dict[str, Any]]] | None,
    ) -> ValidationResult:
        current_validation = package_validation
        for _ in range(self.config.max_package_repair_iterations):
            affected = self._affected_specs(specs, current_validation.issues)
            if not affected:
                break
            for spec in affected:
                dependencies = self._dependency_results(spec.dependency_kinds, specs, materials)
                revised = self.worker.run(
                    task=task,
                    spec=spec,
                    references=references,
                    dependency_results=dependencies,
                    module_material_summaries=module_material_summaries,
                    initial_previous_issues=current_validation.issues,
                )
                for index, material in enumerate(materials):
                    if material.kind == spec.kind:
                        materials[index] = revised
                        break
                validation_reports[spec.kind] = ValidationResult(
                    approved=revised.status == "approved",
                    issues=list(revised.validation_issues),
                    fix_instructions=list(revised.validation_issues),
                )
            current_validation = self.package_validator.validate(task=task, specs=specs, materials=materials)
            if current_validation.approved:
                return current_validation
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
        if not package_validation.approved:
            return "package_needs_human_review"
        return "approved"

    def _agents_called(self, materials: list[MaterialResult]) -> list[str]:
        agents = [item.agent_type for item in materials]
        agents.extend(["MaterialValidatorAgent", "PackageValidatorAgent"])
        return list(dict.fromkeys(agents))

    def _prompt_files_used(self, materials: list[MaterialResult]) -> list[str]:
        files: list[str] = []
        for material in materials:
            files.extend(material.prompt_files)
        return list(dict.fromkeys(files))
