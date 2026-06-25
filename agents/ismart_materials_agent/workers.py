from __future__ import annotations

from typing import Any

from .context import (
    build_generation_prompt,
    build_generator_system_prompt,
    build_package_validation_prompt,
    build_validation_prompt,
    build_validator_system_prompt,
)
from .contracts import (
    IsmartGenerationConfig,
    JsonLLMClient,
    MaterialResult,
    MaterialSpec,
    ReferenceBundle,
    ValidationResult,
)
from .sources import read_prompt_files
from .validators import RuleValidator


class MaterialWorker:
    def __init__(
        self,
        *,
        client: JsonLLMClient,
        config: IsmartGenerationConfig,
        rule_validator: RuleValidator | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.rule_validator = rule_validator or RuleValidator()

    def run(
        self,
        *,
        task: dict[str, Any],
        spec: MaterialSpec,
        references: ReferenceBundle,
        dependency_results: list[MaterialResult],
        module_material_summaries: dict[str, list[dict[str, Any]]] | None = None,
        initial_previous_issues: list[str] | None = None,
    ) -> MaterialResult:
        blocked = [item for item in dependency_results if item.status != "approved"]
        if blocked:
            issues = [f"dependency {item.kind} has status {item.status}" for item in blocked]
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
        previous_content = ""
        previous_issues = list(initial_previous_issues or [])

        for attempt in range(1, self.config.max_generation_iterations + 1):
            generation_prompt = build_generation_prompt(
                task=task,
                spec=spec,
                prompt_contents=prompt_contents,
                references=references,
                dependencies=dependency_results,
                previous_content=previous_content,
                previous_issues=previous_issues,
                module_material_summaries=module_material_summaries,
            )
            generated = self.client.complete_json(
                system=build_generator_system_prompt(spec),
                user=generation_prompt,
            )
            content = str(generated.get("content", "")).strip()
            agent_notes = [str(item) for item in generated.get("agent_notes", [])]
            if not content:
                validation = ValidationResult.fail([f"{spec.agent_type} returned empty content"])
            else:
                rule_result = self.rule_validator.validate_material(content, spec, task)
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

            if validation.approved:
                return MaterialResult(
                    kind=spec.kind,
                    material_type=spec.material_type,
                    agent_type=spec.agent_type,
                    status="approved",
                    iterations=attempt,
                    content=content,
                    prompt_files=spec.prompt_files,
                    agent_notes=agent_notes,
                )

            previous_content = content
            previous_issues = validation.issues

        return MaterialResult(
            kind=spec.kind,
            material_type=spec.material_type,
            agent_type=spec.agent_type,
            status="failed",
            iterations=self.config.max_generation_iterations,
            content=previous_content,
            prompt_files=spec.prompt_files,
            validation_issues=previous_issues,
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
            return ValidationResult.ok()
        validation_prompt = build_validation_prompt(
            task=task,
            spec=spec,
            prompt_contents=prompt_contents,
            references=references,
            dependencies=dependency_results,
            content=content,
            rule_result=rule_result,
        )
        data = self.client.complete_json(
            system=build_validator_system_prompt(),
            user=validation_prompt,
        )
        return ValidationResult(
            approved=bool(data.get("approved")) and not rule_result.issues,
            issues=[str(item) for item in data.get("issues", [])],
            fix_instructions=[str(item) for item in data.get("fix_instructions", data.get("issues", []))],
        )


class PackageValidator:
    def __init__(
        self,
        *,
        client: JsonLLMClient,
        config: IsmartGenerationConfig,
        rule_validator: RuleValidator | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.rule_validator = rule_validator or RuleValidator()

    def validate(
        self,
        *,
        task: dict[str, Any],
        specs: list[MaterialSpec],
        materials: list[MaterialResult],
    ) -> ValidationResult:
        rule_result = self.rule_validator.validate_package(specs, materials)
        if not self.config.use_llm_validator:
            return rule_result
        prompt = build_package_validation_prompt(
            task=task,
            specs=specs,
            materials=materials,
            rule_result=rule_result,
        )
        data = self.client.complete_json(
            system=(
                "Ты PackageValidatorAgent. Проверяешь комплект материалов, не перегенерируешь "
                "и не исправляешь материалы. Верни строго JSON."
            ),
            user=prompt,
        )
        llm_result = ValidationResult(
            approved=bool(data.get("approved")) and not rule_result.issues,
            issues=[str(item) for item in data.get("issues", [])],
            fix_instructions=[str(item) for item in data.get("fix_instructions", data.get("issues", []))],
        )
        return rule_result.merge(llm_result)
