from __future__ import annotations

import re
from typing import Any

from .contracts import MaterialResult, MaterialSpec, ValidationResult
from .task_skip import SKIPPED_MATERIAL_STATUSES


class RuleValidator:
    def validate_material(
        self,
        content: str,
        spec: MaterialSpec,
        task: dict[str, Any],
    ) -> ValidationResult:
        issues: list[str] = []

        issues.extend(self._formatting_issues(content))

        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def validate_package(
        self,
        specs: list[MaterialSpec],
        materials: list[MaterialResult],
    ) -> ValidationResult:
        issues: list[str] = []
        material_by_kind = {item.kind: item for item in materials}
        for spec in specs:
            material = material_by_kind.get(spec.kind)
            if material is None:
                issues.append(f"отсутствует материал: {spec.kind}")
                continue
            if material.status != "approved" and material.status not in SKIPPED_MATERIAL_STATUSES:
                issues.append(f"материал {spec.kind} имеет статус {material.status}")

        for spec in specs:
            for dependency_kind in spec.dependency_kinds:
                if dependency_kind not in {item.kind for item in specs}:
                    continue
                dependency = material_by_kind.get(dependency_kind)
                material = material_by_kind.get(spec.kind)
                if dependency is None or material is None:
                    continue
                if materials.index(material) < materials.index(dependency):
                    issues.append(f"материал {spec.kind} стоит раньше зависимости {dependency_kind}")

        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _formatting_issues(self, content: str) -> list[str]:
        issues: list[str] = []
        if not content.startswith("<style>"):
            issues.append("HTML не начинается с <style>")
        if '<div class="cc-lesson">' not in content:
            issues.append("нет div.cc-lesson")
        if "<script" in content.lower() or "<link" in content.lower():
            issues.append("HTML содержит script/link")
        return issues
