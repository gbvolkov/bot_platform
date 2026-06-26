from __future__ import annotations

import html
import re
from typing import Any

from .contracts import MaterialResult, MaterialSpec, ValidationResult


SERVICE_MARKERS = ("QA-ID", "SHA", "candidate_internal", "leakage", "artifact_id")
SOURCE_EXTENSIONS_FORBIDDEN = (".docx", ".pdf", ".html", ".xlsx", ".xls")


def html_to_text(content: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", content)
    return html.unescape(re.sub(r"\s+", " ", without_tags)).strip()


class RuleValidator:
    def validate_material(
        self,
        content: str,
        spec: MaterialSpec,
        task: dict[str, Any],
    ) -> ValidationResult:
        issues: list[str] = []

        issues.extend(self._formatting_issues(content))
        issues.extend(self._source_boundary_issues(content, spec.validator_kind))

        if spec.validator_kind != "qa":
            for marker in SERVICE_MARKERS:
                if marker in content:
                    issues.append(f"служебный маркер {marker} найден вне QA")

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
            if material.status != "approved":
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

    def _source_boundary_issues(self, content: str, kind: str) -> list[str]:
        text = html_to_text(content)
        issues = []
        for ext in SOURCE_EXTENSIONS_FORBIDDEN:
            if ext in text:
                issues.append(f"материал содержит запрещённую ссылку на исходный формат {ext}")
        if kind != "qa" and "docs/ismart/" in text and "референсы/" in text:
            issues.append("источники/локаторы Markdown попали вне QA")
        return issues
