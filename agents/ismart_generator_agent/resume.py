from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .context import task_identity
from .contracts import IsmartGenerationResult, MaterialResult, MaterialSpec, ValidationResult


@dataclass
class ExistingPackage:
    output_dir: Path
    result: dict[str, Any]
    manifest: dict[str, Any]
    validation_reports: dict[str, ValidationResult]
    html_files: set[str]
    materials_by_kind: dict[str, MaterialResult]
    material_files_by_kind: dict[str, str]
    unusable_reasons_by_kind: dict[str, list[str]] = field(default_factory=dict)

    def material(self, kind: str) -> MaterialResult | None:
        return self.materials_by_kind.get(kind)

    def validation(self, kind: str) -> ValidationResult:
        return self.validation_reports.get(kind) or validation_from_material(self.materials_by_kind[kind])

    def to_generation_result(self) -> IsmartGenerationResult:
        materials = list(self.materials_by_kind.values())
        package_validation = _package_validation_from_result(self.result)
        return IsmartGenerationResult(
            task_id=str(self.result.get("task_id") or ""),
            lesson_number=str(self.result.get("lesson_number") or ""),
            lesson_title=str(self.result.get("lesson_title") or ""),
            course_level=str(self.result.get("resolved_profile") or self.result.get("course_level") or "basic"),
            status=str(self.result.get("status") or "approved"),
            output_dir=str(self.output_dir),
            materials=materials,
            package_validation=package_validation,
            reference_summary=self.result.get("references") if isinstance(self.result.get("references"), dict) else {},
            agents_called=list(self.result.get("agents_called") or []),
            prompt_files_used=list(self.result.get("prompt_files_used") or []),
        )


def load_existing_package(lesson_output_dir: Path) -> ExistingPackage:
    result_path = lesson_output_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"existing lesson output has no result.json: {lesson_output_dir}")
    result = _read_json(result_path)
    manifest_path = lesson_output_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    html_files = {path.name for path in lesson_output_dir.glob("*.html")}
    material_files = _material_files_from_manifest(manifest)
    materials = {
        item.kind: item
        for item in (
            material_from_json(material)
            for material in result.get("materials") or []
            if isinstance(material, dict)
        )
    }
    validation_reports = _read_validation_reports(lesson_output_dir / "validation_reports")
    return ExistingPackage(
        output_dir=lesson_output_dir,
        result=result,
        manifest=manifest,
        validation_reports=validation_reports,
        html_files=html_files,
        materials_by_kind=materials,
        material_files_by_kind=material_files,
    )


def existing_package_from_payload(payload: dict[str, Any], fallback_output_dir: Path) -> ExistingPackage:
    result = dict(payload.get("result") or payload)
    manifest = dict(payload.get("manifest") or {})
    output_dir = Path(str(payload.get("output_dir") or result.get("output_dir") or fallback_output_dir))
    html_files = {str(item) for item in (payload.get("html_files") or [])}
    if not html_files and output_dir.exists():
        html_files = {path.name for path in output_dir.glob("*.html")}
    material_files = _material_files_from_manifest(manifest)
    materials = {
        item.kind: item
        for item in (
            material_from_json(material)
            for material in result.get("materials") or []
            if isinstance(material, dict)
        )
    }
    validation_reports = _validation_reports_from_payload(payload.get("validation_reports"))
    return ExistingPackage(
        output_dir=output_dir,
        result=result,
        manifest=manifest,
        validation_reports=validation_reports,
        html_files=html_files,
        materials_by_kind=materials,
        material_files_by_kind=material_files,
    )


def find_existing_lesson_dir(root_or_lesson_dir: Path, task: dict[str, Any]) -> Path:
    if (root_or_lesson_dir / "result.json").exists():
        package = load_existing_package(root_or_lesson_dir)
        if _package_matches_task(package.result, task):
            return root_or_lesson_dir
        task_id, lesson_number, _ = task_identity(task)
        raise ValueError(
            f"existing lesson output {root_or_lesson_dir} does not match task_id={task_id!r}, "
            f"lesson_number={lesson_number!r}"
        )

    if not root_or_lesson_dir.exists():
        raise FileNotFoundError(f"existing output path does not exist: {root_or_lesson_dir}")
    if not root_or_lesson_dir.is_dir():
        raise ValueError(f"existing output path must be a directory: {root_or_lesson_dir}")

    matches: list[Path] = []
    for child in root_or_lesson_dir.iterdir():
        if not child.is_dir() or not (child / "result.json").exists():
            continue
        try:
            result = _read_json(child / "result.json")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if _package_matches_task(result, task):
            matches.append(child)
    if len(matches) == 1:
        return matches[0]
    task_id, lesson_number, _ = task_identity(task)
    if not matches:
        raise FileNotFoundError(
            f"no existing lesson output found in {root_or_lesson_dir} for task_id={task_id!r}, "
            f"lesson_number={lesson_number!r}"
        )
    raise ValueError(
        f"multiple existing lesson outputs found in {root_or_lesson_dir} for task_id={task_id!r}, "
        f"lesson_number={lesson_number!r}: {matches}"
    )


def material_from_json(data: dict[str, Any]) -> MaterialResult:
    return MaterialResult(
        kind=str(data.get("kind") or ""),
        material_type=str(data.get("type") or data.get("material_type") or ""),
        agent_type=str(data.get("agent") or data.get("agent_type") or ""),
        status=str(data.get("status") or "failed"),  # type: ignore[arg-type]
        iterations=int(data.get("iterations") or 0),
        content=str(data.get("content") or ""),
        prompt_files=tuple(str(item) for item in (data.get("prompt_files") or [])),
        validation_issues=[str(item) for item in (data.get("validation_issues") or [])],
        validation_issues_by_block=[
            dict(item) for item in (data.get("validation_issues_by_block") or []) if isinstance(item, dict)
        ],
        validation_passed_blocks=[
            dict(item) for item in (data.get("validation_passed_blocks") or []) if isinstance(item, dict)
        ],
        agent_notes=[str(item) for item in (data.get("agent_notes") or [])],
        controller_called=bool(data.get("controller_called")),
        controller_decision=dict(data.get("controller_decision") or {}),
        generation_artifacts=dict(data.get("generation_artifacts") or {}),
    )


def validation_from_material(material: MaterialResult) -> ValidationResult:
    return ValidationResult(
        approved=material.status == "approved",
        issues=list(material.validation_issues),
        fix_instructions=list(material.validation_issues),
        issues_by_block=list(material.validation_issues_by_block),
        passed_blocks=list(material.validation_passed_blocks),
    )


def reusable_material(package: ExistingPackage, spec: MaterialSpec) -> tuple[MaterialResult | None, list[str]]:
    material = package.material(spec.kind)
    if material is None:
        return None, ["material is missing in result.json"]
    reasons: list[str] = []
    if material.status != "approved":
        reasons.append(f"material status is {material.status!r}, not 'approved'")
    if not material.content.strip():
        reasons.append("material content is empty")
    filename = package.material_files_by_kind.get(spec.kind)
    if not filename:
        reasons.append("material has no file entry in manifest.json")
    elif filename not in package.html_files:
        reasons.append(f"material file does not exist: {filename}")
    artifact_issue = _required_artifact_issue(material)
    if artifact_issue:
        reasons.append(artifact_issue)
    return (material if not reasons else None), reasons


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _material_files_from_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    files: dict[str, str] = {}
    for material in manifest.get("materials") or []:
        if not isinstance(material, dict):
            continue
        kind = str(material.get("kind") or "").strip()
        filename = str(material.get("file") or "").strip()
        if kind and filename:
            files[kind] = filename
    return files


def _read_validation_reports(reports_dir: Path) -> dict[str, ValidationResult]:
    reports: dict[str, ValidationResult] = {}
    if not reports_dir.exists():
        return reports
    for path in reports_dir.glob("*.json"):
        try:
            data = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        reports[path.stem.replace("-", "_")] = ValidationResult(
            approved=bool(data.get("approved")),
            issues=[str(item) for item in (data.get("issues") or [])],
            fix_instructions=[str(item) for item in (data.get("fix_instructions") or [])],
            issues_by_block=[dict(item) for item in (data.get("issues_by_block") or []) if isinstance(item, dict)],
            passed_blocks=[dict(item) for item in (data.get("passed_blocks") or []) if isinstance(item, dict)],
        )
    return reports


def _validation_reports_from_payload(value: Any) -> dict[str, ValidationResult]:
    if not isinstance(value, dict):
        return {}
    reports: dict[str, ValidationResult] = {}
    for key, data in value.items():
        if isinstance(data, ValidationResult):
            reports[str(key)] = data
            continue
        if not isinstance(data, dict):
            continue
        reports[str(key).replace("-", "_")] = ValidationResult(
            approved=bool(data.get("approved")),
            issues=[str(item) for item in (data.get("issues") or [])],
            fix_instructions=[str(item) for item in (data.get("fix_instructions") or [])],
            issues_by_block=[dict(item) for item in (data.get("issues_by_block") or []) if isinstance(item, dict)],
            passed_blocks=[dict(item) for item in (data.get("passed_blocks") or []) if isinstance(item, dict)],
        )
    return reports


def _package_matches_task(result: Mapping[str, Any], task: dict[str, Any]) -> bool:
    task_id, lesson_number, _ = task_identity(task)
    return str(result.get("task_id") or "") == task_id and str(result.get("lesson_number") or "") == lesson_number


def _package_validation_from_result(result: dict[str, Any]) -> ValidationResult:
    package = result.get("package_validation") if isinstance(result.get("package_validation"), dict) else {}
    return ValidationResult(
        approved=bool(package.get("approved", result.get("status") == "approved")),
        issues=[str(item) for item in (package.get("issues") or [])],
        fix_instructions=[str(item) for item in (package.get("fix_instructions") or [])],
    )


def _required_artifact_issue(material: MaterialResult) -> str:
    artifacts = material.generation_artifacts if isinstance(material.generation_artifacts, dict) else {}
    if material.kind == "practice":
        templates = artifacts.get("practice_templates")
        instances = artifacts.get("practice_instances")
        tasks = instances.get("tasks") if isinstance(instances, dict) else None
        if not isinstance(templates, dict) or not isinstance(instances, dict) or not isinstance(tasks, list) or not tasks:
            return "practice generation_artifacts must include practice_templates and practice_instances.tasks"
    if material.kind == "self_work" and not isinstance(artifacts.get("self_work_autocheck"), dict):
        return "self_work generation_artifacts must include self_work_autocheck"
    if material.kind == "current_control" and not isinstance(artifacts.get("current_control_autocheck"), dict):
        return "current_control generation_artifacts must include current_control_autocheck"
    if material.kind == "intermediate" and not isinstance(artifacts.get("intermediate_assessment"), dict):
        return "intermediate generation_artifacts must include intermediate_assessment"
    return ""
