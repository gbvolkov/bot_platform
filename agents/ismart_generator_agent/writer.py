from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .contracts import IsmartGenerationResult, MaterialResult, ValidationResult
from .context import task_identity
from .task_skip import SKIPPED_MATERIAL_STATUSES


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_slug(value: str, fallback: str = "task") -> str:
    text = value.strip().lower()
    replacements = {
        " ": "-",
        "_": "-",
        "—": "-",
        "–": "-",
        ".": "-",
        ":": "-",
        "/": "-",
        "\\": "-",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[^0-9a-zа-яё-]+", "", text, flags=re.I)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or fallback


def default_run_name(task: dict[str, Any]) -> str:
    task_id, lesson_number, _ = task_identity(task)
    return safe_slug(str(task_id or lesson_number), fallback="task")


def material_filename(index: int, material: MaterialResult) -> str:
    return f"{index:02d}_{safe_slug(material.kind)}.html"


def write_task_output(
    *,
    result: IsmartGenerationResult,
    output_dir: Path,
    validation_reports: dict[str, ValidationResult],
    material_file_overrides: dict[str, str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_error = output_dir / "error.json"
    if stale_error.exists():
        stale_error.unlink()
    material_files: dict[str, str] = dict(material_file_overrides or {})
    used_filenames = {value for value in material_files.values() if value}
    for index, material in enumerate(result.materials, start=1):
        if material.status in SKIPPED_MATERIAL_STATUSES:
            continue
        filename = material_files.get(material.kind)
        if not filename:
            filename = _next_material_filename(output_dir, material.kind, used_filenames, start_index=index)
            material_files[material.kind] = filename
            used_filenames.add(filename)
        (output_dir / filename).write_text(material.content, encoding="utf-8")

    manifest = {
        "task_id": result.task_id,
        "lesson_number": result.lesson_number,
        "lesson_title": result.lesson_title,
        "course_level": result.course_level,
        "resolved_profile": result.course_level,
        "status": result.status,
        "agents_called": result.agents_called,
        "prompt_files_used": result.prompt_files_used,
        "materials": [
            {
                **material.to_public_json(include_content=False),
                "file": material_files.get(material.kind),
            }
            for material in result.materials
        ],
        "package_validation": {
            "approved": result.package_validation.approved,
            "issues": result.package_validation.issues,
            "fix_instructions": result.package_validation.fix_instructions,
        },
        "references": result.reference_summary,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "result.json", result.to_public_json())

    reports_dir = output_dir / "validation_reports"
    for kind, validation in validation_reports.items():
        write_json(
            reports_dir / f"{safe_slug(kind)}.json",
            {
                "approved": validation.approved,
                "issues": validation.issues,
                "fix_instructions": validation.fix_instructions,
                "issues_by_block": validation.issues_by_block,
                "passed_blocks": validation.passed_blocks,
            },
        )


def write_batch_manifest(batch_dir: Path, results: list[IsmartGenerationResult]) -> None:
    write_json(
        batch_dir / "batch_manifest.json",
        {
            "status": _batch_status(results),
            "task_count": len(results),
            "skipped_count": sum(1 for item in results if item.status == "skipped"),
            "completed_with_skips_count": sum(1 for item in results if item.status == "completed_with_skips"),
            "skipped_material_count": sum(
                1
                for result in results
                for material in result.materials
                if material.status in SKIPPED_MATERIAL_STATUSES
            ),
            "tasks": [
                {
                    "task_id": item.task_id,
                    "lesson_number": item.lesson_number,
                    "lesson_title": item.lesson_title,
                    "course_level": item.course_level,
                    "resolved_profile": item.course_level,
                    "status": item.status,
                    "output_dir": item.output_dir,
                }
                for item in results
            ],
        },
    )


def _batch_status(results: list[IsmartGenerationResult]) -> str:
    if any(item.status not in {"approved", "skipped", "completed_with_skips"} for item in results):
        return "has_failures"
    if any(item.status in {"skipped", "completed_with_skips"} for item in results):
        return "completed_with_skips"
    return "approved"


def _next_material_filename(output_dir: Path, kind: str, used_filenames: set[str], *, start_index: int) -> str:
    index = max(start_index, _max_existing_material_index(output_dir, used_filenames) + 1)
    while True:
        filename = f"{index:02d}_{safe_slug(kind)}.html"
        if filename not in used_filenames and not (output_dir / filename).exists():
            return filename
        index += 1


def _max_existing_material_index(output_dir: Path, used_filenames: set[str]) -> int:
    max_index = 0
    for name in {path.name for path in output_dir.glob("*.html")} | set(used_filenames):
        prefix = name.split("_", 1)[0]
        if prefix.isdigit():
            max_index = max(max_index, int(prefix))
    return max_index
