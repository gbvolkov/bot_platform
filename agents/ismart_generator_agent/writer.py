from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .contracts import IsmartGenerationResult, MaterialResult, ValidationResult
from .context import task_identity


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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    material_files: dict[str, str] = {}
    for index, material in enumerate(result.materials, start=1):
        filename = material_filename(index, material)
        (output_dir / filename).write_text(material.content, encoding="utf-8")
        material_files[material.kind] = filename

    manifest = {
        "task_id": result.task_id,
        "lesson_number": result.lesson_number,
        "lesson_title": result.lesson_title,
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
            "status": "approved" if all(item.status == "approved" for item in results) else "has_failures",
            "tasks": [
                {
                    "task_id": item.task_id,
                    "lesson_number": item.lesson_number,
                    "lesson_title": item.lesson_title,
                    "status": item.status,
                    "output_dir": item.output_dir,
                }
                for item in results
            ],
        },
    )
