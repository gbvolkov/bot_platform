from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import ValidationResult
from .writer import safe_slug, write_json


def attempt_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class AttemptArtifactStore:
    root: Path | None = None

    def write_material_attempt(
        self,
        *,
        kind: str,
        attempt: int,
        raw_content: str,
        content: str,
        rule_result: ValidationResult | None,
        llm_result: ValidationResult | None,
        validation: ValidationResult,
        boundary_issues: list[str],
        agent_notes: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.root is None:
            return
        attempt_dir = self.root / safe_slug(kind)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{attempt_timestamp()}__attempt_{attempt:02d}__{safe_slug(kind)}"
        (attempt_dir / f"{prefix}.raw.html").write_text(raw_content, encoding="utf-8")
        (attempt_dir / f"{prefix}.html").write_text(content, encoding="utf-8")
        write_json(
            attempt_dir / f"{prefix}.validation.json",
            {
                "kind": kind,
                "attempt": attempt,
                "metadata": metadata or {},
                "agent_notes": agent_notes,
                "boundary_issues": boundary_issues,
                "rule_validation": _validation_to_json(rule_result),
                "llm_validation": _validation_to_json(llm_result),
                "merged_validation": _validation_to_json(validation),
            },
        )

    def write_package_validation(
        self,
        *,
        rule_result: ValidationResult,
        llm_result: ValidationResult | None,
        validation: ValidationResult,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.root is None:
            return
        package_dir = self.root / "package"
        package_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{attempt_timestamp()}__package_validation"
        write_json(
            package_dir / f"{prefix}.json",
            {
                "metadata": metadata or {},
                "rule_validation": _validation_to_json(rule_result),
                "llm_validation": _validation_to_json(llm_result),
                "merged_validation": _validation_to_json(validation),
            },
        )

    def write_practice_generation_artifacts(
        self,
        *,
        attempt: int,
        templates: dict[str, Any] | None,
        instances: dict[str, Any] | None,
        duplicate_check: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.root is None:
            return
        attempt_dir = self.root / "practice"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{attempt_timestamp()}__attempt_{attempt:02d}__practice"
        common = {"attempt": attempt, "metadata": metadata or {}}
        write_json(attempt_dir / f"{prefix}.practice_templates.json", {**common, "practice_templates": templates or {}})
        write_json(attempt_dir / f"{prefix}.practice_instances.json", {**common, "practice_instances": instances or {}})
        write_json(
            attempt_dir / f"{prefix}.practice_duplicate_check.json",
            {**common, "practice_duplicate_check": duplicate_check or {}},
        )

    def write_self_work_generation_artifacts(
        self,
        *,
        attempt: int,
        autocheck: dict[str, Any] | None,
        structural_check: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.root is None:
            return
        attempt_dir = self.root / "self-work"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{attempt_timestamp()}__attempt_{attempt:02d}__self-work"
        common = {"attempt": attempt, "metadata": metadata or {}}
        write_json(attempt_dir / f"{prefix}.self_work_autocheck.json", {**common, "self_work_autocheck": autocheck or {}})
        write_json(
            attempt_dir / f"{prefix}.self_work_autocheck_check.json",
            {**common, "self_work_autocheck_check": structural_check or {}},
        )

    def write_intermediate_assessment_artifacts(
        self,
        *,
        attempt: int,
        artifact: dict[str, Any] | None,
        structural_check: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.root is None:
            return
        attempt_dir = self.root / "intermediate"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{attempt_timestamp()}__attempt_{attempt:02d}__intermediate"
        common = {"attempt": attempt, "metadata": metadata or {}}
        write_json(
            attempt_dir / f"{prefix}.intermediate_assessment.json",
            {**common, "intermediate_assessment": artifact or {}},
        )
        write_json(
            attempt_dir / f"{prefix}.intermediate_assessment_check.json",
            {**common, "intermediate_assessment_check": structural_check or {}},
        )

    def write_material_controller_review(
        self,
        *,
        kind: str,
        content: str,
        controller_decision: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.root is None:
            return
        review_dir = self.root / safe_slug(kind)
        review_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{attempt_timestamp()}__controller_review__{safe_slug(kind)}"
        write_json(
            review_dir / f"{prefix}.json",
            {
                "kind": kind,
                "metadata": metadata or {},
                "content_chars": len(content),
                "controller_decision": controller_decision,
            },
        )


def _validation_to_json(validation: ValidationResult | None) -> dict[str, Any] | None:
    if validation is None:
        return None
    return {
        "approved": validation.approved,
        "issues": validation.issues,
        "fix_instructions": validation.fix_instructions,
        "issues_by_block": validation.issues_by_block,
        "passed_blocks": validation.passed_blocks,
    }
