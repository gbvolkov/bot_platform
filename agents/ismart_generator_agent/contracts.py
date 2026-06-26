from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


MaterialStatus = Literal["approved", "failed", "blocked_dependency"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


def default_workspace_dir() -> Path:
    docs_root = repo_root() / "docs" / "ismart"
    for prompts_dir in docs_root.glob("**/prompts_skills"):
        if prompts_dir.is_dir():
            return prompts_dir.parent
    return docs_root / "workspace"


@dataclass(frozen=True)
class IsmartGenerationConfig:
    prompts_dir: Path = field(default_factory=lambda: default_workspace_dir() / "prompts_skills")
    output_root: Path = field(default_factory=lambda: default_workspace_dir() / "ismart_agent_outputs")
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    max_generation_iterations: int = 3
    max_package_repair_iterations: int = 2
    max_reference_chars: int = 0
    use_llm_validator: bool = True
    use_validation_controller: bool = True
    validation_controller_accept_score: float = 3.0
    generation_target: str | None = None
    verbose: bool = False


@dataclass(frozen=True)
class MaterialSpec:
    kind: str
    material_type: str
    agent_type: str
    prompt_files: tuple[str, ...]
    validator_kind: str
    dependency_kinds: tuple[str, ...] = ()
    reference_fields: tuple[str, ...] = ()
    json_field_labels: tuple[str, ...] = ()
    prompt_addendum: str = ""


@dataclass(frozen=True)
class ReferenceDocument:
    field: str
    path: str
    resolved_path: str
    sha: str
    truncated: bool
    content: str

    def to_public_json(self, *, include_content: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "field": self.field,
            "path": self.path,
            "resolved_path": self.resolved_path,
            "sha": self.sha,
            "truncated": self.truncated,
        }
        if include_content:
            data["content"] = self.content
        return data


ReferenceBundle = dict[str, list[ReferenceDocument]]


@dataclass
class ValidationResult:
    approved: bool
    issues: list[str] = field(default_factory=list)
    fix_instructions: list[str] = field(default_factory=list)
    issues_by_block: list[dict[str, Any]] = field(default_factory=list)
    passed_blocks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(approved=True)

    @classmethod
    def fail(cls, issues: list[str]) -> "ValidationResult":
        return cls(approved=False, issues=issues, fix_instructions=list(issues))

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        issues = [*self.issues, *other.issues]
        fixes = [*self.fix_instructions, *other.fix_instructions]
        return ValidationResult(
            approved=self.approved and other.approved,
            issues=list(dict.fromkeys(issues)),
            fix_instructions=list(dict.fromkeys(fixes)),
            issues_by_block=[*self.issues_by_block, *other.issues_by_block],
            passed_blocks=[*self.passed_blocks, *other.passed_blocks],
        )


@dataclass
class MaterialResult:
    kind: str
    material_type: str
    agent_type: str
    status: MaterialStatus
    iterations: int
    content: str
    prompt_files: tuple[str, ...]
    validation_issues: list[str] = field(default_factory=list)
    validation_issues_by_block: list[dict[str, Any]] = field(default_factory=list)
    validation_passed_blocks: list[dict[str, Any]] = field(default_factory=list)
    agent_notes: list[str] = field(default_factory=list)
    controller_called: bool = False
    controller_decision: dict[str, Any] = field(default_factory=dict)

    def to_public_json(self, *, include_content: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "type": self.material_type,
            "agent": self.agent_type,
            "status": self.status,
            "iterations": self.iterations,
            "prompt_files": list(self.prompt_files),
            "validation_issues": list(self.validation_issues),
            "validation_issues_by_block": list(self.validation_issues_by_block),
            "validation_passed_blocks": list(self.validation_passed_blocks),
            "agent_notes": list(self.agent_notes),
            "controller_called": self.controller_called,
        }
        if self.controller_decision:
            data["controller_decision"] = self.controller_decision
        if include_content:
            data["content"] = self.content
        return data


@dataclass
class IsmartGenerationResult:
    task_id: str
    lesson_number: str
    lesson_title: str
    status: str
    output_dir: str
    materials: list[MaterialResult]
    package_validation: ValidationResult
    reference_summary: dict[str, list[dict[str, Any]]]
    agents_called: list[str]
    prompt_files_used: list[str]

    def to_public_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "lesson_number": self.lesson_number,
            "lesson_title": self.lesson_title,
            "status": self.status,
            "output_dir": self.output_dir,
            "agents_called": self.agents_called,
            "prompt_files_used": self.prompt_files_used,
            "materials": [item.to_public_json() for item in self.materials],
            "package_validation": {
                "approved": self.package_validation.approved,
                "issues": self.package_validation.issues,
                "fix_instructions": self.package_validation.fix_instructions,
            },
            "references": self.reference_summary,
        }

