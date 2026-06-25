from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


Audience = Literal["8-9", "10-11", "СПО"]
CourseLevel = Literal["базовый", "продвинутый"]
TemplateId = Literal[
    "theory",
    "practice_python",
    "self_study",
    "control_question",
    "interactive_template",
]
InteractiveCode = Literal["6A", "6D", "6G", "8D", "10D", "3H", "3D"]
MaterialStatus = Literal[
    "черновик",
    "проверено",
    "одобрено",
    "на доработку",
    "готово к отгрузке",
]
PreviewStatus = Literal["not_required", "pending", "passed", "failed"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class DonorSource(BaseModel):
    donor_id: str
    donor_mode: Literal["прямой", "ориентир", "ваше"]
    source_title: str | None = None
    source_uri: str | None = None
    attribution_required: bool = False
    rewrite_required: bool = False


class GenerationRequest(BaseModel):
    request_id: str
    course_id: str
    module_id: str = ""
    lesson_id: str = ""
    template_id: TemplateId
    topic: str = ""
    lesson_title: str = ""
    learning_goal: str = ""
    audience: Audience = "8-9"
    level: CourseLevel = "базовый"
    direction: Literal["Python"] = "Python"
    lesson_number: int | None = None
    target_task_level: Literal["L1", "L2", "L3"] | None = None
    target_task_number: int | None = None
    course_tracker_path: str | None = None
    donors_path: str | None = None
    interactive_code: InteractiveCode | None = None
    source_content: str | None = None
    task_spec: dict[str, Any] = Field(default_factory=dict)
    template_payload: dict[str, Any] = Field(default_factory=dict)
    service_spec: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    donor_sources: list[DonorSource] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_interactive_code(self) -> "GenerationRequest":
        if self.template_id == "interactive_template" and not self.interactive_code:
            raise ValueError("interactive_template requires interactive_code")
        if self.template_id != "interactive_template" and self.interactive_code:
            raise ValueError("interactive_code is only valid for interactive_template")
        return self


class BatchGenerationRequest(BaseModel):
    request_id: str
    course_id: str
    audience: Audience = "8-9"
    level: CourseLevel = "базовый"
    direction: Literal["Python"] = "Python"
    first_task_count: int = Field(default=50, ge=1)
    course_tracker_path: str | None = None
    donors_path: str | None = None


class TemplateSpec(BaseModel):
    template_id: TemplateId
    content_type: str
    required_fields: list[str]
    service_fields: list[str] = Field(default_factory=list)
    preview_required: bool = False
    supported_interactive_codes: list[InteractiveCode] = Field(default_factory=list)


class SandboxTest(BaseModel):
    test_id: str
    stdin: str = ""
    expected_stdout: str


class ExecutableSolution(BaseModel):
    language: Literal["python"] = "python"
    runtime: Literal["python3"] = "python3"
    entrypoint: str = "main.py"
    code: str


class SandboxResult(BaseModel):
    content_id: str
    status: Literal["not_applicable", "pass", "fail", "timeout", "runtime_error"]
    tests: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None


class ContentItem(BaseModel):
    content_id: str
    request_id: str
    course_id: str
    module_id: str
    lesson_id: str
    template_id: TemplateId
    content_type: str
    title: str
    audience: Audience
    level: CourseLevel
    requirement_ids: list[str]
    learner_payload: dict[str, Any]
    service_payload: dict[str, Any] = Field(default_factory=dict)
    platform_payload: dict[str, Any] = Field(default_factory=dict)
    status: MaterialStatus = "черновик"
    preview_required: bool = False
    preview_status: PreviewStatus = "not_required"
    preview_artifact: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def validate_preview_state(self) -> "ContentItem":
        if self.preview_required and self.preview_status == "not_required":
            self.preview_status = "pending"
        if not self.preview_required:
            self.preview_status = "not_required"
            self.preview_artifact = None
        return self


class Violation(BaseModel):
    code: str
    content_id: str | None = None
    message: str
    severity: Literal["error", "warning"] = "error"


class ValidationCriterion(BaseModel):
    criterion_id: str
    content_id: str
    requirement_id: str
    category: str
    description: str
    expected: Any
    actual: Any
    status: Literal["passed", "failed", "not_applicable"]
    severity: Literal["error", "warning"] = "error"
    message: str


class ValidationReport(BaseModel):
    request_id: str
    status: Literal["passed", "failed"]
    criteria: list[ValidationCriterion] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    sandbox_results: list[SandboxResult] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utc_now_iso)


class HITLRecord(BaseModel):
    content_id: str
    status: MaterialStatus
    preview_required: bool = False
    preview_status: PreviewStatus = "not_required"
    preview_artifact: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class HITLState(BaseModel):
    request_id: str
    items: dict[str, HITLRecord] = Field(default_factory=dict)


class PublishManifestItem(BaseModel):
    content_id: str
    template_id: TemplateId
    content_type: str
    platform_payload: str
    preview_required: bool
    preview_status: PreviewStatus


class PublishManifest(BaseModel):
    request_id: str
    published_at: str = Field(default_factory=utc_now_iso)
    items: list[PublishManifestItem] = Field(default_factory=list)


class RunDocument(BaseModel):
    schema_version: str = "1.1"
    request: GenerationRequest | BatchGenerationRequest | None = None
    items: list[ContentItem] = Field(default_factory=list)
    validation: ValidationReport | None = None
    hitl: HITLState | None = None
    platform_payloads: dict[str, dict[str, Any]] = Field(default_factory=dict)
    manifest: PublishManifest | None = None
