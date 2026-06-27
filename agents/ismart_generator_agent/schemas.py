from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GeneratedMaterial(BaseModel):
    """Structured output returned by content subagents."""

    content: str = Field(
        default="",
        description="Final user-facing HTML. It must start with <style> and contain div.cc-lesson.",
    )
    agent_notes: list[str] = Field(
        default_factory=list,
        description="Short operational notes about which inputs were used.",
    )


class PracticeTaskTemplate(BaseModel):
    """Structured pattern extracted from a source practice task."""

    id: str = Field(..., min_length=1, description="Stable practice task id, for example P1.")
    level: str = Field(..., min_length=1, description="Task level from the source JSON, for example L1/L2/L3.")
    source_text: str = Field(..., min_length=1, description="Original source task text from lesson.practice_tasks.")
    task_type: str = Field(..., min_length=1, description="Required student action: write, complete, fix, explain, etc.")
    skill_target: str = Field(..., min_length=1, description="The concept or skill the task must assess.")
    invariants: list[str] = Field(default_factory=list, description="Rules that must remain true for any generated variant.")
    slots_to_fill: list[str] = Field(
        default_factory=list,
        description="Variant-specific slots such as scenario, literals, variable names, inputs, outputs, or code shape.",
    )
    constraints: list[str] = Field(default_factory=list, description="Source/reference constraints that variants must obey.")
    test_policy: str = Field(..., min_length=1, description="How tests should be produced or why they require clarification.")


class PracticeTaskTemplateSet(BaseModel):
    """Ordered template set returned by PracticeTaskTemplateAgent."""

    tasks: list[PracticeTaskTemplate] = Field(default_factory=list)
    agent_notes: list[str] = Field(default_factory=list)


class PracticeTaskInstance(BaseModel):
    """Concrete generated variant of a practice task template."""

    id: str = Field(..., min_length=1, description="Stable practice task id, for example P1.")
    template_id: str = Field(..., min_length=1, description="Id of the source template this instance fills.")
    level: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1)
    scenario: str = Field(..., min_length=1, description="New concrete scenario, not copied from theory/references.")
    student_condition: str = Field(..., min_length=1, description="Student-facing task statement.")
    starter_code: str = Field(default="", description="Student-facing starter or faulty code when required by task_type.")
    input_requirements: str = Field(default="", description="Student-facing input data requirements.")
    output_requirements: str = Field(default="", description="Student-facing output/checking requirements.")
    tests: list[dict[str, str]] = Field(
        default_factory=list,
        description="Legacy visible input -> expected output pairs. Keep equal to runtime_tests when runtime tests exist.",
    )
    runtime_tests: list[dict[str, str]] = Field(
        default_factory=list,
        description="Visible stdin -> expected stdout checks for behavior/runtime verification.",
    )
    manual_checks: list[str] = Field(
        default_factory=list,
        description="Visible manual/static checks for requirements that stdout tests cannot prove.",
    )
    run_mode: Literal["single_file", "separate_snippets", "manual_only", "needs_platform_clarification"] = Field(
        default="needs_platform_clarification",
        description="How the generated task should be run and checked.",
    )
    subtasks: list[dict[str, str]] = Field(
        default_factory=list,
        description="Optional student-facing subtask summaries for multi-part practice tasks.",
    )
    hidden_solution: str = Field(
        default="",
        description="Internal reference solution or checking rule for MR/QA. Must not appear in student practice HTML.",
    )
    teacher_explanation: str = Field(
        default="",
        description="Internal teacher explanation for MR/QA. Must not appear in student practice HTML.",
    )
    uniqueness_notes: list[str] = Field(
        default_factory=list,
        description="Short notes explaining how this variant differs from theory/references/dependencies.",
    )


class PracticeTaskInstanceSet(BaseModel):
    """Ordered concrete practice variants returned by PracticeTaskVariantAgent."""

    tasks: list[PracticeTaskInstance] = Field(default_factory=list)
    agent_notes: list[str] = Field(default_factory=list)


class SelfWorkTaskCheck(BaseModel):
    """Internal checking artifact for one independent self-work task."""

    id: str = Field(..., min_length=1, description="Stable task id, for example SW1.")
    student_task_title: str = Field(..., min_length=1, description="Student-facing task title or short label.")
    checked_skill: str = Field(..., min_length=1, description="Skill or concept checked by this task.")
    checking_mode: Literal["autocheck", "manual", "platform", "teacher_review"] = Field(
        default="manual",
        description="How this task can be checked outside the learner-facing HTML.",
    )
    correct_answer: str = Field(
        default="",
        description="Internal reference answer or key. Must not be shown in learner-facing HTML.",
    )
    runtime_tests: list[dict[str, str]] = Field(
        default_factory=list,
        description="Internal or platform stdin -> expected output checks when applicable.",
    )
    manual_check_rules: list[str] = Field(
        default_factory=list,
        description="Internal checking rules for tasks that cannot be fully autochecked.",
    )
    internal_explanation: str = Field(
        default="",
        description="Internal explanation for teachers/QA. Must not be shown in learner-facing HTML.",
    )


class SelfWorkAutocheckQuestion(BaseModel):
    """Internal autocheck configuration for one self-check question."""

    id: str = Field(..., min_length=1, description="Stable question id, for example Q1.")
    template_code: str = Field(..., min_length=1, description="Platform format/template code, for example question, 6A, 6D, 6G, 8D, 10D, 3H.")
    question_type: str = Field(..., min_length=1, description="Human-readable question type.")
    skill_target: str = Field(..., min_length=1, description="Skill or concept checked by this question.")
    student_prompt: str = Field(..., min_length=1, description="Learner-facing question text.")
    options: list[str] = Field(default_factory=list, description="Learner-facing options/items when applicable.")
    template_markup: str = Field(
        default="",
        description="Platform template markup/parameters when applicable. It may contain answer placeholders but must not be rendered as visible learner HTML with filled answers.",
    )
    correct_answers: list[str] = Field(
        default_factory=list,
        description="Internal correct answers/keys for autocheck. Must not be shown in learner-facing HTML.",
    )
    autocheck_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Internal platform autocheck configuration: answer flags, item pairs, order, matching rules, normalization, or rubric.",
    )
    internal_explanation: str = Field(
        default="",
        description="Internal explanation for teachers/QA. Must not be shown in learner-facing HTML.",
    )


class SelfWorkAutocheckSet(BaseModel):
    """Internal self-work checking artifact returned by SelfWorkAutocheckAgent."""

    independent_tasks: list[SelfWorkTaskCheck] = Field(default_factory=list)
    selfcheck_questions: list[SelfWorkAutocheckQuestion] = Field(default_factory=list)
    agent_notes: list[str] = Field(default_factory=list)


class IntermediateClosedQuestion(BaseModel):
    """Internal closed-question configuration for intermediate assessment."""

    id: str = Field(..., min_length=1, description="Stable question id, for example V1-C01.")
    template_code: str = Field(..., min_length=1, description="Platform template code or question type.")
    skill_target: str = Field(..., min_length=1, description="Module skill/concept checked by this question.")
    student_prompt: str = Field(..., min_length=1, description="Learner-facing question text.")
    options: list[str] = Field(default_factory=list, description="Learner-facing options/items when applicable.")
    correct_answers: list[str] = Field(
        default_factory=list,
        description="Internal correct answers/keys. Must not be shown in learner-facing HTML.",
    )
    autocheck_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Internal platform autocheck config: answer flags, ordering, pairs, matching rules, or normalization.",
    )
    internal_explanation: str = Field(
        default="",
        description="Internal explanation for teachers/QA. Must not be shown in learner-facing HTML.",
    )


class IntermediateOpenQuestion(BaseModel):
    """Internal open-question checking artifact for intermediate assessment."""

    id: str = Field(..., min_length=1, description="Stable question id, for example V1-O01.")
    skill_target: str = Field(..., min_length=1)
    student_prompt: str = Field(..., min_length=1)
    reference_answer: str = Field(
        ...,
        min_length=1,
        description="Internal reference/expected answer. Must not be shown in learner-facing HTML.",
    )
    rubric: list[str] = Field(default_factory=list, description="Internal scoring/checking criteria.")
    internal_explanation: str = Field(
        default="",
        description="Internal explanation for teachers/QA. Must not be shown in learner-facing HTML.",
    )


class IntermediateCodeTask(BaseModel):
    """Internal code-task checking artifact for intermediate assessment."""

    id: str = Field(..., min_length=1, description="Stable code task id, for example V1-P01.")
    skill_target: str = Field(..., min_length=1)
    student_condition: str = Field(..., min_length=1, description="Learner-facing task condition.")
    starter_code: str = Field(default="", description="Learner-facing starter/faulty code when needed.")
    input_requirements: str = Field(default="", description="Learner-facing input format.")
    output_requirements: str = Field(default="", description="Learner-facing output format.")
    runtime_tests: list[dict[str, str]] = Field(
        default_factory=list,
        description="Internal or visible stdin -> expected stdout checks when applicable.",
    )
    manual_check_rules: list[str] = Field(
        default_factory=list,
        description="Internal checking rules for tasks that cannot be fully runtime-tested.",
    )
    hidden_solution: str = Field(
        ...,
        min_length=1,
        description="Internal reference solution. Must not be shown in learner-facing HTML.",
    )
    teacher_explanation: str = Field(
        default="",
        description="Internal explanation for teachers/QA. Must not be shown in learner-facing HTML.",
    )


class IntermediateAssessmentVariant(BaseModel):
    """One complete intermediate assessment variant/kit."""

    id: str = Field(..., min_length=1, description="Stable variant id, for example V1.")
    title: str = Field(..., min_length=1)
    closed_questions: list[IntermediateClosedQuestion] = Field(default_factory=list)
    open_questions: list[IntermediateOpenQuestion] = Field(default_factory=list)
    code_tasks: list[IntermediateCodeTask] = Field(default_factory=list)


class IntermediateAssessmentArtifact(BaseModel):
    """Internal assessment artifact returned by IntermediateAssessmentArtifactAgent."""

    variants: list[IntermediateAssessmentVariant] = Field(default_factory=list)
    module_coverage_notes: list[str] = Field(default_factory=list)
    agent_notes: list[str] = Field(default_factory=list)


class BlockIssue(BaseModel):
    block_id: str = Field(default="", description="Nearest heading id, anchor, or heading text.")
    block_heading: str = Field(default="", description="Human-readable block heading.")
    severity: Literal["blocking", "non_blocking"] = "blocking"
    issue: str = Field(default="", description="What is wrong in this block.")
    fix_instruction: str = Field(default="", description="Localized instruction for fixing this block.")


class PassedBlock(BaseModel):
    block_id: str = Field(default="", description="Nearest heading id, anchor, or heading text.")
    block_heading: str = Field(default="", description="Human-readable block heading.")
    reason: str = Field(default="", description="Why this block should be preserved on retry.")


class MaterialValidationDecision(BaseModel):
    """Structured output returned by MaterialValidatorAgent."""

    approved: bool = False
    issues: list[str] = Field(default_factory=list)
    fix_instructions: list[str] = Field(default_factory=list)
    issues_by_block: list[BlockIssue] = Field(default_factory=list)
    passed_blocks: list[PassedBlock] = Field(default_factory=list)


class ValidationControllerDecision(BaseModel):
    """Structured output returned by ValidationControllerAgent."""

    approved: bool = False
    decision: str = Field(default="keep_failed")
    quality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    score_rationale: str = ""
    rationale: str = ""
    blocking_issues: list[str] = Field(default_factory=list)
    non_blocking_issues: list[str] = Field(default_factory=list)
    overruled_validator_issues: list[str] = Field(default_factory=list)
    residual_risks: list[str] = Field(default_factory=list)
    fix_instructions: list[str] = Field(default_factory=list)


class PackageValidationDecision(BaseModel):
    """Structured output returned by PackageValidatorAgent."""

    approved: bool = False
    issues: list[str] = Field(default_factory=list)
    fix_instructions: list[str] = Field(default_factory=list)
