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
    starter_code: str = Field(
        default="",
        description=(
            "Student-facing starter code when required by task_type. For intentionally faulty code, prefer "
            "faulty_code_display for the exact learner-facing rendering and keep faulty_code as the raw internal source."
        ),
    )
    faulty_code: str = Field(
        default="",
        description=(
            "Internal raw faulty code for fix/debug tasks. It may contain intentionally unclosed strings/brackets. "
            "It is used by MR/QA and should not be rendered directly when it can look like broken HTML."
        ),
    )
    faulty_code_display: str = Field(
        default="",
        description=(
            "Safe learner-facing rendering of faulty_code. Use this when raw faulty code contains unclosed strings, "
            "brackets, quotes, or other fragments. It must contain only code the learner should copy into the editor; "
            "do not add marker/comment lines, neutral hints, or explanatory comments such as 'fragment intentionally breaks here'."
        ),
    )
    display_note: str = Field(
        default="",
        description=(
            "Optional learner-facing formatting note. Leave empty by default; do not use it to identify where or why "
            "the faulty code is broken."
        ),
    )
    input_requirements: str = Field(default="", description="Student-facing input data requirements.")
    output_requirements: str = Field(
        default="",
        description=(
            "Student-facing output/checking requirements. For deterministic stdout tasks, define one exact "
            "symbol-for-symbol expected output format without extra labels such as 'Answer:'."
        ),
    )
    tests: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Legacy visible checks. For normal behavior use input + expected_output and keep equal to runtime_tests. "
            "For normal autocheckable tasks provide meaningful typical, boundary, and atypical/special cases when applicable; "
            "do not duplicate the same test row to satisfy count. For error-message diagnostic tasks use "
            "expected_error/error_message or manual_checks instead of normal stdout values."
        ),
    )
    runtime_tests: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Visible runtime checks. For normal behavior use stdin/input -> expected stdout. For tasks whose result is "
            "a Python error message, check expected_error/error_message rather than inventing normal stdout. "
            "When deterministic stdout tests are used, include typical, boundary, and atypical/special cases if applicable "
            "and avoid duplicate input/expected_output pairs."
        ),
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

    lesson_goal: str = Field(
        default="",
        description="Student-facing practice goal, phrased operationally and aligned with the lesson topic.",
    )
    lesson_objectives: list[str] = Field(
        default_factory=list,
        description="Student-facing operational practice objectives. Use learner actions such as finds, runs, checks, fixes, writes.",
    )
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


class CurrentControlAutocheckQuestion(BaseModel):
    """Internal autocheck configuration for one current-control question."""

    id: str = Field(..., min_length=1, description="Stable question id, for example CC1.")
    template_code: str = Field(..., min_length=1, description="Platform format/template code, for example question, 6A, 6D, 6G, 8D, 10D, 3H.")
    question_type: str = Field(..., min_length=1, description="Human-readable question type.")
    skill_target: str = Field(..., min_length=1, description="Skill or concept checked by this question.")
    student_prompt: str = Field(..., min_length=1, description="Learner-facing question text.")
    options: list[str] = Field(default_factory=list, description="Learner-facing options/items when applicable.")
    expected_answer_format: str = Field(
        default="",
        description="Learner-facing answer format or internal normalization hint for open-answer questions.",
    )
    correct_answers: list[str] = Field(
        default_factory=list,
        description="Internal correct answers/keys for autocheck. Must not be shown in learner-facing HTML.",
    )
    autocheck_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Internal platform autocheck configuration: answer flags, matching rules, normalization, or rubric.",
    )
    internal_explanation: str = Field(
        default="",
        description="Internal explanation for teachers/QA. Must not be shown in learner-facing HTML.",
    )


class CurrentControlAutocheckSet(BaseModel):
    """Internal current-control checking artifact returned by CurrentControlAutocheckAgent."""

    questions: list[CurrentControlAutocheckQuestion] = Field(
        default_factory=list,
        description="Question objects only. Do not put agent_notes, strings, markdown, or diagnostics in this list.",
    )
    agent_notes: list[str] = Field(
        default_factory=list,
        description="Top-level diagnostic notes only. Do not place these strings inside questions.",
    )


class IntermediateTestQuestion(BaseModel):
    """Internal closed/test-question configuration for intermediate assessment."""

    id: str = Field(..., min_length=1, description="Stable question id, for example V1-T01.")
    template_code: str = Field(
        ...,
        min_length=1,
        description=(
            "Platform template code or question type. For each intermediate variant, at least 3 distinct "
            "test_questions must use coded template_code values from 6A, 6D, 6G, 8D, 10D."
        ),
    )
    skill_target: str = Field(..., min_length=1, description="Module skill/concept checked by this question.")
    student_prompt: str = Field(..., min_length=1, description="Learner-facing question text.")
    options: list[str] = Field(default_factory=list, description="Learner-facing options/items when applicable.")
    correct_answers: list[str] = Field(
        default_factory=list,
        description="Internal correct answers/keys. Must not be shown in learner-facing HTML.",
    )
    autocheck_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Internal platform autocheck config: answer flags, ordering, pairs, matching rules, or normalization. "
            "For 6A ordering templates, options/display_items are the visible shuffled item order; "
            "ordered_items/correct_order stores the internal key. "
            "For matching/classification templates, left_items and right_items are learner-facing display lists; "
            "right_items must be a derangement relative to left_items/correct_pairs: no right_items[i] may be "
            "the correct pair for left_items[i]. correct_pairs/correct_map stores the key."
        ),
    )
    internal_explanation: str = Field(
        default="",
        description="Internal explanation for teachers/QA. Must not be shown in learner-facing HTML.",
    )


class IntermediateOpenCodeQuestion(BaseModel):
    """Internal open-code question artifact for intermediate assessment."""

    id: str = Field(..., min_length=1, description="Stable open-code question id, for example V1-OC01.")
    skill_target: str = Field(..., min_length=1)
    student_prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Learner-facing open-code prompt. The student must write executable code with a verifiable result; "
            "non-code explanations, output-prediction, matching, and fill-gap prompts do not qualify."
        ),
    )
    starter_code: str = Field(default="", description="Optional learner-facing starter code.")
    input_requirements: str = Field(default="", description="Learner-facing input format when applicable.")
    output_requirements: str = Field(default="", description="Learner-facing expected behavior/output format.")
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
        description="Internal reference solution or answer. Must not be shown in learner-facing HTML.",
    )
    rubric: list[str] = Field(default_factory=list, description="Internal scoring/checking criteria.")
    internal_explanation: str = Field(
        default="",
        description="Internal explanation for teachers/QA. Must not be shown in learner-facing HTML.",
    )


class IntermediateCodeTask(BaseModel):
    """Internal practical code-task checking artifact for intermediate assessment."""

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
    test_questions: list[IntermediateTestQuestion] = Field(default_factory=list)
    open_code_questions: list[IntermediateOpenCodeQuestion] = Field(default_factory=list)
    code_tasks: list[IntermediateCodeTask] = Field(default_factory=list)


class IntermediateAssessmentArtifact(BaseModel):
    """Internal assessment artifact returned by IntermediateAssessmentArtifactAgent."""

    variants: list[IntermediateAssessmentVariant] = Field(default_factory=list)
    module_coverage_notes: list[str] = Field(default_factory=list)
    agent_notes: list[str] = Field(default_factory=list)


class BlockIssue(BaseModel):
    block_id: str = Field(default="", description="Nearest heading id, anchor, or heading text.")
    block_heading: str = Field(default="", description="Human-readable block heading.")
    field_path: str = Field(
        default="",
        description=(
            "JSON path to the structured field that should be fixed, for example "
            "practice_instances.tasks[P3].student_condition. Use this for structured-artifact issues."
        ),
    )
    severity: Literal["blocking", "non_blocking"] = "blocking"
    issue: str = Field(default="", description="What is wrong in this block.")
    fix_instruction: str = Field(default="", description="Localized instruction for fixing this block.")
    evidence_quote: str = Field(
        default="",
        description="Exact short quote from the checked HTML that proves the issue, only when the issue is about rendered HTML.",
    )


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
