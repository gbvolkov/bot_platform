from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from agents.ismart_generator_agent import cli
from agents.ismart_generator_agent.context import (
    build_generation_prompt,
    build_package_validation_prompt,
    build_validation_prompt,
    build_validation_controller_prompt,
    channel_key_visibility_policy_for_spec,
    sanitize_generation_artifacts_for_validation,
    source_contract_for_spec,
    validation_policy_for_spec,
)
from agents.ismart_generator_agent.contracts import IsmartGenerationConfig, MaterialResult, MaterialSpec
from agents.ismart_generator_agent.contracts import ValidationResult
from agents.ismart_generator_agent.runtime import IsmartGeneratorRuntime
from agents.ismart_generator_agent.registry import get_material_spec
from agents.ismart_generator_agent.schemas import (
    GeneratedMaterial,
    IntermediateAssessmentArtifact,
    IntermediateAssessmentVariant,
    IntermediateClosedQuestion,
    IntermediateCodeTask,
    IntermediateOpenQuestion,
    MaterialValidationDecision,
    ValidationControllerDecision,
    PackageValidationDecision,
    PracticeTaskInstance,
    PracticeTaskInstanceSet,
    PracticeTaskTemplate,
    PracticeTaskTemplateSet,
    SelfWorkAutocheckQuestion,
    SelfWorkAutocheckSet,
    SelfWorkTaskCheck,
)
from agents.ismart_generator_agent.subagents import (
    ALL_SUBAGENT_TYPES,
    build_subagent_registry,
)
from agents.ismart_generator_agent.workers import (
    MaterialWorker,
    PackageValidator,
    sanitize_internal_source_locators,
    sanitize_visible_selfcheck_answers,
)


VALID_HTML = '<style>.x{}</style><div class="cc-lesson"><h2 id="concepts">Concepts</h2><p>ok</p></div>'


class FakeStructuredModel:
    def __init__(self, schema: type[Any]) -> None:
        self.schema = schema

    def invoke(self, _messages: list[Any]) -> Any:
        if self.schema is GeneratedMaterial:
            return GeneratedMaterial(content=VALID_HTML, agent_notes=["generated"])
        if self.schema is MaterialValidationDecision:
            return MaterialValidationDecision(approved=True, passed_blocks=[])
        if self.schema is ValidationControllerDecision:
            return ValidationControllerDecision(approved=True, quality_score=4)
        return self.schema(approved=True)


class FakeChatModel:
    def __init__(self) -> None:
        self.schemas: list[type[Any]] = []

    def with_structured_output(self, schema: type[Any]) -> FakeStructuredModel:
        self.schemas.append(schema)
        return FakeStructuredModel(schema)


class FunctionCallingFakeChatModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def with_structured_output(self, schema: type[Any], **kwargs: Any) -> FakeStructuredModel:
        self.calls.append({"schema": schema, "kwargs": kwargs})
        return FakeStructuredModel(schema)


class FakeGraph:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(state))
        if not self.responses:
            raise AssertionError("FakeGraph has no more responses")
        return {"result": self.responses.pop(0)}


def test_subagent_registry_builds_explicit_compiled_graphs() -> None:
    model = FakeChatModel()

    registry = build_subagent_registry(model)  # type: ignore[arg-type]

    assert set(registry) == set(ALL_SUBAGENT_TYPES)
    for agent_type in ALL_SUBAGENT_TYPES:
        assert hasattr(registry[agent_type], "invoke")
    result = registry["TheoryMaterialAgent"].invoke({"system_prompt": "system", "prompt": "prompt"})
    assert isinstance(result["result"], GeneratedMaterial)
    assert GeneratedMaterial in model.schemas
    assert PracticeTaskTemplateSet in model.schemas
    assert PracticeTaskInstanceSet in model.schemas
    assert SelfWorkAutocheckSet in model.schemas
    assert IntermediateAssessmentArtifact in model.schemas
    assert MaterialValidationDecision in model.schemas
    assert ValidationControllerDecision in model.schemas


def test_subagent_registry_uses_function_calling_structured_output_when_supported() -> None:
    model = FunctionCallingFakeChatModel()

    build_subagent_registry(model)  # type: ignore[arg-type]

    assert model.calls
    assert all(call["kwargs"].get("method") == "function_calling" for call in model.calls)


def test_generator_agent_does_not_import_old_generator_runtime() -> None:
    package_dir = Path("agents/ismart_generator_agent")
    for path in package_dir.glob("*.py"):
        assert "generators.ismart_materials_agent" not in path.read_text(encoding="utf-8")


def test_worker_retries_with_previous_content_and_controller_score_accepts(tmp_path: Path) -> None:
    generator = FakeGraph(
        [
            GeneratedMaterial(content=VALID_HTML, agent_notes=["attempt1"]),
            GeneratedMaterial(content=VALID_HTML, agent_notes=["attempt2"]),
            GeneratedMaterial(content=VALID_HTML, agent_notes=["attempt3"]),
        ]
    )
    validator = FakeGraph(
        [
            MaterialValidationDecision(
                approved=False,
                issues=["block needs stronger explanation"],
                fix_instructions=["expand #concepts"],
                issues_by_block=[
                    {
                        "block_id": "#concepts",
                        "block_heading": "Concepts",
                        "severity": "blocking",
                        "issue": "too short",
                        "fix_instruction": "expand this block",
                    }
                ],
                passed_blocks=[],
            ),
            MaterialValidationDecision(
                approved=False,
                issues=["block still needs stronger explanation"],
                fix_instructions=["expand #concepts"],
                issues_by_block=[
                    {
                        "block_id": "#concepts",
                        "block_heading": "Concepts",
                        "severity": "blocking",
                        "issue": "still too short",
                        "fix_instruction": "expand this block",
                    }
                ],
                passed_blocks=[],
            ),
            MaterialValidationDecision(
                approved=False,
                issues=["validator is over-strict"],
                fix_instructions=["rewrite"],
                issues_by_block=[],
                passed_blocks=[{"block_id": "#concepts", "block_heading": "Concepts", "reason": "valid"}],
            ),
        ]
    )
    controller = FakeGraph(
        [
            ValidationControllerDecision(
                approved=True,
                decision="approve_material",
                quality_score=3,
                rationale="Validator objection is not blocking.",
                non_blocking_issues=["editorial improvement"],
            )
        ]
    )
    spec = MaterialSpec(
        kind="theory",
        material_type="Theory",
        agent_type="TheoryMaterialAgent",
        prompt_files=(),
        validator_kind="theory",
    )
    worker = MaterialWorker(
        subagents={
            "TheoryMaterialAgent": generator,
            "MaterialValidatorAgent": validator,
            "ValidationControllerAgent": controller,
        },
        config=IsmartGenerationConfig(
            prompts_dir=tmp_path,
            output_root=tmp_path,
            max_generation_iterations=3,
            validation_controller_accept_score=3,
        ),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"theory": True}, "hours": {"theory": 1}}},
        spec=spec,
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert result.iterations == 3
    assert result.controller_called is True
    assert result.controller_decision["quality_score"] == 3
    assert len(generator.calls) == 3
    assert VALID_HTML in generator.calls[1]["prompt"]
    assert "block needs stronger explanation" in generator.calls[1]["prompt"]
    assert "issues_by_block" in generator.calls[1]["prompt"]


def test_source_locator_sanitizer_replaces_learner_sources_section() -> None:
    spec = MaterialSpec(
        kind="self_work",
        material_type="Self Work",
        agent_type="SelfStudyAgent",
        prompt_files=(),
        validator_kind="self_study",
    )
    content = (
        '<style>.x{}</style><div class="cc-lesson">'
        "<h2>Задания</h2><p>Работа.</p>"
        '<h2 id="sources">Источники</h2>'
        "<ul><li>Материалы: <code>docs/ismart/Материалы для ИИ-агентов/"
        "рабочая область агента/референсы/Шаблоны.md</code></li></ul>"
        "<h2>Итоги</h2><p>Готово.</p></div>"
    )

    sanitized, notes = sanitize_internal_source_locators(content, spec)

    assert notes
    assert "docs/ismart" not in sanitized
    assert ".md" not in sanitized
    assert "рабочая область агента" not in sanitized
    assert "Материал подготовлен на основе содержания занятия" in sanitized
    assert "<h2>Итоги</h2>" in sanitized


def test_source_locator_sanitizer_preserves_qa_paths() -> None:
    spec = MaterialSpec(
        kind="specification_qa",
        material_type="QA",
        agent_type="SpecificationQAAgent",
        prompt_files=(),
        validator_kind="qa",
    )
    content = (
        '<style>.x{}</style><div class="cc-lesson">'
        "<p><code>docs/ismart/референсы/Шаблоны.md</code></p></div>"
    )

    sanitized, notes = sanitize_internal_source_locators(content, spec)

    assert sanitized == content
    assert notes == []


def test_worker_sanitizes_self_work_source_locators_before_validation(tmp_path: Path) -> None:
    generated = (
        '<style>.x{}</style><div class="cc-lesson">'
        "<h2>Задания</h2><p>Работа.</p>"
        '<h2 id="sources">Источники</h2>'
        "<ul><li><code>docs/ismart/Материалы для ИИ-агентов/"
        "рабочая область агента/референсы/Шаблоны.md</code></li></ul>"
        "<h2>Итоги</h2><p>Готово.</p></div>"
    )
    autocheck = FakeGraph([_self_work_autocheck_set()])
    generator = FakeGraph([GeneratedMaterial(content=generated)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    spec = MaterialSpec(
        kind="self_work",
        material_type="Self Work",
        agent_type="SelfStudyAgent",
        prompt_files=(),
        validator_kind="self_study",
    )
    worker = MaterialWorker(
        subagents={
            "SelfWorkAutocheckAgent": autocheck,
            "SelfStudyAgent": generator,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"self_work": True}, "hours": {"self_study": 1}}},
        spec=spec,
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert "docs/ismart" not in result.content
    assert ".md" not in result.content
    assert "internal source locator section was replaced before validation" in result.agent_notes
    assert "docs/ismart" not in validator.calls[0]["prompt"]


def test_selfcheck_answer_sanitizer_removes_visible_keys() -> None:
    spec = MaterialSpec(
        kind="self_work",
        material_type="Self Work",
        agent_type="SelfStudyAgent",
        prompt_files=(),
        validator_kind="self_study",
    )
    content = (
        '<style>.x{}</style><div class="cc-lesson">'
        '<h2 id="selfcheck">Самоконтроль</h2>'
        "<h3>Вопрос 1</h3><p>Что верно?</p>"
        '<div class="cc-note"><div class="cc-note-title">Ключ для автопроверки</div>'
        "<p>Правильный вариант: B</p></div>"
        "<pre><code>{%6A%} {{item:Шаг 1}} {%answer%} {{order:1}} {%/answer%} {%/6A%}</code></pre>"
        "<pre><code>{%3H%} {{input-text:x = x + 2:x+=2}} {%/3H%}</code></pre>"
        "</div>"
    )

    sanitized, notes = sanitize_visible_selfcheck_answers(content, spec)

    assert notes
    assert "Ключ для автопроверки" not in sanitized
    assert "Правильный вариант" not in sanitized
    assert "{%answer%}" not in sanitized
    assert "x = x + 2" not in sanitized
    assert "x+=2" not in sanitized
    assert "{{input-text}}" in sanitized
    assert "Что верно?" in sanitized


def test_selfcheck_answer_sanitizer_preserves_non_self_work() -> None:
    spec = MaterialSpec(
        kind="specification_qa",
        material_type="QA",
        agent_type="SpecificationQAAgent",
        prompt_files=(),
        validator_kind="qa",
    )
    content = '<style>.x{}</style><div class="cc-lesson"><p>{%answer%} {{order:1}} {%/answer%}</p></div>'

    sanitized, notes = sanitize_visible_selfcheck_answers(content, spec)

    assert sanitized == content
    assert notes == []


def test_worker_sanitizes_self_work_visible_selfcheck_answers_before_validation(tmp_path: Path) -> None:
    generated = (
        '<style>.x{}</style><div class="cc-lesson">'
        '<h2 id="selfcheck">Самоконтроль</h2>'
        "<h3>Вопрос 1</h3><p>Что верно?</p>"
        '<div class="cc-note"><div class="cc-note-title">Ключ для автопроверки</div>'
        "<p>Правильный вариант: B</p></div>"
        "<pre><code>{%6G%} {{left:a // b}} {{right:3}} {%answer%} {{pair:a // b=3}} {%/answer%} {%/6G%}</code></pre>"
        "<h2>Итоги</h2><p>Готово.</p></div>"
    )
    autocheck = FakeGraph([_self_work_autocheck_set()])
    generator = FakeGraph([GeneratedMaterial(content=generated)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    spec = MaterialSpec(
        kind="self_work",
        material_type="Self Work",
        agent_type="SelfStudyAgent",
        prompt_files=(),
        validator_kind="self_study",
    )
    worker = MaterialWorker(
        subagents={
            "SelfWorkAutocheckAgent": autocheck,
            "SelfStudyAgent": generator,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"self_work": True}, "hours": {"self_study": 1}}},
        spec=spec,
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert "Ключ для автопроверки" not in result.content
    assert "{%answer%}" not in result.content
    assert "pair:a // b=3" not in result.content
    assert "visible self-check key blocks were removed before validation" in result.agent_notes
    assert "Ключ для автопроверки" not in validator.calls[0]["prompt"]
    assert "{{pair:a // b=3}}" not in validator.calls[0]["prompt"]


def test_practice_validator_policy_accepts_honest_underspecified_tasks() -> None:
    spec = MaterialSpec(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        prompt_files=(),
        validator_kind="practice",
    )

    policy = validation_policy_for_spec(spec)

    assert "honest \"tests are absent/not applicable until source clarification\" status is valid" in policy
    assert "training" in policy
    assert "manual check" in policy
    assert "two variables" in policy
    assert "Visible expected stdout in a student-facing deterministic test table is allowed" in policy
    assert "test oracle, not a forbidden answer key" in policy
    assert "For refactoring tasks, separate runtime tests from manual/static checks" in policy
    assert "stdout tests cannot prove" in policy
    assert "neither runtime tests" in policy


def test_self_work_validation_policy_forbids_visible_keys() -> None:
    spec = MaterialSpec(
        kind="self_work",
        material_type="Self Work",
        agent_type="SelfStudyAgent",
        prompt_files=(),
        validator_kind="self_study",
    )

    policy = validation_policy_for_spec(spec)

    assert "visible answer keys are forbidden" in policy
    assert "Do not require visible keys" in policy
    assert "self_work_autocheck" in policy
    assert "correct_answers" in policy
    assert "{%answer%}" in policy
    assert "filled {{input-text:answer}}" in policy


def test_practice_validation_prompt_allows_visible_expected_stdout() -> None:
    spec = MaterialSpec(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        prompt_files=(),
        validator_kind="practice",
    )

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
    )

    assert "Visible expected stdout in a student-facing deterministic test table is allowed" in prompt
    assert "Do not reject practice merely because deterministic tests show concrete expected stdout" in prompt
    assert "Their absence from student practice is correct" in prompt


def test_practice_validation_prompt_sanitizes_teacher_only_artifacts() -> None:
    spec = MaterialSpec(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        prompt_files=(),
        validator_kind="practice",
    )
    artifacts = {
        "practice_instances": {
            "tasks": [
                {
                    "id": "P1",
                    "student_condition": "Исправьте код.",
                    "runtime_tests": [{"input": "", "expected_output": "ok\n"}],
                    "manual_checks": ["Проверьте читаемые имена переменных."],
                    "hidden_solution": "SECRET_FIXED_CODE",
                    "teacher_explanation": "SECRET_TEACHER_NOTES",
                }
            ]
        },
        "practice_student_leak_check": {"approved": True, "issues": []},
    }

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        generation_artifacts=artifacts,
    )

    assert "VALIDATION VIEW OF GENERATION ARTIFACTS" in prompt
    assert "SECRET_FIXED_CODE" not in prompt
    assert "SECRET_TEACHER_NOTES" not in prompt
    assert "runtime_tests" in prompt
    assert "manual_checks" in prompt
    assert "practice_student_leak_check" in prompt
    assert "Do not infer a learner-facing answer leak from internal artifacts" in prompt


def test_practice_controller_prompt_sanitizes_teacher_only_artifacts() -> None:
    spec = MaterialSpec(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        prompt_files=(),
        validator_kind="practice",
    )
    artifacts = {
        "practice_instances": {
            "tasks": [
                {
                    "id": "P1",
                    "student_condition": "Исправьте код.",
                    "hidden_solution": "SECRET_FIXED_CODE",
                    "teacher_explanation": "SECRET_TEACHER_NOTES",
                }
            ]
        },
        "practice_student_leak_check": {"approved": True, "issues": []},
    }

    prompt = build_validation_controller_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["hidden_solution appears in generation artifacts"]),
        merged_validation=ValidationResult.fail(["hidden_solution appears in generation artifacts"]),
        generation_artifacts=artifacts,
    )

    assert "VALIDATION VIEW OF GENERATION ARTIFACTS" in prompt
    assert "SECRET_FIXED_CODE" not in prompt
    assert "SECRET_TEACHER_NOTES" not in prompt
    assert "overrule validator objections that treat teacher-only fields inside internal generation artifacts" in prompt
    assert "Do not uphold a validator claim of learner-facing key leakage merely because an internal artifact" in prompt


def test_validation_artifact_sanitizer_keeps_full_artifacts_for_teacher_channels() -> None:
    artifacts = {
        "practice_instances": {
            "tasks": [
                {
                    "id": "P1",
                    "student_condition": "task",
                    "hidden_solution": "SECRET_FIXED_CODE",
                    "teacher_explanation": "SECRET_TEACHER_NOTES",
                }
            ]
        }
    }
    practice_spec = MaterialSpec(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        prompt_files=(),
        validator_kind="practice",
    )
    qa_spec = MaterialSpec(
        kind="specification_qa",
        material_type="QA",
        agent_type="SpecificationQAAgent",
        prompt_files=(),
        validator_kind="qa",
    )

    practice_view = sanitize_generation_artifacts_for_validation(practice_spec, artifacts)
    qa_view = sanitize_generation_artifacts_for_validation(qa_spec, artifacts)

    assert "hidden_solution" not in practice_view["practice_instances"]["tasks"][0]
    assert qa_view["practice_instances"]["tasks"][0]["hidden_solution"] == "SECRET_FIXED_CODE"


def test_practice_source_contract_is_template_variant_based() -> None:
    spec = get_material_spec("practice")
    task = {
        "lesson": {
            "hours": {"practice": 1},
            "practice_tasks": {
                "l1": [{"number": 1, "text": "Create a variable with a name and print it"}],
            },
        }
    }

    contract = source_contract_for_spec(task, spec)
    rules = " ".join(contract["generation_rules"])

    assert contract["contract_type"] == "practice_source_contract"
    assert "authoritative task pattern" in rules
    assert "Create a new concrete variant of the same pattern" in rules
    assert "Do not show source_text, source task, source contract" in rules
    assert "level, source task, condition" not in rules
    assert "PracticeTaskTemplateAgent" in " ".join(contract["pipeline"])
    assert "PracticeTaskVariantAgent" in " ".join(contract["pipeline"])


class NamedFakeGraph(FakeGraph):
    def __init__(self, name: str, responses: list[Any], call_order: list[str]) -> None:
        super().__init__(responses)
        self.name = name
        self.call_order = call_order

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.call_order.append(self.name)
        return super().invoke(state)


def _practice_template_set() -> PracticeTaskTemplateSet:
    return PracticeTaskTemplateSet(
        tasks=[
            PracticeTaskTemplate(
                id="P1",
                level="L1",
                source_text="Create a variable with a name and print it",
                task_type="write_code",
                skill_target="variable assignment and print",
                invariants=["student writes Python code"],
                slots_to_fill=["scenario", "variable_name", "literal_value", "expected_output"],
                constraints=["same beginner skill"],
                test_policy="deterministic stdout when generated values are fixed",
            )
        ],
        agent_notes=["templates ok"],
    )


def _practice_instance_set() -> PracticeTaskInstanceSet:
    return PracticeTaskInstanceSet(
        tasks=[
            PracticeTaskInstance(
                id="P1",
                template_id="P1",
                level="L1",
                task_type="write_code",
                scenario="Сохраните название города в переменную и выведите его.",
                student_condition="Создайте переменную city со значением 'Тула' и выведите её.",
                starter_code="",
                input_requirements="Ввод не требуется.",
                output_requirements="Программа выводит Тула.",
                tests=[{"input": "", "expected_output": "Тула"}],
                hidden_solution="city = 'Тула'\nprint(city)",
                teacher_explanation="Проверить присваивание строки переменной city и вывод значения.",
                uniqueness_notes=["Не использует пример из теории про имя."],
            )
        ],
        agent_notes=["instances ok"],
    )


def _practice_worker_spec() -> MaterialSpec:
    return MaterialSpec(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        prompt_files=(),
        validator_kind="practice",
        dependency_kinds=("theory",),
        reference_fields=("requirements", "reference_examples", "goals_and_tasks", "donor_materials"),
    )


def _self_work_worker_spec() -> MaterialSpec:
    return MaterialSpec(
        kind="self_work",
        material_type="Self Work",
        agent_type="SelfStudyAgent",
        prompt_files=(),
        validator_kind="self_study",
        dependency_kinds=("theory", "practice"),
        reference_fields=("requirements", "reference_examples", "template_descriptions"),
    )


def _self_work_autocheck_set() -> SelfWorkAutocheckSet:
    return SelfWorkAutocheckSet(
        independent_tasks=[
            SelfWorkTaskCheck(
                id=f"SW{index}",
                student_task_title=f"Independent task {index}",
                checked_skill="print and variables",
                checking_mode="manual",
                correct_answer=f"Internal self-work task key {index}",
                manual_check_rules=[f"Check task {index} manually."],
                internal_explanation=f"Teacher note for task {index}.",
            )
            for index in range(1, 9)
        ],
        selfcheck_questions=[
            SelfWorkAutocheckQuestion(
                id=f"Q{index}",
                template_code="question" if index < 6 else "6A",
                question_type="single choice" if index < 6 else "ordering",
                skill_target="basic Python syntax",
                student_prompt=f"Self-check question {index}",
                options=["A", "B", "C"],
                template_markup=f"internal template markup {index}",
                correct_answers=[f"Internal self-check key {index}"],
                autocheck_config={"correct": f"Internal self-check key {index}"},
                internal_explanation=f"Teacher explanation {index}.",
            )
            for index in range(1, 11)
        ],
        agent_notes=["autocheck ok"],
    )


def _intermediate_worker_spec() -> MaterialSpec:
    return MaterialSpec(
        kind="intermediate",
        material_type="Intermediate",
        agent_type="IntermediateAssessmentAgent",
        prompt_files=(),
        validator_kind="intermediate",
        reference_fields=("requirements", "template_descriptions", "goals_and_tasks"),
    )


def _intermediate_assessment_artifact() -> IntermediateAssessmentArtifact:
    variants: list[IntermediateAssessmentVariant] = []
    template_codes = ["question", "6A", "6D", "6G", "8D", "10D"]
    for variant_index in range(1, 5):
        variant_id = f"V{variant_index}"
        variants.append(
            IntermediateAssessmentVariant(
                id=variant_id,
                title=f"Variant {variant_index}",
                closed_questions=[
                    IntermediateClosedQuestion(
                        id=f"{variant_id}-C{question_index:02d}",
                        template_code=template_codes[(question_index - 1) % len(template_codes)],
                        skill_target="module concept",
                        student_prompt=f"Closed question {variant_index}.{question_index}",
                        options=["A", "B", "C", "D"],
                        correct_answers=[f"Internal closed key {variant_index}.{question_index}"],
                        autocheck_config={"correct": f"Internal closed key {variant_index}.{question_index}"},
                        internal_explanation="Teacher-only closed explanation.",
                    )
                    for question_index in range(1, 17)
                ],
                open_questions=[
                    IntermediateOpenQuestion(
                        id=f"{variant_id}-O{question_index:02d}",
                        skill_target="explain Python behavior",
                        student_prompt=f"Open question {variant_index}.{question_index}",
                        reference_answer=f"Internal open reference {variant_index}.{question_index}",
                        rubric=["1 point for concept", "1 point for example"],
                        internal_explanation="Teacher-only open explanation.",
                    )
                    for question_index in range(1, 5)
                ],
                code_tasks=[
                    IntermediateCodeTask(
                        id=f"{variant_id}-P{task_index:02d}",
                        skill_target="write Python code",
                        student_condition=f"Code task {variant_index}.{task_index}",
                        input_requirements="stdin contains one number",
                        output_requirements="stdout contains processed number",
                        runtime_tests=[{"input": "2\n", "expected_output": "4\n"}],
                        manual_check_rules=[],
                        hidden_solution="value = int(input())\nprint(value * 2)",
                        teacher_explanation="Teacher-only code explanation.",
                    )
                    for task_index in range(1, 4)
                ],
            )
        )
    return IntermediateAssessmentArtifact(
        variants=variants,
        module_coverage_notes=["covers module topics"],
        agent_notes=["assessment artifact ok"],
    )


def test_self_work_worker_uses_autocheck_pipeline_and_writes_artifacts(tmp_path: Path) -> None:
    call_order: list[str] = []
    autocheck_agent = NamedFakeGraph("SelfWorkAutocheckAgent", [_self_work_autocheck_set()], call_order)
    self_work_agent = NamedFakeGraph(
        "SelfStudyAgent",
        [GeneratedMaterial(content=VALID_HTML, agent_notes=["html ok"])],
        call_order,
    )
    validator = NamedFakeGraph(
        "MaterialValidatorAgent",
        [MaterialValidationDecision(approved=True, passed_blocks=[])],
        call_order,
    )
    worker = MaterialWorker(
        subagents={
            "SelfWorkAutocheckAgent": autocheck_agent,
            "SelfStudyAgent": self_work_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"self_work": True}, "hours": {"self_study": 2}}},
        spec=_self_work_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert call_order == ["SelfWorkAutocheckAgent", "SelfStudyAgent", "MaterialValidatorAgent"]
    assert result.generation_artifacts["self_work_autocheck"]["selfcheck_questions"][0]["correct_answers"]
    assert "GENERATION ARTIFACTS FOR THIS MATERIAL" in self_work_agent.calls[0]["prompt"]
    assert "self_work_autocheck" in self_work_agent.calls[0]["prompt"]
    assert "Internal self-check key 1" in validator.calls[0]["prompt"]
    assert "Internal self-check key" not in result.content
    assert list((tmp_path / "tmp" / "self-work").glob("*.self_work_autocheck.json"))
    assert list((tmp_path / "tmp" / "self-work").glob("*.self_work_autocheck_check.json"))


def test_self_work_worker_blocks_missing_internal_autocheck_keys_before_html(tmp_path: Path) -> None:
    bad = _self_work_autocheck_set()
    bad.selfcheck_questions[0].correct_answers = []
    autocheck_agent = FakeGraph([bad])
    self_work_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "SelfWorkAutocheckAgent": autocheck_agent,
            "SelfStudyAgent": self_work_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"self_work": True}, "hours": {"self_study": 2}}},
        spec=_self_work_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "needs at least one correct answer" in " ".join(result.validation_issues)
    assert self_work_agent.calls == []
    assert validator.calls == []


def test_self_work_validation_view_keeps_internal_autocheck_answers() -> None:
    spec = _self_work_worker_spec()
    artifacts = {"self_work_autocheck": _self_work_autocheck_set().model_dump(mode="json")}

    view = sanitize_generation_artifacts_for_validation(spec, artifacts)

    assert view["self_work_autocheck"]["selfcheck_questions"][0]["correct_answers"] == ["Internal self-check key 1"]
    assert "self_work_autocheck_visibility_note" in view


def test_self_work_source_contract_defines_internal_autocheck_layer() -> None:
    spec = _self_work_worker_spec()

    contract = source_contract_for_spec({"lesson": {"hours": {"self_study": 2}}}, spec)

    assert contract["contract_type"] == "self_work_autocheck_contract"
    assert contract["required_independent_task_count"] == 8
    assert contract["required_selfcheck_question_count"] == 10
    assert "generation_artifacts.self_work_autocheck" in " ".join(contract["generation_rules"])
    assert "Do not require visible keys" in " ".join(contract["validation_rules"])


def test_intermediate_worker_uses_assessment_artifact_pipeline_and_writes_artifacts(tmp_path: Path) -> None:
    call_order: list[str] = []
    artifact_agent = NamedFakeGraph(
        "IntermediateAssessmentArtifactAgent",
        [_intermediate_assessment_artifact()],
        call_order,
    )
    intermediate_agent = NamedFakeGraph(
        "IntermediateAssessmentAgent",
        [GeneratedMaterial(content=VALID_HTML, agent_notes=["html ok"])],
        call_order,
    )
    validator = NamedFakeGraph(
        "MaterialValidatorAgent",
        [MaterialValidationDecision(approved=True, passed_blocks=[])],
        call_order,
    )
    worker = MaterialWorker(
        subagents={
            "IntermediateAssessmentArtifactAgent": artifact_agent,
            "IntermediateAssessmentAgent": intermediate_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {"lessons": []}, "lesson": {"content_flags": {"attestation": True}}},
        spec=_intermediate_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert call_order == [
        "IntermediateAssessmentArtifactAgent",
        "IntermediateAssessmentAgent",
        "MaterialValidatorAgent",
    ]
    assert result.generation_artifacts["intermediate_assessment"]["variants"][0]["closed_questions"][0]["correct_answers"]
    assert "GENERATION ARTIFACTS FOR THIS MATERIAL" in intermediate_agent.calls[0]["prompt"]
    assert "intermediate_assessment" in intermediate_agent.calls[0]["prompt"]
    assert "Internal closed key 1.1" in validator.calls[0]["prompt"]
    assert "Internal closed key" not in result.content
    assert list((tmp_path / "tmp" / "intermediate").glob("*.intermediate_assessment.json"))
    assert list((tmp_path / "tmp" / "intermediate").glob("*.intermediate_assessment_check.json"))


def test_intermediate_worker_blocks_incomplete_assessment_artifact_before_html(tmp_path: Path) -> None:
    bad = _intermediate_assessment_artifact()
    bad.variants[0].closed_questions[0].correct_answers = []
    artifact_agent = FakeGraph([bad])
    intermediate_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "IntermediateAssessmentArtifactAgent": artifact_agent,
            "IntermediateAssessmentAgent": intermediate_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {"lessons": []}, "lesson": {"content_flags": {"attestation": True}}},
        spec=_intermediate_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "needs at least one correct answer" in " ".join(result.validation_issues)
    assert intermediate_agent.calls == []
    assert validator.calls == []


def test_intermediate_worker_blocks_single_key_when_question_declares_multiple_valid_answers(tmp_path: Path) -> None:
    bad = _intermediate_assessment_artifact()
    question = bad.variants[2].closed_questions[10]
    question.id = "V3-C11"
    question.template_code = "10D"
    question.student_prompt = (
        "В выражении c = (a * 2 + b * 3) / 5 выберите «магическое число», которое можно вынести в переменную."
    )
    question.options = ["2", "3", "5", "a"]
    question.correct_answers = ["2"]
    question.autocheck_config = {"type": "10D", "correct_option_index": 0}
    question.internal_explanation = "Любое из 2/3/5 — магическое; здесь проверяем одно."
    artifact_agent = FakeGraph([bad])
    intermediate_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "IntermediateAssessmentArtifactAgent": artifact_agent,
            "IntermediateAssessmentAgent": intermediate_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {"lessons": []}, "lesson": {"content_flags": {"attestation": True}}},
        spec=_intermediate_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "declares multiple valid answers but provides exactly one correct answer" in " ".join(result.validation_issues)
    assert intermediate_agent.calls == []
    assert validator.calls == []


def test_intermediate_validation_view_keeps_internal_assessment_answers() -> None:
    spec = _intermediate_worker_spec()
    artifacts = {"intermediate_assessment": _intermediate_assessment_artifact().model_dump(mode="json")}

    view = sanitize_generation_artifacts_for_validation(spec, artifacts)

    assert view["intermediate_assessment"]["variants"][0]["closed_questions"][0]["correct_answers"] == [
        "Internal closed key 1.1"
    ]
    assert "intermediate_assessment_visibility_note" in view


def test_intermediate_source_contract_defines_internal_assessment_layer() -> None:
    spec = _intermediate_worker_spec()

    contract = source_contract_for_spec({"module": {"lessons": [{"lesson_number": 1}]}, "lesson": {}}, spec)

    assert contract["contract_type"] == "intermediate_assessment_contract"
    assert contract["required_variant_count"] == 4
    assert contract["required_closed_questions_per_variant"] == 16
    assert contract["required_open_questions_per_variant"] == 4
    assert contract["required_code_tasks_per_variant"] == 3
    assert "generation_artifacts.intermediate_assessment" in " ".join(contract["generation_rules"])
    assert "Do not require visible keys" in " ".join(contract["validation_rules"])


def test_intermediate_validation_policy_allows_options_and_publishable_html() -> None:
    policy = validation_policy_for_spec(_intermediate_worker_spec())

    assert "Candidate answer options in closed questions are allowed" in policy
    assert "not answer-key leakage unless the correct option is explicitly marked" in policy
    assert "Do not require platform-import markup" in policy
    assert "Find/fix/explain the error" in policy


def test_intermediate_controller_prompt_overrules_template_and_option_false_positives() -> None:
    spec = _intermediate_worker_spec()
    artifacts = {
        "intermediate_assessment": _intermediate_assessment_artifact().model_dump(mode="json"),
        "intermediate_assessment_check": {"approved": True, "issues": []},
    }

    prompt = build_validation_controller_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["10D options are answer leakage"]),
        merged_validation=ValidationResult.fail(["10D options are answer leakage"]),
        generation_artifacts=artifacts,
    )

    assert "For intermediate, overrule validator objections that treat candidate answer options" in prompt
    assert "platform-import template markup" in prompt
    assert "find/fix/explain the error" in prompt
    assert "generation_artifacts.intermediate_assessment" in prompt


def test_intermediate_appellate_policy_accepts_overstrict_controller_issues(tmp_path: Path) -> None:
    worker = MaterialWorker(
        subagents={},
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, validation_controller_accept_score=3),
    )
    artifacts = {
        "intermediate_assessment": _intermediate_assessment_artifact().model_dump(mode="json"),
        "intermediate_assessment_check": {"approved": True, "issues": []},
    }
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 2,
        "score_rationale": "controller kept failure",
        "rationale": "blocking",
        "blocking_issues": [
            "10D варианты ответов фактически показывают правильный ответ",
            "6A/6D/6G/8D/10D в HTML оформлены как обычные списки без platform template markup",
            "исправьте ошибки раскрывает решение, потому что исправление очевидно",
        ],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [],
    }

    adjusted = worker._apply_intermediate_appellate_policy(
        spec=_intermediate_worker_spec(),
        content=(
            '<style>.x{}</style><div class="cc-lesson">'
            '<h2 id="v1">Variant 1</h2>'
            "<p>Task 10D. Select the fragment that causes SyntaxError.</p>"
            "<ul><li>print</li><li>(\"Hello)</li></ul>"
            "<p>Fix the error in this line: print(\"Hello)</p>"
            "</div>"
        ),
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail(decision["blocking_issues"]),
        generation_artifacts=artifacts,
        decision=decision,
    )

    assert adjusted["approved"] is True
    assert adjusted["decision"] == "approve_material"
    assert adjusted["quality_score"] == 3
    assert adjusted["blocking_issues"] == []
    assert len(adjusted["overruled_validator_issues"]) == 3


def test_intermediate_appellate_policy_keeps_visible_key_failure(tmp_path: Path) -> None:
    worker = MaterialWorker(
        subagents={},
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, validation_controller_accept_score=3),
    )
    artifacts = {
        "intermediate_assessment": _intermediate_assessment_artifact().model_dump(mode="json"),
        "intermediate_assessment_check": {"approved": True, "issues": []},
    }
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 2,
        "score_rationale": "controller kept failure",
        "rationale": "blocking",
        "blocking_issues": ["10D варианты ответов фактически показывают правильный ответ"],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [],
    }

    adjusted = worker._apply_intermediate_appellate_policy(
        spec=_intermediate_worker_spec(),
        content=(
            '<style>.x{}</style><div class="cc-lesson">'
            "<h2>Variant 1</h2><p>Правильный ответ: B</p></div>"
        ),
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail(decision["blocking_issues"]),
        generation_artifacts=artifacts,
        decision=decision,
    )

    assert adjusted["approved"] is False
    assert adjusted["decision"] == "keep_failed"
    assert adjusted["blocking_issues"]


def test_practice_worker_uses_template_variant_pipeline_and_writes_artifacts(tmp_path: Path) -> None:
    call_order: list[str] = []
    template_agent = NamedFakeGraph("PracticeTaskTemplateAgent", [_practice_template_set()], call_order)
    variant_agent = NamedFakeGraph("PracticeTaskVariantAgent", [_practice_instance_set()], call_order)
    practice_agent = NamedFakeGraph(
        "PracticeMaterialAgent",
        [GeneratedMaterial(content=VALID_HTML, agent_notes=["html ok"])],
        call_order,
    )
    validator = NamedFakeGraph(
        "MaterialValidatorAgent",
        [MaterialValidationDecision(approved=True, passed_blocks=[])],
        call_order,
    )
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "PracticeMaterialAgent": practice_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )
    theory = MaterialResult(
        kind="theory",
        material_type="Theory",
        agent_type="TheoryMaterialAgent",
        status="approved",
        iterations=1,
        content='<style></style><div class="cc-lesson"><p>name = "Анна"</p></div>',
        prompt_files=(),
    )

    result = worker.run(
        task={
            "course": {},
            "module": {},
            "lesson": {
                "content_flags": {"practice": True},
                "hours": {"practice": 1},
                "practice_tasks": {"l1": [{"number": 1, "text": "Create a variable with a name and print it"}]},
            },
        },
        spec=_practice_worker_spec(),
        references={},
        dependency_results=[theory],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert call_order == [
        "PracticeTaskTemplateAgent",
        "PracticeTaskVariantAgent",
        "PracticeMaterialAgent",
        "MaterialValidatorAgent",
    ]
    assert result.generation_artifacts["practice_instances"]["tasks"][0]["hidden_solution"]
    assert "GENERATION ARTIFACTS FOR THIS MATERIAL" in practice_agent.calls[0]["prompt"]
    assert "PracticeTaskInstanceSet" not in result.content
    assert list((tmp_path / "tmp" / "practice").glob("*.practice_templates.json"))
    assert list((tmp_path / "tmp" / "practice").glob("*.practice_instances.json"))
    assert list((tmp_path / "tmp" / "practice").glob("*.practice_duplicate_check.json"))


def test_practice_worker_blocks_invalid_instance_ids_before_html(tmp_path: Path) -> None:
    bad_instances = PracticeTaskInstanceSet(
        tasks=[
            *_practice_instance_set().tasks,
            PracticeTaskInstance(
                id="P2",
                template_id="P2",
                level="L1",
                task_type="write_code",
                scenario="extra",
                student_condition="extra",
                hidden_solution="extra solution",
                teacher_explanation="extra explanation",
            ),
        ]
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([bad_instances])
    practice_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "PracticeMaterialAgent": practice_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={
            "course": {},
            "module": {},
            "lesson": {
                "content_flags": {"practice": True},
                "hours": {"practice": 1},
                "practice_tasks": {"l1": [{"number": 1, "text": "Create a variable with a name and print it"}]},
            },
        },
        spec=_practice_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "practice_instances task ids/order mismatch" in " ".join(result.validation_issues)
    assert practice_agent.calls == []
    assert validator.calls == []


def test_practice_worker_blocks_direct_copy_from_theory_before_html(tmp_path: Path) -> None:
    copied_text = "Создайте переменную student_name со значением Анна и выведите её на экран"
    copied_instances = PracticeTaskInstanceSet(
        tasks=[
            PracticeTaskInstance(
                id="P1",
                template_id="P1",
                level="L1",
                task_type="write_code",
                scenario=copied_text,
                student_condition=copied_text,
                hidden_solution="student_name = 'Анна'\nprint(student_name)",
                teacher_explanation="Проверить присваивание и вывод.",
            )
        ]
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([copied_instances])
    practice_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    theory = MaterialResult(
        kind="theory",
        material_type="Theory",
        agent_type="TheoryMaterialAgent",
        status="approved",
        iterations=1,
        content=f'<style></style><div class="cc-lesson"><p>{copied_text}</p></div>',
        prompt_files=(),
    )
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "PracticeMaterialAgent": practice_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={
            "course": {},
            "module": {},
            "lesson": {
                "content_flags": {"practice": True},
                "hours": {"practice": 1},
                "practice_tasks": {"l1": [{"number": 1, "text": "Create a variable with a name and print it"}]},
            },
        },
        spec=_practice_worker_spec(),
        references={},
        dependency_results=[theory],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "directly copies text/code from dependency:theory" in " ".join(result.validation_issues)
    assert result.generation_artifacts["practice_duplicate_check"]["approved"] is False
    assert practice_agent.calls == []
    assert validator.calls == []


def test_practice_worker_blocks_solution_hint_in_instance_before_html(tmp_path: Path) -> None:
    hinted_instances = PracticeTaskInstanceSet(
        tasks=[
            PracticeTaskInstance(
                id="P1",
                template_id="P1",
                level="L1",
                task_type="fix",
                scenario="Программа не запускается.",
                student_condition=(
                    "Дан код с одной ошибкой. Исправьте её: замените prnt на print, "
                    "чтобы программа вывела строку."
                ),
                starter_code="prnt('Готово')",
                input_requirements="Ввод не требуется.",
                output_requirements="Программа выводит Готово.",
                tests=[{"input": "", "expected_output": "Готово\n"}],
                hidden_solution="Заменить prnt на print.",
                teacher_explanation="Это опечатка в имени print.",
            )
        ]
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([hinted_instances])
    practice_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "PracticeMaterialAgent": practice_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={
            "course": {},
            "module": {},
            "lesson": {
                "content_flags": {"practice": True},
                "hours": {"practice": 1},
                "practice_tasks": {"l1": [{"number": 1, "text": "Fix one print typo"}]},
            },
        },
        spec=_practice_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "student-facing fields reveal the exact fix" in " ".join(result.validation_issues)
    assert practice_agent.calls == []
    assert validator.calls == []


def test_practice_solution_hint_check_allows_shared_error_message(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))

    issues = worker._practice_solution_hint_issues(
        {
            "id": "P1",
            "task_type": "fix",
            "source_text": "Fix code with SyntaxError: EOL while scanning string literal",
            "scenario": "IDE показывает сообщение об ошибке.",
            "student_condition": (
                'При запуске кода появляется ошибка: "SyntaxError: EOL while scanning string literal". '
                "Исправьте фрагмент так, чтобы программа запускалась."
            ),
            "input_requirements": "Ввод не требуется.",
            "output_requirements": "Программа должна пройти тесты.",
            "hidden_solution": (
                "Сообщение SyntaxError: EOL while scanning string literal помогает понять тип ошибки; "
                "исправление находится в starter_code."
            ),
        }
    )

    assert issues == []


def test_practice_worker_blocks_internal_source_marker_in_student_fields(tmp_path: Path) -> None:
    marked_instances = PracticeTaskInstanceSet(
        tasks=[
            PracticeTaskInstance(
                id="P1",
                template_id="P1",
                level="L1",
                task_type="fix",
                scenario="Программа не запускается.",
                student_condition=(
                    "Источник (как в задании урока): Fix one print typo.\n\n"
                    "Исправьте код ниже так, чтобы программа вывела строку."
                ),
                starter_code="prnt('Готово')",
                input_requirements="Ввод не требуется.",
                output_requirements="Программа должна пройти тесты.",
                tests=[{"input": "", "expected_output": "Готово\n"}],
                hidden_solution="Заменить prnt на print.",
                teacher_explanation="Это опечатка в имени print.",
            )
        ]
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([marked_instances])
    practice_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "PracticeMaterialAgent": practice_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={
            "course": {},
            "module": {},
            "lesson": {
                "content_flags": {"practice": True},
                "hours": {"practice": 1},
                "practice_tasks": {"l1": [{"number": 1, "text": "Fix one print typo"}]},
            },
        },
        spec=_practice_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "student-facing fields contain internal source/pipeline marker" in " ".join(result.validation_issues)
    assert practice_agent.calls == []
    assert validator.calls == []


def test_practice_worker_normalizes_output_test_key_before_rendering(tmp_path: Path) -> None:
    instances = PracticeTaskInstanceSet(
        tasks=[
            PracticeTaskInstance(
                id="P1",
                template_id="P1",
                level="L1",
                task_type="write_code",
                scenario="Выведите статус.",
                student_condition="Создайте переменную status и выведите её.",
                input_requirements="Ввод не требуется.",
                output_requirements="Программа выводит Готово.",
                tests=[{"input": "", "output": "Готово\n"}],
                hidden_solution="status = 'Готово'\nprint(status)",
                teacher_explanation="Проверяется присваивание и вывод.",
            )
        ]
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([instances])
    practice_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "PracticeMaterialAgent": practice_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={
            "course": {},
            "module": {},
            "lesson": {
                "content_flags": {"practice": True},
                "hours": {"practice": 1},
                "practice_tasks": {"l1": [{"number": 1, "text": "Create a variable with a status and print it"}]},
            },
        },
        spec=_practice_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    task = result.generation_artifacts["practice_instances"]["tasks"][0]
    assert result.status == "approved"
    assert task["tests"][0]["expected_output"] == "Готово\n"
    assert task["runtime_tests"][0]["expected_output"] == "Готово\n"
    assert "GENERATION ARTIFACTS FOR THIS MATERIAL" in practice_agent.calls[0]["prompt"]


def test_practice_worker_mirrors_runtime_tests_to_legacy_tests(tmp_path: Path) -> None:
    instances = PracticeTaskInstanceSet(
        tasks=[
            PracticeTaskInstance(
                id="P1",
                template_id="P1",
                level="L1",
                task_type="write_code",
                scenario="Выведите статус.",
                student_condition="Создайте переменную status и выведите её.",
                input_requirements="Ввод не требуется.",
                output_requirements="Программа выводит Готово.",
                runtime_tests=[{"input": "", "stdout": "Готово\n"}],
                manual_checks=["Проверьте читаемое имя переменной."],
                run_mode="separate_snippets",
                hidden_solution="status = 'Готово'\nprint(status)",
                teacher_explanation="Проверяется присваивание и вывод.",
            )
        ]
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([instances])
    practice_agent = FakeGraph([GeneratedMaterial(content=VALID_HTML)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "PracticeMaterialAgent": practice_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={
            "course": {},
            "module": {},
            "lesson": {
                "content_flags": {"practice": True},
                "hours": {"practice": 1},
                "practice_tasks": {"l1": [{"number": 1, "text": "Create a variable with a status and print it"}]},
            },
        },
        spec=_practice_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    task = result.generation_artifacts["practice_instances"]["tasks"][0]
    assert result.status == "approved"
    assert task["tests"][0]["expected_output"] == "Готово\n"
    assert task["runtime_tests"][0]["expected_output"] == "Готово\n"
    assert task["manual_checks"] == ["Проверьте читаемое имя переменной."]


def test_practice_worker_fails_when_student_html_leaks_hidden_solution(tmp_path: Path) -> None:
    leaked_html = (
        '<style>.x{}</style><div class="cc-lesson"><h2>P1</h2>'
        "<p>city = 'Тула'\nprint(city)</p></div>"
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([_practice_instance_set()])
    practice_agent = FakeGraph([GeneratedMaterial(content=leaked_html)])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "PracticeMaterialAgent": practice_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={
            "course": {},
            "module": {},
            "lesson": {
                "content_flags": {"practice": True},
                "hours": {"practice": 1},
                "practice_tasks": {"l1": [{"number": 1, "text": "Create a variable with a name and print it"}]},
            },
        },
        spec=_practice_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "student practice HTML leaks hidden_solution for P1" in " ".join(result.validation_issues)
    assert len(validator.calls) == 1


def test_practice_student_leak_check_ignores_starter_code_in_hidden_solution(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    content = "<style></style><div class=\"cc-lesson\"><pre><code># Табличка\nprint('Секция)</code></pre></div>"
    instances = {
        "tasks": [
            {
                "id": "P1",
                "scenario": "Исправьте программу.",
                "student_condition": "Код не запускается.",
                "starter_code": "# Табличка\nprint('Секция)",
                "input_requirements": "stdin пустой",
                "output_requirements": "stdout должен совпасть с тестом",
                "hidden_solution": "Нужно закрыть строковый литерал.\nИсправленный код:\n# Табличка\nprint('Секция')",
                "teacher_explanation": "Строка должна быть закрыта кавычкой.",
            }
        ]
    }

    assert worker._practice_student_leak_issues(content, instances) == []


def test_practice_student_leak_check_rejects_source_pipeline_markers(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    content = (
        "<style></style><div class=\"cc-lesson\"><p><strong>Исходный паттерн из JSON:</strong> "
        "Исправить на print.</p></div>"
    )

    issues = worker._practice_student_leak_issues(content, {"tasks": []})

    assert "student practice HTML contains internal source/pipeline marker" in " ".join(issues)


def test_channel_policy_resolves_key_visibility_for_mr_practice() -> None:
    spec = MaterialSpec(
        kind="mr_practice",
        material_type="Teacher Practice Guidance",
        agent_type="TeacherGuidanceAgent",
        prompt_files=(),
        validator_kind="teacher_guidance",
    )

    channel_policy = channel_key_visibility_policy_for_spec(spec)
    validation_policy = validation_policy_for_spec(spec)

    assert "CHANNEL AND KEY VISIBILITY POLICY" in channel_policy
    assert "Learner-facing lesson materials are theory, practice, and self_work" in channel_policy
    assert "Self-check/autocheck keys for learner-facing self_work must not be visible in HTML" in channel_policy
    assert "mr_practice is teacher-facing and is expected to include keys/solutions" in channel_policy
    assert "A \"do not show keys\" instruction applies to learner-facing materials" in channel_policy
    assert "MR_PRACTICE VALIDATION POLICY" in validation_policy
    assert "Do not reject mr_practice for containing solution keys" in validation_policy
    assert "Reject if teacher keys are absent" in validation_policy


def test_mr_practice_source_contract_has_authoritative_task_ids() -> None:
    spec = MaterialSpec(
        kind="mr_practice",
        material_type="Teacher Practice Guidance",
        agent_type="TeacherGuidanceAgent",
        prompt_files=(),
        validator_kind="teacher_guidance",
    )
    task = {
        "lesson": {
            "hours": {"practice": 1},
            "difficulty": {"l1": {"count": 2}, "l2": {"count": 3}},
            "practice_tasks": {
                "l1": [
                    {"number": 1, "text": "Create name"},
                    {"number": 2, "text": "Create age"},
                ],
                "l2": [
                    {"number": 3, "text": "Print name age"},
                    {"number": 4, "text": "Print phrase"},
                    {"number": 5, "text": "Favorite color and animal"},
                ],
            },
        }
    }

    contract = source_contract_for_spec(task, spec)

    assert contract["contract_type"] == "mr_practice_task_key_contract"
    assert contract["authoritative_task_ids"] == ["P1", "P2", "P3", "P4", "P5"]
    assert contract["required_task_count"] == 5
    assert "Do not add, infer, or preserve any P task" in " ".join(contract["generation_rules"])
    assert "Missing keys for task ids outside authoritative_task_ids are not valid validation issues" in " ".join(contract["validation_rules"])


def test_mr_practice_prompts_ignore_non_authoritative_validator_task_feedback() -> None:
    spec = MaterialSpec(
        kind="mr_practice",
        material_type="Teacher Practice Guidance",
        agent_type="TeacherGuidanceAgent",
        prompt_files=(),
        validator_kind="teacher_guidance",
    )
    task = {
        "course": {},
        "module": {},
        "lesson": {
            "practice_tasks": {
                "l1": [{"number": 1, "text": "Create name"}],
                "l2": [{"number": 2, "text": "Create color"}],
            }
        },
    }
    dependency = MaterialResult(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        status="approved",
        iterations=1,
        content='<style></style><div class="cc-lesson"><h3>P1</h3><h3>P2</h3></div>',
        prompt_files=(),
    )

    generation_prompt = build_generation_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[dependency],
        previous_content='<style></style><div class="cc-lesson"><h3>P1</h3><h3>P2</h3></div>',
        previous_issues=["add missing keys for P6 and P7"],
    )
    validation_prompt = build_validation_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[dependency],
        content=dependency.content,
        rule_result=ValidationResult.ok(),
    )

    assert '"authoritative_task_ids": [\n    "P1",\n    "P2"\n  ]' in generation_prompt
    assert "Ignore validator feedback that asks for non-authoritative tasks" in generation_prompt
    assert "validator issue says that keys are missing for a task id that is not listed in authoritative_task_ids" in validation_prompt
    assert "Missing keys for task ids outside authoritative_task_ids are not valid validation issues" in validation_prompt


def test_mr_intermediate_contract_uses_dependency_artifact_without_key_bank_html() -> None:
    spec = get_material_spec("mr_intermediate")

    contract = source_contract_for_spec({"course": {}, "module": {}, "lesson": {}}, spec)

    assert contract["contract_type"] == "mr_intermediate_guidance_contract"
    assert "generation_artifacts.intermediate_assessment" in " ".join(contract["generation_rules"])
    assert "Do not duplicate the full variant-by-variant answer bank" in " ".join(contract["generation_rules"])
    assert "Do not require a full key bank inside mr_intermediate HTML" in " ".join(contract["validation_rules"])


def test_mr_intermediate_prompts_keep_full_keys_in_intermediate_artifact() -> None:
    spec = get_material_spec("mr_intermediate")
    dependency = MaterialResult(
        kind="intermediate",
        material_type="Intermediate",
        agent_type="IntermediateAssessmentAgent",
        status="approved",
        iterations=1,
        content=VALID_HTML,
        prompt_files=(),
        generation_artifacts={"intermediate_assessment": _intermediate_assessment_artifact().model_dump(mode="json")},
    )

    generation_prompt = build_generation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[dependency],
        previous_content="",
        previous_issues=[],
    )
    validation_prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[dependency],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
    )

    assert "MR_INTERMEDIATE VALIDATION POLICY" in validation_prompt
    assert "Do not require a full key bank" in validation_prompt
    assert "Reject mr_intermediate HTML if it duplicates full variant-by-variant keys" in validation_prompt
    assert "intermediate_assessment" in generation_prompt
    assert "Не печатай полный банк ключей" in generation_prompt
    assert "внутреннего QA/artifact-слоя intermediate_assessment" in generation_prompt


def test_controller_prompt_overrules_missing_mr_intermediate_key_bank_when_artifact_exists() -> None:
    spec = get_material_spec("mr_intermediate")
    dependency = MaterialResult(
        kind="intermediate",
        material_type="Intermediate",
        agent_type="IntermediateAssessmentAgent",
        status="approved",
        iterations=1,
        content=VALID_HTML,
        prompt_files=(),
        generation_artifacts={"intermediate_assessment": _intermediate_assessment_artifact().model_dump(mode="json")},
    )

    prompt = build_validation_controller_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[dependency],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["mr_intermediate lacks full keys in HTML"]),
        merged_validation=ValidationResult.fail(["mr_intermediate lacks full keys in HTML"]),
    )

    assert "For mr_intermediate, overrule validator objections that require the full intermediate key bank" in prompt
    assert "MR_INTERMEDIATE VALIDATION POLICY" in prompt
    assert "generation_artifacts.intermediate_assessment" in prompt
    assert "Do not require a full key bank" in prompt


def test_specification_qa_contract_preserves_underspecified_practice_tasks() -> None:
    spec = get_material_spec("specification_qa")
    task = {
        "lesson": {
            "hours": {"practice": 1},
            "difficulty": {"l1": {"count": 1}, "l2": {"count": 1}},
            "practice_tasks": {
                "l1": [{"number": 1, "text": "Create name = \"Анна\" and print it"}],
                "l2": [{"number": 2, "text": "Create favorite color and favorite animal variables and print them"}],
            },
        }
    }

    contract = source_contract_for_spec(task, spec)

    assert contract["contract_type"] == "specification_qa_practice_task_contract"
    assert contract["authoritative_task_ids"] == ["P1", "P2"]
    generation_rules = " ".join(contract["generation_rules"])
    validation_rules = " ".join(contract["validation_rules"])
    assert "Do not invent concrete variable names, concrete values, exact stdout" in generation_rules
    assert "mark it as requiring source clarification or manual checking" in generation_rules
    assert "Reject deterministic tests or keys for underspecified source tasks" in validation_rules
    assert "no deterministic test" in validation_rules


def test_specification_qa_depends_on_generated_materials() -> None:
    spec = get_material_spec("specification_qa")

    assert "practice" in spec.dependency_kinds
    assert "mr_practice" in spec.dependency_kinds
    assert "theory" in spec.dependency_kinds
    assert "mr_theory" in spec.dependency_kinds


def test_specification_qa_validation_prompt_rejects_invented_underspecified_tests() -> None:
    spec = get_material_spec("specification_qa")
    task = {
        "course": {},
        "module": {},
        "lesson": {
            "practice_tasks": {
                "l2": [{"number": 5, "text": "Create favorite color and favorite animal variables and print them"}]
            }
        },
    }
    practice = MaterialResult(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        status="approved",
        iterations=1,
        content=(
            '<style></style><div class="cc-lesson"><h3>P5</h3>'
            "<p>tests are absent/not applicable until source clarification</p></div>"
        ),
        prompt_files=(),
    )

    prompt = build_validation_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[practice],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
    )

    assert "SPECIFICATION_QA VALIDATION POLICY" in prompt
    assert "authoritative_task_ids" in prompt
    assert "Do not require or approve invented concrete values" in prompt
    assert "no deterministic test" in prompt
    assert "reuse that interpretation" in prompt


def test_mr_theory_policy_keeps_teacher_guidance_separate_from_student_theory() -> None:
    spec = MaterialSpec(
        kind="mr_theory",
        material_type="Teacher Theory Guidance",
        agent_type="TeacherGuidanceAgent",
        prompt_files=(),
        validator_kind="teacher_guidance",
    )

    channel_policy = channel_key_visibility_policy_for_spec(spec)
    validation_policy = validation_policy_for_spec(spec)

    assert "mr_theory is teacher-facing and must be judged as teacher guidance" in channel_policy
    assert "MR_THEORY VALIDATION POLICY" in validation_policy
    assert "Do not validate it as a student theory material" in validation_policy
    assert "Do not require a learner-facing \"Проверка себя\" / \"#selfcheck\" section" in validation_policy
    assert "Do not reject mr_theory merely because it contains teacher-facing keys" in validation_policy


def test_generation_prompt_includes_channel_policy_for_mr_practice() -> None:
    spec = MaterialSpec(
        kind="mr_practice",
        material_type="Teacher Practice Guidance",
        agent_type="TeacherGuidanceAgent",
        prompt_files=(),
        validator_kind="teacher_guidance",
    )

    prompt = build_generation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        previous_content="",
        previous_issues=[],
    )

    assert "CHANNEL AND KEY VISIBILITY POLICY" in prompt
    assert "mr_practice is teacher-facing and is expected to include keys/solutions" in prompt
    assert "The checked material kind is mr_practice" in prompt


def test_controller_prompt_includes_mr_theory_validation_policy() -> None:
    spec = MaterialSpec(
        kind="mr_theory",
        material_type="Teacher Theory Guidance",
        agent_type="TeacherGuidanceAgent",
        prompt_files=(),
        validator_kind="teacher_guidance",
    )

    prompt = build_validation_controller_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["mr_theory looks like teacher guidance"]),
        merged_validation=ValidationResult.fail(["mr_theory looks like teacher guidance"]),
    )

    assert "MR_THEORY VALIDATION POLICY" in prompt
    assert "Do not validate it as a student theory material" in prompt
    assert "Do not require a learner-facing \"Проверка себя\" / \"#selfcheck\" section" in prompt
    assert "mr_theory is teacher-facing and must be judged as teacher guidance" in prompt


def test_controller_prompt_overrules_non_authoritative_task_ids() -> None:
    spec = MaterialSpec(
        kind="mr_practice",
        material_type="Teacher Practice Guidance",
        agent_type="TeacherGuidanceAgent",
        prompt_files=(),
        validator_kind="teacher_guidance",
    )
    task = {
        "course": {},
        "module": {},
        "lesson": {
            "practice_tasks": {
                "l1": [{"number": 1, "text": "Create name"}],
                "l2": [{"number": 2, "text": "Create color"}],
            }
        },
    }

    prompt = build_validation_controller_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["missing keys for P6 and P7"]),
        merged_validation=ValidationResult.fail(["missing keys for P6 and P7"]),
    )

    assert "authoritative_task_ids" in prompt
    assert "validator objections about missing keys/content for task ids outside that list are invalid" in prompt
    assert "If a validator issue says that keys are missing for a task id that is not listed in authoritative_task_ids" in prompt


def test_package_validation_prompt_includes_full_final_content() -> None:
    content = '<style>.x{}</style><div class="cc-lesson"><p>' + ("x" * 9000) + "TAIL</p></div>"
    spec = MaterialSpec(
        kind="theory",
        material_type="Theory",
        agent_type="TheoryMaterialAgent",
        prompt_files=(),
        validator_kind="theory",
    )
    material = MaterialResult(
        kind="theory",
        material_type="Theory",
        agent_type="TheoryMaterialAgent",
        status="approved",
        iterations=1,
        content=content,
        prompt_files=(),
    )

    prompt = build_package_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {"lesson_number": 3}},
        specs=[spec],
        materials=[material],
        rule_result=ValidationResult.ok(),
    )

    assert "Package validation is advisory" in prompt
    assert "FULL MATERIALRESULT OBJECTS" in prompt
    assert "content_truncated" in prompt
    assert "TAIL</p></div>" in prompt
    assert "Do not claim that a material is truncated unless content_truncated is true" in prompt


def test_package_validator_is_advisory_and_preserves_issues(tmp_path: Path) -> None:
    spec = MaterialSpec(
        kind="theory",
        material_type="Theory",
        agent_type="TheoryMaterialAgent",
        prompt_files=(),
        validator_kind="theory",
    )
    material = MaterialResult(
        kind="theory",
        material_type="Theory",
        agent_type="TheoryMaterialAgent",
        status="approved",
        iterations=1,
        content=VALID_HTML,
        prompt_files=(),
    )
    package_agent = FakeGraph(
        [
            PackageValidationDecision(
                approved=False,
                issues=["source-data warning should be reviewed"],
                fix_instructions=["review source-data warning"],
            )
        ]
    )
    validator = PackageValidator(
        subagents={"PackageValidatorAgent": package_agent},
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path),
    )

    result = validator.validate(
        task={"course": {}, "module": {}, "lesson": {"lesson_number": 3}},
        specs=[spec],
        materials=[material],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.approved is True
    assert result.issues == ["source-data warning should be reviewed"]
    assert "FULL MATERIALRESULT OBJECTS" in package_agent.calls[0]["prompt"]
    assert "cc-lesson" in package_agent.calls[0]["prompt"]
    assert "Concepts" in package_agent.calls[0]["prompt"]


def test_runtime_result_status_ignores_advisory_package_validation(tmp_path: Path) -> None:
    material = MaterialResult(
        kind="theory",
        material_type="Theory",
        agent_type="TheoryMaterialAgent",
        status="approved",
        iterations=1,
        content=VALID_HTML,
        prompt_files=(),
    )
    runtime = IsmartGeneratorRuntime(
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path),
        subagents={},
    )

    assert runtime._result_status([material], ValidationResult.fail(["advisory warning"])) == "approved"


def test_runtime_agents_called_includes_practice_pipeline_agents(tmp_path: Path) -> None:
    practice = MaterialResult(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        status="approved",
        iterations=1,
        content=VALID_HTML,
        prompt_files=(),
        generation_artifacts={"practice_templates": {"tasks": []}, "practice_instances": {"tasks": []}},
    )
    runtime = IsmartGeneratorRuntime(
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path),
        subagents={},
    )

    assert runtime._agents_called([practice], package_validator_called=False) == [
        "PracticeTaskTemplateAgent",
        "PracticeTaskVariantAgent",
        "PracticeMaterialAgent",
        "MaterialValidatorAgent",
    ]


def test_runtime_agents_called_includes_self_work_autocheck_agent(tmp_path: Path) -> None:
    self_work = MaterialResult(
        kind="self_work",
        material_type="Self Work",
        agent_type="SelfStudyAgent",
        status="approved",
        iterations=1,
        content=VALID_HTML,
        prompt_files=(),
        generation_artifacts={"self_work_autocheck": {"selfcheck_questions": []}},
    )
    runtime = IsmartGeneratorRuntime(
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path),
        subagents={},
    )

    assert runtime._agents_called([self_work], package_validator_called=False) == [
        "SelfWorkAutocheckAgent",
        "SelfStudyAgent",
        "MaterialValidatorAgent",
    ]


def test_runtime_agents_called_includes_intermediate_artifact_agent(tmp_path: Path) -> None:
    intermediate = MaterialResult(
        kind="intermediate",
        material_type="Intermediate",
        agent_type="IntermediateAssessmentAgent",
        status="approved",
        iterations=1,
        content=VALID_HTML,
        prompt_files=(),
        generation_artifacts={"intermediate_assessment": {"variants": []}},
    )
    runtime = IsmartGeneratorRuntime(
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path),
        subagents={},
    )

    assert runtime._agents_called([intermediate], package_validator_called=False) == [
        "IntermediateAssessmentArtifactAgent",
        "IntermediateAssessmentAgent",
        "MaterialValidatorAgent",
    ]


def test_controller_prompt_biases_toward_overruling_over_strict_validator() -> None:
    spec = MaterialSpec(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        prompt_files=(),
        validator_kind="practice",
    )

    prompt = build_validation_controller_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["training task wording is suspicious"]),
        merged_validation=ValidationResult.fail(["training task wording is suspicious"]),
    )

    assert "Default stance: be skeptical of validator strictness" in prompt
    assert "APPELLATE REVIEW METHOD" in prompt
    assert "Treat the validator's factual claims" in prompt
    assert "one local wording or terminology imperfection" in prompt
    assert "Distinguish educational simplification from factual error" in prompt
    assert "prefer approving the material" in prompt
    assert "without autocheck" in prompt
    assert "the score should be at least 3" in prompt
    assert "CHANNEL AND KEY VISIBILITY POLICY" in prompt
    assert "overrule validator objections that treat visible deterministic expected stdout as a forbidden key" in prompt


class FakeCompiledAgent:
    def __init__(self, status: str = "approved") -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"state": state, "config": config})
        return {
            "messages": [AIMessage(content=f"done: {self.status}")],
            "results": [{"status": self.status, "output_dir": "out"}],
        }


def test_cli_runs_langgraph_agent_with_configurable_context(monkeypatch: Any, capsys: Any) -> None:
    fake_agent = FakeCompiledAgent(status="approved")
    monkeypatch.setattr(cli, "initialize_agent", lambda **_kwargs: fake_agent)

    code = cli.main(
        [
            "--input",
            "task.json",
            "--lesson-number",
            "3",
            "--output",
            "docs/generated output",
            "--max-generation-iterations",
            "5",
            "--verbose",
        ]
    )

    assert code == 0
    stdout = capsys.readouterr().out
    assert "done: approved" in stdout
    call = fake_agent.calls[0]
    configurable = call["config"]["configurable"]
    assert configurable["input"] == "task.json"
    assert configurable["output"] == "docs/generated output"
    assert configurable["lesson_number"] == "3"
    assert configurable["max_generation_iterations"] == 5
    assert configurable["verbose"] is True
    assert "thread_id" in configurable


def test_cli_returns_nonzero_for_failed_generation(monkeypatch: Any, capsys: Any) -> None:
    fake_agent = FakeCompiledAgent(status="failed")
    monkeypatch.setattr(cli, "initialize_agent", lambda **_kwargs: fake_agent)

    code = cli.main(["--input", "task.json", "--output", "out"])

    assert code == 1
    assert "done: failed" in capsys.readouterr().out
