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
    source_contract_for_spec,
    validation_policy_for_spec,
)
from agents.ismart_generator_agent.contracts import IsmartGenerationConfig, MaterialResult, MaterialSpec
from agents.ismart_generator_agent.contracts import ValidationResult
from agents.ismart_generator_agent.runtime import IsmartGeneratorRuntime
from agents.ismart_generator_agent.schemas import (
    GeneratedMaterial,
    MaterialValidationDecision,
    ValidationControllerDecision,
    PackageValidationDecision,
)
from agents.ismart_generator_agent.subagents import (
    ALL_SUBAGENT_TYPES,
    build_subagent_registry,
)
from agents.ismart_generator_agent.workers import MaterialWorker, PackageValidator


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
    assert MaterialValidationDecision in model.schemas
    assert ValidationControllerDecision in model.schemas


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
    assert "Self-check/autocheck keys are allowed only when the checked material's own MaterialSpec/prompt explicitly requires them" in channel_policy
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
