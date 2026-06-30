from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage

from agents.ismart_generator_agent import agent as agent_module
from agents.ismart_generator_agent import cli
from agents.ismart_generator_agent import sequential_runner
from agents.ismart_generator_agent.context import (
    build_intermediate_assessment_artifact_prompt,
    build_generation_prompt,
    build_package_validation_prompt,
    build_practice_variant_prompt,
    build_validation_prompt,
    build_validation_controller_prompt,
    channel_key_visibility_policy_for_spec,
    generation_artifacts_for_validation,
    source_contract_for_spec,
    validation_policy_for_spec,
)
from agents.ismart_generator_agent.contracts import (
    IsmartGenerationConfig,
    IsmartGenerationResult,
    MaterialResult,
    MaterialSpec,
    ReferenceDocument,
)
from agents.ismart_generator_agent.contracts import ValidationResult
from agents.ismart_generator_agent.planner import build_material_plan
from agents.ismart_generator_agent.profiles import (
    config_for_task_profile,
    langfuse_agent_name_for_task,
    prompts_dir_for_level,
    resolve_course_class,
    resolve_course_level,
)
from agents.ismart_generator_agent.runtime import IsmartGeneratorRuntime
from agents.ismart_generator_agent.registry import FORMAT_PROMPT, get_material_spec
from agents.ismart_generator_agent.schemas import (
    CurrentControlAutocheckQuestion,
    CurrentControlAutocheckSet,
    GeneratedMaterial,
    IntermediateAssessmentArtifact,
    IntermediateAssessmentVariant,
    IntermediateCodeTask,
    IntermediateOpenCodeQuestion,
    IntermediateTestQuestion,
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
from agents.ismart_generator_agent import subagents as subagents_module
from agents.ismart_generator_agent.sources import read_prompt_files
from agents.ismart_generator_agent.subagents import (
    ALL_SUBAGENT_TYPES,
    build_subagent_registry,
)
from agents.ismart_generator_agent.task_skip import (
    NO_PRACTICE_TASKS_SKIP_REASON,
    practice_task_count,
    skip_reason_for_task,
)
from agents.ismart_generator_agent.workers import (
    HTML_TEMPLATE_PATH,
    MaterialWorker,
    PackageValidator,
    StructuredSubagentError,
    StructuredSubagentInvoker,
    _normalize_practice_instance_tests,
    load_html_format_template,
    render_current_control_material_html,
    render_practice_material_html,
)
from agents.ismart_generator_agent.trace import TraceLogger
from agents.ismart_generator_agent.tracker_converter import detect_course_level, find_references_dir, parse_tasks


VALID_HTML = '<style>.x{}</style><div class="cc-lesson"><h2 id="concepts">Concepts</h2><p>ok</p></div>'


def _profile_task(*, lesson_level: str | None = None, course_level: str | None = None) -> dict[str, Any]:
    lesson: dict[str, Any] = {
        "lesson_number": 1,
        "title": "Profile test",
        "hours": {"practice": 1},
        "content": {"audience": ""},
        "content_flags": {"practice": True},
        "practice_tasks": {"l1": [{"number": 1, "text": "Write a program."}], "l2": [], "l3": []},
    }
    if lesson_level is not None:
        lesson["course_level"] = lesson_level
    course: dict[str, Any] = {"title": "Python"}
    if course_level is not None:
        course["level"] = course_level
    return {"task_id": "profile-test", "course": course, "module": {"title": "M"}, "lesson": lesson}


def test_course_level_resolution_prefers_lesson_then_course_then_basic() -> None:
    assert resolve_course_level(_profile_task(lesson_level="advanced", course_level="basic")) == "advanced"
    assert resolve_course_level(_profile_task(course_level="продвинутый")) == "advanced"
    assert resolve_course_level(_profile_task()) == "basic"


def test_langfuse_agent_name_uses_profile_and_class() -> None:
    advanced_task = _profile_task(lesson_level="advanced")
    advanced_task["lesson"]["content"]["audience"] = "10-11 классы"
    basic_task = _profile_task(lesson_level="basic")
    basic_task["lesson"]["content"]["audience"] = "8-9 классы"

    assert resolve_course_class(advanced_task) == "10"
    assert langfuse_agent_name_for_task(advanced_task) == "ismart_generator_advanced_10_lesson_1"
    assert resolve_course_class(basic_task) == "8"
    assert langfuse_agent_name_for_task(basic_task) == "ismart_generator_basic_8_lesson_1"


def test_practice_lesson_without_practice_tasks_is_skippable() -> None:
    task = _profile_task()
    task["lesson"]["practice_tasks"] = {"l1": [], "l2": [], "l3": []}

    assert practice_task_count(task) == 0
    assert skip_reason_for_task(task) == NO_PRACTICE_TASKS_SKIP_REASON


def test_run_tasks_skips_practice_lesson_without_llm(tmp_path: Path) -> None:
    task = _profile_task()
    task["lesson"]["practice_tasks"] = {"l1": [], "l2": [], "l3": []}

    def fail_subagent_factory() -> Mapping[str, Any]:
        raise AssertionError("subagents should not be built for skipped lessons")

    results = agent_module.run_tasks(
        [task],
        config=IsmartGenerationConfig(output_root=tmp_path),
        subagent_factory=fail_subagent_factory,
    )

    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].skip_reason == NO_PRACTICE_TASKS_SKIP_REASON
    result_path = Path(results[0].output_dir) / "result.json"
    manifest_path = Path(results[0].output_dir) / "manifest.json"
    assert result_path.exists()
    assert manifest_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "skipped"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["skip_reason"] == NO_PRACTICE_TASKS_SKIP_REASON


def test_sequential_runner_skips_empty_practice_lesson_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    task = _profile_task()
    task["lesson"]["practice_tasks"] = {"l1": [], "l2": [], "l3": []}
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([task], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sequential_runner, "build_callback_handlers", lambda _log_name: [])
    monkeypatch.setattr(
        sequential_runner,
        "get_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    exit_code = sequential_runner.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "out"),
            "--run-name",
            "skip-run",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"event": "task.skipped"' in captured.out
    manifest = json.loads((tmp_path / "out" / "skip-run" / "sequential_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_skips"
    assert manifest["skipped_count"] == 1
    assert manifest["tasks"][0]["status"] == "skipped"
    assert manifest["tasks"][0]["skip_reason"] == NO_PRACTICE_TASKS_SKIP_REASON


def test_profile_config_selects_sibling_prompt_directory(tmp_path: Path) -> None:
    basic_dir = tmp_path / "prompts_skills"
    advanced_dir = tmp_path / "prompts_skills_advanced"
    basic_dir.mkdir()
    advanced_dir.mkdir()

    config = IsmartGenerationConfig(prompts_dir=basic_dir, output_root=tmp_path)

    assert config_for_task_profile(config, _profile_task()).prompts_dir == basic_dir
    assert config_for_task_profile(config, _profile_task(lesson_level="advanced")).prompts_dir == advanced_dir
    assert prompts_dir_for_level("basic", base_prompts_dir=advanced_dir) == basic_dir
    traced = config_for_task_profile(config, _profile_task(lesson_level="advanced"))
    assert traced.langchain_config["run_name"] == "ismart_generator_advanced_unknown_lesson_1"
    assert "profile:advanced" in traced.langchain_config["tags"]
    assert "lesson:1" in traced.langchain_config["tags"]


def test_material_plan_uses_advanced_registry_when_task_is_advanced(tmp_path: Path) -> None:
    config = IsmartGenerationConfig(prompts_dir=tmp_path / "prompts_skills", output_root=tmp_path, course_level="advanced")

    specs = build_material_plan(_profile_task(lesson_level="advanced"), config)
    practice = next(spec for spec in specs if spec.kind == "practice")

    assert practice.course_level == "advanced"
    assert "L3 tasks are allowed" in practice.validation_policy_addendum


def test_prompt_isolation_reads_only_resolved_profile_files(tmp_path: Path) -> None:
    basic_dir = tmp_path / "prompts_skills"
    advanced_dir = tmp_path / "prompts_skills_advanced"
    basic_dir.mkdir()
    advanced_dir.mkdir()
    spec = get_material_spec("practice", course_level="advanced")
    for name in spec.prompt_files:
        (basic_dir / name).write_text(f"basic marker for {name}", encoding="utf-8")
        (advanced_dir / name).write_text(f"advanced marker for {name}", encoding="utf-8")

    task_config = config_for_task_profile(
        IsmartGenerationConfig(prompts_dir=basic_dir, output_root=tmp_path),
        _profile_task(lesson_level="advanced"),
    )
    prompt_contents = read_prompt_files(task_config, spec.prompt_files)

    assert prompt_contents
    assert all("advanced marker" in value for value in prompt_contents.values())
    assert all("basic marker" not in value for value in prompt_contents.values())


def test_run_tasks_uses_fresh_subagents_per_task(monkeypatch: Any, tmp_path: Path) -> None:
    task_one = {"task_id": "lesson-1", "course": {}, "module": {}, "lesson": {"lesson_number": 1, "title": "L1"}}
    task_two = {"task_id": "lesson-2", "course": {}, "module": {}, "lesson": {"lesson_number": 2, "title": "L2"}}
    factory_results: list[dict[str, Any]] = []
    received_subagents: list[dict[str, Any]] = []

    def subagent_factory() -> dict[str, Any]:
        subagents = {"marker": len(factory_results)}
        factory_results.append(subagents)
        return subagents

    def fake_run_ismart_task(
        task: dict[str, Any],
        config: IsmartGenerationConfig,
        *,
        subagents: Mapping[str, Any],
        run_dir: Path,
        module_material_summaries: dict[str, list[dict[str, Any]]] | None = None,
    ) -> IsmartGenerationResult:
        received_subagents.append(subagents)  # type: ignore[arg-type]
        return IsmartGenerationResult(
            task_id=str(task["task_id"]),
            lesson_number=str(task["lesson"]["lesson_number"]),
            lesson_title=str(task["lesson"]["title"]),
            course_level="basic",
            status="approved",
            output_dir=str(run_dir),
            materials=[],
            package_validation=ValidationResult(approved=True),
            reference_summary={},
            agents_called=[],
            prompt_files_used=[],
        )

    monkeypatch.setattr(agent_module, "run_ismart_task", fake_run_ismart_task)

    results = agent_module.run_tasks(
        [task_one, task_two],
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path),
        subagent_factory=subagent_factory,
    )

    assert [result.task_id for result in results] == ["lesson-1", "lesson-2"]
    assert len(factory_results) == 2
    assert received_subagents == factory_results
    assert received_subagents[0] is not received_subagents[1]


def test_advanced_registry_does_not_add_per_lesson_practice_quota_rule() -> None:
    advanced_specs = [get_material_spec(kind, course_level="advanced") for kind in ("practice", "specification_qa")]
    joined = "\n".join(
        "\n".join((spec.prompt_addendum, spec.validation_policy_addendum, spec.controller_policy_addendum))
        for spec in advanced_specs
    )

    assert "≥7 задач" not in joined
    assert "7 задач" not in joined
    assert "7 tasks" not in joined.lower()


def test_tracker_converter_detects_course_level_from_workbook_name() -> None:
    assert detect_course_level(Path("Трекер_Python_DOP_Продвинутый.xlsx"), None) == "advanced"
    assert detect_course_level(Path("Трекер_Python_DOP_Базовый.xlsx"), None) == "basic"


def test_tracker_converter_uses_fixed_references_dir(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts_skills"
    advanced_prompts_dir = tmp_path / "prompts_skills_advanced"
    references_dir = tmp_path / "референсы"
    prompts_dir.mkdir()
    advanced_prompts_dir.mkdir()
    references_dir.mkdir()
    (advanced_prompts_dir / "01_prompt.md").write_text("prompt", encoding="utf-8")
    (references_dir / "source.md").write_text("reference", encoding="utf-8")

    assert find_references_dir(tmp_path) == references_dir.resolve()


def test_tracker_converter_parses_tasks_only_from_line_starts() -> None:
    tasks = parse_tasks(
        """
1. Read a name and print a greeting.

2. Read two numbers and print sum.

3. Read a number and classify it.

4. Read n and print all values from 1 to n divisible by 3. Use a for loop.

5. Read a string and print uppercase.

[Extra] Ignore this block.
"""
    )

    assert [task["number"] for task in tasks] == [1, 2, 3, 4, 5]
    assert tasks[3]["text"] == "Read n and print all values from 1 to n divisible by 3. Use a for loop."


def _canonical_style_from_prompt_text(prompt: str) -> str:
    canonical_start = prompt.find("КАНОНИЧЕСКИЙ БЛОК")
    assert canonical_start >= 0
    content_start = prompt.find("\n", canonical_start)
    assert content_start >= 0
    match = re.search(r"\\<style\\>.*?\\</style\\>", prompt[content_start:], flags=re.DOTALL)
    assert match is not None
    return re.sub(r"\\([<>#*])", r"\1", match.group(0)).strip()


def test_saved_cc_lesson_template_matches_format_prompt_style() -> None:
    prompt_path = IsmartGenerationConfig().prompts_dir / FORMAT_PROMPT
    prompt_style = _canonical_style_from_prompt_text(prompt_path.read_text(encoding="utf-8"))
    template = load_html_format_template()

    assert HTML_TEMPLATE_PATH.exists()
    assert template.style_block == prompt_style
    assert "{{ body_html }}" in template.template_html
    assert template.template_html.startswith("<style>")
    assert template.template_html.endswith("</div>")


class FakeChatModel:
    def with_structured_output(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("subagents must use create_agent(response_format=...), not direct with_structured_output")


class FakeGraph:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append({**dict(state), "_config": config})
        if not self.responses:
            raise AssertionError("FakeGraph has no more responses")
        return {"result": self.responses.pop(0)}


class MissingStructuredResponseGraph:
    def invoke(self, _state: dict[str, Any], _config: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="plain raw response without structured output")],
            "structured_response": None,
        }


class FakeCreatedAgent:
    def __init__(self, schema: type[Any]) -> None:
        self.schema = schema
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append({**dict(state), "_config": config})
        if self.schema is GeneratedMaterial:
            structured = GeneratedMaterial(content=VALID_HTML, agent_notes=["generated"])
        elif self.schema is MaterialValidationDecision:
            structured = MaterialValidationDecision(approved=True, passed_blocks=[])
        elif self.schema is ValidationControllerDecision:
            structured = ValidationControllerDecision(approved=True, quality_score=4)
        else:
            structured = self.schema()
        return {
            **state,
            "structured_response": structured,
            "messages": [AIMessage(content="raw model response")],
        }


def test_subagent_registry_builds_explicit_compiled_agents(monkeypatch: Any) -> None:
    create_agent_calls: list[dict[str, Any]] = []

    def fake_create_agent(**kwargs: Any) -> FakeCreatedAgent:
        create_agent_calls.append(kwargs)
        return FakeCreatedAgent(kwargs["response_format"])

    monkeypatch.setattr(subagents_module, "create_agent", fake_create_agent)
    model = FakeChatModel()

    registry = build_subagent_registry(model)  # type: ignore[arg-type]

    assert set(registry) == set(ALL_SUBAGENT_TYPES)
    for agent_type in ALL_SUBAGENT_TYPES:
        assert hasattr(registry[agent_type], "invoke")
    result = registry["TheoryMaterialAgent"].invoke({"system_prompt": "system", "prompt": "prompt"})
    assert isinstance(result["structured_response"], GeneratedMaterial)
    schemas = [call["response_format"] for call in create_agent_calls]
    assert GeneratedMaterial in schemas
    assert PracticeTaskTemplateSet in schemas
    assert PracticeTaskInstanceSet in schemas
    assert SelfWorkAutocheckSet in schemas
    assert CurrentControlAutocheckSet in schemas
    assert IntermediateAssessmentArtifact in schemas
    assert MaterialValidationDecision in schemas
    assert ValidationControllerDecision in schemas
    assert all(call["tools"] is None for call in create_agent_calls)
    assert all(call["state_schema"] is subagents_module.StructuredSubagentState for call in create_agent_calls)


def test_structured_subagent_invoker_logs_raw_response_when_structured_response_is_missing() -> None:
    stream = StringIO()
    invoker = StructuredSubagentInvoker(
        {"SpecificationQAAgent": MissingStructuredResponseGraph()},
        trace=TraceLogger(enabled=True, stream=stream),
    )

    try:
        invoker.invoke(
            "SpecificationQAAgent",
            system="system",
            prompt="prompt",
            schema=GeneratedMaterial,
        )
    except StructuredSubagentError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected StructuredSubagentError")

    trace_output = stream.getvalue()
    assert "plain raw response without structured output" in message
    assert "subagent.structured_output.missing" in trace_output
    assert "plain raw response without structured output" in trace_output


def test_structured_subagent_invoker_invokes_graph_with_isolated_thread_id() -> None:
    graph = FakeGraph([GeneratedMaterial(content=VALID_HTML, agent_notes=[])])
    invoker = StructuredSubagentInvoker(
        {"TheoryMaterialAgent": graph},
        langchain_config={
            "run_name": "ismart_generator_basic_8",
            "tags": ["ismart_generator", "profile:basic", "class:8"],
            "metadata": {"course_level": "basic", "course_class": "8"},
        },
    )

    result = invoker.invoke(
        "TheoryMaterialAgent",
        system="system",
        prompt="prompt",
        schema=GeneratedMaterial,
    )

    assert result.content == VALID_HTML
    assert len(graph.calls) == 1
    call = graph.calls[0]
    assert call["system_prompt"] == "system"
    assert call["prompt"] == "prompt"
    assert len(call["messages"]) == 1
    assert isinstance(call["messages"][0], HumanMessage)
    thread_id = call["_config"]["configurable"]["thread_id"]
    assert thread_id.startswith("ismart-TheoryMaterialAgent-GeneratedMaterial-")
    assert call["_config"]["run_name"] == "ismart_generator_basic_8.TheoryMaterialAgent"
    assert "subagent:TheoryMaterialAgent" in call["_config"]["tags"]
    assert call["_config"]["metadata"]["course_class"] == "8"
    assert call["_config"]["metadata"]["subagent_type"] == "TheoryMaterialAgent"


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


def test_self_work_structured_validation_does_not_receive_rendered_source_paths(tmp_path: Path) -> None:
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
    assert "docs/ismart" in result.content
    assert ".md" in result.content
    assert "docs/ismart" not in validator.calls[0]["prompt"]
    assert "VALIDATION TARGET MODE:\nstructured_artifacts" in validator.calls[0]["prompt"]
    assert "RENDERED HTML IS NOT INCLUDED" in validator.calls[0]["prompt"]


def test_self_work_structured_validation_does_not_receive_rendered_answer_keys(tmp_path: Path) -> None:
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
    assert "Ключ для автопроверки" in result.content
    assert "{%answer%}" in result.content
    assert "pair:a // b=3" in result.content
    assert "visible self-check key blocks were removed before validation" not in result.agent_notes
    assert "Ключ для автопроверки" not in validator.calls[0]["prompt"]
    assert "{{pair:a // b=3}}" not in validator.calls[0]["prompt"]
    assert "VALIDATION TARGET MODE:\nstructured_artifacts" in validator.calls[0]["prompt"]
    assert "RENDERED HTML IS NOT INCLUDED" in validator.calls[0]["prompt"]


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
    assert "visually spans the next line" in policy


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
    assert "do not validate learner-facing faulty_code" in prompt
    assert "These fields are intentionally faulty learning inputs" in prompt
    assert "Their absence from student practice is correct" in prompt


def test_prompt_skills_do_not_recommend_faulty_code_markers() -> None:
    prompts_dir = Path("docs/ismart/Материалы для ИИ-агентов/рабочая область агента/prompts_skills")
    practice_prompt = (prompts_dir / "03_Практика_prompt_skill.md").read_text(encoding="utf-8")
    formatting_prompt = (prompts_dir / "08_Форматирование_заданий_курса_prompt.md").read_text(encoding="utf-8")
    combined = f"{practice_prompt}\n{formatting_prompt}"

    assert "нейтральный маркер/комментарий" not in combined
    assert "фрагмент намеренно обрывается" not in combined
    assert "Валидатор не должен требовать синтаксической корректности" in combined


def test_practice_validation_prompt_includes_full_structured_artifacts() -> None:
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

    assert "STRUCTURED GENERATION ARTIFACTS" in prompt
    assert "SECRET_FIXED_CODE" in prompt
    assert "SECRET_TEACHER_NOTES" in prompt
    assert "runtime_tests" in prompt
    assert "manual_checks" in prompt
    assert "primary validation target for structured material kinds" in prompt
    assert "forbidden internal content appears in structured student-facing fields" in prompt


def test_practice_controller_prompt_includes_full_structured_artifacts() -> None:
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

    assert "STRUCTURED GENERATION ARTIFACTS" in prompt
    assert "SECRET_FIXED_CODE" in prompt
    assert "SECRET_TEACHER_NOTES" in prompt
    assert "overrule validator objections that treat teacher-only fields inside internal generation artifacts" in prompt
    assert "overrule validator objections that treat intentionally faulty starter" in prompt
    assert "not syntactically correct" in prompt
    assert "Do not uphold a validator claim of learner-facing key leakage merely because an internal artifact" in prompt


def test_generation_artifacts_for_validation_preserves_full_artifacts_for_all_channels() -> None:
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

    practice_view = generation_artifacts_for_validation(practice_spec, artifacts)
    qa_view = generation_artifacts_for_validation(qa_spec, artifacts)

    assert practice_view["practice_instances"]["tasks"][0]["hidden_solution"] == "SECRET_FIXED_CODE"
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
    assert contract["authoritative_task_ids"] == ["P1"]
    assert contract["required_task_count"] == 1
    assert "authoritative task pattern" in rules
    assert "lesson.difficulty.*.count is planning context only" in rules
    assert "Create a new concrete variant of the same pattern" in rules
    assert "Do not choose manual_only merely because the starting code is intentionally faulty" in rules
    assert "quote the expected diagnostic message" in rules
    assert "Do not show source_text, source task, source contract" in rules
    assert "level, source task, condition" not in rules
    assert "PracticeTaskTemplateAgent" in " ".join(contract["pipeline"])
    assert "PracticeTaskVariantAgent" in " ".join(contract["pipeline"])


def test_practice_variant_prompt_requires_single_file_for_deterministic_fix_tasks() -> None:
    spec = get_material_spec("practice")
    task = {
        "lesson": {
            "practice_tasks": {
                "l2": [{"number": 3, "text": "Дан код: print(\"Привет). Найти ошибку и исправить."}],
            },
        }
    }

    prompt = build_practice_variant_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        templates={"tasks": [{"id": "P3", "template_id": "P3"}]},
        previous_artifacts={},
        previous_issues=[],
    )

    assert "Classify each fix/debug task before choosing run_mode" in prompt
    assert "correction task with deterministic corrected behavior" in prompt
    assert "Use run_mode=single_file" in prompt
    assert "Do not downgrade this to manual_only just because the initial code is faulty" in prompt
    assert "For no-stdin corrected-output fix/debug tasks, create 3 runtime_tests/tests" in prompt


class NamedFakeGraph(FakeGraph):
    def __init__(self, name: str, responses: list[Any], call_order: list[str]) -> None:
        super().__init__(responses)
        self.name = name
        self.call_order = call_order

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.call_order.append(self.name)
        return super().invoke(state, config)


class RaisingThenFakeGraph(FakeGraph):
    def __init__(self, error: Exception, responses: list[Any]) -> None:
        super().__init__(responses)
        self.error = error
        self.raised = False

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append({**dict(state), "_config": config})
        if not self.raised:
            self.raised = True
            raise self.error
        if not self.responses:
            raise AssertionError("RaisingThenFakeGraph has no more responses")
        return {"result": self.responses.pop(0)}


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
        lesson_goal="Научиться создавать переменную и выводить её значение.",
        lesson_objectives=["создаёт переменную", "выводит значение переменной", "проверяет результат запуска"],
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
                tests=[
                    {"input": "", "expected_output": "Тула"},
                    {"input": "", "expected_output": "Тула"},
                    {"input": "", "expected_output": "Тула"},
                ],
                hidden_solution="city = 'Тула'\nprint(city)",
                teacher_explanation="Проверить присваивание строки переменной city и вывод значения.",
                uniqueness_notes=["Не использует пример из теории про имя."],
            )
        ],
        agent_notes=["instances ok"],
    )


def _practice_instance_set_repaired() -> PracticeTaskInstanceSet:
    repaired = _practice_instance_set().model_copy(deep=True)
    repaired.lesson_goal = "Закрепить вывод значения переменной в редакторе Python."
    repaired.tasks[0].student_condition = "Создайте переменную city со значением 'Казань' и выведите её."
    repaired.tasks[0].output_requirements = "Программа выводит Казань."
    repaired.tasks[0].tests = [
        {"input": "", "expected_output": "Казань"},
        {"input": "", "expected_output": "Казань"},
        {"input": "", "expected_output": "Казань"},
    ]
    repaired.tasks[0].runtime_tests = [
        {"input": "", "expected_output": "Казань"},
        {"input": "", "expected_output": "Казань"},
        {"input": "", "expected_output": "Казань"},
    ]
    repaired.tasks[0].hidden_solution = "city = 'Казань'\nprint(city)"
    return repaired


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


def _current_control_worker_spec() -> MaterialSpec:
    return MaterialSpec(
        kind="current_control",
        material_type="Current Control",
        agent_type="CurrentControlAgent",
        prompt_files=(),
        validator_kind="current_control",
        reference_fields=("template_descriptions", "requirements", "reference_examples"),
        json_field_labels=("lesson.content", "lesson.difficulty", "lesson.hours.raw"),
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


def _current_control_autocheck_set() -> CurrentControlAutocheckSet:
    return CurrentControlAutocheckSet(
        questions=[
            CurrentControlAutocheckQuestion(
                id="CC1",
                template_code="question",
                question_type="single choice",
                skill_target="print syntax",
                student_prompt="Choose the correct print call.",
                options=["print('Hi')", "prnt('Hi')"],
                expected_answer_format="one option",
                correct_answers=["print('Hi')"],
                autocheck_config={"correct_option": 0},
                internal_explanation="The built-in function is print.",
            ),
            CurrentControlAutocheckQuestion(
                id="CC2",
                template_code="6A",
                question_type="ordering",
                skill_target="program execution order",
                student_prompt="Put the actions in execution order.",
                options=["assign value", "call print"],
                expected_answer_format="ordered options",
                correct_answers=["assign value", "call print"],
                autocheck_config={"order": [0, 1]},
                internal_explanation="Assignment happens before print.",
            ),
            CurrentControlAutocheckQuestion(
                id="CC3",
                template_code="3H",
                question_type="open text",
                skill_target="function name recognition",
                student_prompt="Write the exact Python function name used to output text, without parentheses.",
                options=[],
                expected_answer_format="function name only, without parentheses",
                correct_answers=["print"],
                autocheck_config={"normalize": "strip/lower", "accepted": ["print"]},
                internal_explanation="The expected function name is print.",
            ),
        ],
        agent_notes=["current control autocheck ok"],
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
                test_questions=[
                    IntermediateTestQuestion(
                        id=f"{variant_id}-T{question_index:02d}",
                        template_code=template_codes[(question_index - 1) % len(template_codes)],
                        skill_target="module concept",
                        student_prompt=f"Test question {variant_index}.{question_index}",
                        options=["A", "B", "C", "D"],
                        correct_answers=[f"Internal test key {variant_index}.{question_index}"],
                        autocheck_config={"correct": f"Internal test key {variant_index}.{question_index}"},
                        internal_explanation="Teacher-only test explanation.",
                    )
                    for question_index in range(1, 6)
                ],
                open_code_questions=[
                    IntermediateOpenCodeQuestion(
                        id=f"{variant_id}-OC{question_index:02d}",
                        skill_target="write Python expression",
                        student_prompt=f"Напишите код Python для открытого задания {variant_index}.{question_index}.",
                        input_requirements="stdin contains one number",
                        output_requirements="stdout contains processed number",
                        runtime_tests=[
                            {"input": "1\n", "expected_output": "2\n"},
                            {"input": "2\n", "expected_output": "4\n"},
                            {"input": "3\n", "expected_output": "6\n"},
                        ],
                        manual_check_rules=[],
                        hidden_solution="value = int(input())\nprint(value * 2)",
                        rubric=["1 point for concept", "1 point for example"],
                        internal_explanation="Teacher-only open explanation.",
                    )
                    for question_index in range(1, 6)
                ],
                code_tasks=[
                    IntermediateCodeTask(
                        id=f"{variant_id}-P{task_index:02d}",
                        skill_target="write Python code",
                        student_condition=f"Code task {variant_index}.{task_index}",
                        input_requirements="stdin contains one number",
                        output_requirements="stdout contains processed number",
                        runtime_tests=[
                            {"input": "1\n", "expected_output": "2\n"},
                            {"input": "2\n", "expected_output": "4\n"},
                            {"input": "3\n", "expected_output": "6\n"},
                        ],
                        manual_check_rules=[],
                        hidden_solution="value = int(input())\nprint(value * 2)",
                        teacher_explanation="Teacher-only code explanation.",
                    )
                    for task_index in range(1, 6)
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


def test_self_work_worker_freezes_valid_autocheck_on_retry(tmp_path: Path) -> None:
    autocheck_agent = FakeGraph([_self_work_autocheck_set()])
    self_work_agent = FakeGraph(
        [
            GeneratedMaterial(content=VALID_HTML, agent_notes=["attempt 1"]),
            GeneratedMaterial(content=VALID_HTML, agent_notes=["attempt 2"]),
        ]
    )
    validator = FakeGraph(
        [
            MaterialValidationDecision(approved=False, issues=["repair HTML"], fix_instructions=["repair HTML"]),
            MaterialValidationDecision(approved=True, passed_blocks=[]),
        ]
    )
    worker = MaterialWorker(
        subagents={
            "SelfWorkAutocheckAgent": autocheck_agent,
            "SelfStudyAgent": self_work_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=2),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"self_work": True}, "hours": {"self_study": 2}}},
        spec=_self_work_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert len(autocheck_agent.calls) == 1
    assert len(self_work_agent.calls) == 2
    assert "self_work_autocheck" in self_work_agent.calls[1]["prompt"]


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

    view = generation_artifacts_for_validation(spec, artifacts)

    assert view["self_work_autocheck"]["selfcheck_questions"][0]["correct_answers"] == ["Internal self-check key 1"]
    assert set(view) == {"self_work_autocheck"}


def test_self_work_source_contract_defines_internal_autocheck_layer() -> None:
    spec = _self_work_worker_spec()

    contract = source_contract_for_spec({"lesson": {"hours": {"self_study": 2}}}, spec)

    assert contract["contract_type"] == "self_work_autocheck_contract"
    assert contract["required_independent_task_count"] == 8
    assert contract["required_selfcheck_question_count"] == 10
    assert "generation_artifacts.self_work_autocheck" in " ".join(contract["generation_rules"])
    assert "Do not require visible keys" in " ".join(contract["validation_rules"])


def test_current_control_worker_uses_autocheck_pipeline_and_writes_artifacts(tmp_path: Path) -> None:
    call_order: list[str] = []
    autocheck_agent = NamedFakeGraph(
        "CurrentControlAutocheckAgent",
        [_current_control_autocheck_set()],
        call_order,
    )
    validator = NamedFakeGraph(
        "MaterialValidatorAgent",
        [MaterialValidationDecision(approved=True, passed_blocks=[])],
        call_order,
    )
    worker = MaterialWorker(
        subagents={
            "CurrentControlAutocheckAgent": autocheck_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"current_control": True}, "hours": {"raw": "1"}}},
        spec=_current_control_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert call_order == ["CurrentControlAutocheckAgent", "MaterialValidatorAgent"]
    assert result.generation_artifacts["current_control_autocheck"]["questions"][0]["correct_answers"]
    assert "print('Hi')" in validator.calls[0]["prompt"]
    assert "print(&#x27;Hi&#x27;)" in result.content
    assert "The built-in function is print." not in result.content
    assert "autocheck_config" not in result.content
    assert "Current-control HTML rendered deterministically from current_control_autocheck." in result.agent_notes
    assert list((tmp_path / "tmp" / "current-control").glob("*.current_control_autocheck.json"))
    assert list((tmp_path / "tmp" / "current-control").glob("*.current_control_autocheck_check.json"))


def test_current_control_worker_freezes_valid_autocheck_on_retry(tmp_path: Path) -> None:
    autocheck_agent = FakeGraph([_current_control_autocheck_set()])
    validator = FakeGraph(
        [
            MaterialValidationDecision(approved=False, issues=["repair HTML"], fix_instructions=["repair HTML"]),
            MaterialValidationDecision(approved=True, passed_blocks=[]),
        ]
    )
    worker = MaterialWorker(
        subagents={
            "CurrentControlAutocheckAgent": autocheck_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=2),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"current_control": True}, "hours": {"raw": "1"}}},
        spec=_current_control_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert len(autocheck_agent.calls) == 1
    assert len(validator.calls) == 2


def test_current_control_worker_blocks_missing_internal_autocheck_keys_before_html(tmp_path: Path) -> None:
    bad = _current_control_autocheck_set()
    bad.questions[0].correct_answers = []
    autocheck_agent = FakeGraph([bad])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "CurrentControlAutocheckAgent": autocheck_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=1),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"current_control": True}, "hours": {"raw": "1"}}},
        spec=_current_control_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert "needs at least one correct answer" in " ".join(result.validation_issues)
    assert validator.calls == []


def test_current_control_worker_retries_structured_output_parser_error(tmp_path: Path) -> None:
    autocheck_agent = RaisingThenFakeGraph(
        RuntimeError("5 validation errors for CurrentControlAutocheckSet: questions.3 agent_notes=["),
        [_current_control_autocheck_set()],
    )
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "CurrentControlAutocheckAgent": autocheck_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=2),
    )

    result = worker.run(
        task={"course": {}, "module": {}, "lesson": {"content_flags": {"current_control": True}, "hours": {"raw": "1"}}},
        spec=_current_control_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert len(autocheck_agent.calls) == 2
    assert len(validator.calls) == 1
    checks = list((tmp_path / "tmp" / "current-control").glob("*.current_control_autocheck_check.json"))
    assert len(checks) == 2


def test_current_control_validation_view_keeps_internal_autocheck_answers() -> None:
    spec = _current_control_worker_spec()
    artifacts = {"current_control_autocheck": _current_control_autocheck_set().model_dump(mode="json")}

    view = generation_artifacts_for_validation(spec, artifacts)

    assert view["current_control_autocheck"]["questions"][0]["correct_answers"] == ["print('Hi')"]
    assert set(view) == {"current_control_autocheck"}


def test_current_control_renderer_does_not_render_matching_answer_pairs() -> None:
    content = render_current_control_material_html(
        {"lesson": {"lesson_number": 2, "title": "Облачная IDE"}},
        {
            "questions": [
                {
                    "id": "CC2",
                    "template_code": "8D",
                    "question_type": "matching",
                    "student_prompt": "Соедините сообщение с причиной.",
                    "options": [],
                    "expected_answer_format": "4 пары",
                    "correct_answers": ["A -> 1"],
                    "autocheck_config": {
                        "left_items": ["A", "B"],
                        "right_items": ["1", "2"],
                        "correct_pairs": [[0, 0], [1, 1]],
                    },
                    "internal_explanation": "A matches 1.",
                }
            ]
        },
        html_template=load_html_format_template(),
    )

    assert "Список A" in content
    assert "Список B" in content
    assert "<li>A</li>" in content
    assert "<li>1</li>" in content
    assert "<td>A</td>" not in content
    assert "A -&gt; 1" not in content
    assert "correct_pairs" not in content
    assert "A matches 1." not in content


def test_current_control_renderer_hides_expected_format_when_it_contains_answer() -> None:
    content = render_current_control_material_html(
        {"lesson": {"lesson_number": 2, "title": "Облачная IDE"}},
        {
            "questions": [
                {
                    "id": "CC3",
                    "template_code": "3H",
                    "question_type": "open_answer",
                    "student_prompt": "Напишите правильную строку кода.",
                    "options": [],
                    "expected_answer_format": "Одна строка Python-кода: print(\"Привет\")",
                    "correct_answers": ["print(\"Привет\")"],
                    "autocheck_config": {"type": "open_answer"},
                    "internal_explanation": "Correct answer is print.",
                }
            ]
        },
        html_template=load_html_format_template(),
    )

    assert "print(&quot;Привет&quot;)" not in content
    assert "Одна строка Python-кода" not in content
    assert "введите ответ в поле платформы" in content
    assert "Correct answer is print." not in content


def test_current_control_source_contract_defines_internal_autocheck_layer() -> None:
    spec = _current_control_worker_spec()

    contract = source_contract_for_spec({"lesson": {"hours": {"raw": "1"}}}, spec)

    assert contract["contract_type"] == "current_control_autocheck_contract"
    assert contract["required_question_count"] == 3
    assert "generation_artifacts.current_control_autocheck" in " ".join(contract["generation_rules"])
    assert "Do not require visible keys" in " ".join(contract["validation_rules"])


def test_intermediate_artifact_prompt_requires_per_variant_coded_template_diversity() -> None:
    spec = _intermediate_worker_spec()
    prompt = build_intermediate_assessment_artifact_prompt(
        task={"course": {}, "module": {"lessons": []}, "lesson": {"content_flags": {"attestation": True}}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        previous_artifacts={},
        previous_issues=[
            "intermediate_assessment.V1 must include at least 3 coded template types from 6A/6D/6G/8D/10D"
        ],
    )
    contract = source_contract_for_spec({"course": {}, "module": {"lessons": []}, "lesson": {}}, spec)

    assert "For each variant" in prompt
    assert "at least 3 distinct coded template_code values" in prompt
    assert "per-variant requirement" in prompt
    assert "repair only test_questions in the named variant(s)" in prompt
    assert "Do not satisfy this only across the whole artifact" in " ".join(contract["generation_rules"])
    assert "fewer than 3 distinct coded template_code values" in " ".join(contract["validation_rules"])
    assert "Runtime tests must describe the corrected/reference behavior" in prompt
    assert "85.00 rather than 85.0" in prompt
    assert "real left_items/right_items" in prompt
    assert "right_items in deranged display order" in prompt
    assert "right_items[i] must not be the correct pair for left_items[i]" in prompt
    assert "repair only matching/pairing questions by reordering right_items" in prompt
    assert "generic placeholders" in " ".join(contract["validation_rules"])
    assert "even one same-position correct pair is answer-key leakage" in " ".join(contract["validation_rules"])


def test_intermediate_generation_prompt_forbids_matching_placeholders() -> None:
    spec = get_material_spec("intermediate")

    prompt = build_generation_prompt(
        task={"course": {}, "module": {"lessons": []}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        previous_content="",
        previous_issues=[],
        generation_artifacts={"intermediate_assessment": _intermediate_assessment_artifact().model_dump(mode="json")},
    )

    assert "render the actual visible left_items and right_items" in prompt
    assert "Never sort right_items back into the correct-pair order" in prompt
    assert "never leave any list B item on the same row as its correct list A pair" in prompt
    assert "Never replace them with generic placeholders" in prompt
    assert "Action 1" in prompt
    assert "Variant A" in prompt


def test_intermediate_test_question_schema_mentions_coded_template_diversity() -> None:
    schema = IntermediateTestQuestion.model_json_schema()
    description = schema["properties"]["template_code"]["description"]
    autocheck_description = schema["properties"]["autocheck_config"]["description"]

    assert "at least 3 distinct" in description
    assert "6A, 6D, 6G, 8D, 10D" in description
    assert "right_items must be a derangement" in autocheck_description
    assert "no right_items[i] may be the correct pair for left_items[i]" in autocheck_description


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
    assert result.generation_artifacts["intermediate_assessment"]["variants"][0]["test_questions"][0]["correct_answers"]
    assert "GENERATION ARTIFACTS FOR THIS MATERIAL" in intermediate_agent.calls[0]["prompt"]
    assert "intermediate_assessment" in intermediate_agent.calls[0]["prompt"]
    assert "Internal test key 1.1" in validator.calls[0]["prompt"]
    assert "Internal test key" not in result.content
    assert list((tmp_path / "tmp" / "intermediate").glob("*.intermediate_assessment.json"))
    assert list((tmp_path / "tmp" / "intermediate").glob("*.intermediate_assessment_check.json"))


def test_intermediate_worker_freezes_valid_assessment_artifact_on_retry(tmp_path: Path) -> None:
    artifact_agent = FakeGraph([_intermediate_assessment_artifact()])
    intermediate_agent = FakeGraph(
        [
            GeneratedMaterial(content=VALID_HTML, agent_notes=["attempt 1"]),
            GeneratedMaterial(content=VALID_HTML, agent_notes=["attempt 2"]),
        ]
    )
    validator = FakeGraph(
        [
            MaterialValidationDecision(approved=False, issues=["repair HTML"], fix_instructions=["repair HTML"]),
            MaterialValidationDecision(approved=True, passed_blocks=[]),
        ]
    )
    worker = MaterialWorker(
        subagents={
            "IntermediateAssessmentArtifactAgent": artifact_agent,
            "IntermediateAssessmentAgent": intermediate_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=2),
    )

    result = worker.run(
        task={"course": {}, "module": {"lessons": []}, "lesson": {"content_flags": {"attestation": True}}},
        spec=_intermediate_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert len(artifact_agent.calls) == 1
    assert len(intermediate_agent.calls) == 2
    assert "intermediate_assessment" in intermediate_agent.calls[1]["prompt"]


def test_intermediate_worker_does_not_rule_block_visible_matching_answer_pairs(tmp_path: Path) -> None:
    artifact = _intermediate_assessment_artifact()
    question = artifact.variants[0].test_questions[0]
    question.id = "V1-T01"
    question.template_code = "8D"
    question.student_prompt = "Match each error type with its usual cause."
    question.options = ["SyntaxError", "NameError", "Missing quote", "Misspelled name"]
    question.correct_answers = ["SyntaxError->Missing quote", "NameError->Misspelled name"]
    question.autocheck_config = {
        "type": "8D",
        "left": [{"id": "l1", "text": "SyntaxError"}, {"id": "l2", "text": "NameError"}],
        "right": [{"id": "r1", "text": "Missing quote"}, {"id": "r2", "text": "Misspelled name"}],
        "correct_pairs": [["l1", "r1"], ["l2", "r2"]],
    }
    html = (
        '<style>.x{}</style><div class="cc-lesson"><h2 id="variant-1">Variant 1</h2>'
        "<p>Match each error type with its usual cause.</p>"
        "<ul><li>SyntaxError — Missing quote</li><li>NameError — Misspelled name</li></ul>"
        "</div>"
    )
    artifact_agent = FakeGraph([artifact])
    intermediate_agent = FakeGraph([GeneratedMaterial(content=html)])
    worker = MaterialWorker(
        subagents={
            "IntermediateAssessmentArtifactAgent": artifact_agent,
            "IntermediateAssessmentAgent": intermediate_agent,
        },
        config=IsmartGenerationConfig(
            prompts_dir=tmp_path,
            output_root=tmp_path,
            max_generation_iterations=1,
            use_llm_validator=False,
        ),
    )

    result = worker.run(
        task={"course": {}, "module": {"lessons": []}, "lesson": {"content_flags": {"attestation": True}}},
        spec=_intermediate_worker_spec(),
        references={},
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert "leaks matching answer pair" not in " ".join(result.validation_issues)


def test_intermediate_validation_prompt_uses_structured_artifact_not_html() -> None:
    artifact = _intermediate_assessment_artifact().model_dump(mode="json")
    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=_intermediate_worker_spec(),
        prompt_contents={},
        references={},
        dependencies=[],
        content='<style>.x{}</style><div class="cc-lesson"><p>VISIBLE_HTML_SHOULD_NOT_BE_SENT</p></div>',
        rule_result=ValidationResult.ok(),
        generation_artifacts={"intermediate_assessment": artifact},
    )

    assert "VALIDATION TARGET MODE:\nstructured_artifacts" in prompt
    assert "STRUCTURED GENERATION ARTIFACTS" in prompt
    assert "intermediate_assessment" in prompt
    assert "VISIBLE_HTML_SHOULD_NOT_BE_SENT" not in prompt
    assert "RENDERED HTML IS NOT INCLUDED" in prompt


def test_intermediate_artifact_validation_does_not_semantically_reject_decimal_runtime_tests(tmp_path: Path) -> None:
    artifact = _intermediate_assessment_artifact()
    task = artifact.variants[0].code_tasks[0]
    task.skill_target = "rounding to two decimal places"
    task.student_condition = "Read price and print discounted price with exactly two decimal places."
    task.output_requirements = "One number with exactly two decimal places."
    task.runtime_tests = [
        {"stdin": "100\n", "stdout": "85.0\n"},
        {"stdin": "0\n", "stdout": "0.0\n"},
        {"stdin": "199.99\n", "stdout": "169.99\n"},
    ]
    task.hidden_solution = "price = float(input())\nprint(round(price * 0.85, 2))"
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))

    result = worker._validate_intermediate_assessment_artifact(artifact.model_dump(mode="json"))

    assert result.approved
    assert not result.issues


def test_intermediate_worker_blocks_incomplete_assessment_artifact_before_html(tmp_path: Path) -> None:
    bad = _intermediate_assessment_artifact()
    bad.variants[0].test_questions[0].correct_answers = []
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
    question = bad.variants[2].test_questions[3]
    question.id = "V3-T04"
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

    view = generation_artifacts_for_validation(spec, artifacts)

    assert view["intermediate_assessment"]["variants"][0]["test_questions"][0]["correct_answers"] == [
        "Internal test key 1.1"
    ]
    assert set(view) == {"intermediate_assessment"}


def test_intermediate_source_contract_defines_internal_assessment_layer() -> None:
    spec = _intermediate_worker_spec()

    contract = source_contract_for_spec({"module": {"lessons": [{"lesson_number": 1}]}, "lesson": {}}, spec)

    assert contract["contract_type"] == "intermediate_assessment_contract"
    assert contract["required_variant_count"] == 4
    assert contract["required_test_questions_per_variant"] == 5
    assert contract["required_open_code_questions_per_variant"] == 5
    assert contract["required_code_tasks_per_variant"] == 5
    assert contract["minimum_code_item_ratio"] == "10/15"
    assert "generation_artifacts.intermediate_assessment" in " ".join(contract["generation_rules"])
    assert "Do not require visible keys" in " ".join(contract["validation_rules"])


def test_intermediate_artifact_rejects_legacy_closed_open_shape(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    legacy_variant = {
        "id": "V1",
        "title": "Legacy",
        "closed_questions": [{"id": "V1-C01", "template_code": "6A", "skill_target": "x", "student_prompt": "x"}],
        "open_questions": [{"id": "V1-O01", "skill_target": "x", "student_prompt": "x", "reference_answer": "x"}],
        "code_tasks": [
            {
                "id": "V1-P01",
                "skill_target": "x",
                "student_condition": "Напишите код.",
                "hidden_solution": "print(1)",
                "runtime_tests": [
                    {"input": "", "expected_output": "1\n"},
                    {"input": "", "expected_output": "1\n"},
                    {"input": "", "expected_output": "1\n"},
                ],
                "manual_check_rules": [],
            }
        ],
    }

    result = worker._validate_intermediate_assessment_artifact({"variants": [legacy_variant] * 4})

    assert result.approved is False
    assert "test_questions must be a list" in " ".join(result.issues)
    assert "open_code_questions must be a list" in " ".join(result.issues)


def test_intermediate_validation_policy_allows_options_and_publishable_html() -> None:
    policy = validation_policy_for_spec(_intermediate_worker_spec())

    assert "Candidate answer options in closed questions are allowed" in policy
    assert "not answer-key leakage unless the correct option is explicitly marked" in policy
    assert "Matching/classification questions may show the candidate left-side and right-side items" in policy
    assert "Do not require platform-import markup" in policy
    assert "Find/fix/explain the error" in policy
    assert "Критерии оценивания" in policy
    assert "manual_check_rules" in policy


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
    assert "keep failed when a matching/classification question displays solved left-right pairs" in prompt
    assert "even one same-position correct pair" in prompt
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


def test_intermediate_appellate_policy_overrules_html_visible_key_claim_when_artifact_is_valid(tmp_path: Path) -> None:
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
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail(decision["blocking_issues"]),
        generation_artifacts=artifacts,
        decision=decision,
    )

    assert adjusted["approved"] is True
    assert adjusted["decision"] == "approve_material"
    assert adjusted["blocking_issues"] == []


def test_practice_worker_uses_template_variant_pipeline_and_writes_artifacts(tmp_path: Path) -> None:
    call_order: list[str] = []
    template_agent = NamedFakeGraph("PracticeTaskTemplateAgent", [_practice_template_set()], call_order)
    variant_agent = NamedFakeGraph("PracticeTaskVariantAgent", [_practice_instance_set()], call_order)
    validator = NamedFakeGraph(
        "MaterialValidatorAgent",
        [MaterialValidationDecision(approved=True, passed_blocks=[])],
        call_order,
    )
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
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
        "MaterialValidatorAgent",
    ]
    assert result.generation_artifacts["practice_instances"]["tasks"][0]["hidden_solution"]
    assert "PracticeTaskInstanceSet" not in result.content
    assert "Создайте переменную city" in result.content
    assert "city = 'Тула'\nprint(city)" not in result.content
    assert list((tmp_path / "tmp" / "practice").glob("*.practice_templates.json"))
    assert list((tmp_path / "tmp" / "practice").glob("*.practice_instances.json"))


def test_practice_worker_ignores_validator_issues_targeting_internal_reference_fields(tmp_path: Path) -> None:
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([_practice_instance_set()])
    validator = FakeGraph(
        [
            MaterialValidationDecision(
                approved=False,
                issues=[
                    "Learner-facing practice contains hidden_solution and teacher_explanation.",
                ],
                fix_instructions=[
                    "Remove hidden_solution and teacher_explanation from practice_instances.",
                ],
                issues_by_block=[
                    {
                        "block_id": "#P1",
                        "block_heading": "P1",
                        "field_path": "practice_instances.tasks[P1].hidden_solution",
                        "severity": "blocking",
                        "issue": "hidden_solution is present.",
                        "fix_instruction": "Remove hidden_solution.",
                    },
                    {
                        "block_id": "#P1",
                        "block_heading": "P1",
                        "field_path": "practice_instances.tasks[P1].teacher_explanation",
                        "severity": "blocking",
                        "issue": "teacher_explanation is present.",
                        "fix_instruction": "Remove teacher_explanation.",
                    },
                ],
            )
        ]
    )
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=2),
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

    assert result.status == "approved"
    assert result.iterations == 1
    assert result.validation_issues == []
    assert result.validation_issues_by_block == []
    assert len(variant_agent.calls) == 1
    assert result.generation_artifacts["practice_instances"]["tasks"][0]["hidden_solution"]
    assert result.generation_artifacts["practice_instances"]["tasks"][0]["teacher_explanation"]
    assert "hidden_solution" not in result.content
    assert "teacher_explanation" not in result.content


def test_practice_worker_regenerates_instances_after_semantic_failure(tmp_path: Path) -> None:
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([_practice_instance_set(), _practice_instance_set_repaired()])
    validator = FakeGraph(
        [
            MaterialValidationDecision(approved=False, issues=["repair HTML"], fix_instructions=["repair HTML"]),
            MaterialValidationDecision(approved=True, passed_blocks=[]),
        ]
    )
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
            "MaterialValidatorAgent": validator,
        },
        config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path, max_generation_iterations=2),
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

    assert result.status == "approved"
    assert len(template_agent.calls) == 1
    assert len(variant_agent.calls) == 2
    assert len(validator.calls) == 2
    assert result.generation_artifacts["practice_instances"]["tasks"][0]["hidden_solution"] == "city = 'Казань'\nprint(city)"
    assert "Создайте переменную city со значением &#x27;Казань&#x27;" in result.content
    assert "Создайте переменную city со значением &#x27;Тула&#x27;" not in result.content
    assert "PREVIOUS FULL VALIDATION RESULT" in variant_agent.calls[1]["prompt"]
    assert "repair HTML" in variant_agent.calls[1]["prompt"]
    assert "city = 'Казань'\nprint(city)" not in result.content


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
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
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
    assert validator.calls == []


def test_practice_worker_delegates_duplicate_semantics_to_validator(tmp_path: Path) -> None:
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

    assert result.status == "approved"
    assert "practice_duplicate_check" not in result.generation_artifacts
    assert copied_text in result.content
    assert validator.calls


def test_practice_worker_delegates_solution_hint_semantics_to_validator(tmp_path: Path) -> None:
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
    hinted_templates = _practice_template_set()
    hinted_templates.tasks[0].task_type = "fix"
    template_agent = FakeGraph([hinted_templates])
    variant_agent = FakeGraph([hinted_instances])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
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

    assert result.status == "approved"
    assert "замените prnt на print" in result.content
    assert validator.calls


def test_practice_instance_normalization_preserves_faulty_code_marker_lines() -> None:
    instances = {
        "tasks": [
            {
                "id": "P1",
                "faulty_code": 'msg = "Paper collection tomorrow\nprint(msg)\n# фрагмент намеренно обрывается здесь\n',
                "faulty_code_display": (
                    'msg = "Paper collection tomorrow\n'
                    "print(msg)\n"
                    "# фрагмент намеренно обрывается здесь\n"
                ),
                "display_note": "Последняя строка - нейтральный маркер, где фрагмент намеренно обрывается.",
                "tests": [],
                "runtime_tests": [],
            }
        ],
        "agent_notes": [],
    }

    normalized = _normalize_practice_instance_tests(instances)
    task = normalized["tasks"][0]

    assert "# фрагмент намеренно обрывается здесь" in task["faulty_code"]
    assert "# фрагмент намеренно обрывается здесь" in task["faulty_code_display"]
    assert "нейтральный маркер" in task["display_note"]
    assert "Removed marker/comment line(s) from practice_instances.P1" not in " ".join(normalized["agent_notes"])


def test_practice_worker_does_not_block_internal_source_marker_in_student_fields(tmp_path: Path) -> None:
    marked_instances = PracticeTaskInstanceSet(
        tasks=[
            PracticeTaskInstance(
                id="P1",
                template_id="P1",
                level="L1",
                task_type="write_code",
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
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
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

    assert result.status == "approved"
    assert "Источник (как в задании урока)" in result.content
    assert "student-facing fields contain internal source/pipeline marker" not in " ".join(result.validation_issues)
    assert validator.calls


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
                tests=[
                    {"input": "", "output": "Готово\n"},
                    {"input": "", "output": "Готово\n"},
                    {"input": "", "output": "Готово\n"},
                ],
                hidden_solution="status = 'Готово'\nprint(status)",
                teacher_explanation="Проверяется присваивание и вывод.",
            )
        ]
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([instances])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
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
    assert "Готово" in result.content


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
                runtime_tests=[
                    {"input": "", "stdout": "Готово\n"},
                    {"input": "", "stdout": "Готово\n"},
                    {"input": "", "stdout": "Готово\n"},
                ],
                manual_checks=["Проверьте читаемое имя переменной."],
                run_mode="separate_snippets",
                hidden_solution="status = 'Готово'\nprint(status)",
                teacher_explanation="Проверяется присваивание и вывод.",
            )
        ]
    )
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([instances])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
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


def test_practice_instance_validation_does_not_require_three_runtime_tests(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    templates = _practice_template_set().model_dump(mode="json")
    instances = _practice_instance_set().model_dump(mode="json")
    instances["tasks"][0]["runtime_tests"] = [{"input": "", "expected_output": "Тула"}]
    instances["tasks"][0]["tests"] = [{"input": "", "expected_output": "Тула"}]

    result = worker._validate_practice_instances(
        task={
            "lesson": {
                "practice_tasks": {"l1": [{"number": 1, "text": "Create a variable with a name and print it"}]},
            }
        },
        spec=_practice_worker_spec(),
        templates=templates,
        instances=instances,
    )

    assert result.approved is True
    assert result.issues == []


def test_practice_instance_validation_does_not_semantically_judge_error_demonstration_tests(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    templates = _practice_template_set().model_dump(mode="json")
    templates["tasks"][0]["source_text"] = "Прочитать сообщение об ошибке SyntaxError при запуске кода"
    templates["tasks"][0]["task_type"] = "debug_error_message"
    templates["tasks"][0]["skill_target"] = "read Python error message"
    templates["tasks"][0]["test_policy"] = "check expected error message"
    instances = _practice_instance_set().model_dump(mode="json")
    instances["tasks"][0]["task_type"] = "debug_error_message"
    instances["tasks"][0]["scenario"] = "IDE показывает ошибку при запуске программы."
    instances["tasks"][0]["student_condition"] = "Запустите код и прочитайте сообщение об ошибке."
    instances["tasks"][0]["faulty_code"] = 'print("Готово)\n'
    instances["tasks"][0]["faulty_code_display"] = 'print("Готово)\n'
    instances["tasks"][0]["output_requirements"] = "Результат проверки - сообщение ошибки SyntaxError."
    stdout_tests = [
        {"input": "", "expected_output": "Готово\n"},
        {"input": "", "expected_output": "Проверка\n"},
        {"input": "", "expected_output": "Старт\n"},
    ]
    instances["tasks"][0]["runtime_tests"] = stdout_tests
    instances["tasks"][0]["tests"] = stdout_tests

    result = worker._validate_practice_instances(
        task={
            "lesson": {
                "practice_tasks": {
                    "l1": [{"number": 1, "text": "Прочитать сообщение об ошибке SyntaxError при запуске кода"}]
                },
            }
        },
        spec=_practice_worker_spec(),
        templates=templates,
        instances=instances,
    )

    assert result.approved is True


def test_practice_instance_validation_allows_expected_error_for_error_demonstration_task(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    templates = _practice_template_set().model_dump(mode="json")
    templates["tasks"][0]["source_text"] = "Прочитать сообщение об ошибке SyntaxError при запуске кода"
    templates["tasks"][0]["task_type"] = "debug_error_message"
    templates["tasks"][0]["skill_target"] = "read Python error message"
    templates["tasks"][0]["test_policy"] = "check expected error message"
    instances = _practice_instance_set().model_dump(mode="json")
    instances["tasks"][0]["task_type"] = "debug_error_message"
    instances["tasks"][0]["scenario"] = "IDE показывает ошибку при запуске программы."
    instances["tasks"][0]["student_condition"] = "Запустите код и прочитайте сообщение об ошибке."
    instances["tasks"][0]["faulty_code"] = 'print("Готово)\n'
    instances["tasks"][0]["faulty_code_display"] = 'print("Готово)\n'
    instances["tasks"][0]["output_requirements"] = "Результат проверки - сообщение ошибки SyntaxError."
    error_tests = [
        {"input": "", "expected_error": "SyntaxError"},
        {"input": "", "expected_error": "SyntaxError"},
        {"input": "", "expected_error": "SyntaxError"},
    ]
    instances["tasks"][0]["runtime_tests"] = error_tests
    instances["tasks"][0]["tests"] = error_tests

    result = worker._validate_practice_instances(
        task={
            "lesson": {
                "practice_tasks": {
                    "l1": [{"number": 1, "text": "Прочитать сообщение об ошибке SyntaxError при запуске кода"}]
                },
            }
        },
        spec=_practice_worker_spec(),
        templates=templates,
        instances=instances,
    )

    assert result.approved is True


def test_practice_instance_validation_allows_manual_check_for_error_demonstration_task(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    templates = _practice_template_set().model_dump(mode="json")
    templates["tasks"][0]["source_text"] = "Прочитать сообщение об ошибке SyntaxError при запуске кода"
    templates["tasks"][0]["task_type"] = "debug_error_message"
    templates["tasks"][0]["skill_target"] = "read Python error message"
    templates["tasks"][0]["test_policy"] = "manual/static check: code runs without SyntaxError after correction"
    instances = _practice_instance_set().model_dump(mode="json")
    instances["tasks"][0]["task_type"] = "debug_error_message"
    instances["tasks"][0]["scenario"] = "IDE показывает ошибку при запуске программы."
    instances["tasks"][0]["student_condition"] = "Запустите код и прочитайте сообщение об ошибке."
    instances["tasks"][0]["faulty_code"] = 'print("Готово)\n'
    instances["tasks"][0]["faulty_code_display"] = 'print("Готово)\n'
    instances["tasks"][0]["output_requirements"] = (
        "После исправления программа запускается без SyntaxError, связанной с незавершённой строкой."
    )
    instances["tasks"][0]["runtime_tests"] = []
    instances["tasks"][0]["tests"] = []
    instances["tasks"][0]["manual_checks"] = ["Код запускается без SyntaxError, связанной с незавершённой строкой."]

    result = worker._validate_practice_instances(
        task={
            "lesson": {
                "practice_tasks": {
                    "l1": [{"number": 1, "text": "Прочитать сообщение об ошибке SyntaxError при запуске кода"}]
                },
            }
        },
        spec=_practice_worker_spec(),
        templates=templates,
        instances=instances,
    )

    assert result.approved is True


def test_practice_instance_validation_allows_stdout_after_fix_for_non_diagnostic_debug_task(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    templates = _practice_template_set().model_dump(mode="json")
    templates["tasks"][0]["source_text"] = "Дан код: prnt(\"Привет\"). Исправить на print."
    templates["tasks"][0]["task_type"] = "fix_faulty_code (NameError: misspelled print)"
    templates["tasks"][0]["skill_target"] = "Исправлять NameError из-за опечатки в имени функции print."
    templates["tasks"][0]["test_policy"] = (
        "Предпочтительно stdout-автопроверка: после исправления вывод детерминирован. "
        "В learner-facing тестах не требовать демонстрации NameError; демонстрация ошибки — диагностический шаг, "
        "но автопроверка проверяет корректный запуск и stdout после исправления."
    )
    instances = _practice_instance_set().model_dump(mode="json")
    instances["tasks"][0]["task_type"] = "fix_faulty_code (NameError: misspelled print)"
    instances["tasks"][0]["scenario"] = (
        "В проекте есть однострочный скрипт, который должен вывести метку версии. "
        "Сейчас автопроверка падает из-за ошибки при запуске."
    )
    instances["tasks"][0]["student_condition"] = "Скопируйте код и исправьте его так, чтобы программа вывела строку."
    instances["tasks"][0]["faulty_code"] = 'prnint("Версия: 1.0")\n'
    instances["tasks"][0]["faulty_code_display"] = 'prnint("Версия: 1.0")\n'
    instances["tasks"][0]["output_requirements"] = "Программа должна вывести одну строку: Версия: 1.0"
    tests = [
        {"input": "", "expected_output": "Версия: 1.0\n"},
        {"input": "", "expected_output": "Версия: 1.0\n"},
        {"input": "", "expected_output": "Версия: 1.0\n"},
    ]
    instances["tasks"][0]["runtime_tests"] = tests
    instances["tasks"][0]["tests"] = tests
    instances["tasks"][0]["manual_checks"] = ["Программа печатает только указанную строку."]

    result = worker._validate_practice_instances(
        task={
            "lesson": {
                "practice_tasks": {"l1": [{"number": 1, "text": "Дан код: prnt(\"Привет\"). Исправить на print."}]},
            }
        },
        spec=_practice_worker_spec(),
        templates=templates,
        instances=instances,
    )

    assert result.approved is True


def test_practice_worker_deterministic_renderer_excludes_hidden_solution(tmp_path: Path) -> None:
    template_agent = FakeGraph([_practice_template_set()])
    variant_agent = FakeGraph([_practice_instance_set()])
    validator = FakeGraph([MaterialValidationDecision(approved=True)])
    worker = MaterialWorker(
        subagents={
            "PracticeTaskTemplateAgent": template_agent,
            "PracticeTaskVariantAgent": variant_agent,
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

    assert result.status == "approved"
    assert "Создайте переменную city" in result.content
    assert "city = 'Тула'\nprint(city)" not in result.content
    assert "Practice HTML rendered deterministically from practice_instances." in result.agent_notes
    assert len(validator.calls) == 1


def test_practice_renderer_uses_format_prompt_with_verbatim_nameerror_check(tmp_path: Path) -> None:
    html_template = load_html_format_template()

    content = render_practice_material_html(
        {"lesson": {"number": 2, "title": "Облачная IDE"}},
        {
            "tasks": [
                {
                    "id": "P2",
                    "level": "L1",
                    "scenario": "Запустите код и определите тип ошибки.",
                    "student_condition": "Исправьте программу после диагностики.",
                    "starter_code": "pritn('Готово')",
                    "input_requirements": "Ввод не требуется.",
                    "output_requirements": "После исправления программа выводит Готово.",
                    "runtime_tests": [{"input": "", "expected_output": "Готово\n"}],
                    "manual_checks": ["Запуск исходного кода даёт NameError: name 'pritn' is not defined."],
                }
            ]
        },
        html_template=html_template,
    )

    assert ".cc-lesson { max-width: 920px; margin: 0 auto; padding: 24px 18px;" in content
    assert "<h1>Занятие 2. Облачная IDE</h1>" in content
    assert "pritn(&#x27;Готово&#x27;)" in content
    assert "NameError: name &#x27;pritn&#x27; is not defined" in content
    assert "NameError: name &#x27;...&#x27; is not defined" not in content
    assert "<td><pre><code></code></pre></td>" in content


def test_practice_renderer_renders_lesson_goal_and_objectives(tmp_path: Path) -> None:
    content = render_practice_material_html(
        {"lesson": {"number": 2, "title": "Облачная IDE"}},
        _practice_instance_set().model_dump(mode="json"),
        html_template=load_html_format_template(),
    )

    assert '<section id="goals"><h2 id="goals">Цели и задачи</h2>' in content
    assert "<strong>Цель:</strong> Научиться создавать переменную и выводить её значение." in content
    assert "<li>создаёт переменную</li>" in content


def test_practice_renderer_adds_empty_code_block_for_write_code_without_starter(tmp_path: Path) -> None:
    content = render_practice_material_html(
        {"lesson": {"number": 3, "title": "Вывод данных"}},
        {
            "tasks": [
                {
                    "id": "P1",
                    "level": "L1",
                    "task_type": "write_code",
                    "scenario": "Создайте переменную и выведите её.",
                    "student_condition": "Напишите программу.",
                    "input_requirements": "Ввод не требуется.",
                    "output_requirements": "Выведите значение переменной.",
                    "runtime_tests": [{"input": "", "expected_output": "Сокол\n"}],
                    "run_mode": "single_file",
                }
            ]
        },
        html_template=load_html_format_template(),
    )

    assert "<p><strong>Код в редакторе:</strong></p><pre><code></code></pre>" in content


def test_practice_renderer_does_not_add_empty_code_block_for_manual_only_without_code(tmp_path: Path) -> None:
    content = render_practice_material_html(
        {"lesson": {"number": 2, "title": "Ошибки"}},
        {
            "tasks": [
                {
                    "id": "P1",
                    "level": "L1",
                    "task_type": "diagnostic",
                    "scenario": "Прочитайте сообщение об ошибке.",
                    "student_condition": "Назовите тип ошибки.",
                    "manual_checks": ["Назван класс SyntaxError."],
                    "run_mode": "manual_only",
                }
            ]
        },
        html_template=load_html_format_template(),
    )

    assert "<p><strong>Код в редакторе:</strong></p><pre><code></code></pre>" not in content


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
    assert "Learner-facing lesson materials are theory, practice, self_work, current_control, and intermediate" in channel_policy
    assert "Autocheck keys for learner-facing self_work and current_control must not be visible in HTML" in channel_policy
    assert "generation_artifacts.current_control_autocheck" in channel_policy
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
    assert contract["required_assessment_composition"]["variant_count"] == 4
    assert contract["required_assessment_composition"]["items_per_variant"] == 15
    assert contract["required_assessment_composition"]["minimum_code_writing_items_per_variant"] == 10
    assert contract["scoring_policy"]["allow_numeric_points_thresholds_or_grade_conversion"].startswith("only when")
    assert "generation_artifacts.intermediate_assessment" in " ".join(contract["generation_rules"])
    assert "Do not duplicate the full variant-by-variant answer bank" in " ".join(contract["generation_rules"])
    assert "Do not invent a numeric scoring scale" in " ".join(contract["generation_rules"])
    assert "Do not require a full key bank inside mr_intermediate HTML" in " ".join(contract["validation_rules"])
    assert "Reject if mr_intermediate invents numeric scoring" in " ".join(contract["validation_rules"])
    assert "required per-variant composition" in " ".join(contract["validation_rules"])


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
    assert "Не выдумывай числовую шкалу оценивания" in generation_prompt
    assert "4 варианта; в каждом варианте 15 элементов" in generation_prompt
    assert "закрытого учительского проверочного слоя" in generation_prompt
    assert "внутреннего QA/artifact-слоя intermediate_assessment" not in generation_prompt
    assert "Reject invented scoring norms" in validation_prompt
    assert "required approved assessment composition" in validation_prompt


def test_mr_intermediate_validation_prompt_omits_dependency_html_content() -> None:
    spec = get_material_spec("mr_intermediate")
    dependency = MaterialResult(
        kind="intermediate",
        material_type="Intermediate",
        agent_type="IntermediateAssessmentAgent",
        status="approved",
        iterations=1,
        content='<style></style><div class="cc-lesson"><h2 id="v1">SECRET_DEPENDENCY_KIM</h2></div>',
        prompt_files=(),
        generation_artifacts={"intermediate_assessment": _intermediate_assessment_artifact().model_dump(mode="json")},
    )

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[dependency],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
    )

    assert "SECRET_DEPENDENCY_KIM" not in prompt
    assert "DEPENDENCY CONTENT OMITTED: true" in prompt
    assert "Dependency intermediate HTML is not the checked mr_intermediate material" in prompt
    assert "CHECKED MATERIAL HTML START" in prompt


def test_validation_prompt_requires_checked_html_evidence_quote() -> None:
    spec = get_material_spec("specification_qa")

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
    )

    assert '"evidence_quote"' in prompt
    assert "CHECKED MATERIAL HTML START" in prompt
    assert "include an exact short evidence_quote copied from CHECKED MATERIAL HTML" in prompt


def test_validation_prompt_separates_reference_context_without_metadata_paths_or_sha() -> None:
    spec = get_material_spec("specification_qa")
    references = {
        "requirements": [
            ReferenceDocument(
                field="requirements",
                path="docs/ismart/source_rules.md",
                resolved_path="C:\\Projects\\bot_platform\\docs\\ismart\\source_rules.md",
                sha="abc123def456",
                truncated=False,
                content="Reference content for QA validation.",
            )
        ]
    }

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references=references,
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
    )

    assert "Reference content for QA validation." in prompt
    assert "source_rules" in prompt
    assert "PRIMARY CHECKED ARTIFACT START" in prompt
    assert "SOURCE/REFERENCE MATERIALS START" in prompt
    assert "SOURCE DOCUMENT CONTENT START" in prompt
    assert "docs/ismart/source_rules.md" not in prompt
    assert "C:\\Projects\\bot_platform" not in prompt
    assert "abc123def456" not in prompt
    assert '"sha"' not in prompt
    assert '"resolved_path"' not in prompt


def test_validation_prompt_preserves_reference_content_outside_checked_artifact() -> None:
    spec = get_material_spec("specification_qa")
    reference_content = r'<img src="C:\Projects\bot_platform\docs\ismart\reference.png">'
    references = {
        "template_descriptions": [
            ReferenceDocument(
                field="template_descriptions",
                path="docs/ismart/templates.md",
                resolved_path="C:\\Projects\\bot_platform\\docs\\ismart\\templates.md",
                sha="abc123",
                truncated=False,
                content=reference_content,
            )
        ]
    }

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references=references,
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
    )

    checked_section = prompt.split("PRIMARY CHECKED ARTIFACT START", 1)[1].split(
        "PRIMARY CHECKED ARTIFACT END",
        1,
    )[0]
    assert reference_content in prompt
    assert reference_content not in checked_section
    assert "If a string appears only in SOURCE/REFERENCE MATERIALS or DEPENDENCY MATERIALS" in prompt
    assert "Quotes from SOURCE/REFERENCE MATERIALS or DEPENDENCY MATERIALS are invalid evidence" in prompt


def test_validation_prompt_dependency_context_omits_runtime_metadata() -> None:
    spec = get_material_spec("specification_qa")
    dependency = MaterialResult(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        status="approved",
        iterations=7,
        content=VALID_HTML,
        prompt_files=("03_Практика_prompt_skill.md",),
        validation_issues=["old validator issue"],
        agent_notes=["internal generation note"],
        generation_artifacts={"practice_instances": {"tasks": [{"id": "P1", "hidden_solution": "answer = 1"}]}},
    )

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[dependency],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
    )

    assert "DEPENDENCY KIND: practice" in prompt
    assert "DEPENDENCY STATUS: approved" in prompt
    assert "answer = 1" in prompt
    assert "PracticeMaterialAgent" not in prompt
    assert '"iterations": 7' not in prompt
    assert "old validator issue" not in prompt
    assert "internal generation note" not in prompt
    assert "03_Практика_prompt_skill.md" not in prompt


def test_validation_prompt_requires_complete_first_pass_audit() -> None:
    spec = get_material_spec("practice")
    artifacts = {
        "practice_templates": _practice_template_set().model_dump(mode="json"),
        "practice_instances": _practice_instance_set().model_dump(mode="json"),
    }

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content='<style></style><div class="cc-lesson"><p>HTML_SHOULD_NOT_BE_VALIDATED</p></div>',
        rule_result=ValidationResult.ok(),
        generation_artifacts=artifacts,
    )

    assert "VALIDATION COMPLETENESS AND STABILITY" in prompt
    assert "VALIDATION TARGET MODE:\nstructured_artifacts" in prompt
    assert "HTML_SHOULD_NOT_BE_VALIDATED" not in prompt
    assert "RENDERED HTML IS NOT INCLUDED" in prompt
    assert "On the first validation pass, report every material issue" in prompt
    assert "Do not stop after finding the first blocking issue" in prompt
    assert "Keep the validation standard stable across attempts" in prompt
    assert "Top-level issues must be a complete summary of all blocking issues" in prompt


def test_controller_prompt_treats_unquoted_visible_claims_as_unproven() -> None:
    spec = get_material_spec("specification_qa")
    reference_content = r'<img src="C:\Projects\bot_platform\docs\ismart\reference.png">'

    prompt = build_validation_controller_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={
            "template_descriptions": [
                ReferenceDocument(
                    field="template_descriptions",
                    path="docs/ismart/templates.md",
                    resolved_path="C:\\Projects\\bot_platform\\docs\\ismart\\templates.md",
                    sha="abc123",
                    truncated=False,
                    content=reference_content,
                )
            ]
        },
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["visible HTML contains docs/..."]),
        merged_validation=ValidationResult.fail(["visible HTML contains docs/..."]),
    )

    checked_section = prompt.split("PRIMARY CHECKED ARTIFACT START", 1)[1].split(
        "PRIMARY CHECKED ARTIFACT END",
        1,
    )[0]
    assert reference_content in prompt
    assert reference_content not in checked_section
    assert "require a direct quote from CHECKED MATERIAL HTML" in prompt
    assert "If it appears only in SOURCE/REFERENCE MATERIALS or DEPENDENCY MATERIALS" in prompt
    assert "VALIDATION TARGET MODE:\nhtml" in prompt
    assert "Treat a learner-facing leakage claim as unproven" in prompt


def test_controller_prompt_for_structured_materials_excludes_rendered_html() -> None:
    artifacts = {
        "practice_templates": _practice_template_set().model_dump(mode="json"),
        "practice_instances": _practice_instance_set().model_dump(mode="json"),
    }
    prompt = build_validation_controller_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=get_material_spec("practice"),
        prompt_contents={},
        references={},
        dependencies=[],
        content='<style></style><div class="cc-lesson"><p>HTML_CONTROLLER_SHOULD_NOT_SEE</p></div>',
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["HTML formatting issue"]),
        merged_validation=ValidationResult.fail(["HTML formatting issue"]),
        generation_artifacts=artifacts,
    )

    assert "VALIDATION TARGET MODE:\nstructured_artifacts" in prompt
    assert "HTML_CONTROLLER_SHOULD_NOT_SEE" not in prompt
    assert "RENDERED HTML IS NOT INCLUDED" in prompt
    assert "do not keep failed for HTML rendering/assembly issues" in prompt


def test_mr_intermediate_html_first_validation_prompt_keeps_checked_html_visible() -> None:
    spec = get_material_spec("mr_intermediate")
    content = (
        '<style></style><div class="cc-lesson">'
        "<p>Use intermediate_assessment, generation_artifacts.hidden_solution and autocheck_config.</p>"
        "</div>"
    )

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=content,
        rule_result=ValidationResult.ok(),
    )

    assert "VALIDATION TARGET MODE:\nhtml" in prompt
    assert "CHECKED MATERIAL HTML START" in prompt
    assert "intermediate_assessment" in prompt
    assert "autocheck_config" in prompt


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
    assert "keep failed when the checked HTML invents maximum points" in prompt
    assert "approved composition of each variant" in prompt


def test_mr_intermediate_appellate_policy_overrules_dependency_kim_duplication_claim(tmp_path: Path) -> None:
    spec = get_material_spec("mr_intermediate")
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    issue = "mr_intermediate duplicates full KIM variants V1-V4 from the intermediate dependency"
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 1,
        "score_rationale": "",
        "rationale": "",
        "blocking_issues": [issue],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [issue],
    }

    adjusted = worker._apply_mr_intermediate_appellate_policy(
        spec=spec,
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail([issue]),
        decision=decision,
    )

    assert adjusted["approved"] is True
    assert adjusted["decision"] == "approve_material"
    assert adjusted["quality_score"] >= worker.config.validation_controller_accept_score
    assert adjusted["blocking_issues"] == []
    assert issue in adjusted["overruled_validator_issues"]


def test_mr_intermediate_appellate_policy_overrules_full_kim_duplication_claim(tmp_path: Path) -> None:
    spec = get_material_spec("mr_intermediate")
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    issue = "mr_intermediate duplicates full KIM variants V1-V4 in checked HTML"
    full_kim_html = (
        '<style></style><div class="cc-lesson">'
        "<h2>Variant 1</h2><p><strong>Task 1</strong></p><p><strong>Task 2</strong></p>"
        "<h2>Variant 2</h2><p><strong>Task 1</strong></p><p><strong>Task 2</strong></p>"
        "<h2>Variant 3</h2><p><strong>Task 1</strong></p><p><strong>Task 2</strong></p>"
        "<h2>Variant 4</h2><p><strong>Task 1</strong></p><p><strong>Task 2</strong></p>"
        "</div>"
    )
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 1,
        "score_rationale": "",
        "rationale": "",
        "blocking_issues": [issue],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [issue],
    }

    adjusted = worker._apply_mr_intermediate_appellate_policy(
        spec=spec,
        content=full_kim_html,
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail([issue]),
        decision=decision,
    )

    assert adjusted["approved"] is True
    assert adjusted["decision"] == "approve_material"
    assert adjusted["blocking_issues"] == []
    assert issue in adjusted["overruled_validator_issues"]


def test_mr_intermediate_appellate_policy_keeps_internal_marker_leak_blocking(tmp_path: Path) -> None:
    spec = get_material_spec("mr_intermediate")
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    issue = "publishable HTML exposes intermediate_assessment internal marker"
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 1,
        "score_rationale": "",
        "rationale": "",
        "blocking_issues": [issue],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [issue],
    }

    adjusted = worker._apply_mr_intermediate_appellate_policy(
        spec=spec,
        content='<style></style><div class="cc-lesson"><p>intermediate_assessment</p></div>',
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail([issue]),
        decision=decision,
    )

    assert adjusted["approved"] is False
    assert adjusted["blocking_issues"] == [issue]
    assert adjusted["overruled_validator_issues"] == []


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


def test_specification_qa_validation_prompt_allows_visible_qa_ids_and_forbids_process_logs() -> None:
    spec = get_material_spec("specification_qa")

    prompt = build_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content='<style></style><div class="cc-lesson"><h3>P1 - QA-ID: L2-P1</h3></div>',
        rule_result=ValidationResult.ok(),
    )

    assert "QA-ID labels are allowed in specification_qa visible HTML" in prompt
    assert "Do not treat visible QA-ID labels as source-marker leakage" in prompt
    assert "Visible specification_qa HTML must not contain raw local source paths" in prompt
    assert "source hashes/SHA values" in prompt
    assert "исправлено по замечаниям валидатора" in prompt


def test_specification_qa_controller_prompt_overrules_visible_qa_id_leakage_claims() -> None:
    spec = get_material_spec("specification_qa")

    prompt = build_validation_controller_prompt(
        task={"course": {}, "module": {}, "lesson": {}},
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content='<style></style><div class="cc-lesson"><h3>P1 - QA-ID: L2-P1</h3></div>',
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["QA-ID is internal marker leakage"]),
        merged_validation=ValidationResult.fail(["QA-ID is internal marker leakage"]),
    )

    assert "For specification_qa, overrule validator objections that visible QA-ID labels are internal marker leakage" in prompt
    assert "QA-ID is allowed in this internal QA artifact" in prompt
    assert "source hashes/SHA values" in prompt
    assert "process/retry logs" in prompt


def test_specification_qa_appellate_policy_approves_qa_id_only_rejection(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 2.0,
        "score_rationale": "",
        "rationale": "",
        "blocking_issues": ["QA-ID is internal marker leakage in visible HTML"],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": ["Remove QA-ID from headings"],
    }

    adjusted = worker._apply_specification_qa_appellate_policy(
        spec=get_material_spec("specification_qa"),
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail(["QA-ID is internal marker leakage in visible HTML"]),
        decision=decision,
    )

    assert adjusted["approved"] is True
    assert adjusted["decision"] == "approve_material"
    assert adjusted["quality_score"] >= worker.config.validation_controller_accept_score
    assert adjusted["blocking_issues"] == []
    assert adjusted["overruled_validator_issues"] == ["QA-ID is internal marker leakage in visible HTML"]
    assert adjusted["fix_instructions"] == []


def test_specification_qa_appellate_policy_keeps_process_log_blocker(tmp_path: Path) -> None:
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 2.0,
        "score_rationale": "",
        "rationale": "",
        "blocking_issues": [
            "QA-ID is internal marker leakage in visible HTML",
            "Раздел содержит процессную формулировку: исправлено по замечаниям валидатора",
        ],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [
            "Remove QA-ID from headings",
            "Remove process wording",
        ],
    }

    adjusted = worker._apply_specification_qa_appellate_policy(
        spec=get_material_spec("specification_qa"),
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail(decision["blocking_issues"]),
        decision=decision,
    )

    assert adjusted["approved"] is False
    assert adjusted["decision"] == "keep_failed"
    assert adjusted["blocking_issues"] == ["Раздел содержит процессную формулировку: исправлено по замечаниям валидатора"]
    assert adjusted["overruled_validator_issues"] == ["QA-ID is internal marker leakage in visible HTML"]
    assert adjusted["fix_instructions"] == ["Remove process wording"]


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


def test_practice_controller_prompt_overrules_difficulty_count_mismatch() -> None:
    spec = get_material_spec("practice")
    task = {
        "course": {},
        "module": {},
        "lesson": {
            "difficulty": {"l1": {"count": 2}, "l2": {"count": 4}},
            "practice_tasks": {
                "l1": [{"number": 1, "text": "Read SyntaxError"}, {"number": 2, "text": "Read NameError"}],
                "l2": [
                    {"number": 3, "text": "Fix print quote"},
                    {"number": 4, "text": "Fix prnt"},
                    {"number": 5, "text": "Fix missing quotes"},
                ],
            },
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
        llm_result=ValidationResult.fail(["требуется 2 задания L1 и 4 задания L2, отсутствует P6"]),
        merged_validation=ValidationResult.fail(["требуется 2 задания L1 и 4 задания L2, отсутствует P6"]),
    )

    assert "authoritative_task_ids" in prompt
    assert "lesson.difficulty.*.count conflicts with lesson.practice_tasks" in prompt
    assert "overrule validator demands to invent extra tasks such as P6/P7" in prompt


def test_practice_controller_prompt_overrules_unclosed_string_display_objection() -> None:
    spec = get_material_spec("practice")

    prompt = build_validation_controller_prompt(
        task={
            "course": {},
            "module": {},
            "lesson": {"practice_tasks": {"l1": [{"number": 1, "text": "Read SyntaxError"}]}},
        },
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=VALID_HTML,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["faulty_code_display visually spans the next line"]),
        merged_validation=ValidationResult.fail(["faulty_code_display visually spans the next line"]),
    )

    assert "overrule validator objections that the displayed faulty snippet visually becomes multi-line" in prompt
    assert "This is the expected learner-facing faulty input for this error type" in prompt


def test_practice_validation_policy_focuses_on_methodology_not_task_layout() -> None:
    policy = validation_policy_for_spec(get_material_spec("practice"))

    assert "Validate practice methodology, not a rigid task-layout template" in policy
    assert "Do not reject practice solely because a task lacks a specific visual subsection" in policy
    assert "starter code is optional" in policy


def test_practice_appellate_policy_approves_overstrict_difficulty_count_issue(tmp_path: Path) -> None:
    spec = get_material_spec("practice")
    task = {
        "course": {},
        "module": {},
        "lesson": {
            "difficulty": {"l1": {"count": 2}, "l2": {"count": 4}},
            "practice_tasks": {
                "l1": [{"number": 1, "text": "Read SyntaxError"}, {"number": 2, "text": "Read NameError"}],
                "l2": [
                    {"number": 3, "text": "Fix print quote"},
                    {"number": 4, "text": "Fix prnt"},
                    {"number": 5, "text": "Fix missing quotes"},
                ],
            },
        },
    }
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    issue = "Несоответствие обязательному количеству задач по JSON: требуется 2 задания L1 и 4 задания L2 (всего 6), фактически дано 5; отсутствует одно задание уровня L2."
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 2.0,
        "score_rationale": "",
        "rationale": "",
        "blocking_issues": [issue],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [issue],
    }

    adjusted = worker._apply_practice_appellate_policy(
        spec=spec,
        task=task,
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail([issue]),
        generation_artifacts={
            "practice_instances": {
                "tasks": [{"id": f"P{index}"} for index in range(1, 6)]
            }
        },
        decision=decision,
    )

    assert adjusted["approved"] is True
    assert adjusted["decision"] == "approve_material"
    assert adjusted["blocking_issues"] == []
    assert adjusted["overruled_validator_issues"] == [issue]


def test_practice_appellate_policy_approves_subject_entity_and_layout_objections(tmp_path: Path) -> None:
    spec = get_material_spec("practice")
    task = {
        "course": {},
        "module": {},
        "lesson": {
            "practice_tasks": {
                "l1": [{"number": 1, "text": "Create one string variable and print it"}],
                "l2": [
                    {"number": 2, "text": "Create one integer variable and print it"},
                    {"number": 3, "text": "Create string and integer variables and print them"},
                    {"number": 4, "text": "Print text and a variable in one print call"},
                    {"number": 5, "text": "Create favorite color and favorite animal variables and print them"},
                ],
            },
        },
    }
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    subject_issue = (
        "P5: подмена требуемых сущностей («любимый цвет» и «любимое животное») "
        "на «любимый напиток» и «вид спорта» — это изменение паттерна относительно source_text/контракта."
    )
    layout_issue = (
        "Во всех задачах P1–P5 отсутствует явный learner-facing блок «Код» (<pre><code>…</code></pre>) "
        "как место/заготовка для ввода решения."
    )
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 2.0,
        "score_rationale": "",
        "rationale": "",
        "blocking_issues": [subject_issue, layout_issue],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [subject_issue, layout_issue],
    }

    adjusted = worker._apply_practice_appellate_policy(
        spec=spec,
        task=task,
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail([subject_issue, layout_issue]),
        generation_artifacts={
            "practice_instances": {
                "tasks": [{"id": f"P{index}"} for index in range(1, 6)]
            }
        },
        decision=decision,
    )

    assert adjusted["approved"] is True
    assert adjusted["decision"] == "approve_material"
    assert adjusted["blocking_issues"] == []
    assert adjusted["overruled_validator_issues"] == [subject_issue, layout_issue]
    assert adjusted["fix_instructions"] == []


def test_practice_appellate_policy_approves_overstrict_unclosed_string_issue(tmp_path: Path) -> None:
    spec = get_material_spec("practice")
    task = {
        "course": {},
        "module": {},
        "lesson": {
            "practice_tasks": {
                "l1": [{"number": 1, "text": "Прочитать сообщение об ошибке SyntaxError"}],
                "l2": [{"number": 2, "text": "Дан код: print(\"Привет). Найти ошибку и исправить"}],
            },
        },
    }
    worker = MaterialWorker(subagents={}, config=IsmartGenerationConfig(prompts_dir=tmp_path, output_root=tmp_path))
    issue = (
        "P1: learner-facing faulty_code_display содержит незакрытую строку, из-за чего фрагмент "
        "в HTML фактически разрывается на две строки, что делает код невалидным и не соответствует "
        "инварианту «ровно одна ошибка в кавычках»."
    )
    decision = {
        "approved": False,
        "decision": "keep_failed",
        "quality_score": 2.0,
        "score_rationale": "",
        "rationale": "",
        "blocking_issues": [issue],
        "non_blocking_issues": [],
        "overruled_validator_issues": [],
        "residual_risks": [],
        "fix_instructions": [issue],
    }

    adjusted = worker._apply_practice_appellate_policy(
        spec=spec,
        task=task,
        rule_result=ValidationResult.ok(),
        validation=ValidationResult.fail([issue]),
        generation_artifacts={
            "practice_instances": {
                "tasks": [
                    {"id": "P1"},
                    {"id": "P2"},
                ]
            }
        },
        decision=decision,
    )

    assert adjusted["approved"] is True
    assert adjusted["decision"] == "approve_material"
    assert adjusted["blocking_issues"] == []
    assert adjusted["overruled_validator_issues"] == [issue]


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


def test_package_validation_prompt_omits_html_for_structured_materials() -> None:
    content = '<style>.x{}</style><div class="cc-lesson"><p>STRUCTURED_HTML_SHOULD_BE_OMITTED</p></div>'
    spec = get_material_spec("practice")
    material = MaterialResult(
        kind="practice",
        material_type="Practice",
        agent_type="PracticeMaterialAgent",
        status="approved",
        iterations=1,
        content=content,
        prompt_files=(),
        generation_artifacts={
            "practice_templates": _practice_template_set().model_dump(mode="json"),
            "practice_instances": _practice_instance_set().model_dump(mode="json"),
        },
    )

    prompt = build_package_validation_prompt(
        task={"course": {}, "module": {}, "lesson": {"lesson_number": 2}},
        specs=[spec],
        materials=[material],
        rule_result=ValidationResult.ok(),
    )

    assert "STRUCTURED_HTML_SHOULD_BE_OMITTED" not in prompt
    assert "full_final_content_omitted" in prompt
    assert "primary_structured_artifact_keys" in prompt
    assert "do not semantically review rendered HTML" in prompt


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
