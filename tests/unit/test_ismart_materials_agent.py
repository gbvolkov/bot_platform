from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from generators.ismart_materials_agent.agent import run_ismart_task
from generators.ismart_materials_agent.cli import filter_tasks, tasks_from_payload
from generators.ismart_materials_agent.context import (
    build_generation_prompt,
    build_validation_controller_prompt,
    build_validation_prompt,
)
from generators.ismart_materials_agent.contracts import IsmartGenerationConfig, MaterialResult, ValidationResult
from generators.ismart_materials_agent.planner import build_material_plan, teacher_material_required
from generators.ismart_materials_agent.registry import MATERIAL_SPEC_REGISTRY
from generators.ismart_materials_agent.sources import ReferenceLoader
from generators.ismart_materials_agent.validators import RuleValidator
from generators.ismart_materials_agent.workers import MaterialWorker


WORKSPACE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ismart"
    / "Материалы для ИИ-агентов"
    / "рабочая область агента"
)
SOURCE_JSON = WORKSPACE / "для_теста_существенные_данные.json"


def load_course() -> dict[str, Any]:
    return json.loads(SOURCE_JSON.read_text(encoding="utf-8"))


def task_for_lesson(lesson_number: int) -> dict[str, Any]:
    course = load_course()
    for module in course["modules"]:
        for lesson in module["lessons"]:
            if lesson["lesson_number"] == lesson_number:
                return {
                    "task_id": f"lesson-{lesson_number}",
                    "course": course["course"],
                    "module": module,
                    "lesson": lesson,
                    "modules": course["modules"],
                    "markdown_references_base": course.get("markdown_references_base"),
                }
    raise AssertionError(f"missing lesson {lesson_number}")


class FakeClient:
    def __init__(self, *, bad_first_for: set[str] | None = None) -> None:
        self.bad_first_for = set(bad_first_for or set())
        self.generator_calls: dict[str, int] = {}

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        if "MaterialValidatorAgent" in system:
            return {"approved": "RULE VALIDATOR ISSUES:\n[]" in user, "issues": [], "fix_instructions": []}
        if "PackageValidatorAgent" in system:
            return {"approved": "RULE VALIDATOR ISSUES:\n[]" in user, "issues": [], "fix_instructions": []}

        kind = _kind_from_prompt(user)
        self.generator_calls[kind] = self.generator_calls.get(kind, 0) + 1
        if kind in self.bad_first_for and self.generator_calls[kind] == 1:
            return {"content": "<div>bad</div>", "agent_notes": ["bad first"]}
        return {"content": html_for_kind(kind, user), "agent_notes": [kind]}


class TailClient(FakeClient):
    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        if "MaterialValidatorAgent" in system:
            approved = "RULE VALIDATOR ISSUES:\n[]" in user
            return {
                "approved": approved,
                "issues": [] if approved else ["rule issues present"],
                "fix_instructions": [] if approved else ["fix rule issues"],
            }
        if "PackageValidatorAgent" in system:
            approved = "RULE VALIDATOR ISSUES:\n[]" in user
            return {
                "approved": approved,
                "issues": [] if approved else ["package rule issues present"],
                "fix_instructions": [] if approved else ["fix package rule issues"],
            }

        kind = _kind_from_prompt(user)
        self.generator_calls[kind] = self.generator_calls.get(kind, 0) + 1
        return {
            "content": f"{html_for_kind(kind, user)}\nReturn only JSON:\n{{\"approved\": true}}",
            "agent_notes": [kind, "tail"],
        }


class OverstrictValidatorClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.controller_calls = 0

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        if "ValidationControllerAgent" in system:
            self.controller_calls += 1
            return {
                "approved": True,
                "decision": "approve_material",
                "quality_score": 4,
                "score_rationale": "usable material; validator issue is non-blocking",
                "rationale": "validator issue is editorial and non-blocking",
                "blocking_issues": [],
                "non_blocking_issues": ["minor style preference"],
                "overruled_validator_issues": ["overstrict style issue"],
                "residual_risks": ["editor may still adjust wording"],
                "fix_instructions": [],
            }
        if "MaterialValidatorAgent" in system:
            return {
                "approved": False,
                "issues": ["overstrict style issue"],
                "fix_instructions": ["revise style"],
            }
        return super().complete_json(system=system, user=user)


class ScoredControllerClient(FakeClient):
    def __init__(self, score: float, *, approved: bool = False) -> None:
        super().__init__()
        self.score = score
        self.approved = approved
        self.controller_calls = 0

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        if "ValidationControllerAgent" in system:
            self.controller_calls += 1
            return {
                "approved": self.approved,
                "decision": "keep_failed",
                "quality_score": self.score,
                "score_rationale": "score-driven controller review",
                "rationale": "remaining issue is non-blocking enough for score threshold",
                "blocking_issues": ["validator still reports an issue"],
                "non_blocking_issues": ["conceptual overlap only"],
                "overruled_validator_issues": ["overstrict validator issue"],
                "residual_risks": ["editor may revise later"],
                "fix_instructions": ["optional edit"],
            }
        if "MaterialValidatorAgent" in system:
            return {
                "approved": False,
                "issues": ["overstrict validator issue"],
                "fix_instructions": ["revise optional issue"],
            }
        return super().complete_json(system=system, user=user)


class BlockAwareRetryClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.generation_prompts: list[str] = []

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        if "MaterialValidatorAgent" in system:
            if self.generator_calls.get("theory", 0) == 1:
                return {
                    "approved": False,
                    "issues": ["selfcheck needs answers"],
                    "fix_instructions": ["Add answers only to the selfcheck block."],
                    "issues_by_block": [
                        {
                            "block_id": "#selfcheck",
                            "block_heading": "Проверка себя",
                            "severity": "blocking",
                            "issue": "Questions have no answers.",
                            "fix_instruction": "Add short answers.",
                        }
                    ],
                    "passed_blocks": [
                        {
                            "block_id": "#intro",
                            "block_heading": "Введение",
                            "reason": "Intro is correct and should not change.",
                        }
                    ],
                }
            return {
                "approved": True,
                "issues": [],
                "fix_instructions": [],
                "issues_by_block": [],
                "passed_blocks": [{"block_id": "#intro", "block_heading": "Введение"}],
            }
        if "PackageValidatorAgent" in system:
            return {"approved": True, "issues": [], "fix_instructions": []}

        kind = _kind_from_prompt(user)
        self.generation_prompts.append(user)
        self.generator_calls[kind] = self.generator_calls.get(kind, 0) + 1
        if self.generator_calls[kind] == 1:
            filler_before = " ".join(["intro"] * 700)
            filler_after = " ".join(["body"] * 700)
            content = (
                '<style></style><div class="cc-lesson">'
                '<h2 id="intro">Введение</h2>'
                f"<p>{filler_before} FULL_PREVIOUS_HTML_MARKER {filler_after}</p>"
                '<h2 id="selfcheck">Проверка себя</h2><ol><li>Question?</li></ol>'
                "</div>"
            )
            return {"content": content, "agent_notes": ["first"]}

        assert "FULL_PREVIOUS_HTML_MARKER" in user
        assert "PREVIOUS FAILED CONTENT HTML START" in user
        assert '"block_id": "#selfcheck"' in user
        assert '"block_id": "#intro"' in user
        assert "Preserve blocks listed in PREVIOUS VALIDATION RESULT.passed_blocks" in user
        return {
            "content": (
                '<style></style><div class="cc-lesson">'
                '<h2 id="intro">Введение</h2><p>FULL_PREVIOUS_HTML_MARKER preserved.</p>'
                '<h2 id="selfcheck">Проверка себя</h2><ol><li>Question? Answer.</li></ol>'
                "</div>"
            ),
            "agent_notes": ["second"],
        }


def _kind_from_prompt(prompt: str) -> str:
    marker = "MATERIAL KIND:\n"
    start = prompt.index(marker) + len(marker)
    return prompt[start:].splitlines()[0].strip()


def html_for_kind(kind: str, prompt: str = "") -> str:
    prefix = '<style></style><div class="cc-lesson">'
    suffix = "</div>"
    if kind == "theory":
        body = "".join(f"<h2>{title}</h2><p>Текст</p>" for title in [
            "Цель занятия",
            "Задачи занятия",
            "Ключевые понятия",
            "Конспект",
            "Задачи-примеры для разбора",
            "Типичные ошибки",
            "Проверка себя",
            "Итоговые выводы",
        ])
    elif kind == "practice":
        expected_count = 6 if '"lesson_number": 2' in prompt else 5
        tasks = "".join(f"<h3>P{index}. Задание</h3><p>Вход: {index}</p><p>Ожидаемый вывод: {index}</p>" for index in range(1, expected_count + 1))
        body = f"<h2>Цель работы</h2><h2>Указания по выполнению</h2><h2>Задания</h2>{tasks}"
    elif kind in {"mr_theory", "mr_practice", "mr_intermediate"}:
        body = "".join(f"<h2>{title}</h2><p>Текст</p>" for title in [
            "Цель и задачи",
            "Методическая опора",
            "Подготовка",
            "Сценарий",
            "Ключи и пояснения",
            "Типичные ошибки и реакция",
        ])
    elif kind == "current_control":
        body = "<h2>Контроль</h2>" + "".join(f"<h3>Вопрос {index}</h3><p>Ключ: {index}</p>" for index in range(1, 4))
    elif kind == "self_work":
        tasks = "".join(f"<h3>Задача {index}</h3><p>Условие</p>" for index in range(1, 9))
        questions = "".join(f"<h3>Вопрос самоконтроля {index}</h3><p>Ключ: {index}</p>" for index in range(1, 11))
        body = "<h2>Тема</h2><h2>Цели и задачи</h2><h2>Порядок выполнения</h2><h2>Самоконтроль</h2>"
        body += f"{tasks}{questions}<h2>Требования к результату</h2><h2>Источники</h2>"
    elif kind == "intermediate":
        body = "".join(f"<h2>Комплект {index}</h2><p>Ключ: {index}</p>" for index in range(1, 5))
    elif kind == "specification_qa":
        body = "<h2>Паспорт</h2><p>QA-ID 1 SHA abc</p><h2>Источники</h2><h2>Ключи и тесты</h2><h2>Критерии QA</h2>"
    elif kind == "final_project":
        body = "<h2>Варианты проекта</h2><p>Вариант 1</p>"
    else:
        body = "<h2>Материал</h2>"
    return f"{prefix}{body}{suffix}"


def test_registry_uses_exact_existing_prompt_files():
    assert MATERIAL_SPEC_REGISTRY["theory"].prompt_files == (
        "01_Общее_prompt_skill.md",
        "02_Теория_prompt_skill.md",
        "08_Форматирование_заданий_курса_prompt.md",
        "91_skill_map.md",
        "92_описание_json.md",
    )
    assert MATERIAL_SPEC_REGISTRY["current_control"].prompt_files == (
        "01_Общее_prompt_skill.md",
        "08_Форматирование_заданий_курса_prompt.md",
        "91_skill_map.md",
        "92_описание_json.md",
    )
    for spec in MATERIAL_SPEC_REGISTRY.values():
        for prompt_file in spec.prompt_files:
            assert (WORKSPACE / "prompts_skills" / prompt_file).exists()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", False), ("Н/П", False), ("n/a", False), ("нет", False), ("false", False), ("-", False), ("Заполнить МР", True)],
)
def test_teacher_material_required(value: str, expected: bool):
    assert teacher_material_required(value) is expected


def test_planner_builds_order_and_dependency_specs():
    kinds = [spec.kind for spec in build_material_plan(task_for_lesson(3))]

    assert kinds == ["theory", "practice", "mr_theory", "mr_practice", "specification_qa"]
    assert MATERIAL_SPEC_REGISTRY["practice"].dependency_kinds == ("theory",)
    assert MATERIAL_SPEC_REGISTRY["mr_practice"].dependency_kinds == ("practice",)


def test_source_loader_reads_markdown_references_once():
    task = task_for_lesson(2)
    bundle = ReferenceLoader(IsmartGenerationConfig()).load(task)

    assert bundle["requirements"]
    assert bundle["template_descriptions"]
    assert all(document.resolved_path.endswith(".md") for docs in bundle.values() for document in docs)
    assert all(document.content for docs in bundle.values() for document in docs)


def test_practice_generation_prompt_includes_source_contract():
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["practice"]

    prompt = build_generation_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        previous_content="",
        previous_issues=[],
    )

    assert "SOURCE CONTRACT FROM JSON:" in prompt
    assert '"id": "P5"' in prompt
    assert '"level": "L2"' in prompt
    assert "Создать переменные: любимый цвет, любимое животное. Вывести их на экран." in prompt
    assert "Do not invent concrete values" in prompt
    assert "Expected output is not invented when it is deterministically derived" in prompt
    assert "do not render a fake input -> expected output table" in prompt


def test_theory_generation_prompt_treats_practice_tasks_as_boundaries():
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]

    prompt = build_generation_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        previous_content="",
        previous_issues=[],
    )

    assert "theory_boundary_contract" in prompt
    assert "related_practice_tasks" in prompt
    assert "Do not include complete solved examples" in prompt
    assert "не превращай их в готовые решённые примеры" in prompt


def test_practice_validation_prompts_allow_derived_stdout():
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["practice"]
    content = (
        '<style></style><div class="cc-lesson">'
        '<h2 id="tasks">Задания</h2>'
        "<h3>P1</h3><p>Вход: пусто</p><p>Ожидаемый вывод: Анна</p>"
        "</div>"
    )

    validation_prompt = build_validation_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=content,
        rule_result=ValidationResult.ok(),
    )
    controller_prompt = build_validation_controller_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=content,
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["overstrict expected output issue"]),
        merged_validation=ValidationResult.fail(["overstrict expected output issue"]),
    )

    for prompt in (validation_prompt, controller_prompt):
        assert "PRACTICE VALIDATION POLICY" in prompt
        assert "Do not treat an expected output as invented" in prompt
        assert "explicit source_text literals, assignments, and exact print(...)" in prompt
        assert "Do not create an impossible requirement" in prompt
        assert "placeholder expected output" in prompt
    assert '"quality_score": 0' in controller_prompt
    assert "QUALITY SCORE:" in controller_prompt
    assert ">= 3" not in controller_prompt


def test_theory_validation_prompt_rejects_practice_answer_leakage():
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]
    content = (
        '<style></style><div class="cc-lesson">'
        '<h2 id="examples">Задачи-примеры для разбора</h2>'
        '<pre><code>name = "Анна"\nprint(name)</code></pre>'
        '<pre class="cc-console"><code>Анна</code></pre>'
        "</div>"
    )

    prompt = build_validation_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content=content,
        rule_result=ValidationResult.ok(),
    )

    assert "THEORY VALIDATION POLICY" in prompt
    assert "Practice tasks in JSON are boundaries for the theory" in prompt
    assert "not reveal practice answers" in prompt


def test_validation_controller_prompt_has_tolerant_theory_boundary_policy():
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]
    prompt = build_validation_controller_prompt(
        task=task,
        spec=spec,
        prompt_contents={},
        references={},
        dependencies=[],
        content='<style></style><div class="cc-lesson"><h2>Теория</h2></div>',
        rule_result=ValidationResult.ok(),
        llm_result=ValidationResult.fail(["conceptual overlap with practice"]),
        merged_validation=ValidationResult.fail(["conceptual overlap with practice"]),
    )

    assert "Conceptual overlap with practice skills is expected and allowed" in prompt
    assert "complete solved answer to a specific practice task" in prompt
    assert "Self-check answers are allowed" in prompt
    assert "audit the validator decision" in prompt
    assert "Do not independently hunt for new defects" in prompt
    assert 'print("label:", value)' in prompt
    assert "without a concrete match to source-specific practice task data" in prompt
    assert "Assign the score after discounting validator objections" in prompt
    assert "QUALITY SCORE:" in prompt
    assert ">= 3" not in prompt


def test_worker_retries_after_validation_failure(tmp_path):
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]
    client = FakeClient(bad_first_for={"theory"})
    config = IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True)
    references = ReferenceLoader(config).load(task)

    result = MaterialWorker(client=client, config=config).run(
        task=task,
        spec=spec,
        references=references,
        dependency_results=[],
    )

    assert result.status == "approved"
    assert result.iterations == 2
    assert client.generator_calls["theory"] == 2


def test_worker_retry_receives_full_previous_content_and_block_reports(tmp_path):
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]
    client = BlockAwareRetryClient()
    config = IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True)
    references = ReferenceLoader(config).load(task)

    result = MaterialWorker(client=client, config=config).run(
        task=task,
        spec=spec,
        references=references,
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert result.iterations == 2
    assert len(client.generation_prompts) == 2
    validation = json.loads(
        sorted((tmp_path / "tmp" / "theory").glob("*attempt_01*.validation.json"))[0].read_text(encoding="utf-8")
    )
    assert validation["merged_validation"]["issues_by_block"][0]["block_id"] == "#selfcheck"
    assert validation["merged_validation"]["passed_blocks"][0]["block_id"] == "#intro"


def test_worker_controller_can_approve_after_overstrict_validation(tmp_path):
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]
    client = OverstrictValidatorClient()
    config = IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True, max_generation_iterations=1)
    references = ReferenceLoader(config).load(task)

    result = MaterialWorker(client=client, config=config).run(
        task=task,
        spec=spec,
        references=references,
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert result.iterations == 1
    assert result.controller_called
    assert result.controller_decision["approved"] is True
    assert result.controller_decision["quality_score"] == 4
    assert client.generator_calls == {"theory": 1}
    assert client.controller_calls == 1
    controller_reviews = list((tmp_path / "tmp" / "theory").glob("*controller_review*.json"))
    assert controller_reviews


def test_worker_accepts_controller_score_even_when_controller_approved_false(tmp_path):
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]
    client = ScoredControllerClient(score=3, approved=False)
    config = IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True, max_generation_iterations=1)
    references = ReferenceLoader(config).load(task)

    result = MaterialWorker(client=client, config=config).run(
        task=task,
        spec=spec,
        references=references,
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "approved"
    assert result.controller_called
    assert result.controller_decision["approved"] is False
    assert result.controller_decision["quality_score"] == 3
    assert "quality_score=3" in result.agent_notes[-1]


def test_worker_rejects_controller_score_below_threshold(tmp_path):
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]
    client = ScoredControllerClient(score=2.9, approved=True)
    config = IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True, max_generation_iterations=1)
    references = ReferenceLoader(config).load(task)

    result = MaterialWorker(client=client, config=config).run(
        task=task,
        spec=spec,
        references=references,
        dependency_results=[],
        attempts_dir=tmp_path / "tmp",
    )

    assert result.status == "failed"
    assert result.controller_called
    assert result.controller_decision["approved"] is True
    assert result.controller_decision["quality_score"] == 2.9


def test_rule_validator_does_not_require_formal_theory_headings():
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["theory"]
    content = (
        '<style></style><div class="cc-lesson">'
        "<h2>Как устроена тема</h2>"
        "<p>Содержательный материал без фиксированных названий разделов.</p>"
        "</div>"
    )

    result = RuleValidator().validate_material(content, spec, task)

    assert result.approved


def test_failed_dependency_blocks_dependent_material(tmp_path):
    task = task_for_lesson(3)
    spec = MATERIAL_SPEC_REGISTRY["practice"]
    failed_theory = MaterialResult(
        kind="theory",
        material_type="Материалы занятия — теория",
        agent_type="TheoryMaterialAgent",
        status="failed",
        iterations=3,
        content="",
        prompt_files=MATERIAL_SPEC_REGISTRY["theory"].prompt_files,
        validation_issues=["bad"],
    )

    result = MaterialWorker(client=FakeClient(), config=IsmartGenerationConfig(output_root=tmp_path)).run(
        task=task,
        spec=spec,
        references=ReferenceLoader(IsmartGenerationConfig()).load(task),
        dependency_results=[failed_theory],
    )

    assert result.status == "blocked_dependency"
    assert "dependency theory" in result.validation_issues[0]


def test_run_ismart_task_writes_html_and_manifests(tmp_path):
    task = task_for_lesson(2)
    result = run_ismart_task(
        task,
        IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True),
        client=FakeClient(),
        run_dir=tmp_path / "run",
    )

    assert result.status == "approved"
    assert (tmp_path / "run" / "manifest.json").exists()
    assert (tmp_path / "run" / "result.json").exists()
    assert list((tmp_path / "run").glob("*.html"))
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert "PracticeMaterialAgent" in manifest["agents_called"]
    assert "PackageValidatorAgent" in manifest["agents_called"]
    assert "03_Практика_prompt_skill.md" in manifest["prompt_files_used"]


def test_runtime_manifest_includes_validation_controller_when_called(tmp_path):
    task = task_for_lesson(2)
    result = run_ismart_task(
        task,
        IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True, max_generation_iterations=1),
        client=OverstrictValidatorClient(),
        run_dir=tmp_path / "run",
    )

    assert result.status == "approved"
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert "ValidationControllerAgent" in manifest["agents_called"]
    assert "PackageValidatorAgent" in manifest["agents_called"]


def test_attempt_tmp_stores_raw_clean_and_validation_for_failed_tail(tmp_path):
    task = task_for_lesson(3)
    result = run_ismart_task(
        task,
        IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True, max_generation_iterations=1),
        client=TailClient(),
        run_dir=tmp_path / "run",
    )

    assert result.status == "failed"
    assert [material.kind for material in result.materials] == ["theory"]
    assert "non-HTML tail" in result.materials[0].validation_issues[0]

    attempts_dir = tmp_path / "run" / "tmp" / "theory"
    raw_html = next(attempts_dir.glob("*.raw.html")).read_text(encoding="utf-8")
    clean_html = next(attempts_dir.glob("*__theory.html")).read_text(encoding="utf-8")
    validation = json.loads(next(attempts_dir.glob("*.validation.json")).read_text(encoding="utf-8"))

    assert "Return only JSON" in raw_html
    assert "Return only JSON" not in clean_html
    assert validation["boundary_issues"]
    assert validation["merged_validation"]["approved"] is False


def test_runtime_stops_after_material_exhausts_validation(tmp_path):
    task = task_for_lesson(3)
    client = TailClient()
    result = run_ismart_task(
        task,
        IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True, max_generation_iterations=1),
        client=client,
        run_dir=tmp_path / "run",
    )

    assert result.status == "failed"
    assert [material.kind for material in result.materials] == ["theory"]
    assert client.generator_calls == {"theory": 1}
    assert "execution stopped" in result.package_validation.issues[0]
    assert not (tmp_path / "run" / "tmp" / "practice").exists()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert "MaterialValidatorAgent" in manifest["agents_called"]
    assert "ValidationControllerAgent" not in manifest["agents_called"]
    assert "PackageValidatorAgent" not in manifest["agents_called"]


def test_verbose_mode_prints_trace_events(tmp_path, capsys):
    task = task_for_lesson(2)
    run_ismart_task(
        task,
        IsmartGenerationConfig(output_root=tmp_path, use_llm_validator=True, verbose=True),
        client=FakeClient(),
        run_dir=tmp_path / "run",
    )

    stdout = capsys.readouterr().out
    assert "[ismart-materials] task.start" in stdout
    assert "[ismart-materials] planner.done" in stdout
    assert "[ismart-materials] references.load.file" in stdout
    assert "[ismart-materials] worker.attempt.start" in stdout
    assert "[ismart-materials] output.write.done" in stdout


def test_cli_payload_shapes_and_selectors():
    course = load_course()
    single = task_for_lesson(2)
    from_array = tasks_from_payload([single])
    from_tasks = tasks_from_payload({"tasks": [single]})
    from_course = tasks_from_payload(course)

    assert len(from_array) == 1
    assert len(from_tasks) == 1
    assert len(from_course) == sum(len(module["lessons"]) for module in course["modules"])
    assert filter_tasks(from_course, lesson_number="2")[0]["lesson"]["lesson_number"] == 2
    with pytest.raises(ValueError, match="No tasks matched"):
        filter_tasks(from_course, task_id="missing")
