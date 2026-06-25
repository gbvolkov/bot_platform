from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.ismart_materials_agent.agent import run_ismart_task
from agents.ismart_materials_agent.cli import filter_tasks, tasks_from_payload
from agents.ismart_materials_agent.contracts import IsmartGenerationConfig, MaterialResult, ValidationResult
from agents.ismart_materials_agent.planner import build_material_plan, teacher_material_required
from agents.ismart_materials_agent.registry import MATERIAL_SPEC_REGISTRY
from agents.ismart_materials_agent.sources import ReferenceLoader
from agents.ismart_materials_agent.validators import RuleValidator
from agents.ismart_materials_agent.workers import MaterialWorker


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
    assert "03_Практика_prompt_skill.md" in manifest["prompt_files_used"]


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
