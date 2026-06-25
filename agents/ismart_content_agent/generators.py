from __future__ import annotations

import hashlib
import re
from typing import Any

from .contracts import ContentItem, ExecutableSolution, GenerationRequest
from .templates import get_template, validate_interactive_code


def _content_id(request: GenerationRequest) -> str:
    raw = "|".join(
        [
            request.request_id,
            request.course_id,
            request.module_id,
            request.lesson_id,
            request.template_id,
            request.interactive_code or "",
            str(request.lesson_number or ""),
            str(request.target_task_number or ""),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"ismart-{digest}"


def generate_content_item(request: GenerationRequest) -> ContentItem:
    spec = get_template(request.template_id)
    learner_payload, service_payload = _generate_payloads(request)
    content_id = _content_id(request)
    platform_payload = _build_platform_payload(
        request=request,
        content_id=content_id,
        learner_payload=learner_payload,
    )
    return ContentItem(
        content_id=content_id,
        request_id=request.request_id,
        course_id=request.course_id,
        module_id=request.module_id,
        lesson_id=request.lesson_id,
        template_id=request.template_id,
        content_type=spec.content_type,
        title=_title(request),
        audience=request.audience,
        level=request.level,
        requirement_ids=_requirement_ids(request),
        learner_payload=learner_payload,
        service_payload=service_payload,
        platform_payload=platform_payload,
        preview_required=spec.preview_required,
    )


def _generate_payloads(request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    if request.template_id == "theory":
        return _theory_payload(request), {}
    if request.template_id == "practice_python":
        return _practice_payload(request)
    if request.template_id == "self_study":
        return _self_study_payload(request)
    if request.template_id == "control_question":
        return _control_question_payload(request)
    if request.template_id == "interactive_template":
        return _interactive_payload(request)
    raise ValueError(f"Unsupported template_id: {request.template_id}")


def _base_payload(request: GenerationRequest) -> dict[str, Any]:
    return {
        "course_id": request.course_id,
        "module_id": request.module_id,
        "lesson_id": request.lesson_id,
        "topic": request.topic,
        "audience": request.audience,
        "level": request.level,
        "source_refs": request.source_refs,
    }


def _theory_payload(request: GenerationRequest) -> dict[str, Any]:
    task = request.task_spec
    if not task and not request.source_content:
        raise ValueError("theory requires lesson source_content or task_spec from the course tracker")

    return {
        **_base_payload(request),
        "sections": [
            {
                "heading": request.lesson_title or task.get("lesson_title", "Теория"),
                "content": request.source_content or task.get("common_content", ""),
            },
            {
                "heading": f"Материал для ЦА {request.audience}",
                "content": task.get("audience_content", "") if task else "",
            },
        ],
        "summary": [
            request.learning_goal or "Материал связан с практикой и самостоятельной работой занятия.",
            _source_note(task),
        ],
        "source_binding": _source_binding(request, task) if task else {},
    }


def _practice_payload(request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    task = _require_task_spec(request)
    service_payload = _service_payload(request, task)
    learner_payload = {
        **_base_payload(request),
        "module_number": task["module_number"],
        "lesson_number": task["lesson_number"],
        "program_task_level": task["task_level"],
        "program_task_number": task["task_number"],
        "assignment_title": _assignment_title(task),
        "goal": request.learning_goal,
        "condition": task["task_text"],
        "input_data": _input_data_description(task["task_text"]),
        "expected_result": _expected_result_description(task["task_text"]),
        "criteria": [
            "Программа запускается без синтаксических ошибок.",
            "Ввод соответствует условию задания.",
            "Вывод соответствует требуемому результату.",
            "Код написан на Python 3.",
        ],
        "difficulty_level": task["difficulty_level"],
        "assessment_alignment": "Практическое задание по форме таблицы 4; уровень сложности взят из L-маркера программы.",
        "language": "Python 3",
        "source_binding": _source_binding(request, task),
    }
    return learner_payload, service_payload


def _self_study_payload(request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    task = _require_task_spec(request)
    learner_payload = {
        **_base_payload(request),
        "task": task["task_text"],
        "algorithm": [
            "Повторите материалы занятия.",
            "Выполните задание из программы курса.",
            "Проверьте результат по критериям перед отправкой.",
        ],
        "submission_requirements": "Краткий ответ или файл с кодом, если задание требует Python.",
        "source_binding": _source_binding(request, task),
    }
    return learner_payload, {"answer_key": request.service_spec.get("answer_key", "")}


def _control_question_payload(request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    task = _require_task_spec(request)
    learner_payload = {
        **_base_payload(request),
        "question": task["task_text"],
        "answer_format": "Свободный ответ по материалу занятия.",
        "source_binding": _source_binding(request, task),
    }
    return learner_payload, {"answer_key": request.service_spec.get("answer_key", "")}


def _interactive_payload(request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    code = validate_interactive_code(request.interactive_code)
    if not request.template_payload:
        raise ValueError("interactive_template requires template_payload; the agent must not infer template answers")
    learner_payload, answer_key = _split_interactive_payload(code, request.template_payload)
    learner_payload = {
        **_base_payload(request),
        **learner_payload,
        "source_binding": _source_binding(request, request.task_spec) if request.task_spec else {},
    }
    return learner_payload, {"answer_key": answer_key}


def _split_interactive_payload(code: str, template_payload: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    payload = dict(template_payload)
    prompt = payload.get("prompt") or payload.get("text") or "Выполните интерактивное задание."
    if code == "6A":
        source_items = payload.get("items", [])
        items = [{k: v for k, v in item.items() if k != "order"} for item in source_items]
        answer_key = [
            item["id"]
            for item in sorted(source_items, key=lambda item: item.get("order", 0))
            if "id" in item
        ]
        return {"template_code": code, "type": "ordering", "prompt": prompt, "items": items}, answer_key
    if code == "6D":
        source_items = payload.get("items", [])
        items = [{k: v for k, v in item.items() if k != "correct_column_id"} for item in source_items]
        answer_key = {
            item["id"]: item["correct_column_id"]
            for item in source_items
            if "id" in item and "correct_column_id" in item
        }
        return {
            "template_code": code,
            "type": "columns",
            "prompt": prompt,
            "columns": payload.get("columns", []),
            "items": items,
        }, answer_key
    if code == "6G":
        pairs = payload.get("pairs", [])
        return {
            "template_code": code,
            "type": "matching",
            "prompt": prompt,
            "left": [{"id": f"l{index}", "text": pair.get("left", "")} for index, pair in enumerate(pairs, start=1)],
            "right": [{"id": f"r{index}", "text": pair.get("right", "")} for index, pair in enumerate(pairs, start=1)],
        }, {f"l{index}": f"r{index}" for index, _ in enumerate(pairs, start=1)}
    if code == "8D":
        return {
            "template_code": code,
            "type": "connect",
            "prompt": prompt,
            "left": payload.get("left", []),
            "right": payload.get("right", []),
        }, payload.get("answers", [])
    if code == "10D":
        return {
            "template_code": code,
            "type": "highlight",
            "prompt": prompt,
            "text": payload.get("text", ""),
        }, payload.get("answers", [])

    text, answers = _strip_input_answers(payload.get("text", ""))
    return {
        "template_code": code,
        "type": "input_text",
        "prompt": prompt,
        "text": text,
    }, payload.get("answers") or answers


def _strip_input_answers(text: str) -> tuple[str, list[str]]:
    answers: list[str] = []

    def replace(match: re.Match[str]) -> str:
        raw_answers = [part for part in match.group("answers").split(":") if part]
        for answer_group in raw_answers:
            answers.extend([answer for answer in answer_group.split(",") if answer])
        return "{{input-text}}"

    stripped = re.sub(r"\{\{input-text:(?P<answers>[^}]+)\}\}", replace, text)
    return stripped, answers


def _service_payload(request: GenerationRequest, task: dict[str, Any]) -> dict[str, Any]:
    if request.service_spec:
        if "service_solution" in request.service_spec and "tests" in request.service_spec:
            raw_solution = request.service_spec["service_solution"]
            solution = (
                ExecutableSolution(code=raw_solution).model_dump(mode="json")
                if isinstance(raw_solution, str)
                else ExecutableSolution.model_validate(raw_solution).model_dump(mode="json")
            )
            return {
                "service_solution": solution,
                "tests": request.service_spec["tests"],
            }
    return _infer_service_payload(task["task_text"])


def _infer_service_payload(task_text: str) -> dict[str, Any]:
    text = task_text.lower()
    if "удво" in text:
        return {
            "service_solution": _executable_solution(
                "def main():\n"
                "    a = int(input())\n"
                "    print(a * 2)\n\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
            ),
            "tests": [{"test_id": "source-task-1", "stdin": "7\n", "expected_stdout": "14\n"}],
        }
    if "два числа" in text and ("сумм" in text or "слож" in text):
        return {
            "service_solution": _executable_solution(
                "def main():\n"
                "    a = int(input())\n"
                "    b = int(input())\n"
                "    print(a + b)\n\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
            ),
            "tests": [{"test_id": "source-task-1", "stdin": "3\n4\n", "expected_stdout": "7\n"}],
        }
    if "возраст" in text and "через 5" in text:
        return {
            "service_solution": _executable_solution(
                "def main():\n"
                "    age = int(input())\n"
                "    print(age + 5)\n\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
            ),
            "tests": [{"test_id": "source-task-1", "stdin": "14\n", "expected_stdout": "19\n"}],
        }
    if "минут" in text and "час" in text:
        return {
            "service_solution": _executable_solution(
                "def main():\n"
                "    minutes = int(input())\n"
                "    print(f\"{minutes // 60} часа {minutes % 60} минут\")\n\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
            ),
            "tests": [{"test_id": "source-task-1", "stdin": "130\n", "expected_stdout": "2 часа 10 минут\n"}],
        }
    raise ValueError(
        "Cannot build hidden Python service solution for this source task; provide service_spec explicitly"
    )


def _executable_solution(code: str) -> dict[str, Any]:
    return ExecutableSolution(code=code).model_dump(mode="json")


def _require_task_spec(request: GenerationRequest) -> dict[str, Any]:
    if not request.task_spec:
        raise ValueError(f"{request.template_id} requires task_spec resolved from the course tracker")
    return request.task_spec


def _assignment_title(task: dict[str, Any]) -> str:
    return f"{task['lesson_title']}: задание {task['task_level']}-{task['task_number']}"


def _input_data_description(task_text: str) -> str:
    text = task_text.lower()
    if "два числа" in text:
        return "Два числа, каждое с новой строки."
    if "три числа" in text:
        return "Три числа, каждое с новой строки."
    if "минут" in text:
        return "Одно целое число: количество минут."
    if "возраст" in text:
        return "Одно целое число: возраст."
    if "число" in text:
        return "Одно число."
    return "Входные данные берутся из условия задания."


def _expected_result_description(task_text: str) -> str:
    text = task_text.lower()
    if "вывести" in text:
        return "Программа выводит результат, указанный в условии задания."
    return "Результат соответствует условию задания."


def _source_binding(request: GenerationRequest, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "course_tracker": {
            "document": task.get("source_document"),
            "sheet": task.get("sheet_name"),
            "row_index": task.get("row_index"),
            "lesson_number": task.get("lesson_number"),
            "task_level": task.get("task_level"),
            "task_number": task.get("task_number"),
            "source_task_text": task.get("task_text"),
        },
        "donors": [donor.model_dump() for donor in request.donor_sources],
    }


def _source_note(task: dict[str, Any] | None) -> str:
    if not task:
        return "Источник содержания передан во входном запросе."
    return f"Источник содержания: {task['source_document']}, лист {task['sheet_name']}, строка {task['row_index']}."


def _title(request: GenerationRequest) -> str:
    if request.lesson_title and request.topic:
        return f"{request.lesson_title}: {request.topic}"
    return request.lesson_title or request.topic or request.template_id


def _build_platform_payload(
    *,
    request: GenerationRequest,
    content_id: str,
    learner_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "content_id": content_id,
        "template_id": request.template_id,
        "interactive_code": request.interactive_code,
        "payload": learner_payload,
    }


def _requirement_ids(request: GenerationRequest) -> list[str]:
    base = ["FR-A05a"]
    mapping = {
        "theory": ["FR-THEORY01"],
        "practice_python": ["FR-PRACT07", "FR-PRACT08", "FR-PRACT13", "FR-RUN01"],
        "self_study": ["FR-SR01"],
        "control_question": ["FR-CUR01"],
        "interactive_template": ["FR-TPL02", "FR-TPL03", "FR-EXP10"],
    }
    return [*base, *mapping[request.template_id]]
