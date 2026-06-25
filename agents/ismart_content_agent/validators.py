from __future__ import annotations

import json
import re
from typing import Any

from .contracts import (
    ContentItem,
    SandboxResult,
    ValidationCriterion,
    ValidationReport,
    Violation,
)
from .sandbox import run_sandbox
from .templates import get_template


SECRET_FIELD_NAMES = {
    "service_payload",
    "service_solution",
    "answer_key",
    "expected_stdout",
    "hidden_tests",
    "tests",
}

V_MATRIX: dict[str, tuple[str, str]] = {
    "V1": ("модуль", "Количество практических заданий: базовый >=6, продвинутый >=10"),
    "V2": ("модуль", "Доля заданий L1 <=30%"),
    "V3": ("задание/модуль", "Базовый уровень допускает {1,2}, продвинутый {1,2,3}"),
    "V4": ("модуль", "В модуле представлены минимум два уровня сложности"),
    "V5": ("задание", "Практическое задание содержит действие с измеримым результатом"),
    "V6": ("ДОП", "Практика занимает >=50% часов ДОП"),
    "V7": ("задание", "Заполнены все поля таблицы 4"),
    "V8": ("ДОП", "Есть минимум один проект с работающим продуктом"),
    "V9": ("модуль", "Самостоятельных работ >=7"),
    "V10": ("модуль", "Заданий самоконтроля >=5"),
    "V11": ("модуль", "Заданий текущего контроля >=10"),
    "V12": ("вариант", "Промежуточная аттестация: >=15 вопросов, >=1 практика, >=20% открытых"),
    "V13": ("ДОП", "Итоговый контроль содержит >=7 вариантов"),
    "V14": ("модуль Python", "Кодовых заданий в промежуточной аттестации >=50%"),
    "V15": ("ДОП Python", "Заданий на алгоритмизацию >=5"),
    "V16": ("ДОП Python", "Тем про ИИ-инструмент >=2"),
    "V17": ("ДОП Python", "Заданий исправления/улучшения ИИ-кода >=2"),
    "V18": ("УМК Python", "Политика ИИ содержит разрешения, запреты и последствия"),
    "V19": ("ДОП ИИ базовый", ">=50% тем про принципы алгоритмов/моделей ИИ"),
    "V20": ("ДОП ИИ продвинутый", ">=50% тем про математику/архитектуры ИИ"),
    "V21": ("ДОП ИИ", "Доля тем только про генеративные сервисы не превышает лимит"),
    "V22": ("ДОП ИИ", "Используются минимум два реальных датасета"),
    "V23": ("ДОП ИИ", "Есть минимум один полный цикл ML"),
    "V24": ("ДОП", "Общий объём >=144 академических часов"),
}


def validate_items(items: list[ContentItem]) -> ValidationReport:
    request_id = items[0].request_id if items else "unknown"
    criteria: list[ValidationCriterion] = []
    sandbox_results: list[SandboxResult] = []

    for item in items:
        item_criteria = _base_criteria(item)
        if item.template_id == "practice_python":
            item_criteria.extend(_practice_criteria(item))
        elif item.template_id == "interactive_template":
            item_criteria.extend(_interactive_criteria(item))
        elif item.template_id == "control_question":
            item_criteria.extend(_control_criteria(item))

        sandbox_result = run_sandbox(item)
        sandbox_results.append(sandbox_result)
        item_criteria.append(_sandbox_criterion(item, sandbox_result))
        item_criteria.extend(_v_matrix_criteria(item, item_criteria))
        criteria.extend(item_criteria)

    violations = [
        Violation(
            code=criterion.criterion_id,
            content_id=criterion.content_id,
            message=criterion.message,
            severity=criterion.severity,
        )
        for criterion in criteria
        if criterion.status == "failed"
    ]
    failed = any(c.status == "failed" and c.severity == "error" for c in criteria)
    return ValidationReport(
        request_id=request_id,
        status="failed" if failed else "passed",
        criteria=criteria,
        violations=violations,
        sandbox_results=sandbox_results,
    )


def _base_criteria(item: ContentItem) -> list[ValidationCriterion]:
    criteria: list[ValidationCriterion] = []
    spec = get_template(item.template_id)
    requirement = item.requirement_ids[0] if item.requirement_ids else "FR-TRACE"
    for field in spec.required_fields:
        value = item.learner_payload.get(field)
        criteria.append(
            _criterion(
                item,
                criterion_id=f"SCHEMA_REQUIRED_{field.upper()}",
                requirement_id=requirement,
                category="schema",
                description=f"Обязательное поле learner_payload.{field} заполнено",
                expected="непустое значение",
                actual=_safe_actual(value),
                passed=not _is_blank(value),
                message=f"Поле {field} {'заполнено' if not _is_blank(value) else 'не заполнено'}.",
            )
        )

    for field in ["course_id", "module_id", "lesson_id", "topic", "audience", "level"]:
        value = item.learner_payload.get(field)
        criteria.append(
            _criterion(
                item,
                criterion_id=f"CONTEXT_{field.upper()}",
                requirement_id="FR-IN01",
                category="context",
                description=f"Контекстное поле {field} присутствует",
                expected="непустое значение",
                actual=_safe_actual(value),
                passed=not _is_blank(value),
                message=f"Контекст {field} {'присутствует' if not _is_blank(value) else 'отсутствует'}.",
            )
        )

    criteria.append(
        _criterion(
            item,
            criterion_id="TRACE_REQUIREMENT_IDS",
            requirement_id="FR-TRACE01",
            category="traceability",
            description="Материал содержит ссылки на требования",
            expected="минимум один requirement_id",
            actual=item.requirement_ids,
            passed=bool(item.requirement_ids),
            message="Трассировка требований заполнена." if item.requirement_ids else "Трассировка требований отсутствует.",
        )
    )
    leaks = sorted(find_secret_leaks(item))
    criteria.append(
        _criterion(
            item,
            criterion_id="SECURITY_NO_SERVICE_LEAK",
            requirement_id="FR-A05a",
            category="security",
            description="Learner/platform payload не содержит служебных решений и ключей",
            expected=[],
            actual=leaks,
            passed=not leaks,
            message="Служебные данные не обнаружены." if not leaks else f"Обнаружены служебные поля: {', '.join(leaks)}.",
        )
    )
    preview_scope_ok = not item.preview_required or item.template_id == "interactive_template"
    criteria.append(
        _criterion(
            item,
            criterion_id="PREVIEW_SCOPE",
            requirement_id="FR-TPL08",
            category="preview",
            description="Preview требуется только для интерактивного шаблона",
            expected=True,
            actual=preview_scope_ok,
            passed=preview_scope_ok,
            message="Preview-конфигурация корректна." if preview_scope_ok else "Preview включён для неподдерживаемого типа.",
        )
    )
    return criteria


def _practice_criteria(item: ContentItem) -> list[ValidationCriterion]:
    payload = item.learner_payload
    source_binding = payload.get("source_binding")
    tracker = source_binding.get("course_tracker", {}) if isinstance(source_binding, dict) else {}
    donors = source_binding.get("donors", []) if isinstance(source_binding, dict) else []
    source_present = bool(tracker)
    condition_matches = source_present and tracker.get("source_task_text") == payload.get("condition")
    difficulty = payload.get("difficulty_level")
    program_level = payload.get("program_task_level")
    expected_difficulty = {"L1": 1, "L2": 2, "L3": 3}.get(program_level)
    condition = str(payload.get("condition", ""))
    raw_solution = item.service_payload.get("service_solution")
    solution_shape_ok = (
        isinstance(raw_solution, dict)
        and raw_solution.get("language") == "python"
        and raw_solution.get("runtime") == "python3"
        and bool(raw_solution.get("entrypoint"))
        and bool(raw_solution.get("code"))
    )
    code = raw_solution.get("code", "") if isinstance(raw_solution, dict) else ""
    script_complete = "if __name__ == \"__main__\":" in code and "main()" in code
    tests = item.service_payload.get("tests")

    return [
        _criterion(item, "PRACT_SOURCE_BINDING", "FR-IN01", "source", "Задание привязано к строке программы курса", "course_tracker binding", _safe_actual(tracker), source_present, "Источник программы найден." if source_present else "Источник программы отсутствует."),
        _criterion(item, "PRACT_SOURCE_TASK_MATCH", "FR-IN01", "source", "Условие совпадает с выбранным заданием программы", tracker.get("source_task_text"), payload.get("condition"), condition_matches, "Условие совпадает с источником." if condition_matches else "Условие изменено относительно источника."),
        _criterion(item, "PRACT_DONOR_BINDING", "FR-DON01", "source", "Для занятия определены доноры или режим «ваше»", "минимум одна запись", _safe_actual(donors), bool(donors), "Донорская привязка найдена." if donors else "Донорская привязка отсутствует."),
        _criterion(item, "PRACT_DIFFICULTY_ENUM", "V3", "rule", "Уровень сложности входит в допустимый enum", [1, 2, 3], difficulty, difficulty in {1, 2, 3}, "Уровень сложности допустим." if difficulty in {1, 2, 3} else "Недопустимый уровень сложности."),
        _criterion(item, "PRACT_DIFFICULTY_MATCH", "V3", "rule", "Уровень сложности совпадает с L-маркером программы", expected_difficulty, difficulty, expected_difficulty == difficulty, "Уровень соответствует L-маркеру." if expected_difficulty == difficulty else "Уровень не соответствует L-маркеру."),
        _criterion(item, "PRACT_NO_PLACEHOLDER", "FR-A05a", "content", "Условие взято из программы, а не создано из topic-placeholder", False, condition.startswith("Напишите программу на Python по теме"), not condition.startswith("Напишите программу на Python по теме"), "Placeholder отсутствует." if not condition.startswith("Напишите программу на Python по теме") else "Обнаружен placeholder задания."),
        _criterion(item, "PRACT_SERVICE_SOLUTION_OBJECT", "FR-PRACT08", "service", "Эталонное решение имеет исполняемую структуру", {"language": "python", "runtime": "python3", "entrypoint": "main.py", "code": "non-empty"}, _solution_actual(raw_solution), solution_shape_ok, "Структура эталона корректна." if solution_shape_ok else "Структура эталона некорректна."),
        _criterion(item, "PRACT_SERVICE_SCRIPT_COMPLETE", "FR-PRACT08", "service", "Код является самостоятельным Python-скриптом", "main() и __main__ guard", {"main_guard": script_complete}, script_complete, "Эталон можно выполнить напрямую." if script_complete else "Эталон не является полным исполняемым скриптом."),
        _criterion(item, "PRACT_HIDDEN_TESTS", "FR-PRACT13", "service", "Для эталона заданы stdin/stdout тесты", "непустой список тестов", {"test_count": len(tests) if isinstance(tests, list) else 0}, isinstance(tests, list) and bool(tests), "Скрытые тесты заданы." if isinstance(tests, list) and tests else "Скрытые тесты отсутствуют."),
    ]


def _interactive_criteria(item: ContentItem) -> list[ValidationCriterion]:
    payload = item.learner_payload
    code = payload.get("template_code")
    template_type = payload.get("type")
    expected_types = {"6A": "ordering", "6D": "columns", "6G": "matching", "8D": "connect", "10D": "highlight", "3H": "input_text", "3D": "input_text"}
    supported = code in expected_types
    criteria = [
        _criterion(item, "TPL_CODE_SUPPORTED", "FR-TPL02", "template", "Код интерактивного шаблона поддерживается", list(expected_types), code, supported, "Код шаблона поддерживается." if supported else "Код шаблона не поддерживается."),
        _criterion(item, "TPL_TYPE_MATCH", "FR-TPL03", "template", "Тип payload соответствует коду шаблона", expected_types.get(code), template_type, supported and expected_types.get(code) == template_type, "Тип payload корректен." if supported and expected_types.get(code) == template_type else "Тип payload не соответствует коду."),
    ]
    if code == "6D":
        columns = payload.get("columns", [])
        criteria.append(_criterion(item, "TPL_6D_COLUMNS", "FR-TPL04", "template", "6D содержит от 2 до 6 колонок", "2..6", len(columns) if isinstance(columns, list) else None, isinstance(columns, list) and 2 <= len(columns) <= 6, "Количество колонок допустимо." if isinstance(columns, list) and 2 <= len(columns) <= 6 else "Количество колонок недопустимо."))
    if code == "8D":
        left = payload.get("left", [])
        right = payload.get("right", [])
        equal = isinstance(left, list) and isinstance(right, list) and len(left) == len(right)
        within_limit = isinstance(left, list) and 2 <= len(left) <= 4
        criteria.append(_criterion(item, "TPL_8D_EQUAL_COUNTS", "FR-TPL06", "template", "8D имеет одинаковое число вариантов слева и справа", "equal counts", {"left": len(left) if isinstance(left, list) else None, "right": len(right) if isinstance(right, list) else None}, equal, "Количество вариантов совпадает." if equal else "Количество вариантов не совпадает."))
        criteria.append(_criterion(item, "TPL_8D_LIMIT", "FR-TPL06", "template", "8D содержит от 2 до 4 вариантов", "2..4", len(left) if isinstance(left, list) else None, within_limit, "Лимит вариантов соблюдён." if within_limit else "Лимит вариантов нарушен."))
    if code in {"3H", "3D"}:
        leaked = "{{input-text:" in str(payload.get("text", ""))
        criteria.append(_criterion(item, "TPL_INPUT_NO_ANSWER_LEAK", "FR-TPL07", "security", "Learner payload не содержит ответы внутри input-text", False, leaked, not leaked, "Ответы скрыты." if not leaked else "Ответы присутствуют в learner payload."))
    answer_key = item.service_payload.get("answer_key")
    criteria.append(_criterion(item, "TPL_HIDDEN_ANSWER_KEY", "FR-TPL03", "service", "Правильный ответ хранится в service_payload", "непустой answer_key", _safe_actual(answer_key), not _is_blank(answer_key), "Скрытый ключ задан." if not _is_blank(answer_key) else "Скрытый ключ отсутствует."))
    return criteria


def _control_criteria(item: ContentItem) -> list[ValidationCriterion]:
    answer_key = item.service_payload.get("answer_key")
    return [
        _criterion(
            item,
            "CONTROL_HIDDEN_ANSWER_KEY",
            "FR-CUR03",
            "service",
            "Контрольный вопрос содержит скрытый эталон ответа",
            "непустой answer_key",
            _safe_actual(answer_key),
            not _is_blank(answer_key),
            "Скрытый эталон ответа задан." if not _is_blank(answer_key) else "Скрытый эталон ответа отсутствует.",
        )
    ]


def _sandbox_criterion(item: ContentItem, result: SandboxResult) -> ValidationCriterion:
    if item.template_id != "practice_python":
        return ValidationCriterion(
            criterion_id="PRACT_SANDBOX_EXECUTION",
            content_id=item.content_id,
            requirement_id="FR-RUN01",
            category="sandbox",
            description="Эталонное Python-решение проходит sandbox",
            expected="pass",
            actual=result.status,
            status="not_applicable",
            message="Sandbox не применяется к этому типу материала.",
        )
    passed = result.status == "pass"
    return _criterion(item, "PRACT_SANDBOX_EXECUTION", "FR-RUN01", "sandbox", "Эталонное Python-решение выполняется напрямую и проходит все тесты", "pass", result.status, passed, "Sandbox пройден." if passed else f"Sandbox не пройден: {result.reason or result.status}.")


def _v_matrix_criteria(item: ContentItem, current: list[ValidationCriterion]) -> list[ValidationCriterion]:
    results: list[ValidationCriterion] = []
    required_fields = [c for c in current if c.criterion_id.startswith("SCHEMA_REQUIRED_")]
    difficulty = item.learner_payload.get("difficulty_level")
    allowed = {1, 2} if item.level == "базовый" else {1, 2, 3}
    action_text = str(item.learner_payload.get("condition", "")).lower()
    measurable_action = bool(re.search(r"\b(ввести|вывести|написать|создать|реализовать|исправить|добавить|найти|рассчитать|преобразовать)\b", action_text)) and not _is_blank(item.learner_payload.get("expected_result"))

    for rule_id, (scope, description) in V_MATRIX.items():
        if rule_id == "V3" and item.template_id == "practice_python":
            results.append(_criterion(item, rule_id, rule_id, "V-matrix", description, sorted(allowed), difficulty, difficulty in allowed, "V3 выполнен." if difficulty in allowed else "V3 нарушен."))
        elif rule_id == "V5" and item.template_id == "practice_python":
            results.append(_criterion(item, rule_id, rule_id, "V-matrix", description, True, {"action_detected": measurable_action, "expected_result_present": not _is_blank(item.learner_payload.get("expected_result"))}, measurable_action, "V5 выполнен." if measurable_action else "V5 нарушен."))
        elif rule_id == "V7" and item.template_id == "practice_python":
            passed = bool(required_fields) and all(c.status == "passed" for c in required_fields)
            results.append(_criterion(item, rule_id, rule_id, "V-matrix", description, "все поля таблицы 4 заполнены", {"passed": sum(c.status == "passed" for c in required_fields), "total": len(required_fields)}, passed, "V7 выполнен." if passed else "V7 нарушен."))
        else:
            results.append(
                ValidationCriterion(
                    criterion_id=rule_id,
                    content_id=item.content_id,
                    requirement_id=rule_id,
                    category="V-matrix",
                    description=description,
                    expected=f"Проверка на scope: {scope}",
                    actual="В run присутствует одно задание, агрегат scope недоступен",
                    status="not_applicable",
                    severity="warning" if rule_id in {"V4", "V16", "V17", "V18", "V22", "V23"} else "error",
                    message=f"{rule_id} не применяется на уровне одного ContentItem; требуется пакет scope «{scope}».",
                )
            )
    return results


def _criterion(
    item: ContentItem,
    criterion_id: str,
    requirement_id: str,
    category: str,
    description: str,
    expected: Any,
    actual: Any,
    passed: bool,
    message: str,
    severity: str = "error",
) -> ValidationCriterion:
    return ValidationCriterion(
        criterion_id=criterion_id,
        content_id=item.content_id,
        requirement_id=requirement_id,
        category=category,
        description=description,
        expected=expected,
        actual=actual,
        status="passed" if passed else "failed",
        severity=severity,
        message=message,
    )


def _safe_actual(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_actual(child) for key, child in value.items() if key not in SECRET_FIELD_NAMES}
    if isinstance(value, list):
        return [_safe_actual(child) for child in value]
    return value


def _solution_actual(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    return {
        "language": value.get("language"),
        "runtime": value.get("runtime"),
        "entrypoint": value.get("entrypoint"),
        "code_present": bool(value.get("code")),
    }


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) == 0
    return False


def find_secret_leaks(item: ContentItem) -> set[str]:
    haystacks = [item.learner_payload, item.platform_payload]
    leaks: set[str] = set()
    for payload in haystacks:
        leaks.update(_find_secret_keys(payload))
    return leaks


def assert_publishable(items: list[ContentItem], report: ValidationReport | None = None) -> None:
    if report is None:
        raise ValueError("Cannot publish run without passed validation")
    if report.status != "passed":
        raise ValueError("Cannot publish run with failed validation")
    for item in items:
        if item.status != "одобрено":
            raise ValueError(f"Content item {item.content_id} is not approved")
        if item.preview_required and item.preview_status != "passed":
            raise ValueError(f"Interactive content item {item.content_id} has not passed preview")
        leaks = find_secret_leaks(item)
        if leaks:
            raise ValueError(f"Content item {item.content_id} leaks service-only data: {', '.join(sorted(leaks))}")


def _find_secret_keys(value: Any) -> set[str]:
    leaks: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in SECRET_FIELD_NAMES:
                leaks.add(key)
            leaks.update(_find_secret_keys(child))
    elif isinstance(value, list):
        for child in value:
            leaks.update(_find_secret_keys(child))
    elif isinstance(value, str):
        for key in SECRET_FIELD_NAMES:
            if key in value:
                leaks.add(key)
        if value.strip().startswith("{"):
            try:
                leaks.update(_find_secret_keys(json.loads(value)))
            except json.JSONDecodeError:
                pass
    return leaks
