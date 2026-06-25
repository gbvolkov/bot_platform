from __future__ import annotations

import html
import re
from typing import Any

from .contracts import MaterialResult, MaterialSpec, ValidationResult


SERVICE_MARKERS = ("QA-ID", "SHA", "candidate_internal", "leakage", "artifact_id")
SOURCE_EXTENSIONS_FORBIDDEN = (".docx", ".pdf", ".html", ".xlsx", ".xls")


def html_to_text(content: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", content)
    return html.unescape(re.sub(r"\s+", " ", without_tags)).strip()


def headings(content: str, level: int | None = None) -> list[str]:
    if level is None:
        pattern = r"<h([123])[^>]*>(.*?)</h\1>"
    else:
        pattern = rf"<h{level}[^>]*>(.*?)</h{level}>"
    found = []
    for match in re.finditer(pattern, content, re.S | re.I):
        body = match.group(2) if level is None else match.group(1)
        found.append(html.unescape(re.sub(r"<[^>]+>", "", body)).strip())
    return found


def count_regex(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, re.I | re.S))


class RuleValidator:
    def validate_material(
        self,
        content: str,
        spec: MaterialSpec,
        task: dict[str, Any],
    ) -> ValidationResult:
        lesson = task.get("lesson") or {}
        issues: list[str] = []
        text = html_to_text(content)
        h2 = headings(content, 2)

        issues.extend(self._formatting_issues(content))
        issues.extend(self._source_boundary_issues(content, spec.validator_kind))

        if spec.validator_kind != "qa":
            for marker in SERVICE_MARKERS:
                if marker in content:
                    issues.append(f"служебный маркер {marker} найден вне QA")

        if spec.validator_kind == "theory":
            issues.extend(
                self._require_h2(
                    h2,
                    [
                        "Цель занятия",
                        "Задачи занятия",
                        "Ключевые понятия",
                        "Конспект",
                        "Задачи-примеры для разбора",
                        "Типичные ошибки",
                        "Проверка себя",
                        "Итоговые выводы",
                    ],
                )
            )
        elif spec.validator_kind == "practice":
            issues.extend(self._require_h2(h2, ["Цель работы", "Указания по выполнению", "Задания"]))
            expected = self._expected_practice_count(lesson)
            found = count_regex(r"\bP\d+[\.\s]", text)
            if expected and found != expected:
                issues.append(f"практика: ожидалось {expected} задач P1..PN, найдено {found}")
            if "Решение:" in text or "Ключ:" in text or "Ответ для автопроверки" in text:
                issues.append("ученическая практика содержит ключи/решения")
            if "Ожидаемый вывод" not in text or "Вход" not in text:
                issues.append("практика не содержит тест-кейсы вход -> ожидаемый вывод")
        elif spec.validator_kind == "teacher_guidance":
            issues.extend(
                self._require_h2(
                    h2,
                    [
                        "Цель и задачи",
                        "Методическая опора",
                        "Подготовка",
                        "Сценарий",
                        "Ключи и пояснения",
                        "Типичные ошибки и реакция",
                    ],
                )
            )
            if any(term in text for term in ("rc=", "stderr", "stdout")):
                issues.append("МР содержит dev-жаргон rc/stderr/stdout")
        elif spec.validator_kind == "self_study":
            issues.extend(
                self._require_h2(
                    h2,
                    ["Тема", "Цели и задачи", "Порядок выполнения", "Самоконтроль", "Требования к результату", "Источники"],
                )
            )
            task_count = len([item for item in headings(content, 3) if re.match(r"^Задача\s+\d+\b", item)])
            question_count = len([item for item in headings(content, 3) if re.match(r"^Вопрос самоконтроля\s+\d+\b", item)])
            if task_count != 8:
                issues.append(f"самостоятельная: ожидалось 8 задач, найдено {task_count}")
            if question_count != 10:
                issues.append(f"самостоятельная: ожидалось 10 вопросов самоконтроля, найдено {question_count}")
            if any(item.strip().lower() == "краткая теория" for item in headings(content, 2) + headings(content, 3)):
                issues.append("самостоятельная содержит запрещённый раздел краткой теории")
            if "Ключ" not in text and "Ответ" not in text:
                issues.append("самоконтроль не содержит ключи для автопроверки")
        elif spec.validator_kind == "current_control":
            question_count = count_regex(r"\b(?:Вопрос|Задача)\s+\d+[\.\s]", text)
            if question_count != 3:
                issues.append(f"текущий контроль: ожидалось 3 вопроса, найдено {question_count}")
            if "Ключ" not in text and "Ответ для автопроверки" not in text:
                issues.append("текущий контроль не содержит ключи для автопроверки")
        elif spec.validator_kind == "intermediate":
            for index in range(1, 5):
                if f"Комплект {index}" not in h2 and f"Комплект {index}" not in text:
                    issues.append(f"промежуточная: отсутствует комплект {index}")
            if "Ключ" not in text and "Ответ для автопроверки" not in text and "Эталон" not in text:
                issues.append("промежуточная не содержит ключи/эталоны")
        elif spec.validator_kind == "qa":
            issues.extend(self._require_h2(h2, ["Паспорт", "Источники", "Ключи и тесты", "Критерии QA"]))
            if "QA-ID" not in text:
                issues.append("QA не содержит QA-ID")
            if "SHA" not in text:
                issues.append("QA не содержит SHA")
        elif spec.validator_kind == "final_project":
            if "вариант" not in text.lower():
                issues.append("итоговая аттестация не содержит варианты проекта")

        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def validate_package(
        self,
        specs: list[MaterialSpec],
        materials: list[MaterialResult],
    ) -> ValidationResult:
        issues: list[str] = []
        material_by_kind = {item.kind: item for item in materials}
        for spec in specs:
            material = material_by_kind.get(spec.kind)
            if material is None:
                issues.append(f"отсутствует материал: {spec.kind}")
                continue
            if material.status != "approved":
                issues.append(f"материал {spec.kind} имеет статус {material.status}")

        for spec in specs:
            for dependency_kind in spec.dependency_kinds:
                if dependency_kind not in {item.kind for item in specs}:
                    continue
                dependency = material_by_kind.get(dependency_kind)
                material = material_by_kind.get(spec.kind)
                if dependency is None or material is None:
                    continue
                if materials.index(material) < materials.index(dependency):
                    issues.append(f"материал {spec.kind} стоит раньше зависимости {dependency_kind}")

        return ValidationResult.fail(issues) if issues else ValidationResult.ok()

    def _formatting_issues(self, content: str) -> list[str]:
        issues: list[str] = []
        if not content.startswith("<style>"):
            issues.append("HTML не начинается с <style>")
        if '<div class="cc-lesson">' not in content:
            issues.append("нет div.cc-lesson")
        if "<script" in content.lower() or "<link" in content.lower():
            issues.append("HTML содержит script/link")
        return issues

    def _source_boundary_issues(self, content: str, kind: str) -> list[str]:
        text = html_to_text(content)
        issues = []
        for ext in SOURCE_EXTENSIONS_FORBIDDEN:
            if ext in text:
                issues.append(f"материал содержит запрещённую ссылку на исходный формат {ext}")
        if kind != "qa" and "docs/ismart/" in text and "референсы/" in text:
            issues.append("источники/локаторы Markdown попали вне QA")
        return issues

    def _require_h2(self, h2: list[str], required: list[str]) -> list[str]:
        missing = [item for item in required if item not in h2]
        return [f"отсутствует обязательный H2: {item}" for item in missing]

    def _expected_practice_count(self, lesson: dict[str, Any]) -> int:
        difficulty = lesson.get("difficulty") or {}
        l1 = ((difficulty.get("l1") or {}).get("count")) or 0
        l2 = ((difficulty.get("l2") or {}).get("count")) or 0
        if isinstance(l1, int) and isinstance(l2, int) and (l1 + l2):
            return l1 + l2
        tasks = lesson.get("practice_tasks") or {}
        return len(tasks.get("l1") or []) + len(tasks.get("l2") or [])
