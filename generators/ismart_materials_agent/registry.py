from __future__ import annotations

from dataclasses import replace

from .contracts import MaterialSpec


COMMON_PROMPT = "01_Общее_prompt_skill.md"
FORMAT_PROMPT = "08_Форматирование_заданий_курса_prompt.md"
SKILL_MAP_PROMPT = "91_skill_map.md"
JSON_DESCRIPTION_PROMPT = "92_описание_json.md"

THEORY_PROMPT = "02_Теория_prompt_skill.md"
PRACTICE_PROMPT = "03_Практика_prompt_skill.md"
SELF_WORK_PROMPT = "04_Самостоятельная_prompt_skill.md"
INTERMEDIATE_PROMPT = "05_Промежуточная_prompt_skill.md"
FINAL_PROJECT_PROMPT = "06_Итоговая_prompt_skill.md"
TEACHER_GUIDANCE_PROMPT = "07_Методические_указания_prompt_skill.md"

REFERENCE_FIELDS = (
    "requirements",
    "reference_examples",
    "goals_and_tasks",
    "donor_materials",
    "template_descriptions",
)

def _files(*type_specific: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                COMMON_PROMPT,
                *type_specific,
                FORMAT_PROMPT,
                SKILL_MAP_PROMPT,
                JSON_DESCRIPTION_PROMPT,
            )
        )
    )


MATERIAL_SPEC_REGISTRY: dict[str, MaterialSpec] = {
    "theory": MaterialSpec(
        kind="theory",
        material_type="Материалы занятия — теория",
        agent_type="TheoryMaterialAgent",
        prompt_files=_files(THEORY_PROMPT),
        validator_kind="theory",
        reference_fields=("requirements", "reference_examples", "goals_and_tasks", "donor_materials"),
        json_field_labels=("course", "module", "lesson.topic/title", "lesson.hours.theory", "lesson.content", "lesson.difficulty"),
        prompt_addendum=(
            "Создай ученический материал теории. Содержательно покрой цель занятия, задачи, "
            "ключевые понятия, объяснение темы, примеры для разбора, типичные ошибки, "
            "важные моменты, итоговые выводы и мостик к практике. Не добавляй learner-facing "
            "самопроверку, вопросы для обсуждения, предсказание вывода или задания ученику. "
            "Все примеры должны быть полностью разобранными демонстрациями, а не упражнениями. "
            "Названия разделов выбирай по смыслу и по "
            "reference_examples. Не добавляй ключи практики, QA-ID или SHA. Если в JSON есть "
            "lesson.practice_tasks, используй их только как границы темы: не превращай их в "
            "готовые решённые примеры с полным кодом и выводом. Для демонстраций бери другие "
            "нейтральные значения/имена переменных или показывай неполный разбор без финального ответа."
        ),
    ),
    "practice": MaterialSpec(
        kind="practice",
        material_type="Материалы занятия — практика",
        agent_type="PracticeMaterialAgent",
        prompt_files=_files(PRACTICE_PROMPT),
        validator_kind="practice",
        dependency_kinds=("theory",),
        reference_fields=("requirements", "reference_examples", "goals_and_tasks", "donor_materials"),
        json_field_labels=("course", "module", "lesson.practice_tasks", "lesson.difficulty", "lesson.content", "lesson.hours.practice"),
        prompt_addendum=(
            "Создай ученическую практику строго по SOURCE CONTRACT FROM JSON. lesson.practice_tasks/source_contract.tasks "
            "задают авторитетный P id, уровень, тип и паттерн задания, но не обязательно финальный текст. "
            "Для каждого задания явно покажи уровень L1/L2/L3, условие для ученика, входные данные, "
            "требование к выводу и тесты. Не показывай source_text/source contract/JSON wording. Конкретный сценарий, "
            "значения, имена переменных, входы/выходы и код должны быть новым вариантом того же паттерна. "
            "Если deterministic runtime tests source-supported, дай минимум 3 пары stdin→stdout. Если исходное условие не задаёт "
            "достаточно данных для точного stdout-теста, явно напиши, что эталон автопроверки требует уточнения источника, "
            "и не подставляй вымышленные значения; при этом сохрани для такого задания поля «Как проверить» и «Тесты» "
            "со статусом «не заданы/не применимы до уточнения». Для такого задания не делай таблицу или строку "
            "«вход → ожидаемый вывод» и не ставь заглушки в колонку ожидаемого вывода. Исправленный код, ответы и ключи не показывай."
        ),
    ),
    "mr_theory": MaterialSpec(
        kind="mr_theory",
        material_type="МР-теория",
        agent_type="TeacherGuidanceAgent",
        prompt_files=_files(THEORY_PROMPT, TEACHER_GUIDANCE_PROMPT),
        validator_kind="teacher_guidance",
        dependency_kinds=("theory",),
        reference_fields=("requirements", "reference_examples", "goals_and_tasks", "donor_materials"),
        json_field_labels=("teacher_materials.theory", "lesson.hours.theory", "lesson.content"),
        prompt_addendum=(
            "Создай МР-теорию для учителя-неэксперта. Не пересказывай ученический материал. "
            "Содержательно покрой цели и задачи, методическую опору, подготовку, сценарий, "
            "ключи и пояснения, типичные ошибки и реакцию учителя. Названия разделов "
            "выбирай по смыслу и по reference_examples."
        ),
    ),
    "mr_practice": MaterialSpec(
        kind="mr_practice",
        material_type="МР-практика",
        agent_type="TeacherGuidanceAgent",
        prompt_files=_files(PRACTICE_PROMPT, TEACHER_GUIDANCE_PROMPT),
        validator_kind="teacher_guidance",
        dependency_kinds=("practice",),
        reference_fields=("requirements", "reference_examples", "goals_and_tasks", "donor_materials"),
        json_field_labels=("teacher_materials.practice", "lesson.practice_tasks", "lesson.difficulty", "lesson.hours.practice"),
        prompt_addendum=(
            "Создай МР-практику. Включи ключи и пояснения ко всем задачам P1..PN. "
            "Критерии проверки формулируй словами, без rc/stderr/stdout."
        ),
    ),
    "self_work": MaterialSpec(
        kind="self_work",
        material_type="Материалы занятия — самостоятельная работа",
        agent_type="SelfStudyAgent",
        prompt_files=_files(SELF_WORK_PROMPT),
        validator_kind="self_study",
        dependency_kinds=("theory", "practice"),
        reference_fields=("requirements", "reference_examples", "template_descriptions"),
        json_field_labels=("lesson.content", "lesson.hours.self_study", "lesson.practice_tasks", "lesson.difficulty"),
        prompt_addendum=(
            "Создай самостоятельную работу. Содержательно покрой тему, цели и задачи, порядок выполнения, "
            "самоконтроль, требования к результату и источники. Названия разделов выбирай по смыслу "
            "и по reference_examples. Раздел краткой теории запрещён. "
            "Нужны ровно 8 практических задач и ровно 10 вопросов самоконтроля. Ключи и настройки "
            "автопроверки должны быть internal/QA data, а не видимыми learner-facing ответами."
        ),
    ),
    "current_control": MaterialSpec(
        kind="current_control",
        material_type="Текущий контроль",
        agent_type="CurrentControlAgent",
        prompt_files=_files(),
        validator_kind="current_control",
        reference_fields=("template_descriptions", "requirements", "reference_examples"),
        json_field_labels=("lesson.content", "lesson.difficulty", "lesson.hours.raw"),
        prompt_addendum=(
            "Render a separate learner-facing current_control material from "
            "GENERATION ARTIFACTS FOR THIS MATERIAL.current_control_autocheck when that artifact is available. "
            "Use exactly the 3 structured questions from the artifact, preserve their ids/order, "
            "and show only student_prompt, options, template/question type, and answer-format guidance. "
            "Do not show correct_answers, answer flags, autocheck_config, internal_explanation, QA-ID, SHA, "
            "or any source/service locator. Internal keys for autocheck remain only in the generation artifact."
        ),
    ),
    "intermediate": MaterialSpec(
        kind="intermediate",
        material_type="Промежуточная аттестация",
        agent_type="IntermediateAssessmentAgent",
        prompt_files=_files(INTERMEDIATE_PROMPT),
        validator_kind="intermediate",
        reference_fields=("requirements", "template_descriptions", "goals_and_tasks"),
        json_field_labels=("course", "module.lessons", "module.hours/l1/l2/totals", "attestation lesson"),
        prompt_addendum=(
            "Создай промежуточную аттестацию уровня модуля по v34: 4 комплекта. В каждом комплекте "
            "5 тестовых вопросов, 5 открытых-код вопросов и 5 практико-ориентированных задач на код. "
            "Открытые-код вопросы засчитываются только если ученик пишет исполняемый код с проверяемым результатом."
        ),
    ),
    "mr_intermediate": MaterialSpec(
        kind="mr_intermediate",
        material_type="МР к промежуточной аттестации",
        agent_type="TeacherGuidanceAgent",
        prompt_files=_files(INTERMEDIATE_PROMPT, TEACHER_GUIDANCE_PROMPT),
        validator_kind="teacher_guidance",
        dependency_kinds=("intermediate",),
        reference_fields=("requirements", "template_descriptions", "goals_and_tasks"),
        json_field_labels=("module", "attestation lesson"),
        prompt_addendum="Создай методические указания к готовой промежуточной аттестации.",
    ),
    "specification_qa": MaterialSpec(
        kind="specification_qa",
        material_type="Спецификация+QA",
        agent_type="SpecificationQAAgent",
        prompt_files=_files(),
        validator_kind="qa",
        reference_fields=REFERENCE_FIELDS,
        json_field_labels=("full task JSON", "all MaterialResult", "validation reports"),
        prompt_addendum=(
            "Создай Спецификацию+QA. Содержательно покрой паспорт, источники, ключи и тесты, "
            "faulty code и патчи, критерии QA, рубрику и результат валидации. "
            "QA-ID и SHA разрешены только здесь."
        ),
    ),
    "final_project": MaterialSpec(
        kind="final_project",
        material_type="Итоговая проектная аттестация",
        agent_type="FinalProjectAssessmentAgent",
        prompt_files=_files(FINAL_PROJECT_PROMPT),
        validator_kind="final_project",
        reference_fields=REFERENCE_FIELDS,
        json_field_labels=("course", "modules", "final assessment lesson/row"),
        prompt_addendum=(
            "Создай итоговую аттестацию как проект: не менее 7 вариантов, работающий продукт, "
            "условия и оценивание только из данных JSON."
        ),
    ),
}


MATERIAL_SPEC_REGISTRY["current_control"] = replace(
    MATERIAL_SPEC_REGISTRY["current_control"],
    prompt_addendum=(
        "Render a separate learner-facing current_control material from "
        "GENERATION ARTIFACTS FOR THIS MATERIAL.current_control_autocheck when that artifact is available. "
        "Use exactly the 3 structured questions from the artifact, preserve their ids/order, "
        "and show only student_prompt, options, template/question type, and answer-format guidance. "
        "Do not show correct_answers, answer flags, autocheck_config, internal_explanation, QA-ID, SHA, "
        "or any source/service locator. Internal keys for autocheck remain only in the generation artifact."
    ),
)


def get_material_spec(kind: str) -> MaterialSpec:
    try:
        return MATERIAL_SPEC_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown material_kind: {kind}") from exc
