from __future__ import annotations

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
            "Создай ученический материал теории. Обязательные H2: Цель занятия; "
            "Задачи занятия; Ключевые понятия; Конспект; Задачи-примеры для разбора; "
            "Типичные ошибки; Проверка себя; Итоговые выводы. Не добавляй ключи практики, QA-ID или SHA."
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
            "Создай ученическую практику. Используй типы задач и количество из lesson.practice_tasks "
            "и lesson.difficulty. Каждое задание оформи H3 с префиксом P1, P2, ...; "
            "у каждого задания должны быть тест-кейсы вход -> ожидаемый вывод. Исправленный код и ключи не показывай."
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
            "Обязательные H2: Цель и задачи; Методическая опора; Подготовка; Сценарий; "
            "Ключи и пояснения; Типичные ошибки и реакция."
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
            "Создай самостоятельную работу. Обязательные H2: Тема; Цели и задачи; Порядок выполнения; "
            "Самоконтроль; Требования к результату; Источники. Раздел краткой теории запрещён. "
            "Нужны ровно 8 практических задач и ровно 10 вопросов самоконтроля с ключами для автопроверки."
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
            "Создай отдельный текущий контроль: ровно 3 вопроса, разные шаблоны из template_descriptions. "
            "Внутри файла должны быть ключи для автопроверки. Не добавляй QA-ID/SHA."
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
            "Создай промежуточную аттестацию уровня модуля: 4 комплекта. В каждом комплекте "
            "16 закрытых заданий, 4 открытых задания и 3 практико-ориентированные задачи на код."
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
            "Создай Спецификацию+QA. Обязательные H2: Паспорт; Источники; Ключи и тесты; "
            "Faulty code и патчи; Критерии QA; Рубрика; Результат валидации. QA-ID и SHA разрешены только здесь."
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


def get_material_spec(kind: str) -> MaterialSpec:
    try:
        return MATERIAL_SPEC_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown material_kind: {kind}") from exc
