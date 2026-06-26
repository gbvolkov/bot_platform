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
            "Создай ученический материал теории. Содержательно покрой цель занятия, задачи, "
            "ключевые понятия, объяснение темы, примеры для разбора, типичные ошибки, "
            "самопроверку и итоговые выводы. Названия разделов выбирай по смыслу и по "
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
            "Создай ученическую практику строго по SOURCE CONTRACT FROM JSON. Используй ровно те задания, "
            "которые перечислены в lesson.practice_tasks/source_contract.tasks, в том же порядке и с теми же P id. "
            "Для каждого задания явно покажи уровень L1/L2/L3, исходное условие из JSON, условие для ученика, входные данные, "
            "требование к выводу и тесты. Не придумывай конкретные значения, имена переменных, входные данные, ожидаемый вывод "
            "или общий запрет на input(), если этого нет в JSON или Markdown references. Ожидаемый вывод не считается "
            "придуманным, если он детерминированно следует из явных литералов, присваиваний и точных команд print(...) "
            "в исходном условии по стандартной семантике Python. Если исходное условие не задаёт "
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
            "Создай МР-теорию для учителя-неэксперта. Это teacher-facing материал, не ученическая теория. "
            "Не пересказывай ученический материал и не перестраивай документ под структуру ученической теории. "
            "Содержательно покрой цели и задачи, методическую опору, подготовку, сценарий, "
            "ключи и пояснения к теоретическим примерам, типичные ошибки и реакцию учителя. "
            "Не добавляй ученический раздел «Проверка себя»; если нужен контроль понимания, оформи его как "
            "вопросы/чек-лист для учителя. Названия разделов "
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
            "Создай МР-практику для учителя. Включи раздел «Ключи и пояснения» только к задачам из "
            "SOURCE CONTRACT FROM JSON.authoritative_task_ids, в том же порядке и с теми же P id. "
            "Не добавляй задачи, которых нет в authoritative_task_ids, даже если валидатор или reference_examples "
            "упоминают другие номера. "
            "Для детерминированных задач дай минимальный эталонный Python-код и ожидаемый вывод или "
            "однозначный критерий проверки. Для недоопределённых задач не выдумывай единственный обязательный "
            "ответ ученика: дай один допустимый вариант учителя, помеченный как пример, и правила ручной проверки. "
            "Ключи в МР-практике разрешены и обязательны, потому что это teacher-facing материал; запрет на ключи "
            "относится к ученической практике. Не ссылайся на QA-артефакт, не добавляй QA-ID/SHA/служебные локаторы. "
            "Критерии проверки формулируй словами, без rc/stderr/stdout logs."
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


def get_material_spec(kind: str) -> MaterialSpec:
    try:
        return MATERIAL_SPEC_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown material_kind: {kind}") from exc
