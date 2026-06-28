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
            "Создай ученическую практику строго из GENERATION ARTIFACTS FOR THIS MATERIAL.practice_instances. "
            "lesson.practice_tasks/source_contract.tasks задают авторитетный P id, уровень, тип и паттерн задания; "
            "конкретная ученическая формулировка, сценарий, значения, имена переменных, входы/выходы и код берутся "
            "из PracticeTaskInstanceSet и должны быть новым вариантом, а не копией теории или reference_examples. "
            "Предметные сущности в source_text являются примерами слотов, если источник явно не требует именно их: "
            "можно заменить «любимый цвет/любимое животное» на параллельные категории, если сохраняются навык, "
            "количество и тип переменных, действие и проверяемый результат. "
            "Используй ровно те P id, которые перечислены в source_contract.tasks, в том же порядке. "
            "Для каждого задания явно покажи поля «Уровень», «Условие», «Код в редакторе» при наличии starter_code, "
            "«Входные данные», «Требование к выводу», «Как проверить» и «Тесты». Не показывай source_text, "
            "«исходный паттерн из JSON», внутренний source contract, generation artifacts или любые ссылки на JSON/pipeline. "
            "Не добавляй в learner-facing текст собственные подсказки о том, "
            "какую именно правку нужно сделать: не пиши «добавить кавычку», «заменить X на Y», «неверно написано имя функции» "
            "и аналогичные подсказки-ключи, если они находятся только в hidden_solution/teacher_explanation или source hint. "
            "Не показывай hidden_solution, teacher_explanation, corrected code, ответы или ключи. Если task instance не задаёт "
            "достаточно данных для точного stdout-теста, явно напиши, что эталон автопроверки требует уточнения источника, "
            "и не подставляй вымышленные значения; при этом сохрани для такого задания поля «Как проверить» и «Тесты» "
            "со статусом «не заданы/не применимы до уточнения». Для такого задания не делай таблицу или строку "
            "«вход → ожидаемый вывод» и не ставь заглушки в колонку ожидаемого вывода. Если runtime_tests заданы, выводи их как "
            "минимум 3 точные пары stdin→stdout: поле input как stdin, поле expected_output как stdout. Поле tests является legacy alias "
            "для runtime_tests. Если manual_checks заданы, покажи их отдельным learner-facing чек-листом ручной/статической "
            "проверки; для задач на рефакторинг такой чек-лист обязателен для требований, которые stdout не доказывает "
            "(имена переменных, комментарии, устранение повторов, именованные константы). Рендери проверку строго по "
            "форме structured artifacts: если runtime_tests/tests содержат expected_output, это обычная таблица stdin→stdout; "
            "если содержат expected_error/error_message, это диагностическая проверка сообщения ошибки, а не stdout-таблица; "
            "если run_mode='manual_only' и runtime_tests/tests пусты, не пиши «Запустить тесты», «все тесты Успех», "
            "«stdin», «stdout» или «Статус тестов», а покажи только раздел «Как проверить вручную» с manual_checks. "
            "Никогда не пиши в ученическом HTML внутренние оговорки вроде «не заданы в исходных данных», "
            "«требуется уточнить», «поддерживает ли платформа», «source clarification»; такие замечания допустимы "
            "только в agent_notes/internal artifacts. Если PracticeTaskInstance содержит "
            "faulty_code_display, рендери именно его как learner-facing код; raw faulty_code используй только как внутренний "
            "artifact для МР/QA. Не добавляй display_note, если он не требуется источником. "
            "The code block rendered from faulty_code_display must contain only editor code. Do not add marker/comment lines such as "
            "\"# fragment intentionally breaks here\". Do not add learner-facing notes that point to where or why the code is broken. "
            "В HTML не печатай символы "
            "backslash+n как текст; если expected_output заканчивается \\n, покажи содержимое stdout в <pre><code> с реальным "
            "переводом строки перед </code></pre>, а рядом можно текстом указать «stdout завершается переводом строки»."
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
            "Если dependency practice содержит generation_artifacts.practice_instances, используй эти instances, "
            "их tests, hidden_solution и teacher_explanation как источник правды; не реконструируй задачи заново. "
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
            "В разделе источников не показывай внутренние пути, имена Markdown-файлов, docs/ismart, "
            "референсы, рабочую область агента или любые локаторы исходных материалов. Если источник нужен, "
            "пиши нейтрально: «материалы курса», «учебные требования курса», «содержание занятия». "
            "Нужны ровно 8 практических задач и ровно 10 вопросов самоконтроля. Не показывай ученику ключи, "
            "правильные варианты, готовые ответы, блоки «Ключ для автопроверки», {%answer%} и заполненные "
            "{{input-text:...}} с ответами. Если автопроверке нужны ключи, считай их внутренним слоем платформы, "
            "не отображаемым в HTML. Рендери ученический HTML на основе GENERATION ARTIFACTS FOR THIS MATERIAL.self_work_autocheck: "
            "используй student_task_title, checked_skill, student_prompt, options и template_code как источник состава заданий, "
            "но не показывай correct_answer, correct_answers, runtime_tests, autocheck_config или internal_explanation. "
            "Не пиши, что ключи отсутствуют; они находятся во внутреннем artifact-слое и доступны валидатору/QA."
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
            "GENERATION ARTIFACTS FOR THIS MATERIAL.current_control_autocheck. "
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
            "Открытые-код вопросы засчитываются только если ученик пишет исполняемый код с проверяемым результатом; "
            "не превращай их в выбор ответа, matching, заполнение пропуска, подчёркивание или «что выведет код». "
            "Подсказки в learner-facing полях запрещены. "
            "Рендери ученический HTML на основе GENERATION ARTIFACTS FOR THIS MATERIAL.intermediate_assessment: "
            "используй visible поля из test_questions, open_code_questions и code_tasks: student_prompt, options, "
            "student_condition, starter_code, input_requirements "
            "и output_requirements. Не показывай correct_answers, reference_answer, rubric, manual_check_rules, "
            "runtime_tests, hidden_solution, teacher_explanation, internal_explanation или autocheck_config. "
            "Не добавляй в ученический HTML блоки 'Критерии оценивания', 'Рубрика', 'Правила проверки' "
            "и фразы вида 'это проверяется по коду': все критерии, rubrics и правила проверки остаются "
            "только во внутреннем artifact-слое. "
            "For 6A ordering questions, render the visible item list from options or "
            "autocheck_config.display_items only; never render correct_answers, ordered_items, "
            "items_in_correct_order, or correct_order as the visible order. "
            "For matching/classification test questions, never render correct pairs such as 'left — right' "
            "or 'left -> right'. Render two separate labeled item lists plus answer-format guidance; the pair map "
            "must remain only in correct_answers/autocheck_config inside the internal artifact. "
            "For matching/classification questions, render the actual visible left_items and right_items from "
            "autocheck_config or options in the order stored in the artifact. Never sort right_items back into "
            "the correct-pair order and never leave any list B item on the same row as its correct list A pair. "
            "Never replace them with generic placeholders such as Action 1, Variant A, Example 1, Действие 1, "
            "Вариант A, or Пример 1 unless those exact labels are present in the artifact. "
            "Ключи, эталоны, рубрики и тесты должны оставаться во внутреннем artifact-слое для валидатора/QA."
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
        prompt_addendum=(
            "Создай методические указания к готовой промежуточной аттестации. Если dependency intermediate "
            "содержит generation_artifacts.intermediate_assessment, используй этот artifact как источник ключей, "
            "эталонов, рубрик, тестов и решений; не реконструируй комплекты заново. "
            "HTML МР должен быть publishable методическим документом: цель/задачи, подготовка, сценарий 45+45 минут, "
            "явная структура аттестации, общая методика проверки закрытых, открытых и кодовых заданий, порядок фиксации результата, "
            "типичные ошибки и реакция учителя. "
            "Обязательно напиши структуру аттестации: 4 варианта; в каждом варианте 15 элементов: "
            "5 тестовых вопросов, 5 открытых-код вопросов и 5 практико-ориентированных задач на код; "
            "не менее 10 из 15 элементов требуют написания кода. "
            "Не выдумывай числовую шкалу оценивания, максимальные баллы, проходной порог, проценты или перевод в 5-балльную "
            "отметку. Используй такие числа только если они явно есть в JSON или Markdown references. Если источники "
            "не задают шкалу, опиши проверку по группам критериев и закрытому учительскому проверочному слою, а итоговую "
            "фиксацию результата оставь на локальные правила организации без числовых порогов. "
            "Не печатай полный банк ключей V1–V4, эталонные ответы по каждому открытому вопросу, эталонный код, "
            "stdin/stdout тесты, hidden solutions или autocheck_config. Разрешено указать, что конкретные ключи/эталоны/тесты "
            "используются из закрытого учительского проверочного слоя и не выдаются ученикам."
        )
        + (
            " In publishable mr_intermediate HTML, never print literal internal field names such as "
            "intermediate_assessment, generation_artifacts, hidden_solution, or autocheck_config. "
            "Use neutral wording such as closed teacher checking layer / internal teacher-only checking layer."
        ),
    ),
    "specification_qa": MaterialSpec(
        kind="specification_qa",
        material_type="Спецификация+QA",
        agent_type="SpecificationQAAgent",
        prompt_files=_files(),
        validator_kind="qa",
        dependency_kinds=(
            "theory",
            "practice",
            "mr_theory",
            "mr_practice",
            "self_work",
            "current_control",
            "intermediate",
            "mr_intermediate",
            "final_project",
        ),
        reference_fields=REFERENCE_FIELDS,
        json_field_labels=("full task JSON", "all MaterialResult", "validation reports"),
        prompt_addendum=(
            "Создай Спецификацию+QA. Содержательно покрой паспорт, источники, ключи и тесты, "
            "faulty code и патчи, критерии QA, рубрику и результат валидации. "
            "QA-ID можно показывать в HTML для трассировки задач, потому что specification_qa является внутренним QA-артефактом. "
            "Не показывай в HTML SHA/source hashes, локальные пути, имена файлов, prompt/runtime paths, agent class names, "
            "generation_artifacts.*, JSON field names или другие технические координаты пайплайна. "
            "Если нужна трассировка источников, используй только человекочитаемые названия материалов и утверждённых артефактов. "
            "Не добавляй в HTML процессный лог генерации, "
            "историю попыток или формулировки вроде «исправлено по замечаниям валидатора»; итог валидации пиши "
            "как нейтральную QA-сводку по фактическому состоянию материалов. Для практических задач используй SOURCE CONTRACT FROM JSON: "
            "не добавляй задачи вне authoritative_task_ids и не выдумывай конкретные значения, имена переменных, "
            "stdin/stdout или обязательный формат вывода, если их нет в source_text или approved dependencies. "
            "Если dependency practice содержит generation_artifacts.practice_instances, используй эти instances, "
            "их tests, hidden_solution и teacher_explanation как источник правды для QA; в HTML называй это "
            "«утверждённые материалы практики и закрытый QA-набор ключей/тестов», без внутренних имён полей; не реконструируй "
            "практические задания заново. "
            "Если dependency self_work содержит generation_artifacts.self_work_autocheck, используй этот artifact "
            "как источник внутренних ключей, correct_answers, runtime_tests и autocheck_config для самостоятельной работы; "
            "в HTML называй это «закрытый набор самопроверки», без внутренних имён полей; не требуй, чтобы эти ключи были показаны в ученическом HTML. "
            "Если dependency intermediate содержит generation_artifacts.intermediate_assessment, используй этот artifact "
            "как источник ключей, эталонов, rubrics, runtime_tests, hidden_solution и autocheck_config для промежуточной аттестации; "
            "в HTML называй это «закрытый набор проверки аттестации», без внутренних имён полей; не требуй, чтобы эти ключи были показаны в ученическом HTML. "
            "Если задача недоопределена для детерминированной автопроверки, явно пометь её как требующую "
            "уточнения источника или ручной проверки; не превращай пример в обязательный ключ/тест."
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
        "GENERATION ARTIFACTS FOR THIS MATERIAL.current_control_autocheck. "
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
