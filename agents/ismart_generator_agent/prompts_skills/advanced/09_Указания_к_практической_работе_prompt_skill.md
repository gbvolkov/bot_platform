---
skill_id: practice_guidance_generation
source_data: PracticeGuidanceInput
source_level: approved lesson output
profile: advanced
---

# Указания к практической работе: prompt/skill

## Назначение

Сгенерируй отдельный ученический материал "Указания к практической работе" для одного занятия Python-трека.

Это не сами практические задания и не методические рекомендации для преподавателя. Материал объясняет ученику:

- зачем выполняется практика;
- что нужно вспомнить перед выполнением;
- как разобрать аналогичный пример;
- как пошагово выполнить уже утвержденные задания практики;
- какой результат должен получиться;
- как этот результат связан с дальнейшим обучением.

## Источник истины

Получай только `PracticeGuidanceInput`.

Не используй raw HTML практики, raw HTML теории, manifest целиком, result целиком, локальные пути, SHA, внутренние логи и validation retry logs.

Практические задания не создаются заново. Основной источник заданий:

- `practice_tasks[]`;
- `approved_materials.practice.practice_instances`;
- `approved_materials.practice.practice_templates`.

Краткая теория строится по:

- `theory_brief_source.sections[]`, если они есть;
- `references.requirements[]`;
- `references.reference_examples[]`;
- `references.goals_and_tasks[]`;
- `references.donor_materials[]`;
- `references.template_descriptions[]`.

Связка с предыдущими занятиями строится только по:

- `previous_lessons_context[]`.

Если `previous_lessons_context[]` пустой, не пиши "на прошлом занятии", "ранее вы изучили", "вспомните занятие N" и любые другие ссылки на предыдущие занятия. В этом случае готовь указания автономно, без выдуманной ретроспективы.

Если `previous_lessons_context[]` непустой, используй его для коротких явных связок: "на занятии N вы...", "вспомните, как...". Не добавляй сведения о прошлых занятиях из предположений.

## Обязательная структура документа

Верни structured output `PracticeGuidanceArtifact`.

Заполни структуру так:

1. `header`
   - `work_title`: название вида "Указания к практической работе N" или близкое по смыслу.
   - `topic`: тема занятия из `task_meta`, `practice_instances.lesson_goal`, references или входных данных.
   - `lesson_number`: номер занятия из `task_meta.lesson_number`.
   - `audience`: обязательное непустое поле. Бери значение из `task_meta.audience`.

2. `goals`
   - `goal`: цель работы.
   - `objectives[]`: операциональные задачи ученика: "проектирует", "реализует", "тестирует", "отлаживает", "оптимизирует", "объясняет".
   - Цель и задачи связывай с `practice_instances.lesson_goal`, `practice_instances.lesson_objectives`, `practice_tasks[]`, `references.goals_and_tasks[]`.

3. `theory_brief`
   - Это краткая актуализация, а не полный пересказ теории.
   - Включай только инструменты, необходимые для выполнения текущей практики.
   - Если `previous_lessons_context[]` непустой, добавь короткую связь с предыдущими занятиями.
   - Если теоретических данных недостаточно, не выдумывай; оставь компактный самодостаточный блок по доступным references и практике, без служебных фраз о проверке или уточнении.

4. `methodical_guidance`
   - `problem_statement`: зачем выполняется практика и какой учебный результат ожидается.
   - `environment`: Python 3, редактор платформы, стандартная библиотека. Сторонние библиотеки допускаются только если они явно указаны во входе.
   - `before_start.steps[]`: что проверить перед началом.
   - `before_start.checkpoint`: короткий контроль готовности.
   - `stages[]`: этапы выполнения практики.

5. `stages[]`
   - Строй этапы по уровням и методической близости задач.
   - `source_task_ids[]` должны ссылаться только на id из `practice_tasks[]`.
   - Не добавляй новые P-id.
   - Не удаляй и не переименовывай P-id.
   - Не меняй порядок и смысл утвержденных задач.
   - Для advanced используй L3 только если L3-задачи есть во входе.

6. `worked_example`
   - Для каждого этапа дай один разобранный аналогичный пример того же типа.
   - Пример должен быть похож по методу, но отличаться сценарием, значениями и формулировкой от задач практики, references и теории.
   - Для примера допустима петля "условие -> код/действие -> результат -> правило".
   - Не копируй пример из теории дословно.
   - Не превращай пример в ответ на одну из задач практики.

7. `module_tasks[]`
   - Используй только student-facing поля из `practice_tasks[]`.
   - Не показывай `hidden_solution`, `teacher_explanation`, внутренние ключи и эталоны.
   - Для задач на поиск ошибки не называй саму ошибку в условии и не подсказывай точную правку.
   - Если задача содержит `faulty_code_display`, используй его как learner-facing код в `module_tasks[].code_cell`. Raw `faulty_code` не выводи, если вместо него есть `faulty_code_display`.
   - Если `faulty_code_display` пустой, но есть `starter_code`, используй `starter_code` как learner-facing код в `module_tasks[].code_cell`.
   - Не оставляй `module_tasks[].code_cell` пустым при наличии `faulty_code_display` или `starter_code`.
   - `starter_code` не является ключом и не является решением; это ученическая заготовка кода.

8. `result_requirements`
   - Обязательный непустой раздел.
   - `deliverable`: что ученик должен получить после выполнения практики.
   - `criteria[]`: минимум 2 проверяемых критерия результата без раскрытия ключей.

9. `self_check_questions[]`
   - Обязательный непустой раздел.
   - Минимум 3 вопроса для самопроверки без ответов и без ключей.

10. `requires_check[]`
   - Для обычной генерации оставляй пустым.
   - Не используй это поле для отсутствующей теории, отсутствующих точных названий интерфейса, пустого `previous_lessons_context[]` или иных нормальных пробелов источников.
   - Не пиши фразы "Требует проверки", "требует уточнения", "отсутствует утверждённый источник", "при необходимости уточнить" и аналогичные служебные предупреждения.
   - Если точных данных нет, опусти неподтверждённую деталь и сформулируй нейтрально. Внутренние ограничения источников фиксируй только в `consistency_notes[]` или `agent_notes[]`.

11. `consistency_notes[]` и `agent_notes[]`
   - Внутренние заметки о согласованности источников.
   - Не выводи их как ученический контент.

## Привязка входных данных к содержанию

- Шапка: `task_meta.lesson_number`, `task_meta.lesson_title`, `approved_materials.practice.practice_instances.lesson_goal`.
- Аудитория: `task_meta.audience`.
- Количество и состав задач: ровно `practice_tasks[]`; количество берется из утвержденной практики, а не из общей нормы.
- Цели и задачи: `approved_materials.practice.practice_instances.lesson_goal`, `lesson_objectives`, `references.goals_and_tasks[]`.
- Краткая теория: `theory_brief_source.sections[]` и references.
- Связка с прошлым: только `previous_lessons_context[]`.
- Практические действия: `practice_tasks[].student_condition`, `starter_code`, `faulty_code_display`, `input_requirements`, `output_requirements`, `checks`, `manual_checks`.
- Методическая логика и подсказки: references и approved practice artifacts.
- Фактура/данные: `references.donor_materials[]`; если фактуры нет, не выдумывай.

## Запреты

- Не раскрывай ключи, исправленный код, hidden_solution, teacher_explanation.
- Не показывай внутренние имена JSON-полей, SHA, локальные пути, process wording.
- Не реконструируй задания.
- Не добавляй задания сверх `practice_tasks[]`.
- Не дублируй raw HTML.
- Не делай ссылки на предыдущие занятия при пустом `previous_lessons_context[]`.
- Не используй общую норму количества задач как требование к конкретному занятию.
- Не возвращай пустые обязательные разделы: `result_requirements.deliverable`, `result_requirements.criteria[]`, `self_check_questions[]`, `header.audience`.

## Правило результата

Верни только structured output `PracticeGuidanceArtifact`.
