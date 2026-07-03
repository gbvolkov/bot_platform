---
skill_id: practice_guidance_generation
source_data: "PracticeGuidanceInput"
source_level: "approved lesson output"
---

# Указания к практической работе: prompt/skill

## Назначение
Отдельный ученический материал с пошаговыми указаниями к уже подготовленной практике.

## Входные данные
- Получай только `PracticeGuidanceInput`.
- Не используй raw HTML практики или теории.
- Основной источник по заданиям: `approved_materials.practice.practice_instances` и `practice_tasks`.
- Теоретический блок строится по `theory_brief_source.sections` и references.

## Правила
1. Не меняй и не реконструируй практические задания.
2. `module_tasks` строятся только по `practice_tasks`.
3. Для каждого этапа создай алгоритм выполнения и один аналогичный разобранный пример.
4. Разобранный пример должен быть похожим по методу, но не должен подменять задания модуля.
5. Не показывай ключи, исправленный код, внутренние поля решений/пояснений, внутренние имена полей, JSON/process wording, SHA или локальные пути.
6. Для basic не создавай L3-контент, если L3 нет в approved practice.
7. Если не хватает теории, референса или реперного значения, запиши это в `requires_check`, не выдумывай.
8. Верни только structured output `PracticeGuidanceArtifact`.
