# Advanced prompt set changelog

Generated: 2026-06-29T15:50:46
Source workbook: `docs/ismart/Материалы для ИИ-агентов/Промпт_генерации_УМК_Python_продвинутый_v16.xlsx`
Baseline directory: `docs/ismart/Материалы для ИИ-агентов/рабочая область агента/prompts_skills`
Advanced directory: `docs/ismart/Материалы для ИИ-агентов/рабочая область агента/prompts_skills_advanced`

## Method

- Each advanced prompt file starts from the current working basic prompt file with the same filename.
- Controlled additions were appended only where v16 changes are needed for this iteration.
- The prompts do not contain runtime profile switches; profile isolation is done by directory/registry selection before generation.
- Rules that would require inventing extra practice tasks were intentionally not transferred; `lesson.practice_tasks` remains authoritative for task count, ids, order, and levels.

## Files with controlled additions

- `02_Теория_prompt_skill.md`: appended controlled advanced addendum derived from the matching v16 sheet.
- `03_Практика_prompt_skill.md`: appended controlled advanced addendum derived from the matching v16 sheet.
- `06_Итоговая_prompt_skill.md`: appended controlled advanced addendum derived from the matching v16 sheet.
- `07_Методические_указания_prompt_skill.md`: appended controlled advanced addendum derived from the matching v16 sheet.

## Files copied unchanged

- `01_Общее_prompt_skill.md`: Copied unchanged to avoid common cross-material profile branching and accidental changes to stable artifact boundaries.
- `04_Самостоятельная_prompt_skill.md`: Copied unchanged by plan: self-work flow is not changed in this iteration.
- `05_Промежуточная_prompt_skill.md`: Copied unchanged by plan: intermediate flow is not changed in this iteration.
- `08_Форматирование_заданий_курса_prompt.md`: Copied unchanged: HTML renderer/output structure is not changed.
- `91_skill_map.md`: Copied unchanged: skill map remains stable.
- `92_описание_json.md`: Copied unchanged: JSON structure remains common for basic and advanced.

## Excluded from prompt transfer

- Per-lesson or generic practice task quota rules from v16 are not copied into advanced prompts or policies.
- Module-level percentage diagnostics are not used to create or require additional tasks in a single lesson.
- Basic-vs-advanced branching text from v16 is not copied into prompts; the advanced directory contains only advanced-ready instructions.
