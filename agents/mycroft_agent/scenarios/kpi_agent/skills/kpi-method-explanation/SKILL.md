---
name: kpi-method-explanation
description: Explain named KPI meaning, frequency, calculation detail, assignment caveats, metric terms, or follow-up references. For formula, calculation method, method note, numerator, denominator, formula component, or formula value requests, use kpi-calculation-method-details instead.
---

# KPI Method Explanation

Use this skill when the user asks what a KPI means, what a metric term means,
or refers to a KPI from the previous answer.

If the user asks to disclose, expand, explain, or quote a KPI formula,
calculation method, method note, numerator, denominator, calculation component,
or formula value, use `kpi-calculation-method-details` instead.

Reference documents:
- KPI services: `../references/kpi-agent-service-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Tools And Services Used

| Situation | Tool/subagent | Service |
|---|---|---|
| Follow-up reference like "second KPI" | none first | Resolve from conversation history |
| Named KPI explanation | `task` -> `kpi_bi_int` | KPI Method Lookup |
| Metric term without formula-detail request | `task` -> `kpi_bi_int` | KPI Metric Text Search |
| Formula, method, note, numerator, denominator, component, or value detail | use `kpi-calculation-method-details` | KPI Method Lookup / KPI Metric Text Search |
| Assignment-specific method | `task` -> `kpi_bi_int` with resolved assignment context | KPI Method Lookup |

## Workflow

1. Identify whether the user named a KPI, referenced a previous item, or asked
   about a metric term.
2. If the request is about a formula, calculation method, method note,
   numerator, denominator, calculation component, or formula value, stop this
   workflow and use `kpi-calculation-method-details`.
3. Resolve references such as "second", "this one", or "yes" from
   conversation history before calling BI.
4. `kpi_bi_int` is stateless. Do not ask BI about "item 1", "point 10",
   "point 13", "the previous list", or "the same row as before" unless the
   same BI request restates the resolved assignment context.
5. For assignment-specific follow-ups, reconstruct the exact assignment from
   Mycroft's own conversation history before calling BI.
6. Call `task` with `subagent_type: kpi_bi_int` using KPI Method Lookup or KPI
   Metric Text Search.
7. For assignment-specific answers, make the BI request self-contained: include
   the resolved `staff_structure_id` and the distinguishing assignment
   attributes already known from the selected row, such as `kpi_name`,
   `business_line`, `pool_flag`, `calculation_detail`, `specifics`,
   `responsibility_center`, `position_group`, or `kpi_method_ref`, and ask BI
   to return the fields needed to explain the KPI meaning, metric term, or
   assignment caveat.
8. When the user asks for several numbered items from a previous KPI list,
   Mycroft must restate each selected assignment inside the BI request instead
   of assuming that BI remembers the earlier list. Example of a correct
   self-contained BI request:

   `Дай развернутый текст методики (method text) и примечания (method notes)
   для KPI-назначений staff_structure_id=8 по строкам, соответствующим пунктам
   1, 10 и 13 из ранее возвращенного списка (те же KPI name + business line +
   pool flag + calculation detail + specifics). Верни дословно method text и
   method notes полностью (без сокращений), и повтори ключевые атрибуты
   назначения, чтобы различать строки. Если в базе методика/примечание пустые
   — так и укажи.`
9. After BI returns, explain only what is supported by method text, note,
   calculation detail, calculation specifics, or assignment fields.

## Rules

- A named KPI can be explained without first resolving position if the question
  is KPI-centric.
- If the KPI appears in several contexts but the method is the same, explain the
  common method first.
- If the answer depends on assignment context, ask for position or department
  only after giving any confirmed common information.
- Never assume that BI remembers the previous KPI list or previous numbering.
  Mycroft must resolve and restate the assignment context itself.
- If database text does not define a term, say that the term appears in the
  database but its meaning is not disclosed there.
- If the user explicitly asks for a general definition outside KPI context,
  label it as general and separate it from database-backed KPI meaning.
