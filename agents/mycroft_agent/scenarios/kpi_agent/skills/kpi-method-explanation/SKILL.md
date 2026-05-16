---
name: kpi-method-explanation
description: Explain a named KPI, formula, note, frequency, calculation detail, metric term, or follow-up reference.
---

# KPI Method Explanation

Use this skill when the user asks how a KPI is calculated, what a KPI means,
what a metric term means, or refers to a KPI from the previous answer.

Reference documents:
- KPI services: `../references/kpi-agent-service-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Tools And Services Used

| Situation | Tool/subagent | Service |
|---|---|---|
| Follow-up reference like "second KPI" | none first | Resolve from conversation history |
| Named KPI explanation | `task` -> `kpi_bi_int` | KPI Method Lookup |
| Formula or metric term | `task` -> `kpi_bi_int` | KPI Metric Text Search |
| Assignment-specific method | `task` -> `kpi_bi_int` with context | KPI Method Lookup through `kpi_method_ref` |

## Mandatory BI Request For Method Details

When the user asks to disclose, expand, explain, or quote any KPI calculation
method, formula, method note, numerator, denominator, calculation component, or
formula value, Mycroft must call `task` with `subagent_type: kpi_bi_int` before
answering.

Conversation history may be used only to resolve which KPI assignment, KPI name,
or formula term the user means and to make the BI request self-contained. Do not
answer method-detail questions from prior KPI-list output, prior method snippets,
remembered context, or Mycroft's own interpretation unless current BI output for
this user request has returned the method or metric text.

## Workflow

1. Identify whether the user named a KPI, referenced a previous item, or asked
   about a metric term.
2. Resolve references such as "second", "this one", "formula", or "yes" from
   conversation history before calling BI.
3. `kpi_bi_int` is stateless. Do not ask BI about "item 1", "point 10",
   "point 13", "the previous list", or "the same row as before" unless the
   same BI request restates the resolved assignment context.
4. For assignment-specific follow-ups, reconstruct the exact assignment from
   Mycroft's own conversation history before calling BI.
5. Call `task` with `subagent_type: kpi_bi_int` using KPI Method Lookup or KPI
   Metric Text Search.
6. For assignment-specific answers, make the BI request self-contained: include
   the resolved `staff_structure_id` and the distinguishing assignment
   attributes already known from the selected row, such as `kpi_name`,
   `business_line`, `pool_flag`, `calculation_detail`, `specifics`,
   `responsibility_center`, `position_group`, or `kpi_method_ref`, and ask BI
   to return the calculation method text, method note, method gaps, and any
   related KPI or metric definitions needed to answer the user's method-detail
   question.
7. When the user asks for several numbered items from a previous KPI list,
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
8. After BI returns, explain only what is supported by method text, note,
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
- Never treat prior conversation context as complete evidence for calculation
  method details. A fresh `kpi_bi_int` result is required for each user request
  that asks for method, formula, note, or calculation-component details.
- If database text does not define a term, say that the term appears in the
  database but its meaning is not disclosed there.
- If the user explicitly asks for a general definition outside KPI context,
  label it as general and separate it from database-backed KPI meaning.
