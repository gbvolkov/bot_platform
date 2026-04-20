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

## Workflow

1. Identify whether the user named a KPI, referenced a previous item, or asked
   about a metric term.
2. Resolve references such as "second", "this one", "formula", or "yes" from
   conversation history before calling BI.
3. Call `task` with `subagent_type: kpi_bi_int` using KPI Method Lookup or KPI
   Metric Text Search.
4. For assignment-specific answers, include the resolved staff context or KPI
   assignment identifier and ask BI to join methods through `kpi_method_ref`.
5. Explain only what is supported by method text, note, calculation detail,
   calculation specifics, or assignment fields.

## Rules

- A named KPI can be explained without first resolving position if the question
  is KPI-centric.
- If the KPI appears in several contexts but the method is the same, explain the
  common method first.
- If the answer depends on assignment context, ask for position or department
  only after giving any confirmed common information.
- If database text does not define a term, say that the term appears in the
  database but its meaning is not disclosed there.
- If the user explicitly asks for a general definition outside KPI context,
  label it as general and separate it from database-backed KPI meaning.
