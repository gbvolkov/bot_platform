---
name: kpi-calculation-method-details
description: "Must be read before answering any KPI calculation-method detail request: formula, calculation method, method note, numerator, denominator, formula component, formula value, or term inside a formula. Always call task with subagent_type kpi_bi_int; prior conversation context is not enough."
---

# KPI Calculation Method Details

Use this skill when the user asks to disclose, expand, explain, or quote a KPI
calculation method, formula, method note, numerator, denominator, calculation
component, formula value, or a term that appears inside a formula.

Reference documents:
- KPI services: `../references/kpi-agent-service-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Tools And Services Used

| Situation | Tool/subagent | Service |
|---|---|---|
| Method, formula, note, numerator, denominator, component, or formula value | `task` -> `kpi_bi_int` | KPI Method Lookup |
| Term inside a formula may refer to another KPI | `task` -> `kpi_bi_int` | KPI Metric Text Search |
| Follow-up reference like "this value", "the numerator", or "item 5" | none first | Resolve from conversation history |

## Workflow

1. Identify the referenced KPI assignment, KPI name, method row, formula term,
   or calculation component.
2. Use conversation history only to resolve the reference and make the BI
   request self-contained.
3. Do not answer from prior KPI-list output, prior method snippets, remembered
   context, or Mycroft's own interpretation.
4. Call `task` with `subagent_type: kpi_bi_int` before answering.
5. In the BI request, restate the resolved context: selected KPI name, selected
   assignment attributes, staff context, method identifier, and the exact term
   or component the user asked about when those are known.
6. Ask BI to return the calculation method text, method note, method gaps, and
   any related KPI or metric definitions needed to answer the user's question.
7. If a term or component inside the selected method may be a separate KPI,
   ask BI to search method names, method texts, and notes globally for that
   term and return candidate definitions.
8. If several candidate definitions remain plausible, ask the user one focused
   clarification instead of choosing silently.
9. Answer only after current BI output for this user request has returned the
   method or metric text.

## Rules

- A fresh `kpi_bi_int` result is required for every method-detail request.
- Do not treat a previous KPI list or previous answer as complete evidence for
  calculation method details.
- Do not tell BI how to build SQL. Ask for the business fields and evidence
  needed; BI owns retrieval.
- If BI says the method text, note, or related definition is missing, state the
  gap directly.
- Preserve official method wording when quoting or explaining formula elements.
