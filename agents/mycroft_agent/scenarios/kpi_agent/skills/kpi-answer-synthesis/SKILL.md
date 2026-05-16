---
name: kpi-answer-synthesis
description: Combine kpi_bi_int output into compact user-facing KPI answers with source boundaries and guardrails.
---

# KPI Answer Synthesis

Use this skill after receiving `kpi_bi_int` output, or when the correct next
step is a user clarification rather than a database lookup.

Reference documents:
- KPI services: `../references/kpi-agent-service-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Tools And Services Used

This skill uses no tools directly. It formats outputs from other skills and
asks user-facing business clarifications when needed.

## Synthesis Rules

1. Answer the user's current question first.
2. Keep database-backed facts separate from assumptions and missing data.
3. Preserve official database wording for positions, departments, KPI names, and
   method text.
4. If the position context is exact, include every KPI assignment returned by BI.
5. For KPI-list answers, preserve the BI assignment row count: if BI returns N
   assignment rows, the numbered user-facing list must contain exactly N KPI
   assignment items before any optional summary or follow-up.
6. Do not merge, deduplicate, group, aggregate, hide, or turn repeated KPI names
   into variants before showing the returned assignments to the user.
7. Repeated KPI names with different calculation detail, business line, pool
   flag, frequency, specifics, responsibility center, or position level are
   separate assignments and must stay separate list items.
8. "Compact" means concise wording inside each assignment item, not fewer
   assignment items.
9. If the exact position context was inferred from fuzzy search or BI
   disambiguation and several official contexts still remain plausible, ask the
   user to confirm or choose before producing a KPI list.
10. Every KPI-list answer must state the official applied position name before
   the list.
11. If scope is ambiguous, ask one focused clarification instead of producing a
   mixed KPI list.
12. If data is missing, state the specific missing layer.

## User-Facing Privacy

- Do not expose tool names, BI calls, fuzzy-search calls, subagent calls, SQL, or
  internal coordination unless the user explicitly asks how the answer was
  produced.
- Do not ask the user for permission to use internal tools or subagents.
- Ask only for business clarification, confirmation of an inferred official
  context, or missing information.

## Guardrails

- Never invent an official split into primary, secondary, or additional KPI
  categories.
- If asked for such a split and no database field supports it, say that the
  available data has no official KPI split into primary and secondary.
- Do not infer KPI meaning from name alone.
- Treat "Нетто-комиссионное вознаграждение" as a company expense for payments to
  external agents or intermediaries, not employee income.
- Do not expose database field names, table names, or technical identifiers in
  user-facing answers unless the user explicitly asks for schema or SQL details.
- Convert technical names to human labels: `department_1` -> upper-level
  department, `department_2` -> second-level department, deeper `department_*`
  -> subordinate department, `employee_group` -> employee group, `position` ->
  position, `position_group` -> "уровень должности", `kpi_method_ref` -> linked
  calculation method.
- Treat `employee_group` as a group of staff within a department, `position` as
  the job or role in that department, and `position_group` as the position level
  for KPI assignment context.
- Do not add sources or references footers.

## Position KPI Lists

- Do not return KPI assignments until the exact official position context is
  confirmed.
- The opening line must include the applied official position name.
- KPI-list answers are compact assignment lists. Do not require or wait for full
  calculation method text or method notes before listing the returned KPI
  assignments.
- Include department, employee group, and position level when they distinguish
  the KPI assignment.
- The KPI list must use one numbered item per BI assignment row, not one item
  per unique KPI name.
- Before sending the answer, compare the number of numbered KPI items with the
  number of assignment rows returned by BI. If they differ, rewrite the answer
  so the counts match.
- If BI returned only compact assignment fields and method references, that is
  enough for the KPI-list answer. Fetch full method details only when the user
  asks to disclose or explain a specific KPI calculation method.
- If exactly one official BI-validated context remains suitable, do not ask the
  user for a number or an extra "yes" confirmation; return the KPI list for
  that official context immediately.
- If the user asks for KPI after only an inferred candidate match and several
  official contexts still remain plausible, ask whether that exact official
  position context is correct and wait for confirmation.

## Output Shapes

- Missing personal context: ask for position and department/employee group.
- Position KPI list: short intro with official applied position context, then a
  numbered KPI list.
- Particular KPI: KPI name, plain meaning, method/formula, note, and assignment
  caveats.
- Metric explanation: term, KPI-context meaning, where it appears, and
  uncertainty if the database text is silent.
- Org structure: compact official list or table-like bullets.
- Data gap: one clear sentence plus the nearest available database-backed
  alternative.
