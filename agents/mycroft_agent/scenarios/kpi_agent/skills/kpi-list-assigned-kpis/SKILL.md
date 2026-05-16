---
name: kpi-list-assigned-kpis
description: Retrieve and explain the complete assigned KPI list for a confirmed staff-structure context.
---

# KPI List Assigned KPIs

Use this skill only after the official staff-structure context is resolved. If
the context was inferred from candidates, that still counts as resolved when BI
validation leaves exactly one suitable official context; user confirmation is
still required only when several official contexts remain plausible.

Reference documents:
- KPI services: `../references/kpi-agent-service-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Tools And Services Used

| Step | Tool/subagent | Service |
|---|---|---|
| Staff context missing or unconfirmed | none | Return to Position Identification |
| Staff context confirmed | `task` -> `kpi_bi_int` | KPI Assignment Lookup |
| Assignment method details needed after list | use `kpi-calculation-method-details` | KPI Method Lookup |
| Final response | none | Answer Synthesis |

## Workflow

1. Confirm that an exact official staff-structure context exists.
2. Prefer a validated `staff_structure_id` as the KPI lookup scope. Also pass
   the full position name and row fields when available.
3. If the staff context is missing, guessed, or unconfirmed, do not request or
   return KPI assignments; return to `kpi-position-identification`.
4. Call `task` with `subagent_type: kpi_bi_int` using KPI Assignment Lookup.
5. Ask BI for all KPI assignments for the resolved context as a row-level list
   of KPI items with the required fields below.
6. Do not ask BI to include full calculation method text, full method notes, or
   formula details in the KPI-list request. Keep the list payload compact.
7. Count the BI assignment rows.
8. Preserve every assignment row returned for the exact context.
9. Synthesize a compact user-facing answer without changing the assignment row
   count.

## BI Request Shape

When the user requests KPIs for a position, the request to `kpi_bi_int` must ask
for a row-level list of KPI assignments for the resolved official staff context.

The request must explicitly ask BI to return:
- one item per assigned KPI row;
- no grouping, deduplication, merging, or aggregation by KPI name;
- all returned items in stable assignment order;
- the official applied position context;
- compact assignment fields needed for Mycroft to distinguish each assignment;
- method references when available, so a later method-detail request can fetch
  the method through `kpi-calculation-method-details`.

Each KPI item should include the required fields below when available.

The KPI-list request must not ask for full method text, full method notes,
formula expansion, or calculation-method details. Those belong to
`kpi-calculation-method-details` when the user asks to disclose or explain a
specific KPI's calculation method.

## Required Fields To Request

Request these fields when available:
- KPI assignment identifier;
- `staff_structure_id`;
- official applied position;
- official department path;
- employee group;
- responsibility center;
- position level (`position_group`);
- KPI name;
- calculation detail;
- business line;
- pool flag;
- calculation frequency;
- calculation specifics;
- calculation method reference, if available.

## Answer Rules

- Always state the official position name to which the KPI list is applied.
- Include department and employee group when they distinguish the context.
- Use a numbered list.
- Use one numbered item per assignment row returned by BI, not one item per
  unique KPI name.
- If BI returns N assignment rows, the numbered list must contain exactly N
  KPI assignment items before any optional summary or follow-up.
- Never merge, deduplicate, group, aggregate, hide, or turn repeated KPI names
  into variants before showing the returned assignments to the user.
- If the same KPI name appears several times, repeat the KPI name and show the
  distinguishing business attributes, such as calculation detail, business
  line, pool flag, frequency, specifics, responsibility center, or position
  level.
- Compact wording is allowed only inside each assignment item; it must not
  reduce the number of listed assignments.
- Add frequency, pool, or specifics only when asked or needed for
  understanding.
- Do not include full method text or method notes in the KPI list. If the user
  asks to disclose or explain a method, formula, note, numerator, denominator,
  or calculation component, use `kpi-calculation-method-details`.
- Do not introduce primary, secondary, important, or optional categories unless
  they are explicitly in the database.
- Never synthesize KPI rows from examples, skill text, fuzzy output, or
  generated SQL. If `kpi_bi_int` has not returned assignment rows, ask BI before
  answering.
