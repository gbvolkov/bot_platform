---
name: kpi-position-identification
description: Resolve official staff-structure context before explaining KPIs for a person, role, or position.
---

# KPI Position Identification

Use this skill when the user asks for their KPI, KPI for a role, KPI for a job
title, or KPI for an employee context.

Reference documents:
- KPI services: `../references/kpi-agent-service-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Required Outcome

Before KPI assignment lookup, resolve the official staff-structure context:
- department path;
- employee group when it distinguishes contexts;
- official position;
- `staff_structure_id`;
- later, after staff context is resolved, position level and responsibility
  center from KPI assignments when they distinguish rows.

Terminology:
- Employee group means a group of staff within a department.
- Position means the job or role the person has within that department.
- `position_group` means position level for KPI assignment context.

## Tools And Services Used

| Step | Tool/subagent | Service |
|---|---|---|
| No usable clue in personal KPI request | none | Ask user for missing position and department |
| Usable structural clue exists | `kpi_staff_structure_fuzzy_search` | Candidate Lookup |
| Candidate IDs or exact names need validation | `task` -> `kpi_bi_int` | Staff Structure Validation / Position Lookup |
| Several official contexts remain | none after BI data is available | Clarify Ambiguous Scope |
| Exact context confirmed | hand off to `kpi-list-assigned-kpis` | Explain KPIs For Position |

## Workflow

1. Extract usable staff-structure clues from the user and recent conversation.
2. If the request is personal, such as "какие у меня КПЭ?", and contains no
   usable clue, ask: "Уточните, пожалуйста, вашу должность и подразделение
   или группу работников." Do not call fuzzy search. Do not call BI.
3. If a usable clue exists, call `kpi_staff_structure_fuzzy_search` with the
   clue phrase only.
4. Read fuzzy output as full-position data: `candidate_values` are full position
   names, `staff_structure_ids` are matching row identifiers in the same order,
   and `candidates[]` provides `staff_structure_id`, `full_position_name`, and
   row fields.
5. Before asking the user about candidates, call `task` with
   `subagent_type: kpi_bi_int` for all candidate IDs returned by fuzzy search.
   Request complete `kpi_staff_structure` rows for every candidate:
   `staff_structure_id`, all department levels, employee group, position, and a
   human-readable full structure. Batch candidate IDs in one request when
   possible.
6. Call `task` with `subagent_type: kpi_bi_int` only with a concrete
   staff-structure request, for example:
   - validate these `staff_structure_id` values and return exact rows;
   - return complete hierarchy for these candidate IDs;
   - return official rows for this exact department and position;
   - list official positions in this exact department;
   - list official departments for this exact position.
7. These BI calls must ask about `kpi_staff_structure`, not `kpi_values`.
8. User-facing options must come from BI-validated rows. Do not show candidate
   choices, ask confirmation, or proceed with inferred context using
   fuzzy-only fields.
9. If BI validation leaves exactly one official row that matches the
   user-provided structural components, treat that context as resolved and hand
   off immediately to `kpi-list-assigned-kpis`. Do not ask the user to type a
   number and do not ask for an extra confirmation like "yes" in this case.
10. If BI returns partial context or ambiguity, decide the next step in Mycroft:
   either run another fuzzy search for a new user-provided clue, make another
   concrete BI staff-structure request, or ask the user one focused
   clarification with official options.
11. Do not request KPI Assignment Lookup and do not ask for `kpi_values` rows
   until the exact official staff-structure context is resolved.
12. If several official contexts still remain after BI validation, ask the user
   to choose between those official options. Ask for confirmation only when more
   than one official context is still plausible.

## Clarification Rules

- If the position is missing, ask for the position.
- If both position and department are missing, ask for both in one short
  question.
- If a position exists in several departments, ask for the department using
  official options.
- If a department contains several employee groups or positions that affect KPI
  applicability, ask for the relevant group or position using official options.
- When fuzzy search found several candidates, first enrich every candidate
  through BI by `staff_structure_id`; then ask the user using the full
  BI-validated department structure and position names.
- If only one BI-validated candidate remains suitable after applying the
  user-provided structural components, do not ask the user to choose by number;
  continue directly to KPI explanation for that official context.
- Do not ask the user for permission to use fuzzy search, BI, tools, or
  subagents.
- Do not mention internal tool or subagent calls in the user-facing
  clarification.

## Forbidden Behavior

- Do not call fuzzy search with "мне", "у меня", "я", "КПЭ", "KPI", or other
  generic words as the query.
- Do not ask BI what Mycroft should ask the user.
- Do not ask BI to list all possible positions for a personal KPI request with
  no usable clue.
- Do not resolve official position context from fuzzy-search output alone. BI
  must validate candidate rows before Mycroft treats them as official options.
  The final staff-structure context must be confirmed by `kpi_bi_int`.

## Output

Return either:
- a resolved context summary and, if needed, a confirmation question before KPI
  explanation; or
- one short clarification question with official options; or
- a missing-information question when no usable staff clue is present.
