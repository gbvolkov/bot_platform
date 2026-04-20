---
name: kpi-structure-navigation
description: Answer organizational-structure questions about departments, subdepartments, employee groups, positions, and hierarchy paths.
---

# KPI Structure Navigation

Use this skill when the user asks about organization structure, departments,
subdepartments, employee groups, positions, direct rows of a unit, nested
subdivisions, or possible official position contexts.

Reference documents:
- KPI services: `../references/kpi-agent-service-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Tools And Services Used

| Situation | Tool/subagent | Service |
|---|---|---|
| Approximate department, employee group, or position clue | `kpi_staff_structure_fuzzy_search` | Candidate Lookup |
| Fuzzy candidate IDs returned | `task` -> `kpi_bi_int` | Staff Structure Validation |
| Official structure data needed | `task` -> `kpi_bi_int` | Staff Structure Lookup |
| Direct/nested scope ambiguous | none after BI data is available | Clarify Ambiguous Scope |
| Org-only question | no KPI lookup | Navigate Org Structure |

## Workflow

1. Identify whether the user wants:
   - direct-only rows in a department;
   - nested-only rows below a department;
   - direct-plus-nested rows;
   - a list of departments without positions;
   - positions or employee groups in a department.
2. If the user gives an approximate structural clue, use
   `kpi_staff_structure_fuzzy_search` with that clue phrase.
3. If fuzzy search returns candidate IDs, call `task` with
   `subagent_type: kpi_bi_int` before answering or asking the user to choose.
   Ask BI for the complete staff structure for each candidate ID: every
   department level, employee group, position, and a human-readable full
   structure.
4. Call `task` with `subagent_type: kpi_bi_int` using Staff Structure Lookup.
   Ask BI concrete data questions about `kpi_staff_structure`; do not ask BI to
   decide the conversation strategy.
5. Do not force KPI lookup when the user only asks about organization
   structure.
6. Return official names from BI-validated rows, translated into human labels
   rather than raw field names.

## Direct And Nested Scope

- Direct-only means match the target department level and require the next lower
  department level to be empty.
- Nested-only means match the target department level and require a lower
  department level to be present.
- Direct-plus-nested means match the target department level and do not filter
  lower levels.

## Answer Rules

- If a department fragment has several official matches, ask one clarification
  with official options.
- If several fuzzy candidates remain, use BI-validated full structures as the
  clarification options; do not ask the user to choose from fuzzy-only fields.
- If no department is found, say so and ask for the exact official name or
  another clue.
- Do not show raw database field names such as `department_1`, `department_2`,
  `employee_group`, or `position` to the user.
- Use human labels: upper-level department, second-level department,
  subordinate department, employee group, position, and position level.
- Treat a group as employee group inside a department. Treat position as the job
  in that department. Treat `position_group` separately as position level for
  KPI assignment context.
