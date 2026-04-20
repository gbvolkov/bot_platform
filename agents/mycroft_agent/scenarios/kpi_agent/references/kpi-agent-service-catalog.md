# Mycroft KPI Agent Service Catalog

This catalog defines user-facing services owned by the Mycroft KPI scenario.

The primary workflow is position-first KPI explanation: identify the official
staff-structure context, then explain the complete KPI set assigned to that
context.

## Service Catalog

| Service | Trigger | Mycroft-owned behavior | Tools and subagent services |
|---|---|---|---|
| Identify Position Context | User asks for "my KPI" or KPI for a role/job/person. | Extract position, department, employee group, and prior context. If no usable clue exists, ask the user for position and department/employee group without calling tools. When fuzzy returns candidates, request the full structure for each candidate before asking the user to choose or confirm. | `kpi_staff_structure_fuzzy_search` only when a usable structural clue exists; `kpi_bi_int` Staff Structure Validation for every returned `staff_structure_id`, or for exact constraints. |
| Explain KPIs For Position | Official staff context is resolved and confirmed. | Request and explain the complete KPI list for that context. | `kpi_bi_int` KPI Assignment Lookup; optionally KPI Method Lookup via `kpi_method_ref`. |
| Explain Particular KPI | User names a KPI or refers to a previous KPI. | Explain the KPI first; request context only if method/assignment differs by context. | `kpi_bi_int` KPI Method Lookup or KPI Metric Text Search. |
| Explain KPI Metrics | User asks about formula terms, method text, notes, calculation detail, or specifics. | Explain only database-backed meaning; clarify if the user wants a general definition. | `kpi_bi_int` KPI Metric Text Search and KPI Method Lookup. |
| Navigate Org Structure | User asks about departments, subdepartments, employee groups, positions, or hierarchy. | Answer org-structure question without forcing KPI lookup. | Optional fuzzy candidate lookup; `kpi_bi_int` Staff Structure Lookup. |
| Search KPI Data | User asks for search, filters, counts, or summaries. | Produce factual summaries by KPI name, department, employee group, position level, calculation detail, pool, frequency, business line, or method availability. | `kpi_bi_int` KPI Assignment Lookup, KPI Metric Text Search, Aggregated Counts. |
| Clarify Ambiguous Scope | Several official candidates remain after data lookup. | Ask one focused user-facing clarification with official options. Mycroft chooses the clarification question. | Uses previously returned fuzzy/BI data; do not ask BI to decide the conversation strategy. |
| Report Data Gaps | User asks for actual performance, targets, salary/bonus, or external policy. | State the missing layer when absent and offer the nearest database-backed alternative. | `kpi_bi_int` Data Availability Check when needed. |

## Entity Gate

Mycroft must not call tools for a personal KPI question that contains no usable
staff-structure clue. For example, for "какие у меня КПЭ?" Mycroft asks the
user for position and department/employee group.

Usable structural clues are department names, office names, sector names,
employee groups, positions, role titles, and concrete context from previous
turns. Pronouns and generic KPI words are not usable structural clues.

## Scenario Rules

- Do not jump to KPI explanation before resolving official staff context.
- Position context is not only a job title; it may require department path,
  employee group, position level, or responsibility center.
- Employee group means a group of staff within a department.
- Position means the job or role in that department.
- `position_group` means position level for KPI assignment and should be shown
  to users as "уровень должности".
- If the user asks only about org structure, answer the org-structure question
  without KPI lookup.
- Fuzzy search is only a candidate generator. Before Mycroft uses candidates in
  user dialogue, it must ask `kpi_bi_int` for complete `kpi_staff_structure`
  rows for all returned `staff_structure_id` values and use those full
  structures as the official options.
- If the user asks about a particular KPI, answer the KPI-centric question first.
- Do not invent official KPI categories or missing data.
