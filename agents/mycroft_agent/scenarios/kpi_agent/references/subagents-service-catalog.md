# Mycroft KPI Subagents Service Catalog

This catalog defines specialist subagent services available to the Mycroft KPI
scenario. Mycroft calls the KPI BI specialist through `task` with exact
`subagent_type: kpi_bi_int`. Do not route KPI database facts through
`general-purpose`.

## Subagent: kpi_bi_int

`kpi_bi_int` works with the normalized KPI SQLite database
`data/kpi/kpi.sqlite`.

Use it for official staff-structure data, KPI assignments, KPI methods, metric
text search, disambiguation data, counts, and data availability checks.

Do not use it for conversation planning, deciding what Mycroft should ask the
user, external HR policy, actual KPI performance values, compensation
calculations, or interpretations unsupported by database text.

## kpi_bi_int Service Catalog

| Service | What to pass | What it returns | What it must not do |
|---|---|---|---|
| Staff Structure Validation | All `staff_structure_id` values returned by fuzzy search, full position names, or exact row fields. | One official `kpi_staff_structure` row per candidate ID, including every department level, employee group, position, full human-readable structure, and ambiguity information. | Does not decide Mycroft's next user question and does not query `kpi_values`. |
| Position Lookup | Concrete position fragment plus known department/employee-group constraints. | Official matching positions and contexts. | Does not list all positions for a no-clue personal KPI request. |
| Staff Structure Lookup | Concrete department, employee group, position, direct/nested scope, or candidate IDs. | Departments, hierarchy paths, employee groups, positions, direct/nested matches. | Does not require KPI lookup for org-only questions. |
| KPI Assignment Lookup | Resolved staff context, preferably `staff_structure_id`, plus assignment filters if known. | Complete KPI assignments with responsibility center, position level (`position_group`), KPI name, calculation detail, business line, pool, frequency, specifics, and staff context. | Does not omit rows for brevity when scope is exact. |
| KPI Method Lookup | KPI assignment ID, KPI name, method ID, or resolved assignment context. | Method name, calculation method, note, linked assignment context, and method gaps. | Does not join by KPI name when `kpi_method_ref` exists. |
| KPI Metric Text Search | Metric term, KPI phrase, formula phrase, or business term. | Matching KPI assignments and method excerpts with context. | Does not infer metric meaning when database text is silent. |
| Scope Disambiguation Data | Concrete partial position/department plus known constraints. | Official candidate values and discriminating fields. | Does not choose the user-facing clarification wording. |
| Aggregated Counts | Grouping field, filters, and requested scope. | SQL-backed counts and compact summaries. | Does not treat counts as actual KPI performance values. |
| Data Availability Check | Requested field, metric, or business question. | Whether data exists and which table/field owns it. | Does not fill missing information from external sources. |

## Forbidden Request Shapes

Never ask `kpi_bi_int`:
- "what should I ask the user?";
- "propose clarification options for this conversation";
- "list possible positions for the user" when the only user input is "какие у
  меня КПЭ?";
- to query `kpi_values` before staff context is resolved;
- to invent business meaning not present in the database.

## Database Ownership

- `kpi_staff_structure` owns department hierarchy, employee groups, positions,
  and `staff_structure_id`.
- `kpi_values` owns KPI assignments, responsibility centers, position levels
  (`position_group`), KPI names, method references, calculation detail, business
  line, pool, frequency, and specifics.
- `kpi_method` owns KPI method names, calculation method text, and notes.
- `employee_group` is a group of staff within a department.
- `position` is a job or role within that department.
- `position_group` is the position level used in KPI assignment context.

## Delegation Discipline

- Use exact subagent name `kpi_bi_int`.
- Pass concrete data questions and known constraints.
- After fuzzy search, pass all returned `staff_structure_id` values in one Staff
  Structure Validation request when possible. Ask for the full structure for
  each candidate before Mycroft presents options to the user.
- Use `staff_structure_id` for staff-context validation and KPI lookup whenever
  it is available.
- State whether department scope is direct-only, nested-only, or
  direct-plus-nested.
- For complete KPI lists, ask for all rows for the resolved context.
- For method explanations, ask BI to join `kpi_method` via `kpi_method_ref`.
- If BI reports missing or ambiguous data, Mycroft asks the user one focused
  clarification or states the gap.
