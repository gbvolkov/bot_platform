---
name: kpi-source-policy
description: Route KPI requests between Mycroft, fuzzy search, and kpi_bi_int without delegating conversation strategy.
---

# KPI Source Policy

Use this skill before answering any factual KPI, position, org-structure,
method, metric, count, or data-availability question.

Reference documents:
- KPI services: `../references/kpi-agent-service-catalog.md`
- Subagent services: `../references/subagents-service-catalog.md`

## Responsibility Split

- Mycroft owns intent classification, slot extraction, follow-up handling, and
  user clarifications.
- `kpi_staff_structure_fuzzy_search` owns approximate matching of usable
  staff-structure clues to full position candidates.
- `kpi_bi_int` owns factual database retrieval from `data/kpi/kpi.sqlite`.
- `kpi_bi_int` must not be asked to plan the conversation or decide what
  Mycroft should ask the user.

## Entity Gate

Before any fuzzy or BI call, decide whether the user provided a usable
staff-structure clue.

Usable clues:
- department, office, sector, or unit names;
- employee-group names;
- position or role titles;
- named KPI contexts from prior turns.

Not usable clues:
- pronouns: "я", "мне", "у меня", "мой", "my", "me";
- generic KPI words: "КПЭ", "КПИ", "KPI", "показатели";
- empty or catch-all requests such as "какие у меня КПЭ?".

If a personal KPI request has no usable clue, ask the user for their position
and department/employee group. Do not call fuzzy search and do not call BI.

## Tool And Service Matrix

| Situation | Mycroft action | Tool/subagent | Service |
|---|---|---|---|
| Personal KPI request with no usable clue | Ask user for position and department/employee group | none | Identify Position Context |
| Approximate structural clue exists | Find full position candidates | `kpi_staff_structure_fuzzy_search` | Candidate Lookup |
| Fuzzy candidates returned | Enrich and validate every candidate by ID before user dialogue | `task` -> `kpi_bi_int` | Staff Structure Validation |
| Exact staff context confirmed | Retrieve all KPI assignments | `task` -> `kpi_bi_int` | KPI Assignment Lookup |
| Org-structure-only question | Return official structure data | optional fuzzy, then `kpi_bi_int` | Staff Structure Lookup |
| Named KPI or formula question | Retrieve method/metric text | `task` -> `kpi_bi_int` | KPI Method Lookup / KPI Metric Text Search |
| Search/count/data gap | Retrieve factual aggregate or availability | `task` -> `kpi_bi_int` | Aggregated Counts / Data Availability Check |

## Routing Rules

1. Classify the user request.
2. Apply the Entity Gate.
3. Use fuzzy search only when there is a usable staff-structure clue. Pass the
   clue phrase, not pronouns or the whole underspecified message.
4. Interpret fuzzy output as candidate data: `candidate_values` are full
   position names, `staff_structure_ids` are row IDs in the same order, and
   `candidates[]` contains fields for BI validation.
5. After fuzzy returns candidates, call BI with all returned
   `staff_structure_id` values. Request the complete staff structure for each
   candidate: every department level, employee group, position, and a
   human-readable full structure. Batch IDs in one request when possible.
6. Use BI-validated staff-structure rows, not fuzzy-only fields, for
   user-facing candidate choices, clarification questions, and confirmation
   questions.
7. For position-first KPI requests, BI calls before KPI lookup must ask only
   about `kpi_staff_structure`.
8. Do not request `kpi_values` rows until the exact staff-structure context is
   resolved. If BI validation leaves exactly one suitable official context,
   treat it as resolved and continue to KPI lookup without asking for an extra
   confirmation. If several official contexts still remain after BI validation,
   ask the user to confirm or choose between them before KPI lookup.
9. For KPI database facts, `task` with `subagent_type: kpi_bi_int` is mandatory.
10. Never call `task` with `subagent_type: general-purpose` for KPI database
   facts.

## Forbidden BI Requests

Do not ask `kpi_bi_int`:
- what Mycroft should ask the user next;
- to propose a conversation strategy;
- to list all possible positions when the user only said "какие у меня КПЭ?";
- to search `kpi_values` before the staff context is resolved;
- to invent interpretations unsupported by database text.

## Internal Tool Privacy

- Do not tell the user that you are calling BI, fuzzy search, tools, subagents,
  or internal requests.
- Do not ask permission before using internal tools or subagents.
- Ask the user only for business clarification, confirmation of an inferred
  official staff context, or missing details that cannot be resolved from the
  database.

## Terminology

- `employee_group` means a group of staff within a department.
- `position` means the person's job or role within that department.
- `position_group` means position level for KPI assignment context, not
  employee group.
