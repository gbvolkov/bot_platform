# 04. State, Tools, and Trace Contracts

## State model

The graph state should contain:

- chat transcript / summaries
- current stage
- active branch
- branch conflict metadata
- extracted slots
- derived problem summary
- evidence pack
- read document results
- shortlist
- follow-up pack
- request flags
- counters
- audit fields

## Required slot set

Recommended slot keys:

- `customer_goal`
- `transport_type`
- `route_type`
- `route_mode`
- `body_type`
- `capacity_or_payload`
- `competitor`
- `decision_criterion`
- `decision_role`
- `special_conditions`

## Tools contract

### 1. `classify_problem_branch`
Purpose:
- map current conversation state to exactly one business branch or detect conflict.

### 2. `get_tco_pack`
Purpose:
- return TCO/economics-oriented document pack.

### 3. `get_configuration_pack`
Purpose:
- return options/configuration-oriented pack.

### 4. `get_comparison_pack`
Purpose:
- return direct comparison documents and supporting evidence.

### 5. `get_service_pack`
Purpose:
- return manuals, service books, and quality docs.

### 6. `get_internal_approval_pack`
Purpose:
- return a concise decision-ready set for the identified role.

### 7. `get_passenger_route_pack`
Purpose:
- return route/capacity-fit passenger materials.

### 8. `get_special_body_pack`
Purpose:
- return function-first special body materials and likely chassis families.

### 9. `get_special_conditions_pack`
Purpose:
- return materials for offroad / severe conditions.

### 10. `read_doc`
Purpose:
- read only the needed section/focus of a selected material.

### 11. `build_solution_shortlist`
Purpose:
- convert problem + evidence context into 1–3 candidate product families.

### 12. `build_followup_pack`
Purpose:
- assemble a structured role-aware follow-up pack.

## Tool execution policy

### Tools not allowed in:
- `OPENING`
- `DISCOVERY`

### Only allowed in `BRANCH_LOCK`
- `classify_problem_branch`

### In `EVIDENCE`
Allow only:
- one branch-pack tool determined by branch
- `read_doc`

### In `SHORTLIST`
Allow only:
- `build_solution_shortlist`

### In `NEXT_STEP`
Allow only:
- `build_followup_pack`

## Trace requirements

Every turn trace should log:
- `turn_index`
- `stage_before`
- `stage_after`
- updated slots
- allowed tools
- actual tool calls
- tool result summaries
- policy checks passed/failed
- agent response summary

## Must-have trace assertions

- product recommendation never appears before branch lock
- shortlist never appears before evidence
- branch conflict is resolved before branch-pack call
- follow-up package includes rationale per document

## Sample trace event shape

```json
{
  "turn_index": 4,
  "stage_before": "BRANCH_LOCK",
  "stage_after": "EVIDENCE",
  "slots_changed": {
    "decision_criterion": "configuration"
  },
  "allowed_tools": ["classify_problem_branch"],
  "tool_calls": [
    {"tool": "classify_problem_branch", "args": {...}}
  ],
  "response_summary": "Agent locked branch and moved into configuration evidence",
  "policy_checks_passed": [
    "no_product_before_branch_lock"
  ]
}
```
