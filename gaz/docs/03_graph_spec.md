# 03. Graph Spec

## Stage machine

### Stages
- `OPENING`
- `DISCOVERY`
- `BRANCH_LOCK`
- `EVIDENCE`
- `SHORTLIST`
- `NEXT_STEP`

## Stage responsibilities

### OPENING
Goal:
- get consent for a short diagnostic conversation.

Allowed behavior:
- brief framing only.

Forbidden:
- product recommendation,
- price discussion,
- evidence retrieval.

### DISCOVERY
Goal:
- collect minimum routing slots.

Required slots:
- `customer_goal`
- `transport_type`
- `decision_criterion`
- plus at least one of:
  - `route_type`
  - `body_type`
  - `competitor`
  - a positive special condition flag

### BRANCH_LOCK
Goal:
- determine one active problem branch.

If multiple branches compete:
- do not call branch tools;
- ask one priority question;
- re-enter branch classification.

### EVIDENCE
Goal:
- retrieve and explain proof, not product list.

Rules:
- exactly one branch-pack call per evidence cycle;
- maximum two read_doc calls;
- no shortlist unless evidence exists.

### SHORTLIST
Goal:
- propose 1–3 product families maximum.

Rules:
- no shortlist before evidence;
- every family must have `fit_reason`;
- if useful, include `risk_note`.

### NEXT_STEP
Goal:
- build role-aware package and align next action.

Rules:
- do not “send everything”;
- explain each document’s role.

## Branch rules

### `tco`
When:
- client centers the conversation on operating economics, total cost, value over time.

Tool:
- `get_tco_pack`

### `configuration`
When:
- client needs the right configuration / base / body fit.

Tool:
- `get_configuration_pack`

### `comparison`
When:
- client explicitly compares against a competitor.

Tool:
- `get_comparison_pack`

### `service_risk`
When:
- reliability, serviceability, downtime, quality concerns dominate.

Tool:
- `get_service_pack`

### `internal_approval`
When:
- client needs a package for management, procurement, finance, or technical review.

Tool:
- `get_internal_approval_pack`

### `passenger_route`
When:
- client selects passenger transport for a route/capacity profile.

Tool:
- `get_passenger_route_pack`

### `special_body`
When:
- client comes for a function-first need:
  refrigerator, crane, tow, garbage, tipper, tank, etc.

Tool:
- `get_special_body_pack`

### `special_conditions`
When:
- offroad / severe operating environment / municipal / heavy-load constraints dominate.

Tool:
- `get_special_conditions_pack`

### `unknown_selection`
When:
- information is insufficient.

Behavior:
- ask targeted question, do not use branch tools.

## Conflict resolution examples

### comparison + configuration
Ask:
> What is primary right now: selecting the right configuration, or proving it wins versus the competitor?

### tco + service_risk
Ask:
> Should we first validate operating economics, or first de-risk service and downtime?

### internal_approval + anything
Ask:
> Is the immediate need to understand the solution, or to assemble a package for internal approval?

## Core invariants

1. No product recommendation before branch lock.
2. No shortlist before evidence.
3. No final price before configuration + criterion are understood.
4. No branch-pack use when conflict is unresolved.
5. Follow-up package must be role-aware.
