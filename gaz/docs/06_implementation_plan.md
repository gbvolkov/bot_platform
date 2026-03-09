# 06. Implementation Plan

## Delivery strategy

Use parallel Codex workstreams, but merge in this order:

1. Foundation
2. Routing + policy
3. Evidence layer
4. Rendering + follow-up
5. Tests + evals
6. Cleanup + docs

## Phase 1 — Foundation

### Goals
- package layout
- enums and typed state
- config module
- graph skeleton
- placeholders for all nodes and tool adapters

### Deliverables
- importable package
- graph compiles with stub nodes
- mypy / ruff / pytest baseline

## Phase 2 — Routing and policies

### Goals
- slot completeness checks
- stage machine
- branch classification wrapper
- branch conflict resolution
- policy guards

### Deliverables
- tested `advance_stage_if_possible`
- tested `readiness_gate`
- tested `branch_router`
- tested “no shortlist before evidence”

## Phase 3 — Evidence layer

### Goals
- implement all pack adapters against mocked registry
- evidence subgraph
- read plan logic
- doc reading and evidence synthesis

### Deliverables
- subgraph compiles
- one happy-path integration test per major branch family

## Phase 4 — Shortlist and next-step

### Goals
- shortlist adapter
- follow-up pack adapter
- answer rendering nodes
- optional approval gate abstraction

### Deliverables
- end-to-end flow from discovery to follow-up

## Phase 5 — Evals

### Goals
- implement 7 scenario tests
- assert trace-level invariants
- add failure snapshots for debugging

### Deliverables
- eval harness
- pass/fail reporting

## Phase 6 — Hardening

### Goals
- tighten prompts
- tune edge cases
- improve traces
- simplify duplicated code
- document runbook

### Deliverables
- stable graph
- concise developer docs
- reproducible test instructions

## Recommended task split for Codex agents

### Codex Agent 1 — Core graph
Owns:
- state
- graph assembly
- stage machine
- transitions

### Codex Agent 2 — Router/policies
Owns:
- slot extraction wrappers
- branch router
- conflict resolution
- policy guards

### Codex Agent 3 — Tool adapters / registry
Owns:
- materials registry loader
- all pack builders
- read_doc
- shortlist
- follow-up pack builder

### Codex Agent 4 — Eval harness
Owns:
- fixtures
- transcript scenarios
- integration tests
- trace assertions

## Merge order

1. Agent 1
2. Agent 2
3. Agent 3
4. Agent 4

## Recommended checkpoints

- Checkpoint A: graph compiles
- Checkpoint B: router + policies stable
- Checkpoint C: evidence subgraph works
- Checkpoint D: 7 evals pass
