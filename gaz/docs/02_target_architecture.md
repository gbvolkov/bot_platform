# 02. Target Architecture

## Chosen approach

Use a **deterministic parent graph** in LangGraph with:
- explicit stages,
- branch-specific evidence selection,
- graph-controlled tool execution,
- optional human-in-the-loop only for external actions.

This is preferred over a single free-form tool-calling agent because the sales workflow has hard ordering constraints.

## Why this architecture

### Problems with a simple agent
A single ReAct-style agent tends to:
- jump to product suggestions;
- overuse tools;
- mix multiple branches;
- produce non-auditable traces.

### Benefits of graph orchestration
A graph gives:
- explicit stage management;
- explicit routing;
- hard guardrails;
- testable invariants;
- inspectable traces.

## Parent graph

Canonical sequence:

`OPENING -> DISCOVERY -> BRANCH_LOCK -> EVIDENCE -> SHORTLIST -> NEXT_STEP`

But the graph must also support controlled fallback:
- `BRANCH_LOCK -> DISCOVERY`
- `EVIDENCE -> BRANCH_LOCK`
- `SHORTLIST -> EVIDENCE`
- `NEXT_STEP -> EVIDENCE`

## Evidence subgraph

One subgraph reused for all business branches.

It does four things:
1. Selects the correct pack tool for the active branch.
2. Plans which 1–2 documents are worth reading.
3. Reads those documents.
4. Synthesizes evidence into a user-facing explanation.

## Key design choice: deterministic tool execution

The model should **not** be free to choose business tools in most nodes.

Preferred split:
- LLM nodes for interpretation and synthesis.
- Python nodes for deterministic tool calls.

This keeps tool usage aligned with graph state.

## Use of LangGraph primitives

### `StateGraph`
Use `StateGraph` as the main orchestration primitive.

### `add_conditional_edges`
Use for simple routing decisions based only on state.

### `Command`
Use when a node needs to:
- update state,
- and route to a different node in the same step.

### Subgraphs
Use a dedicated evidence subgraph invoked from the parent graph.

### Interrupts / HITL
Do not use interrupts in the main sales reasoning flow.
Reserve interrupts for external actions such as:
- send email,
- create CRM record,
- trigger quote workflow.

## Why subgraphs should be called from nodes

Subgraphs should be invoked from node functions, not hidden behind tool-wrapped subagents, because:
- nested state remains inspectable;
- traces stay readable;
- resume semantics around interrupts are clearer;
- orchestration stays explicit.

## High-level node inventory

Parent graph:
- `ingest_user_turn`
- `slot_extractor`
- `readiness_gate`
- `ask_clarifying_question`
- `branch_router`
- `resolve_branch_conflict`
- `evidence_subgraph`
- `evidence_gate`
- `ask_targeted_question`
- `shortlist_builder`
- `answer_with_evidence`
- `answer_with_shortlist`
- `followup_pack_builder`
- `approval_gate` (optional)

Evidence subgraph:
- `select_pack`
- `plan_doc_reads`
- `read_doc_1`
- `read_doc_2`
- `synthesize_evidence`

## Non-functional requirements

- durable execution ready
- inspectable traces
- deterministic policy enforcement
- type-safe state updates
- test coverage for stage transitions and branch conflicts
