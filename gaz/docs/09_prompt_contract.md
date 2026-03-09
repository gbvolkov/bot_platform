# 09. Prompt Contract

This file explains how the system prompt is used inside the LangGraph runtime.

The most important implementation rule is this:

> **The graph controls behavior first. The prompt refines behavior second.**

The model must not be expected to enforce stage logic by itself.

---

## 1. Where the system prompt is used

The full system prompt from `08_system_prompt.md` should be attached only to **LLM-driven nodes**.

Recommended LLM nodes:

- `slot_extractor`
- `branch_router` (if using LLM-assisted routing)
- `ask_clarifying_question`
- `ask_targeted_question`
- `resolve_branch_conflict`
- `plan_doc_reads`
- `synthesize_evidence`
- `answer_with_shortlist`
- `answer_with_next_step`

Recommended deterministic Python nodes that should **not** use the system prompt directly:

- `ingest_user_turn`
- `readiness_gate`
- `select_pack`
- `read_doc_1`
- `read_doc_2`
- `shortlist_builder`
- `followup_pack_builder`
- `approval_gate`

---

## 2. Prompt layering model

Use a layered prompt structure.

### Layer A — global system prompt
The full prompt from `08_system_prompt.md`.

Purpose:
- overall agent identity,
- conversation discipline,
- anti-catalog behavior,
- high-level stage behavior.

### Layer B — node-specific instruction
Each LLM node should append a very short local instruction.

Examples:

#### `slot_extractor`
```text
Extract slots from the latest customer message. Do not recommend anything. Return structured data only.
```

#### `resolve_branch_conflict`
```text
The current case fits multiple branches. Ask one prioritization question that forces the customer to choose what is primary right now.
```

#### `synthesize_evidence`
```text
You already have the evidence documents. Explain the implications for the customer's problem in 2-4 concise points. Do not move into shortlist unless explicitly told by the graph.
```

#### `answer_with_shortlist`
```text
Present at most 3 product families. Each must include a fit reason. If relevant, include a risk note. Keep the explanation problem-first.
```

### Layer C — stage/context payload
Provide the model with a concise structured state summary, not the raw state dump.

---

## 3. Recommended state summary passed to LLM nodes

Use a compact summary object like this:

```json
{
  "stage": "EVIDENCE",
  "branch": "configuration",
  "problem_summary": "Customer needs a city refrigerator solution and wants to choose the right configuration before comparing economics.",
  "slots": {
    "transport_type": "cargo",
    "route_type": "city",
    "body_type": "refrigerator",
    "capacity_or_payload": "10-12 m3",
    "competitor": "sollers_atlant",
    "decision_criterion": "configuration",
    "decision_role": "owner"
  },
  "evidence_titles": [
    "САТ_Изотермические фургоны и рефрижераторы",
    "ГАЗель NEXT_База опции"
  ],
  "shortlist": [],
  "client_flags": {
    "requested_price": true,
    "requested_materials": false
  }
}
```

Do **not** pass huge raw trace dumps into every LLM node. Keep context narrow.

---

## 4. Prompt behavior by node

## `slot_extractor`

### Input
- latest user message,
- recent short conversation window,
- compact state summary.

### Output
Structured slot extraction only.

### Must not do
- recommend products,
- interpret branch,
- use evidence language.

---

## `branch_router`

### Input
- compact state summary,
- extracted slots.

### Output
Either:
- one locked branch,
- or explicit branch conflict,
- or `unknown_selection`.

### Must not do
- shortlist,
- document explanation,
- send next-step package.

---

## `ask_clarifying_question`

### Input
- missing slots list,
- state summary.

### Output
Exactly one focused question.

### Must not do
- ask multiple unrelated questions,
- introduce products,
- mention documents.

---

## `resolve_branch_conflict`

### Input
- branch candidates,
- state summary.

### Output
One prioritization question.

### Good example
"What is primary for you right now: choosing the right execution format, or proving that it is better than the alternative on economics?"

### Bad example
"We can do both, let me show you a few options..."

---

## `plan_doc_reads`

### Input
- evidence pack,
- state summary.

### Output
Select up to 2 documents and focus areas.

### Must optimize for
- minimal document reads,
- relevance to the current branch,
- no redundant reads.

---

## `synthesize_evidence`

### Input
- read doc summaries,
- state summary.

### Output
2-4 evidence-grounded conclusions.

### Must not do
- move into shortlist unless the graph has already transitioned,
- quote large blocks,
- over-explain product details.

---

## `answer_with_shortlist`

### Input
- shortlist output,
- state summary.

### Output
1-3 families, each with fit reason, optional risk note.

### Must not do
- list too many families,
- revert to catalog mode,
- reopen branch selection.

---

## `answer_with_next_step`

### Input
- follow-up pack,
- state summary.

### Output
Explain what will be sent and why, then ask for or confirm the next action.

### Must not do
- say "I'll send everything",
- provide an unstructured list,
- skip the next action.

---

## 5. When the prompt must be overridden by code

The graph must override the prompt whenever these conditions occur:

1. **No tools in early stages**
   - `OPENING` and `DISCOVERY` are tool-free by design.
2. **No shortlist before evidence**
   - even if the model tries to jump.
3. **No product recommendation before branch lock**
   - even if the model is confident.
4. **No final price before configuration logic**
   - unless a dedicated pricing action flow exists.
5. **No more than two document reads per evidence cycle**
   - enforce in code.

---

## 6. Prompt versioning policy

Create a dedicated module for prompt text, for example:

- `src/prompts/system_prompt.py`
- `src/prompts/node_prompts.py`

Expose explicit version tags:

```python
SYSTEM_PROMPT_VERSION = "v1.0.0"
```

Every prompt change must trigger:

- eval reruns,
- branch-regression check,
- evidence discipline check,
- shortlist timing check.

---

## 7. What Codex must not do when implementing prompts

Codex must not:

- inline huge prompt strings inside random node files,
- duplicate the system prompt in multiple places,
- mix deterministic routing rules into prompt-only logic,
- treat prompts as the primary policy layer.

Codex should:

- keep prompts centralized,
- keep node instructions short,
- keep policy in orchestration,
- keep prompts focused on linguistic behavior and reasoning quality.

---

## 8. Acceptance criteria for prompt integration

Prompt integration is considered correct only if:

1. LLM nodes receive the global system prompt plus node-local instructions.
2. Deterministic nodes do not rely on prompt behavior.
3. The graph can pass eval cases where:
   - the model does not jump into product too early,
   - the model asks for prioritization when branches conflict,
   - the model produces evidence before shortlist,
   - the model produces structured next steps.
