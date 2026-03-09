# 05. Recommended Repo Structure

```text
repo/
├─ AGENTS.md
├─ README.md
├─ pyproject.toml
├─ src/
│  └─ sales_agent/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ state.py
│     ├─ graph.py
│     ├─ router.py
│     ├─ guards.py
│     ├─ reducers.py
│     ├─ orchestration/
│     │  ├─ stage_machine.py
│     │  ├─ transitions.py
│     │  └─ trace.py
│     ├─ nodes/
│     │  ├─ ingest.py
│     │  ├─ slot_extractor.py
│     │  ├─ clarifying.py
│     │  ├─ branch_router.py
│     │  ├─ conflict_resolution.py
│     │  ├─ answers.py
│     │  ├─ shortlist.py
│     │  └─ followup.py
│     ├─ subgraphs/
│     │  ├─ __init__.py
│     │  └─ evidence.py
│     ├─ prompts/
│     │  ├─ slot_extractor.py
│     │  ├─ branch_router.py
│     │  ├─ read_plan.py
│     │  ├─ evidence_synthesis.py
│     │  └─ answer_rendering.py
│     ├─ tools/
│     │  ├─ __init__.py
│     │  ├─ base.py
│     │  ├─ registry.py
│     │  ├─ classify_branch.py
│     │  ├─ tco_pack.py
│     │  ├─ configuration_pack.py
│     │  ├─ comparison_pack.py
│     │  ├─ service_pack.py
│     │  ├─ approval_pack.py
│     │  ├─ passenger_route_pack.py
│     │  ├─ special_body_pack.py
│     │  ├─ special_conditions_pack.py
│     │  ├─ read_doc.py
│     │  ├─ shortlist.py
│     │  └─ followup_pack.py
│     └─ schemas/
│        ├─ branch.py
│        ├─ packs.py
│        ├─ shortlist.py
│        └─ followup.py
├─ tests/
│  ├─ unit/
│  │  ├─ test_stage_machine.py
│  │  ├─ test_router.py
│  │  ├─ test_guards.py
│  │  └─ test_pack_selectors.py
│  ├─ integration/
│  │  ├─ test_parent_graph.py
│  │  ├─ test_evidence_subgraph.py
│  │  └─ test_shortlist_flow.py
│  └─ evals/
│     ├─ fixtures/
│     │  ├─ materials_registry.json
│     │  └─ transcripts/
│     ├─ test_eval_city_refrigerator.py
│     ├─ test_eval_vague_best_option.py
│     ├─ test_eval_service_risk.py
│     ├─ test_eval_internal_approval.py
│     ├─ test_eval_passenger_route.py
│     ├─ test_eval_special_body.py
│     └─ test_eval_offroad_conditions.py
└─ docs/
   ├─ 01_project_brief.md
   ├─ 02_target_architecture.md
   ├─ 03_graph_spec.md
   ├─ 04_state_tools_contracts.md
   ├─ 05_repo_structure.md
   ├─ 06_implementation_plan.md
   └─ 07_eval_and_acceptance.md
```

## Notes

### `src/sales_agent/state.py`
Single source of truth for stage, slots, branch, evidence, shortlist and audit.

### `src/sales_agent/graph.py`
Graph assembly only. Avoid business logic here.

### `src/sales_agent/router.py`
Deterministic + LLM-assisted branch routing logic.

### `src/sales_agent/subgraphs/evidence.py`
Reusable evidence retrieval and synthesis subgraph.

### `src/sales_agent/tools/`
Keep adapters thin. They should transform state inputs into registry queries and normalized outputs.

### `tests/evals/`
These are not generic unit tests. They are behavior tests over traces and invariants.
