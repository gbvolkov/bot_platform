# AGENTS.md

## Mission

Построить production-grade sales-agent orchestration layer на **LangGraph/LangChain** для консультативных продаж коммерческого транспорта, пассажирского транспорта и спецтехники.

Система должна быть **problem-first**, а не product-first.

Главный разговорный протокол:

1. Понять задачу клиента.
2. Зафиксировать критерий выбора.
3. Выбрать проблемную ветку.
4. Поднять доказательства из материалов.
5. Сформировать shortlist только после evidence.
6. Согласовать следующий шаг и пакет документов.

## Non-goals

Не делать:
- “простого” ReAct-агента с кучей tools;
- one-shot chatbot без orchestration;
- UI, CRM-интеграции и отправку email в первой версии;
- генерацию коммерческих предложений;
- embedding/retrieval платформу “с нуля”, если достаточно mocked adapters.

## Architecture rules (must follow)

1. Использовать **LangGraph `StateGraph`** как основной orchestration layer.
2. Использовать **deterministic Python nodes** там, где выбор действия можно вычислить без LLM.
3. Использовать **LLM only where it adds value**:
   - slot extraction,
   - ambiguous branch classification,
   - read-plan selection,
   - evidence synthesis,
   - answer rendering.
4. Не давать LLM свободно выбирать business tools, если выбор однозначно определяется stage/branch.
5. Реализовать:
   - parent graph,
   - evidence subgraph,
   - policy guards,
   - traceable state,
   - eval harness.
6. Если в коде появляется HITL/interrupt:
   - закладывать checkpointer-aware API,
   - делать идемпотентный код до точки interrupt.

## Required stages

`OPENING`
`DISCOVERY`
`BRANCH_LOCK`
`EVIDENCE`
`SHORTLIST`
`NEXT_STEP`

Нельзя перескакивать стадии произвольно.

## Required branches

- `tco`
- `configuration`
- `comparison`
- `service_risk`
- `internal_approval`
- `passenger_route`
- `special_body`
- `special_conditions`
- `unknown_selection`

## Required invariants

1. До `BRANCH_LOCK` нельзя рекомендовать продукт.
2. До `EVIDENCE` нельзя делать shortlist.
3. До понимания конфигурации и критерия выбора нельзя обсуждать финальную цену.
4. При конфликте веток нельзя вызывать branch-pack tools.
5. На один evidence-цикл:
   - максимум 1 branch-pack call,
   - максимум 2 `read_doc`.
6. Пакет follow-up должен быть role-aware.

## Coding standards

- Python 3.11+
- Pydantic v2 for schemas where useful
- Strong typing
- Small, pure functions where possible
- No hidden global state
- State transitions must be testable
- All routers and guards must have unit tests
- Graph compilation must be covered by integration tests
- Use docstrings on every public node/tool adapter

## Project conventions

- Parent graph in `src/sales_agent/graph.py`
- State model in `src/sales_agent/state.py`
- Branch router in `src/sales_agent/router.py`
- Evidence subgraph in `src/sales_agent/subgraphs/evidence.py`
- Tools adapters in `src/sales_agent/tools/`
- Prompts in `src/sales_agent/prompts/`
- Policies / guards in `src/sales_agent/policies/`
- Eval scenarios in `tests/evals/`
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`

## Work strategy for Codex agents

### Agent A — Foundation
Build:
- package layout
- state types
- enums
- config
- graph skeleton
- checkpointer abstraction
- typed interfaces

### Agent B — Routing & Policy
Build:
- slot completeness checks
- branch router
- branch conflict logic
- stage transitions
- policy guards

### Agent C — Evidence Layer
Build:
- branch pack adapters
- read_doc adapter
- evidence subgraph
- shortlist builder adapter
- followup pack adapter

### Agent D — Testing & Evals
Build:
- fixtures
- mocked materials registry
- 7 eval scenarios
- invariants tests
- integration tests over graph traces

## Definition of done

A task is done only if:
1. Code is typed and tested.
2. Invariants are enforced in code, not only in prompts.
3. Parent graph compiles.
4. At least 7 end-to-end eval scenarios pass.
5. Trace demonstrates:
   - no product recommendation before branch lock,
   - no shortlist before evidence,
   - conflict resolution before tool use,
   - role-aware follow-up pack.

## Required deliverables

- working LangGraph implementation
- mocked registry-backed tool adapters
- test suite
- sample trace outputs
- concise developer README section on how to run tests

## External references for implementers

- Codex + AGENTS.md: https://openai.com/index/introducing-codex/
- LangGraph Graph API / Command: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangChain HITL middleware: https://docs.langchain.com/oss/python/langchain/human-in-the-loop
