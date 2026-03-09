# Codex Delivery Pack: LangGraph Sales Agent

Этот пакет подготовлен как стартовый набор для Codex-агентов-разработчиков.

## Что внутри

- `AGENTS.md` — правила работы Codex в репозитории, роли, ограничения, definition of done.
- `docs/01_project_brief.md` — продуктовая постановка задачи и ожидаемый результат.
- `docs/02_target_architecture.md` — целевая архитектура решения на LangGraph/LangChain.
- `docs/03_graph_spec.md` — детальная спецификация графа, стадий, маршрутизации и subgraph-ов.
- `docs/04_state_tools_contracts.md` — контракты state, tools, guardrails и trace.
- `docs/05_repo_structure.md` — рекомендуемая структура репозитория и модулей.
- `docs/06_implementation_plan.md` — план реализации по этапам для нескольких Codex-агентов.
- `docs/07_eval_and_acceptance.md` — eval-план, acceptance criteria и тестовые сценарии.
- gaz-docs - набор документов с информацией о доступных продуктах. На основе этой информации надо построить индекс и RAG

## Ключевые архитектурные решения

1. Не строить одного “свободного” tool-calling sales-агента.
2. Построить **детерминированный parent graph** с явными стадиями:
   `OPENING -> DISCOVERY -> BRANCH_LOCK -> EVIDENCE -> SHORTLIST -> NEXT_STEP`.
3. Использовать **branch-specific evidence subgraph** вместо хаотического retrieval.
4. Делать **tool execution в Python nodes**, а не отдавать выбор tools LLM там, где логика может быть детерминирована.
5. Использовать `Command` там, где node должен одновременно обновить state и сменить маршрут.
6. Если появятся внешние действия (CRM, email, quote) — вынести их в отдельный action-layer с HITL/interrupt.

## Полезные внешние источники

- OpenAI: Codex может управляться файлами `AGENTS.md`: https://openai.com/index/introducing-codex/
- OpenAI: Codex как multi-agent coding environment: https://openai.com/codex/
- LangGraph Graph API / `Command`: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph use-subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- LangGraph interrupts / HITL: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangChain HITL middleware: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

## Как использовать этот пакет

1. Дайте Codex прочитать `AGENTS.md`.
2. Пусть первый Codex-агент создаст skeleton проекта по `docs/05_repo_structure.md`.
3. Второй агент реализует state + router + graph.
4. Третий агент реализует tools/adapters и fake test fixtures.
5. Четвертый агент собирает eval harness и acceptance tests.

## Added prompt docs

The package now also includes:

- `docs/08_system_prompt.md` — canonical production system prompt for the sales agent.
- `docs/09_prompt_contract.md` — rules for how prompts are attached to LangGraph nodes and where prompt logic ends and orchestration logic begins.
