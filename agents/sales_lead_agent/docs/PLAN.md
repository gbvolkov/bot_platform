# Implementation Plan for `sales_lead_agent`

## Summary

Реализация ведется по текущему [TR.md](/C:/Projects/bot_platform/agents/sales_lead_agent/docs/TR.md) как по единственному source of truth. Агент реализуется в `agents/sales_lead_agent` как user-facing `deepagent` с free tool-calling loop, unified conversational context between calls, и internal classifier runnable для semantic decisions.

Ключевая модель состояния:
- `thread/session state` живет между пользовательскими calls и хранит conversational context, `active_run_id`, последние факты, последние structured classifier outputs и normalized answer artifacts.
- `acquisition_run` создается, когда запрос инициирует procurement/open-source acquisition; он создает `run_id`, workspace и `index_id`.
- Один ответ формируется на один turn, но `active_run_id` и связанные артефакты могут переиспользоваться в последующих turns до `reset`, явного нового поиска или явного переключения контекста.

## Implementation Changes

### A. Runtime, state, reuse patterns

- `F00 Bootstrap Validation`
  - Проверить canonical environment в активированном `.venv`.
  - Обязательные команды:
    - `.venv\Scripts\python.exe -c "import deepagents, rag_lib"`
    - `.venv\Scripts\python.exe -c "import zakupki_crawler"`
  - Дополнительно выполнить минимальный `create_deep_agent(...)` smoke.
  - Это validation step, не feature разработки; implementation начинается только после green bootstrap.

- `F01 Package Skeleton and Runtime Wiring`
  - Создать пакет `agents/sales_lead_agent` с `agent.py`, `prompts.py`, `state.py`, `schemas.py`, `tools.py`/`tools/`, `settings.py`, internal `services/`.
  - Runtime должен переиспользовать deepagent bootstrap pattern из [agent.py#L21](/C:/Projects/bot_platform/agents/mycroft_agent/agent.py#L21) и [agent.py#L52](/C:/Projects/bot_platform/agents/mycroft_agent/agent.py#L52), если нет документированного основания отклониться.
  - Main runtime должен хранить `thread/session state` отдельно от `acquisition_run state`.
  - `active_run_id` должен жить между calls и использоваться по умолчанию для follow-up document/evidence questions.
  - Legacy `tests/unit/test_sales_lead_logic.py` должен быть удален как часть feature work.

- `F02 Internal Classifier Runnable`
  - Classifier — это `per-call internal runnable`.
  - Его `state` — ephemeral внутри одной classifier invocation.
  - У него нет собственного `checkpointer`.
  - У него нет собственного `tool loop`.
  - Его outputs записываются в `main runtime state` как structured artifacts.
  - Он используется минимум для:
    - `TaskUnderstandingResult`
    - enrichment interpretation
    - risk verification
    - procurement-hit relevance filtering
  - Любая semantic classification выполняется только через него; deterministic classification запрещена.

### B. Procurement and retrieval flow

- `F03 Procurement Query Builder & Relevance Filter`
  - Подсистема 1: deterministic URL/template builder.
    - Базовый procurement URL template должен использовать approved `zakupki.gov.ru` extended-search template, предоставленный пользователем.
    - Agent не должен копировать user message verbatim в `searchString`.
    - Agent должен строить `searchString` из контекста задачи, с нормализацией в query fragments вида `substr1+substr2+substr3`.
    - `searchString` должен выражать search intent, а не raw phrasing пользователя.
  - Подсистема 2: LLM-based relevance classifier.
    - После получения hits от procurement search агент не должен сохранять в `main state` и не должен включать в `lead_list` нерелевантные procurement results.
    - Relevance filtering должно идти через internal classifier runnable, не через deterministic rules.
    - Пример: при теме страховых услуг агент может получить результаты с общим словом `услуг`, но должен сохранять только semantically relevant insurance-related hits.
  - На выходе feature должно быть четко разделено:
    - `raw procurement hits`
    - `classified relevant hits`
    - `dropped non-relevant hits` с reason artifact для debugging/testability.

- `F04 Document Preparation Service`
  - Реализовать internal parse/index service на `rag_lib`.
  - Использовать `services/kb_manager/utils/loader.py` для `PDF/DOCX/XLSX`.
  - Хранить unified `run-scoped index` и metadata per chunk.
  - Все procurement/open-source artifacts проходят через этот service до использования в reasoning.

- `F05 purchase_search_tool`
  - Реализовать procurement acquisition через internal `purchase_adapter` поверх `zakupki_crawler`.
  - Поддержать:
    - direct `search_url`
    - deterministic build from contextualized `search_filters`
  - Tool обязан:
    - получить procurement hits
    - прогнать relevance classifier
    - скачать документы только для релевантных hits
    - вызвать document preparation
    - записать результат в `current_run.index_id`
  - В `main state` и в final answer попадают только classified relevant hits.

- `F06 open_source_fetch_tool`
  - Реализовать web fetch + document preparation + unified index write.
  - Использовать `rag_lib` loader path.
  - Возвращать prepared artifacts, пригодные для `doc_search_tool`.

- `F07 doc_search_tool`
  - Искать по `active_run_id.index_id` по умолчанию.
  - Поддерживать `source_kind` и `bundle_id` filters.
  - Для paginated sources возвращать `page`.
  - Для non-paginated sources возвращать `page=null` и `locator`.

- `F08 Enrichment Tools`
  - Реализовать `counterparty_scoring_tool` и `counterparty_fssp_tool`.
  - Вызовы допускаются только после появления нормализованного INN в main state.
  - Runtime guard “facts first, enrichment second” обязателен.

### C. Main agent behavior and answering

- `F09 Main Deepagent Runtime`
  - Собрать user-facing `deepagent` runtime.
  - Logical phases `Understand Task`, `Collect Facts`, `Enrich & Verify`, `Compose Answer` остаются behavioral, а не graph nodes.
  - Runtime обязан поддерживать continuous discussion:
    - reuse `active_run_id`
    - reuse normalized facts/evidence between calls
    - не терять procurement/document context на follow-up question
    - создавать новый run только при новом acquisition context, `reset` или explicit user switch

- `F10 Final Answer Contract`
  - Формировать exact normalized answer contract по `TR.md`.
  - Human-readable response строить только из normalized contract.
  - Никаких новых фактов поверх structured artifacts.

- `F11 Simulator`
  - Реализовать scripted + interactive CLI simulator.
  - Simulator CLI/testing pattern должен переиспользовать existing repo pattern из [test_simulate_sysadmin_agent_dialog.py#L12](/C:/Projects/bot_platform/tests/unit/test_simulate_sysadmin_agent_dialog.py#L12), если нет документированного основания отклониться.
  - Simulator должен покрывать все 5 сценариев из TR.
  - Для simulator tests использовать stubbed adapters/classifier outputs там, где это ускоряет локальную разработку, но это не заменяет real integration smoke.

- `F12 Traceability, Acceptance, Post-Implementation Config Note`
  - Подготовить final implementation matrix:
    - `TR -> code path`
    - `TR -> tests`
    - `TR -> simulator scenario`
  - После завершения implementation и тестов отдельно выполнить note-level config alignment:
    - привести `data/load.json` в соответствие с TR
    - убрать `allow_raw_attachments`
    - обновить description/supported content types под pilot scope
  - Это делается после implementation, но до final acceptance.

## Acceptance Layers

- `L1 Local/Stubbed Verification`
  - Fast local unit and component verification.
  - Допускает stubs/mocks для procurement, scoring, fssp, classifier.
  - Нужен для feature-by-feature progress.

- `L2 Real Integration Smoke`
  - Реальный smoke для `purchase_adapter`/EIS path.
  - Проверяет:
    - deterministic query builder
    - procurement hit retrieval
    - relevance filtering path
    - document download/preparation handoff
  - Не заменяется stubbed tests.

- `L3 Simulator End-to-End`
  - Полный simulator behavior по всем 5 сценариям.
  - Должен подтверждать conversational behavior, evidence handling, answer typing и missing-data policy.

Final acceptance requires all three layers: `L1 + L2 + L3`.

## Command Matrix

- `F00 Bootstrap Validation`
  - `.venv\Scripts\python.exe -c "import deepagents, rag_lib"`
  - `.venv\Scripts\python.exe -c "import zakupki_crawler"`
  - `.venv\Scripts\python.exe -c "from agents.mycroft_agent.agent import _import_create_deep_agent; print(callable(_import_create_deep_agent()))"`

- `F01 Package Skeleton and Runtime Wiring`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent or registry"`
  - `.venv\Scripts\python.exe -m compileall agents/sales_lead_agent`

- `F02 Internal Classifier Runnable`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and classifier"`

- `F03 Procurement Query Builder & Relevance Filter`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and procurement_query"`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and relevance_filter"`

- `F04 Document Preparation Service`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and document_preparation"`

- `F05 purchase_search_tool`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and purchase_search_tool"`

- `F06 open_source_fetch_tool`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and open_source_fetch_tool"`

- `F07 doc_search_tool`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and doc_search_tool"`

- `F08 Enrichment Tools`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and (scoring_tool or fssp_tool or enrichment)"`

- `F09 Main Deepagent Runtime`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and runtime"`

- `F10 Final Answer Contract`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and answer_contract"`

- `F11 Simulator`
  - `.venv\Scripts\python.exe -m pytest -q tests/unit -k "sales_lead_agent and simulator"`

- `L2 Real Integration Smoke`
  - `.venv\Scripts\python.exe -m pytest -q tests/integration -k "sales_lead_agent and procurement"`

- `Milestone A`
  - `.venv\Scripts\python.exe simulate_sales_lead_agent.py --scenario procurement_search`
  - `.venv\Scripts\python.exe simulate_sales_lead_agent.py --scenario fact_lookup`

- `Milestone B`
  - `.venv\Scripts\python.exe simulate_sales_lead_agent.py --scenario procurement_analysis`
  - `.venv\Scripts\python.exe simulate_sales_lead_agent.py --scenario company_check`
  - `.venv\Scripts\python.exe simulate_sales_lead_agent.py --scenario comparison`

- `Final Pass`
  - `.venv\Scripts\python.exe simulate_sales_lead_agent.py --all-scenarios`

Command names for new tests/scripts are part of the implementation contract and should be created accordingly if they do not yet exist.

## Test and Scenario Expectations

- Procurement query builder:
  - builds `searchString` from context, not raw user text
  - supports multi-substring query via `+`
  - preserves deterministic template structure

- Procurement relevance filter:
  - keeps semantically relevant hits
  - drops semantically irrelevant hits even if lexical overlap exists
  - uses classifier runnable, not deterministic rules

- Continuous discussion:
  - follow-up turns reuse prior `active_run_id`
  - evidence lookup can happen after a prior procurement answer without forced reacquisition
  - new acquisition starts only on explicit/new search context

- Simulator pass/fail:
  - correct `answer_type`
  - expected tools actually used
  - searchable prepared index exists where required
  - evidence/provenance present
  - missing data not hallucinated
  - explicit next step when data insufficient
  - fact statuses correctly labeled

## Assumptions

- Current `TR.md` is the only normative spec for implementation.
- User-confirmed active `.venv` environment is the canonical environment for bootstrap validation.
- Procurement dependency path is `zakupki_crawler`, used via internal `purchase_adapter`.
- `load.json` alignment is intentionally postponed until implementation and tests are complete, but must be completed before final acceptance.
