# Технические требования к `sales_lead_agent`

## 1. Основание документа

Документ подготовлен на основании функционального задания `FZ_Pilot_simplified_v2.docx` для предпродажного пилота "Интеллектуальный агент поиска и первичной квалификации потенциальных клиентов по закупкам, открытым источникам и внешним API".

Цель документа: зафиксировать технические требования к реализации агента `sales_lead_agent` в репозитории `bot_platform` без выхода за рамки пилота и с учетом принятых архитектурных решений по ходу проектирования.

### 1.1. Обозначения

- `FR-*` — функциональные требования из раздела 6 исходного ФЗ.
- `SR-*` — остальные обязательные требования исходного ФЗ: стек, интеграции, контракты, workflow, сценарии и критерии приемки.
- `UR-*` — дополнительные требования, добавленные пользователем после первичной редакции документа.
- `TR-*` — технические требования этого документа.

## 2. Границы реализации

### 2.1. Что входит в реализацию

- Один чат-агент для одного пользователя.
- Работа по задачам в свободной форме без профилей поиска и профилей пользователя.
- Main runtime на `deepagent` со свободным tool-calling loop как основным orchestration path.
- Internal classifier subagent как внутренний structured runnable/service со своим state и summary-only input.
- Минимальный публичный toolset из 5 tools:
  - `purchase_search_tool`;
  - `open_source_fetch_tool`;
  - `doc_search_tool`;
  - `counterparty_scoring_tool`;
  - `counterparty_fssp_tool`.
- Встроенная подготовка документов внутри `purchase_search_tool` и `open_source_fetch_tool`: acquisition/download/fetch, parse, index, извлечение сущностей.
- Единый `run-scoped index` как default search scope.
- Краткий, фактический и объяснимый ответ в одном из четырех типов: `lead_list`, `lead_card`, `company_check`, `comparison`.
- Локальный симулятор со scripted mode и interactive CLI mode.

### 2.2. Что явно не входит в реализацию

- Профили поиска.
- Профили пользователя и ролевая модель.
- Интеграции с CRM, АИС, Spark, Контур, SCAN, Outlook.
- Автоматическая постановка задач, рассылка, сложная маршрутизация.
- Отдельный data-platform контур, BI-витрины и длительное версионирование документов.
- User-uploaded raw attachments в пилоте.

### 2.3. Граница по пользовательским файлам

В пилоте агент не должен принимать raw attachments от пользователя. На уровне регистрации агента и платформенной конфигурации должен быть зафиксирован boundary-level reject этого входного канала, а не deferred обработка внутри бизнес-логики агента. Для пилота `allow_raw_attachments` должен быть выключен, а `supported_content_types` в registry должны быть пустыми, чтобы платформа не рекламировала неподдерживаемый пользовательский upload path.

## 3. Требования к размещению в репозитории и структуре пакета

### TR-001. Расположение агента

Реализация должна находиться в каталоге `agents/sales_lead_agent/`.

### TR-002. Точка входа агента

Пакет должен предоставлять `initialize_agent(...)` в файле `agents/sales_lead_agent/agent.py` в соответствии с общими правилами репозитория.

### TR-003. Разделение по файлам

Минимально должны быть предусмотрены:

- `agent.py` — сборка main runtime на `deepagent`, регистрация публичных tools и wiring internal services.
- `prompts.py` — системный prompt main agent и prompts для internal classifier subagent.
- `state.py` — схема состояния main runtime и схемы состояния internal classifier subagent.
- `schemas.py` — typed request/response schemas публичных tools, internal classifier outputs и final answer contract.
- `tools.py` или `tools/` — реализации 5 публичных tools.
- `services/` или эквивалентный внутренний модуль:
  - internal classifier runnable/service;
  - internal document-preparation service;
  - purchase/open-source/scoring/fssp adapters.
- `settings.py` — agent-local configuration surface.
- `docs/TR.md` — данный документ.

### TR-004. Регистрация агента

При внедрении агент должен быть зарегистрирован в конфигурации платформы с `id="sales_lead_agent"`, отдельными `name`, `description`, `factory`, `default_provider`, `supported_content_types` и с отключенным raw attachment passthrough для пилота. В pilot configuration `supported_content_types` должны быть пустыми, а описание агента не должно обещать export workflows или иной scope за пределами procurement discovery, document-backed evidence, open-source enrichment и counterparty checks.

### TR-005. Использование общих механизмов репозитория

Для создания модели и платформенной интеграции необходимо использовать общие механизмы репозитория, в первую очередь `agents/utils.py`, `agents/llm_utils.py` и контракт `bot_service/agent_registry.py`. Реализация не должна вводить отдельную, параллельную платформу исполнения.

## 4. Обязательный стек и интеграции

### TR-010. Базовый runtime и оркестрация

Пользовательский runtime агента должен быть реализован на `deepagent` с основным свободным tool-calling loop.

### TR-011. Платформа исполнения

Агент должен работать внутри `bot_platform` как штатный агент платформы. Архитектура реализации должна быть гибридной:

- main runtime на `deepagent`;
- internal classifier subagent как structured LangChain runnable/service;
- публичные tools и внутренние adapters внутри пакета агента.

### TR-012. Обязательные внешние интеграции

В реализации должны присутствовать интеграции со следующими обязательными компонентами:

- `purchase_adapter`;
- `rag_lib`;
- `API-Скоринг`;
- `API-ФССП`.

`rag_pipeline` может использоваться как будущая совместимая оболочка, но не является базовым implementation path для пилота.

### TR-013. Источник доступа к закупкам

`purchase_search_tool` должен работать через внутренний `purchase_adapter`, а `purchase_adapter` должен использовать `purchase_scraper` как базовую библиотеку обхода ЕИС.

### TR-014. Источник разбора документов

Подготовка документов и поиск по ним должны выполняться через `rag_lib` напрямую:

- загрузка документов через `services/kb_manager/utils/loader.py` для `PDF/DOCX/XLSX`;
- web fetch через `rag_lib` loader path;
- embeddings/vector store/indexer через `rag_lib`.

### TR-015. Обязательная подготовка закупочных документов

Каждый файл, загруженный `purchase_search_tool` в рамках закупки, должен автоматически проходить internal document-preparation service до того, как этот артефакт может быть использован в reasoning, search или enrichment.

### TR-016. Запрет обхода этапа подготовки документов

Использование `downloaded_files`, `attachments` или иных acquisition artifacts напрямую, без обязательного этапа `parse/index/retrieval` через internal document-preparation service на базе `rag_lib`, не допускается.

### TR-017. Конфигурация секретов

Ключи и параметры подключения к внешним API должны поступать из переменных окружения и конфигурации платформы. Секреты не должны быть зашиты в код или в репозиторий.

### TR-018. Запрет предопределенных semantic-pattern классификаций

Агент не должен использовать предопределенные паттерны, rule-based классификаторы, keyword-матрицы, жесткие decision trees или иные hardcoded эвристики для семантической классификации на любом этапе работы.

Это требование распространяется как минимум на:

- определение интента и типа пользовательской задачи;
- семантическую интерпретацию результатов enrichment;
- верификацию и классификацию рисков;
- присвоение приоритета и формирование рекомендаций, если они требуют смысловой оценки, а не простого копирования результата инструмента.

Все такие классификации должны выполняться через structured response от облегченной LLM-модели класса `mini` или `nano`, вызываемой internal classifier subagent.

Детерминированная логика допускается только для технических операций, не являющихся семантической классификацией: валидация схемы, нормализация форматов, дедупликация, file routing, evidence grouping и вызов внешних API.

### TR-019. Обязательная проверка runtime/bootstrap в `.venv`

Разработка, тестирование, smoke-проверки и симулятор должны опираться на `.venv` как на обязательное execution environment. До начала реализации должны быть подтверждены:

- импорт `deepagents` из `.venv`;
- импорт `rag_lib` из `.venv`;
- доступность procurement dependency path из `.venv`;
- успешный минимальный smoke создания main runtime на `deepagent`.

## 5. Архитектура агента

### TR-020. Модель взаимодействия

Агент должен поддерживать ровно один пользовательский контур: один пользователь ставит задачи агенту в чате в свободной форме.

### TR-021. Отсутствие профилей

Все фильтры и условия поиска должны задаваться пользователем в конкретном диалоге. Управление профилями поиска и профилями пользователя в реализацию не включается.

### TR-022. Основной orchestration path

Логика агента должна быть реализована как свободный `deepagent` tool-calling loop. Жесткий node-by-node graph для пилота не допускается как основной orchestration path.

### TR-023. Логические фазы без жесткого графа

Логические фазы `Understand Task`, `Collect Facts`, `Enrich & Verify`, `Compose Answer` сохраняются только как поведенческая дисциплина prompts, state и acceptance criteria. Они не должны реализовываться как обязательные отдельные nodes графа.

### TR-024. Internal classifier subagent: intent classification

Определение типа задачи и типа ответа должно выполняться internal classifier subagent. Этот компонент обязан:

- быть внутренним runnable/service, а не публичным агентом и не публичным tool;
- иметь собственный state;
- получать только summary-only input, необходимый для конкретной классификации;
- не иметь собственного свободного tool loop;
- возвращать только typed structured output.

### TR-025. Сбор фактов и подготовка источников

На фазе `Collect Facts` main runtime должен вызывать `purchase_search_tool` и/или `open_source_fetch_tool`. Эти tools обязаны самостоятельно:

- скачать или прочитать source artifacts;
- запустить internal document-preparation service;
- записать артефакты в текущий `run-scoped index`;
- вернуть ready-to-search результат без отдельного публичного шага parse/index.

### TR-026. Enrichment и risk verification через classifier subagent

Семантическая интерпретация результатов `doc_search_tool`, `counterparty_scoring_tool`, `counterparty_fssp_tool` и `open_source_fetch_tool` должна выполняться только через internal classifier subagent с typed structured outputs. Main runtime не должен принимать semantic decisions на основе pattern tables или keyword heuristics.

### TR-027. Сборка ответа

На фазе `Compose Answer` main runtime должен формировать итог в одном из утвержденных форматов, прикладывать причины, приоритет, подтверждающие фрагменты и корректную маркировку provenance.

### TR-028. Обработка недостаточности данных

Если данных недостаточно для уверенного ответа, агент обязан:

- явно указать, каких данных не хватает;
- не придумывать недостающие сведения;
- предложить следующий шаг в рамках того же диалога;
- сохранить `missing_data` и `recommended_next_step` во внутреннем normalized answer contract.

### TR-029. Состояние runtime и unified run-scoped index

Схема состояния main runtime должна хранить как минимум:

- исходный запрос пользователя;
- `run_id`;
- распознанный тип задачи;
- structured outputs internal classifier subagent;
- исходные критерии поиска;
- `search_url`, если он передан пользователем;
- найденные закупки;
- данные open-source fetch;
- `prepared_documents`;
- текущий `index_id`;
- `acquisition_status`, `acquisition_attempts[]`, `last_acquisition_error`;
- `active_run_ready`;
- извлеченные факты из документов;
- найденные ИНН и названия компаний;
- результаты `API-Скоринг`;
- результаты `API-ФССП`;
- массив `evidence`;
- список `missing_data`;
- `recommended_next_step`;
- normalized final answer.

`run-scoped index` должен быть единственным default search scope для одного acquisition run. Один acquisition run создается запросом, который инициировал acquisition и привел к формированию подготовленного корпуса. Этот `active_run_id` и связанный `index_id` должны сохраняться между последующими turns в том же thread/session и переиспользоваться для follow-up вопросов до `reset`, явного нового acquisition context или явного переключения контекста. `doc_search_tool` по умолчанию ищет по `active_run_id.index_id`, если иной `index_id` не передан явно.

Если acquisition attempt завершился `failed` или не сформировал `active_run_ready=true`, такой run не должен заменять текущий `active_run_id`/`index_id` в conversational context. Агент обязан сохранять structured acquisition artifacts (`acquisition_status`, `acquisition_attempts[]`, `last_acquisition_error`) и использовать их как источник истины для explicit partial answers.

Reuse follow-up context должен опираться на `active_run_ready=true` и наличие searchable prepared corpus. Наличие только `procurement_hits` или других незавершенных acquisition artifacts без готового searchable corpus не должно считаться достаточным основанием для reuse.

Повторяющиеся идентичные user messages в разных turns должны все равно инициировать новый turn reset для turn-scoped validation/tool state; агент не должен считать такие turns одинаковыми только по совпадению raw text.

## 6. Требования к инструментам и внутренним сервисам

### TR-030. Минимальный обязательный набор public tools

В агенте должны быть реализованы и доступны для вызова следующие публичные инструменты:

- `purchase_search_tool`;
- `open_source_fetch_tool`;
- `doc_search_tool`;
- `counterparty_scoring_tool`;
- `counterparty_fssp_tool`.

Подготовка документов должна быть реализована как internal document-preparation service и не должна публиковаться как отдельный public tool.

### TR-031. Exact request/response schema: `purchase_search_tool`

`purchase_search_tool` должен принимать строгий request schema:

```json
{
  "run_id": "string | null",
  "search_url": "string | null",
  "query_text": "string | null",
  "law": "44-FZ | 223-FZ | null",
  "region": "string | null",
  "min_price": "number | null",
  "max_price": "number | null",
  "published_from": "YYYY-MM-DD | null",
  "published_to": "YYYY-MM-DD | null",
  "submission_deadline_from": "YYYY-MM-DD | null",
  "submission_deadline_to": "YYYY-MM-DD | null",
  "customer_name": "string | null",
  "customer_inn": "string | null",
  "supplier_hint": "string | null",
  "max_pages": "integer | null",
  "headless": "boolean | null"
}
```

Правила вызова:

- должен быть задан либо `search_url`, либо хотя бы один flat search-parameter;
- если передан `search_url`, он используется как есть без модификации;
- если переданы только критерии, URL строится helper-логикой `purchase_adapter`, а не LLM.

`purchase_search_tool` должен возвращать exact response schema:

```json
{
  "source": "purchase_adapter",
  "run_id": "string",
  "index_id": "string",
  "status": "success | partial | failed",
  "errors": ["string"],
  "items": [
    {
      "bundle_id": "string",
      "registry_number": "string",
      "law": "44-FZ | 223-FZ | null",
      "purchase_title": "string",
      "customer_name": "string",
      "price_text": "string | null",
      "published_at": "YYYY-MM-DD | null",
      "updated_at": "YYYY-MM-DD | null",
      "submission_deadline": "YYYY-MM-DD | null",
      "detail_url": "string",
      "common_info_url": "string | null",
      "documents_url": "string | null",
      "document_urls": ["string"],
      "downloaded_files": ["string"],
      "prepared_document_ids": ["string"],
      "documents_json": "string | null",
      "common_info_json": "string | null",
      "lots_json": "string | null",
      "crawl_status": "success | partial | failed",
      "crawl_error": "string | null",
      "crawl_ts_utc": "ISO-8601"
    }
  ],
  "prepared_documents": [
    {
      "$ref": "#/definitions/prepared_document"
    }
  ]
}
```

### TR-032. Обязательная запись procurement artifacts в current run index

После успешного или частично успешного ответа `purchase_search_tool` все загруженные procurement artifacts должны быть автоматически подготовлены и записаны в `current_run.index_id`. Агент не должен принимать дополнительное решение о том, индексировать ли их.

### TR-033. Exact request/response schema: `open_source_fetch_tool`

`open_source_fetch_tool` должен принимать exact request schema:

```json
{
  "run_id": "string | null",
  "url": "string",
  "depth": "integer | null",
  "follow_download_links": "boolean | null",
  "max_concurrency": "integer | null"
}
```

`open_source_fetch_tool` должен:

- быть реализован через `rag_lib` web loader path;
- загружать страницы и attachments из открытых источников;
- запускать internal document-preparation service для полученных artifacts;
- записывать их в `current_run.index_id`.

`open_source_fetch_tool` должен возвращать exact response schema:

```json
{
  "source": "rag_lib",
  "run_id": "string",
  "index_id": "string",
  "status": "success | partial | failed",
  "errors": ["string"],
  "pages": [
    {
      "bundle_id": "string",
      "url": "string",
      "title": "string | null",
      "text": "string",
      "attachments": ["string"],
      "prepared_document_ids": ["string"]
    }
  ],
  "prepared_documents": [
    {
      "$ref": "#/definitions/prepared_document"
    }
  ]
}
```

### TR-034. Internal document-preparation service

Internal document-preparation service должен:

- принимать acquisition artifacts от `purchase_search_tool` и `open_source_fetch_tool`;
- поддерживать `PDF`, `DOCX`, `XLSX`, `HTML` и другие plain-text-compatible источники пилота;
- извлекать текст и сущности;
- индексировать chunks в local Chroma или эквивалентном `rag_lib` vector store;
- возвращать normalized `prepared_document` объекты;
- обеспечивать одинаковый pipeline для procurement и open-source artifacts.

### TR-035. Exact `prepared_document` schema and extracted entities

`prepared_document` должен иметь exact schema:

```json
{
  "document_id": "string",
  "origin": "purchase | open_source",
  "bundle_id": "string",
  "registry_number": "string | null",
  "source_url": "string | null",
  "original_source_url": "string | null",
  "original_file_name": "string | null",
  "original_content_type": "string | null",
  "derived_artifact_path": "string | null",
  "file_path": "string",
  "file_name": "string",
  "file_type": "pdf | docx | xlsx | html | other",
  "parse_status": "success | partial | failed",
  "index_status": "ready | failed",
  "text_excerpt": "string",
  "entities": {
    "inn": ["string"],
    "company_names": ["string"],
    "emails": ["string"],
    "phones": ["string"],
    "dates": ["string"],
    "amounts": ["string"]
  },
  "chunks_count": "integer",
  "error": "string | null"
}
```

Если документ успешно загружен, но не дал ни одного indexable/searchable chunk, `parse_status` должен быть `partial`, `index_status` должен быть `failed`, и такой document не должен считаться searchable support ни в runtime logic, ни в simulator checks.

Для transformed web/open-source artifacts `file_path` / `file_name` / `file_type` могут описывать derived indexed artifact, но `prepared_document` обязан сохранять original-vs-derived provenance через `original_source_url`, `original_file_name`, `original_content_type` и `derived_artifact_path`.

### TR-036. Exact request/response schema: `doc_search_tool`

`doc_search_tool` должен принимать exact request schema:

```json
{
  "index_id": "string | null",
  "query": "string",
  "top_k": "integer | null",
  "source_kind": "purchase | open_source | null",
  "bundle_id": "string | null"
}
```

`doc_search_tool` должен:

- по умолчанию использовать `current_run.index_id`, если `index_id` не передан;
- при явном `index_id` разрешать его через explicit index-to-run mapping, а не через допущение `index_id == run_id`;
- поддерживать optional narrowing по `source_kind` и `bundle_id`;
- искать только по уже подготовленному индексу;
- не генерировать snippet свободным пересказом.

`doc_search_tool` должен возвращать exact response schema:

```json
{
  "index_id": "string",
  "matches": [
    {
      "document_id": "string",
      "bundle_id": "string",
      "file_path": "string",
      "page": "integer | null",
      "locator": "string | null",
      "snippet": "string",
      "score": "number",
      "source_kind": "purchase | open_source",
      "source_url": "string | null"
    }
  ]
}
```

Для paginated sources `page` должен быть заполнен. Для non-paginated sources `page=null`, а позиция должна быть возвращена через `locator`.

### TR-037. Exact request/response schema: `counterparty_scoring_tool`

`counterparty_scoring_tool` должен принимать exact request schema:

```json
{
  "inn": "string",
  "model": "string | null",
  "include_fincoefs": "boolean | null"
}
```

`counterparty_scoring_tool` должен вызывать обязательные endpoints `GET /scoring/score` и, при необходимости, `GET /scoring/fincoefs`.

`counterparty_scoring_tool` должен возвращать exact response schema:

```json
{
  "source": "damia_scoring",
  "status": "success | failed",
  "error": "string | null",
  "inn": "string",
  "score": {
    "risk_value": "number | null",
    "risk_zone": "string | null",
    "score_value": "number | null",
    "score_zone": "string | null",
    "reliability_value": "number | null",
    "reliability_zone": "string | null",
    "top_factors": [
      {
        "name": "string",
        "value": "number | null",
        "nwoe": "number | null"
      }
    ]
  },
  "fincoefs": [
    {
      "name": "string",
      "value": "number | null",
      "norm": "number | null",
      "comparison": "string | null"
    }
  ]
}
```

### TR-038. Exact request/response schema: `counterparty_fssp_tool`

`counterparty_fssp_tool` должен принимать exact request schema:

```json
{
  "inn": "string",
  "from_date": "YYYY-MM-DD | null",
  "to_date": "YYYY-MM-DD | null",
  "format": "1 | 2 | null"
}
```

`counterparty_fssp_tool` должен вызывать обязательный endpoint `GET /fssp/isps` и возвращать exact response schema:

```json
{
  "source": "damia_fssp",
  "status": "success | failed",
  "error": "string | null",
  "inn": "string",
  "grouped": [
    {
      "year": "integer",
      "status": "string",
      "subject": "string",
      "amount": "number | null",
      "count": "integer",
      "proceeding_ids": ["string"]
    }
  ],
  "raw_format": "1 | 2"
}
```

### TR-039. Нормализация ошибок tools и partial results

Ошибки вызова инструментов должны отражаться в состоянии агента и в итоговом ответе без фабрикации данных. Каждый public tool обязан возвращать нормализованный `status`, структурированную ошибку и partial result, если часть данных все же была получена.

Recoverable provider/classifier/tool-validation failures не должны hard-abort'ить turn. Они должны записываться в turn-scoped validation artifacts, отражаться в `missing_data` и `recommended_next_step`, а итоговый ответ должен деградировать в явный partial answer. Отсутствие обязательных tools для текущего turn также должно приводить к explicit partial answer, а не к nominally complete answer.

Для acquisition tools (`purchase_search_tool`, `open_source_fetch_tool`) агент обязан сохранять нормализованное structured failure state в runtime state и строить degraded answer только из trusted state artifacts, а не из unsupported model-authored explanations о причинах ошибок.

Если main runtime получает recoverable provider error или structured-output validation error при формировании final answer, turn должен деградировать в explicit partial answer через trusted state artifacts. Main runtime не должен скрыто переключать response strategy или silently retry structured-output path за пользователя.

## 7. Требования к извлечению, нормализации и проверке фактов

### TR-040. Обязательные опорные факты

Агент должен уметь извлекать и нормализовывать как минимум:

- ИНН;
- название компании;
- предмет закупки;
- сумму;
- сроки;
- регион;
- контакты;
- дополнительные подтверждающие факты из закупочной документации.

### TR-041. Источник истины по документам

Для анализа содержания документов источником истины должен быть только parsed/indexed content из `run-scoped index`. Raw acquisition artifacts не могут считаться источником истины для reasoning.

### TR-042. Поиск подтверждений

Если пользователь просит указать, где именно в документах находится конкретный факт, агент обязан использовать `doc_search_tool`, а не пересказывать найденное без ссылки на источник и locator.

### TR-043. Runtime ordering guard

Main runtime обязан соблюдать следующий guard:

- сначала acquisition и preparation sources;
- затем документный поиск и извлечение опорных фактов;
- затем enrichment через внешние API;
- затем semantic interpretation и final answer.

`counterparty_scoring_tool` и `counterparty_fssp_tool` не должны вызываться, пока в state не появился нормализованный ИНН. Если acquisition path уже вернул документы или attachments, enrichment не должен выполняться до завершения их подготовки и записи в индекс.

### TR-044. Обогащение по ИНН

При наличии нормализованного ИНН агент должен уметь:

- запросить риск-профиль компании через `counterparty_scoring_tool`;
- запросить сводку по исполнительным производствам через `counterparty_fssp_tool`.

### TR-045. Дополнение открытыми источниками

`open_source_fetch_tool` должен использоваться только как дополнение к карточке лида или проверке компании, когда пользователь просит дополнить закупку или проверить компанию по публичным материалам.

### TR-046. Явная маркировка статуса факта

Для каждого существенного факта агент должен уметь отразить один из статусов:

- `document`;
- `external_api`;
- `open_source`;
- `not_found`.

### TR-047. Internal classifier-only enrichment and risk semantics

При интерпретации результатов `counterparty_scoring_tool`, `counterparty_fssp_tool`, `open_source_fetch_tool` и `doc_search_tool` агент должен выполнять смысловую классификацию только через internal classifier subagent. Использование предопределенных шаблонов, фиксированных паттернов и жестких правил классификации для этих решений не допускается.

### TR-048. Exact structured schema: `TaskUnderstandingResult`

Internal classifier subagent в режиме `intent` должен возвращать exact structured schema:

```json
{
  "answer_type": "lead_list | lead_card | company_check | comparison",
  "task_kind": "procurement_search | procurement_analysis | company_check | fact_lookup | comparison",
  "search_url": "string | null",
  "search_filters": {
    "query_text": "string | null",
    "law": "44-FZ | 223-FZ | null",
    "region": "string | null",
    "min_price": "number | null",
    "max_price": "number | null",
    "published_from": "YYYY-MM-DD | null",
    "published_to": "YYYY-MM-DD | null",
    "submission_deadline_from": "YYYY-MM-DD | null",
    "submission_deadline_to": "YYYY-MM-DD | null",
    "customer_name": "string | null",
    "customer_inn": "string | null",
    "supplier_hint": "string | null"
  },
  "requested_company_inns": ["string"],
  "comparison_targets": ["string"],
  "document_questions": ["string"],
  "needs_purchase_search": "boolean",
  "needs_open_source": "boolean",
  "needs_doc_search": "boolean",
  "needs_enrichment": "boolean",
  "missing_data": ["string"]
}
```

### TR-049. Exact structured schemas: `EnrichmentAssessmentResult` и `RiskVerificationResult`

Internal classifier subagent должен возвращать exact structured schema в режимах `enrichment` и `risk_verification`:

```json
{
  "priority": "high | medium | low | unknown",
  "reasons": ["string"],
  "risk_summary": "string",
  "manual_review_required": "boolean",
  "significant_signals": ["string"],
  "fact_statuses": [
    {
      "fact_key": "string",
      "status": "document | external_api | open_source | not_found"
    }
  ],
  "recommended_next_step": "string | null"
}
```

## 8. Требования к финальному ответу

### TR-050. Exact normalized final answer schema

Финальный ответ агента должен быть приводим к exact normalized contract:

```json
{
  "answer_type": "lead_list | lead_card | company_check | comparison",
  "summary": "string",
  "items": [
    {
      "$ref": "#/definitions/lead_answer_item"
    }
  ],
  "missing_data": ["string"],
  "recommended_next_step": "string | null"
}
```

### TR-051. Поддерживаемые типы ответа

`answer_type` должен принимать только следующие значения:

- `lead_list`;
- `lead_card`;
- `company_check`;
- `comparison`.

### TR-052. Exact answer item schema

Каждый элемент `items` должен иметь exact schema:

```json
{
  "company_name": "string | null",
  "inn": "string | null",
  "event_title": "string | null",
  "source_url": "string | null",
  "region": "string | null",
  "amount_text": "string | null",
  "contacts": ["string"],
  "scoring": "object | null",
  "fssp": "object | null",
  "priority": "high | medium | low | unknown",
  "reasons": ["string"],
  "evidence": [
    {
      "$ref": "#/definitions/evidence_item"
    }
  ],
  "fact_statuses": [
    {
      "fact_key": "string",
      "status": "document | external_api | open_source | not_found"
    }
  ]
}
```

### TR-053. Exact `evidence` schema

Каждый элемент `evidence` должен иметь exact schema:

```json
{
  "source": "purchase | document | open_source | scoring | fssp",
  "source_url": "string | null",
  "file_path": "string | null",
  "page": "integer | null",
  "locator": "string | null",
  "snippet": "string"
}
```

### TR-054. Обязательная объяснимость ответа

Итоговый ответ должен содержать:

- краткое заключение;
- причины присвоения приоритета;
- подтверждающие фрагменты;
- явное разделение между подтвержденными и неподтвержденными данными;
- следующий шаг, если данных недостаточно.

### TR-055. Запрет на выдуманные данные

Агент не должен придумывать отсутствующие данные. Если факт не найден или инструмент не вернул подтверждение, это должно быть явно отражено в `fact_statuses`, `missing_data` и/или `recommended_next_step`.

User-visible answer должен явно рендерить `fact_statuses` по каждому item, а не хранить их только во внутреннем normalized contract.

### TR-056. Форма ответа

Ответ пользователю должен быть кратким, деловым и фактическим. Даже если внутренне используется structured contract, человекочитаемый ответ в чате должен оставаться компактным и не содержать новых фактов поверх нормализованного контракта.

## 9. Требования по поддерживаемым сценариям и симулятору

### TR-060. Сценарий "найти новые релевантные закупки"

Агент должен поддерживать сценарий поиска новых закупок с выдачей списка лидов. В этом сценарии `purchase_search_tool` обязан вернуть уже подготовленные procurement documents и записать их в `current_run.index_id` до формирования итогового списка.

Если для сценария `procurement_search` `purchase_search_tool` не был вызван в текущем turn или acquisition завершился recoverable failure without trusted result, агент обязан вернуть explicit partial answer и не должен формировать nominally complete `lead_list`.

### TR-061. Сценарий "разобрать одну закупку и документы"

Агент должен поддерживать сценарий детального разбора одной закупки с выдачей `lead_card`, где документные факты подтверждаются через `doc_search_tool` и `evidence`.

### TR-062. Сценарий "проверить компанию по ИНН"

Агент должен поддерживать сценарий `company_check` с обязательным использованием `counterparty_scoring_tool` и `counterparty_fssp_tool`, а `open_source_fetch_tool` использовать опционально.

### TR-063. Сценарий "найти подтверждение факта в документах"

Агент должен поддерживать сценарий точечного поиска факта в документах и возвращать короткие snippets с указанием файла и `page`/`locator`.

### TR-064. Сценарий "сравнить несколько компаний"

Агент должен поддерживать сценарий `comparison` с ранжированием компаний, краткой сравнительной выдачей, причинами и рекомендацией по приоритету передачи продавцу.

### TR-065. Поддержка примеров без жесткого ограничения

Агент должен поддерживать примеры сценариев из ФЗ, но не ограничиваться только ими.

### TR-066. Обязательная разработка симулятора агента

Для `sales_lead_agent` должен быть разработан симулятор, предназначенный для локальной проверки поведения агента по сценариям пилота и для ручного интерактивного прогона.

Симулятор должен строиться по паттернам, уже используемым для симуляторов других агентов в репозитории.

### TR-067. Покрытие всех обязательных сценариев симулятором

Симулятор должен содержать преднастроенные сценарии прогона как минимум для всех обязательных сценариев агента:

- поиск новых релевантных закупок;
- разбор одной закупки и документов;
- проверка компании по ИНН;
- поиск подтверждения факта в документах;
- сравнение нескольких компаний.

Для каждого такого сценария симулятор должен уметь запускать полный диалоговый прогон и сохранять результат прогона в виде транскрипта, пригодного для ручной проверки.

Помимо этих 5 обязательных single-turn сценариев симулятор должен содержать scripted multi-turn follow-up сценарии, которые проверяют:

- reuse одного и того же `thread_id` внутри сценария;
- корректное reuse `active_run_id` для follow-up document/evidence turns;
- отсутствие silent finalization без обязательных tools и evidence.

### TR-068. CLI-режим симулятора

Симулятор должен поддерживать CLI-режим, в котором пользователь взаимодействует с инструментом вручную через терминал.

CLI-режим должен поддерживать как минимум:

- выбор сценария или запуск без заранее выбранного сценария;
- старт новой сессии;
- отправку произвольных пользовательских сообщений агенту;
- просмотр ответов агента по ходам;
- команды `help`, `reset`, `exit`;
- сохранение или вывод транскрипта диалога.

### TR-069. Технический формат симулятора

Симулятор должен быть реализован как локальный инженерный инструмент репозитория и использовать общие паттерны существующих симуляторов:

- отдельный запускаемый entry point на уровне репозитория;
- конфигурацию через аргументы командной строки;
- изолированный `thread_id` и `run_id` для каждого прогона;
- сохранение логов или markdown/plain-text транскриптов в файловую систему;
- наличие как scripted mode, так и interactive CLI mode.

## 10. Требования к временному хранению и артефактам пилота

### TR-070. Unified run workspace

Для каждого run должен создаваться отдельный рабочий контур с:

- папкой скачанных procurement documents;
- папкой web/open-source artifacts;
- единым индексом `current_run.index_id`;
- связкой между `run_id`, workdir, `bundle_id`, `document_id` и `index_id`.

Связь между `run_id` и `index_id` должна храниться явно в run metadata и/или отдельном index registry. Реализация не должна полагаться на неявное равенство `run_id == index_id` в публичных контрактах.

Один run начинается при обработке пользовательского запроса, который инициировал acquisition, и завершается после формирования одного финального ответа. Новый scripted run, CLI `reset` или новая сессия должны создавать новый `run_id`.

### TR-071. Source metadata в индексе

Каждый chunk в `run-scoped index` должен содержать как минимум следующие metadata-поля:

- `source_kind`;
- `source_url`;
- `bundle_id`;
- `document_id`;
- `registry_number`, если источник относится к закупке;
- `page` и/или `locator`.

### TR-072. Хранение временных документов и отсутствие обязательной выгрузки

Срок хранения скачанных документов и временных индексов не должен быть зашит в бизнес-логику агента. Он должен задаваться конфигурацией.

Экспорт результата в `JSON` или `CSV` может быть подготовлен позднее, но не является обязательной частью пилота и не должен блокировать запуск базовой чат-версии агента.

## 11. Критерии приемки и правила верификации

### TR-080. Работа в чате

Один пользователь должен иметь возможность поставить задачу и получить ответ без настройки профилей.

### TR-081. Поиск закупок

Агент должен уметь получать результаты через `purchase_search_tool`, включая автоматическую подготовку документов и запись в `run-scoped index`.

### TR-082. Подготовка документов

Агент должен уметь подготавливать `PDF/DOCX/XLSX` и извлекать опорные факты; для open-source path должен поддерживаться также `HTML`.

### TR-083. Поиск по документу

Агент должен уметь показывать подтверждающий фрагмент и источник через `doc_search_tool`, включая `page` для paginated sources и `locator` для non-paginated sources.

### TR-084. Обогащение

Агент должен уметь вызывать `API-Скоринг` и `API-ФССП` по нормализованному ИНН.

### TR-085. Объяснимость

Ответ должен содержать причины, приоритет, `evidence`, `fact_statuses` и не должен скрывать `missing_data`.

### TR-086. Гибкость

Агент должен поддерживать примеры сценариев из ФЗ, но не ограничиваться только ими.

### TR-087. Наличие симулятора

В поставке пилота должен присутствовать рабочий симулятор `sales_lead_agent`, запускаемый локально из репозитория.

### TR-088. Покрытие сценариев симулятором

Симулятор должен уметь воспроизводить все пять обязательных сценариев из `TR-060` - `TR-064`, а также дополнительные scripted multi-turn follow-up сценарии, и сохранять результаты каждого прогона в виде проверяемого транскрипта.

### TR-089. Работоспособность CLI-режима

Симулятор должен предоставлять CLI-режим ручного взаимодействия, в котором пользователь может вести диалог с агентом через терминал и управлять сессией командами `help`, `reset`, `exit`.

### TR-090. Все команды разработки, тестов и симулятора выполняются через `.venv`

Все команды реализации, unit/integration tests, registry smoke checks и simulator runs должны выполняться через интерпретатор и зависимости из `.venv`.

### TR-091. Verification gate after each implemented feature

После завершения каждой feature обязательно должны быть выполнены и пройти ее feature-specific tests или smoke checks. Переход к следующей feature без успешного verification gate не допускается.

### TR-092. Milestone simulator run A и B

Помимо финального simulator pass должны выполняться два промежуточных прогона:

- milestone A после реализации acquisition + preparation + `doc_search_tool`;
- milestone B после реализации enrichment + classifier + final answer contract.

### TR-093. Final simulator run across all 5 scenarios

После завершения разработки должен быть выполнен полный simulator pass по всем 5 обязательным сценариям.

### TR-094. Simulator pass/fail rubric

Каждый simulator scenario считается успешно пройденным только если одновременно выполняются все условия:

- сформирован корректный `answer_type`;
- использованы обязательные для сценария tools;
- подготовленный индекс searchable;
- ответ содержит `evidence` и provenance;
- отсутствующие данные не выдуманы;
- при недостатке данных указан явный следующий шаг;
- источник факта корректно помечен как `document`, `external_api`, `open_source` или `not_found`.

### TR-095. Обязательные действия при провале финального simulator pass

Если финальный simulator pass дает некорректный результат, команда обязана:

- подготовить test report с входами, ожидаемым поведением, фактическим поведением и suspected root cause;
- подготовить новый development plan на доработку;
- не считать разработку завершенной до повторного успешного полного simulator pass.

## 12. Матрица покрытия требований

### 12.1. Функциональные требования раздела 6 ФЗ

| ID | Функциональное требование | Покрывающие технические требования |
| --- | --- | --- |
| FR-01 | Агент должен принимать задания в свободной форме от одного пользователя в чате. | TR-020, TR-080 |
| FR-02 | Агент должен уметь работать без заранее созданных профилей поиска: критерии задаются прямо в запросе пользователя. | TR-021, TR-048, TR-080 |
| FR-03 | Агент должен уметь инициировать поиск закупок через `purchase_search_tool`. | TR-025, TR-030, TR-031, TR-081 |
| FR-04 | Агент должен уметь скачивать и разбирать документы закупки. | TR-014, TR-015, TR-016, TR-025, TR-032, TR-034, TR-035, TR-082 |
| FR-05 | Агент должен уметь искать подтверждающие фрагменты в документах через `doc_search_tool`. | TR-036, TR-042, TR-083 |
| FR-06 | Агент должен уметь обогащать компанию через `counterparty_scoring_tool` и `counterparty_fssp_tool`. | TR-037, TR-038, TR-043, TR-044, TR-084 |
| FR-07 | Агент должен уметь дополнять карточку данными из открытых источников через `open_source_fetch_tool`. | TR-025, TR-033, TR-045 |
| FR-08 | Агент должен давать краткий, фактический и объяснимый ответ без придумывания отсутствующих данных. | TR-027, TR-028, TR-050, TR-054, TR-055, TR-056, TR-085 |
| FR-09 | Агент должен явно показывать, какие данные подтверждены документами, какие — внешними API, какие — открытыми источниками, а какие не найдены. | TR-046, TR-049, TR-052, TR-053, TR-085 |

### 12.2. Дополнительная матрица покрытия остальных обязательных требований исходного ФЗ

| ID | Обязательное требование исходного ФЗ | Покрывающие технические требования |
| --- | --- | --- |
| SR-01 | В пилот входит поиск по `zakupki.gov.ru` через обязательный инструмент `purchase_adapter`. | TR-012, TR-013, TR-031 |
| SR-02 | В пилот входит получение и разбор документов закупки через `rag_lib` или совместимый pipeline. | TR-012, TR-014, TR-015, TR-016, TR-034, TR-035, TR-041 |
| SR-03 | Агент должен извлекать ИНН, название компании, предмет закупки, сумму, сроки, регион, контакты и другие опорные факты. | TR-035, TR-040, TR-041, TR-042 |
| SR-04 | Агент должен обогащать найденную компанию через `API-Скоринг` и `API-ФССП`. | TR-012, TR-037, TR-038, TR-044, TR-084 |
| SR-05 | Ответ пользователю должен быть в виде краткой объяснимой карточки лида и/или списка лидов. | TR-027, TR-050, TR-051, TR-054, TR-056 |
| SR-06 | Агент должен поддерживать примеры диалогов из документа без ограничения только этими сценариями. | TR-065, TR-086 |
| SR-07 | В пилоте должен использоваться `bot_platform` и approved orchestration stack. | TR-010, TR-011, TR-019 |
| SR-08 | Замена обязательных компонентов без отдельного согласования не допускается. | TR-012, TR-013, TR-014, TR-016 |
| SR-09 | Все procurement artifacts должны быть переданы на downstream-разбор и анализ. | TR-015, TR-016, TR-025, TR-032 |
| SR-10 | В пилоте должен быть минимальный инструментальный контур для поиска, подготовки документов, document search и enrichment. | TR-030, TR-031, TR-033, TR-034, TR-036, TR-037, TR-038 |
| SR-11 | `purchase_search_tool` работает через `purchase_adapter`, реализованный на базе `purchase_scraper`. | TR-013, TR-031 |
| SR-12 | Финальный ответ должен быть приводим к внутреннему структурированному контракту. | TR-050, TR-051, TR-052, TR-053 |
| SR-13 | Сначала собирать факты из закупки и документов, потом обогащать через внешние API. | TR-025, TR-043, TR-044 |
| SR-14 | Использовать поиск по документам, если ответ зависит от содержимого PDF/DOCX/XLSX/HTML. | TR-025, TR-034, TR-036, TR-042 |
| SR-15 | Агент должен использовать упрощенную логическую схему работы без жесткого графа. | TR-022, TR-023, TR-024, TR-025, TR-026, TR-027 |
| SR-16 | Агент должен поддерживать 5 примеров сценариев из документа. | TR-060, TR-061, TR-062, TR-063, TR-064, TR-065 |
| SR-17 | Один пользователь должен получить ответ без настройки профилей. | TR-020, TR-021, TR-080 |
| SR-18 | Агент должен уметь показать подтверждающий фрагмент и источник. | TR-036, TR-042, TR-053, TR-083 |
| SR-19 | Ответ должен содержать причины, приоритет и `evidence`. | TR-049, TR-052, TR-053, TR-054, TR-085 |
| SR-20 | Пилот должен оставаться гибким и не ограничиваться только примерами. | TR-023, TR-065, TR-086 |

### 12.3. Матрица покрытия дополнительных пользовательских требований

| ID | Дополнительное требование | Покрывающие технические требования |
| --- | --- | --- |
| UR-01 | Агент не должен использовать предопределенные паттерны на этапах intent classification, enrichment и verifying risks; такие классификации должны выполняться через structured responses от lightweight LLM класса `mini` или `nano`. | TR-018, TR-024, TR-026, TR-047, TR-048, TR-049 |
| UR-02 | Для агента должен быть разработан симулятор по паттернам симуляторов других агентов; симулятор должен покрывать все сценарии и иметь CLI-режим ручного взаимодействия. | TR-066, TR-067, TR-068, TR-069, TR-087, TR-088, TR-089 |

## 13. Проверка полноты покрытия

Проверка на уровне данного документа показывает следующее:

- все 9 функциональных требований раздела 6 ФЗ покрыты явными `TR-*` пунктами;
- approved runtime architecture зафиксирована непосредственно в ТЗ, без внешнего addendum:
  - `sales_lead_agent` как единственный package/module path;
  - free `deepagent` loop как основной orchestration path;
  - internal classifier subagent как отдельный structured runnable/service;
  - 5 public tools и internal document-preparation service;
  - unified `run-scoped index` как default search scope;
  - boundary-level reject raw attachments для пилота;
  - `.venv`-only execution policy для разработки, тестов и симулятора;
  - feature verification gates, milestone simulator runs и final failure policy.
- все обязательные tool contracts, structured classifier schemas и final answer schemas зафиксированы в явном виде;
- дополнительное пользовательское требование по запрету предопределенных паттернов и обязательному использованию structured responses от lightweight LLM класса `mini` или `nano` отражено в `TR-018`, `TR-024`, `TR-026`, `TR-047`, `TR-048` и `TR-049`;
- дополнительное пользовательское требование по разработке симулятора с покрытием всех сценариев и CLI-режимом отражено в `TR-066`, `TR-067`, `TR-068`, `TR-069`, `TR-087`, `TR-088` и `TR-089`;
- дополнительные требования по качеству реализации и верификации отражены в `TR-019`, `TR-090`, `TR-091`, `TR-092`, `TR-093`, `TR-094` и `TR-095`.

Следовательно, документ покрывает исходные функциональные требования, закрепляет согласованные архитектурные решения пилота и не оставляет конфликтующих legacy-требований из прежней редакции.
