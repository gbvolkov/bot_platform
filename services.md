# Архитектура Bot Platform (расширенное описание)

Документ описывает три ключевых сервиса: `bot_service`, `openai_proxy` и `task_worker`. Для каждого сервиса приведены назначение, конфигурация, структуры данных, последовательности действий, а также детальные API (назначение, внутренняя логика, взаимодействия, форматы запросов и ответов).

---

## 1. bot_service — внутренний API и исполнение агентов

### 1.1 Назначение
- Принимает запросы на создание разговоров и сообщений.
- Лениво инициализирует агентов, хранит их в реестре, управляет статусами готовности.
- Нормализует и сохраняет историю диалога (разговоры и сообщения) в БД.
- Обрабатывает вложения: определяет типы, извлекает текст, прокидывает поддерживаемые типы в агент.
- Синхронно вызывает агента (LangGraph/LCEL) с поддержкой прерываний и резюма.

### 1.2 Конфигурация (bot_service/config.py)
- `BOT_SERVICE_DATABASE_URL` — SQLAlchemy URL (async, MySQL/MariaDB).
- `BOT_SERVICE_DEFAULT_MODEL_PROVIDER` — `openai|yandex|mistral|gigachat`.
- `BOT_SERVICE_DEFAULT_USER_ROLE` — роль по умолчанию при отсутствии заголовка.
- `BOT_SERVICE_ALLOW_RESET_COMMAND` — разрешает `type="reset"`.
- `BOT_SERVICE_REQUEST_TIMEOUT_SECONDS` — таймаут вызова агента (используется в инфраструктуре агента).
- `BOT_SERVICE_API_BASE_PATH` — префикс API (`/api`).

### 1.3 Модель данных (bot_service/models.py)
- Таблица `conversations`:
  - `id` (UUID str), `agent_id`, `user_id`, `user_role`.
  - `status`: `pending` (агент инициализируется), `active`, `waiting_user` (агент вернул interrupt).
  - `metadata` (JSON) — хранит, например, `pending_interrupt`.
  - `title`, `created_at`, `updated_at`, `last_message_at`.
- Таблица `messages`:
  - `id`, `conversation_id` (FK, каскадное удаление).
  - `role`: `user` | `assistant`.
  - `content`: нормализованный payload (segments/text/attachments).
  - `raw_text`: строковое представление сообщения.
  - `metadata`: произвольные поля (для ассистента сюда кладутся `agent_status`, `attachments`, `interrupt_payload`, `question/content`).
  - `created_at`.

### 1.4 Реестр агентов (bot_service/agent_registry.py)
- Структура `AgentDefinition`: `id`, `name`, `description`, `factory(provider)`, `default_provider`, `supported_content_types`.
- Провайдеры маппятся из строк в `ModelType`: `openai|yandex|mistral|gigachat`.
- Генерируются динамические агенты по продуктовым каталогам в `data/docs` (`product_<name>`).
- Состояния инициализации: `pending` → `initializing` (executor) → `ready` или `error`.
- Ключевые методы:
  - `ensure_agent_ready(agent_id)` — запускает инициализацию при первом обращении, возвращает готовность, поднимает ошибку при падении.
  - `get_agent(agent_id)` — отдаёт готовый инстанс или ошибку, если не готов/сломался.
  - `supported_content_types(agent_id)` — список допустимых типов вложений.
  - `preload_all()` — стартует инициализацию всех агентов при запуске приложения.

### 1.5 Нормализация сообщений и вложений
- Схемы (bot_service/schemas.py):
  - `AttachmentPayload { filename, content_type?, data?, text? }`.
  - `MessagePayload { type: text|reset, text?, metadata: dict, attachments: [AttachmentPayload] }`.
  - `MessageCreate { payload: MessagePayload }`.
- Нормализация (bot_service/service.py):
  - `_normalise_content`, `_extract_text`, `_extract_attachments`, `serialise_message` приводят LangChain-сообщения к JSON-формату для хранения.
  - `build_human_message` собирает `HumanMessage` из текста, сегментов вложений и `attachment_text_segments`.
- Обработка вложений (bot_service/attachments.py):
  - Типизация по расширению/миме (`ContentType`: images/pdfs/text/md/docx/csv/excels/sounds/videos).
  - Поддерживаемые типы берутся из `AgentDefinition.supported_content_types`. Неподдерживаемые пытаются конвертироваться в текст через `services.kb_manager.utils.load_single_document` (Unstructured/JSONLoader/etc).
  - Метаданные `ProcessedAttachment`: категория, поддерживается ли агентом, есть ли текст, была ли конверсия, ошибка.
  - `attachment_to_text_segment` генерирует текстовый блок `[Attachment: <name> (...)]\n<text>` и кладёт его в `metadata.attachment_text_segments`.

### 1.6 API bot_service (префикс `/api`)

#### GET /agents/
- **Назначение:** вернуть список доступных агентов и их возможности.
- **Внутри:** использует список готовых агентов (инициализированные экземпляры) и сериализует в `AgentInfo`.
- **Поведение:** только готовые агенты; инициализирующиеся/ожидающие не показываются; на старте список может быть пустым.
- **Формат запроса:** без тела, без аутентификации (используется внутренними сервисами).
- **Ответ 200:** `[{ id, name, description, provider, supported_content_types[] }]`.
- **Ошибки:** нет специфичных.

#### POST /conversations/
- **Назначение:** создать разговор под конкретного агента.
- **Внутри:** вызывает `ensure_agent_ready(agent_id)`. Если агент ещё инициализируется — разговор помечается `status=pending`.
- **Кого вызывает/чего ждёт:** только реестр агентов; БД создаёт запись.
- **Headers:** `X-User-Id` (обязательный), `X-User-Role` (опционально).
- **Тело запроса (ConversationCreate):**
  ```json
  { "agent_id": "find_job", "title": "optional", "user_role": "optional", "metadata": { } }
  ```
- **Ответы:**
  - `201 Created`: `ConversationView` с `status=active` если агент уже готов.
  - `202 Accepted`: `ConversationView` с `status=pending` если инициализация в процессе.
  - `404`: неизвестный агент.
  - `500`: ошибка инициализации агента.

#### GET /conversations/
- **Назначение:** вернуть список разговоров пользователя, обновить их статусы если агент стал готов.
- **Внутри:** выборка `Conversation` по `user_id`; для `pending` проверяет `agent_registry.is_ready` и переключает на `active` при необходимости; коммит изменений.
- **Headers:** `X-User-Id`, `X-User-Role` (опционально).
- **Ответ 200:** `list[ConversationView]` (отсортированы по `last_message_at desc`).
- **Ошибки:** 401 при пустом `X-User-Id` (обрабатывается в deps), иных специфичных нет.

#### GET /conversations/{id}
- **Назначение:** вернуть разговор и все сообщения пользователя.
- **Внутри:** получает `Conversation` по id и `user_id`; вызывает `ensure_agent_ready` (может переключить статус с `pending` на `active`); подгружает связанные `messages`.
- **Headers:** `X-User-Id`, `X-User-Role` (опционально).
- **Ответ 200:** `ConversationDetail` с `messages: [MessageView]`.
- **Ошибки:** `404` если нет доступа или не найден; `500` при ошибке инициализации агента.

#### POST /conversations/{id}/messages
- **Назначение:** добавить пользовательское сообщение и получить ответ агента.
- **Внутри и последовательность:**
  1) Загружает разговор с `for update`; проверяет `user_id`.
  2) `ensure_agent_ready`; если не готов — `409`.
  3) Валидирует `reset` + вложения (запрещено).
  4) Обрабатывает вложения через `process_attachments`: конвертирует неподдерживаемые в текст, собирает `metadata.attachments` и `attachment_text_segments`.
  5) Проверяет `conversation.metadata.pending_interrupt`; если есть — подготавливает резюмирование.
  6) Строит `HumanMessage`, `RunnableConfig` (`user_id`, `user_role`, `thread_id`).
  7) Вызывает `invoke_agent` (threadpool): либо `agent.invoke({"messages":[human]})`, либо `agent.invoke(Command(resume=raw_user_text))` при резюме.
  8) Формирует `Message` пользователя и ассистента; ассистенту в `metadata` добавляет `agent_status`, `attachments`, `interrupt_payload`, `question/content` если есть.
  9) Обновляет `conversation.status`:
     - `interrupted` → `waiting_user`, пишет `pending_interrupt` (interrupt_id/question/content/artifact_*).
     - иначе `active`, удаляет `pending_interrupt`.
  10) Коммит и возврат `SendMessageResponse`.
- **Headers:** `X-User-Id` (обяз.), `X-User-Role` (опц.).
- **Тело запроса (MessageCreate):**
  ```json
  {
    "payload": {
      "type": "text",
      "text": "привет",
      "metadata": { "raw_user_text": "..." },
      "attachments": [
        { "filename": "a.pdf", "content_type": "application/pdf", "data": "<base64>", "text": null }
      ]
    }
  }
  ```
- **Ответ 201:** `SendMessageResponse { conversation: ConversationView, user_message: MessageView, agent_message: MessageView }`.
- **Ошибки:** `404` (нет разговора/агента), `409` (агент не готов), `400` (reset+attachments), `500/502` (ошибки вложений/агента).

### 1.7 Прерывания (interrupt/resume)
- Агент может вернуть `{"__interrupt__": [...]}`; `invoke_agent` отмечает `agent_status=interrupted`, генерирует вопрос в `AIMessage`.
- Сообщение ассистента сохраняется с `metadata.agent_status=interrupted` и `interrupt_payload` (добавляется `interrupt_id`, если отсутствует).
- Разговор ставится в `waiting_user`, `metadata.pending_interrupt` содержит `interrupt_id/question/content/artifact_*`.
- При следующем пользовательском сообщении этот `pending_interrupt` считывается, агент вызывается с `Command(resume=raw_user_text)`, после успешного ответа метаданные очищаются.

### 1.8 Синхронное выполнение
- Все вызовы агента в `bot_service` синхронны относительно HTTP-запроса: клиент получает полный ответ сразу или ошибку/interrupt.
- Таймаут задаётся конфигом агента (`BOT_SERVICE_REQUEST_TIMEOUT_SECONDS` в настройках платформы).

---

## 2. openai_proxy — публичный фасад OpenAI API

### 2.1 Назначение
- Принимает OpenAI-совместимые запросы (`/v1/chat/completions`, `/v1/models`).
- Создаёт разговоры в `bot_service` при необходимости.
- Ставит задания в Redis очередь через `RedisTaskQueue`.
- Транслирует события из Pub/Sub Redis в SSE-ответ клиенту или формирует конечный JSON-ответ.

### 2.2 Конфигурация (openai_proxy/config.py)
- `OPENAI_PROXY_BOT_SERVICE_BASE_URL` — адрес `bot_service` (`/api`).
- `OPENAI_PROXY_DEFAULT_USER_ID` / `DEFAULT_USER_ROLE` — подставляются, если нет `user` в запросе.
- `OPENAI_PROXY_REQUEST_TIMEOUT_SECONDS` / `CONNECT_TIMEOUT_SECONDS` — таймауты httpx клиента.
- `OPENAI_PROXY_LOG_LEVEL`.
- `OPENAI_PROXY_DEFAULT_ATTACHMENT_PROMPT` — текст по умолчанию, если в последнем user-сообщении нет явного текста, но есть вложения.

### 2.3 Клиент к bot_service (services/bot_client.py)
- Кэширует агентов (`refresh_agents`), но `list_agents` обновляет список на каждый вызов; `ensure_agent` использует обновленный кэш (ready-only).
- Методы: `list_agents`, `get_agent`, `create_conversation`, `send_message`, `get_conversation`.
- Заголовки: `X-User-Id`, `X-User-Role`.
- Логирование редактирует base64 (`<base64 N chars>`).

### 2.4 API openai_proxy

#### GET /v1/models
- **Назначение:** отдать список моделей/агентов, доступных через прокси.
- **Внутри:** вызывает `client.list_agents()` (GET /api/agents/ на bot_service), маппит в `ModelCard`.
- **Поведение:** только готовые (ready-only); на старте список может быть пустым.
- **Ответ 200 (ModelList):**
  ```json
  { "object": "list", "data": [ { "id": "...", "object": "model", "owned_by": "bot-service", "name": "...", "description": "...", "provider": "...", "metadata": { ... } } ] }
  ```
- **Ошибки:** 502 при ошибке bot_service.

#### GET /v1/models/{id}
- **Назначение:** вернуть информацию по конкретной модели/агенту.
- **Внутри:** `client.get_agent(id)` (валидация через bot_service).
- **Ответ 200:** `ModelCard` как выше.
- **Ошибки:** `404` если агент неизвестен или еще не готов (ready-only).

#### POST /v1/chat/completions
- **Назначение:** запустить диалог или продолжить существующий, с возможностью стрима.
- **Headers:** обычные HTTP, авторизаций нет (предполагается фронт доверенной зоны).
- **Тело запроса (ChatCompletionRequest):**
  ```json
  {
    "model": "theodor_agent",
    "messages": [
      { "role": "system", "content": "..." },
      { "role": "user", "content": "Привет", "attachments": [ { "filename": "...", "content_type": "...", "data": "<base64>", "text": null } ] }
    ],
    "user": "external-user-123",
    "conversation_id": "optional-existing-id",
    "stream": true
  }
  ```
  - Валидаторы собирают текст из массива частей, переносят вложения с `type=input_*` в `attachments`.
- **Последовательность внутри:**
  1) Определение `user_id` (`request.user` или `default_user_id`), `user_role` (`default_user_role`).
  2) Если `conversation_id` не передан: `POST /api/conversations/` с заголовками пользователя. При `202 pending` — опрос до 30 секунд `/api/conversations/{id}` до `status=active`, иначе 503 (detail + Retry-After: 1).
  3) Если `conversation_id` передан: `GET /api/conversations/{id}`. При `pending` — тот же опрос; `404` если не найден, 503 при таймауте.
  4) Построение промпта `build_prompt(messages, default_attachment_prompt)`: склейка system + история + последний user; если у последнего нет текста, но есть вложения — подставляется `default_attachment_prompt`, фиксируется `default_prompt_used`.
  5) Извлекаются `raw_user_text` (последний user.content или default prompt) и `attachments` из последнего user.
  6) Формируется `EnqueuePayload` с `job_id=job-<uuid>`, `model`, `conversation_id`, `user_id`, `user_role`, `text` (промпт), `raw_user_text`, `attachments`.
  7) `task_queue.enqueue(payload)` кладёт JSON в Redis list и публикует событие `status=queued`.
  8) Ветвление по `stream`:
     - **stream=true:** создаётся `StreamingResponse`, которая читает Pub/Sub события (`iter_events`) и транслирует в SSE чанки.
     - **stream=false:** `wait_for_completion` ждёт терминальное событие (`completed|failed|interrupt`) с таймаутом `TASK_QUEUE_COMPLETION_WAIT_TIMEOUT_SECONDS`.
- **Формат SSE (stream=true):**
  - Каждое событие отдаётся как `data: <json>\n\n`.
  - Отображение событий Redis (`QueueEvent`) в SSE:
    - `status` → `chat.completion.chunk` с пустым `delta`, `agent_status` = stage.
    - `chunk` → при первом чанке отправляется роль ассистента (`delta.role="assistant"`), далее `delta.content=<text>`.
    - `completed` → `finish_reason="stop"`, `agent_status="completed"`, `usage`, `message_metadata.attachments` если есть.
    - `interrupt` → `finish_reason="stop"`, `agent_status="interrupted"`, `message_metadata` (question/content/interrupt_id/artifact_*), `delta.content` содержит вопрос.
    - `failed` → `data: {"error":{...},"conversation_id":...,"job_id":...}`.
    - `heartbeat` → комментарий `: heartbeat <status>`.
  - После терминального события отправляется `data: [DONE]\n\n`.
- **Формат ответа (stream=false):**
  - `completed` → `ChatCompletionResponse { id: job_id, model, choices:[{index:0, message:{role:"assistant", content, metadata?}, finish_reason:"stop"}], usage, conversation_id }`. Если `attachments` были, кладутся в `message.metadata.attachments`.
  - `interrupt` → аналогично, но контент — вопрос, `finish_reason=stop`, `metadata` из события.
  - `failed` → HTTP 502 с текстом ошибки.
- **Ошибки:** `404` (неизвестный model или не готов), `400` (нет user-сообщения/ошибка промпта), `503` (agent initializing; `{ "detail": "Agent is initializing. Retry the request shortly." }`, `Retry-After: 1`), `502` (сбой воркера/бот-сервиса), `422` (валидация).

---

## 3. task_worker / task_queue — очередь, воркер, события

### 3.1 Назначение
- Хранит задания в Redis (`agent:jobs`), статусы и результаты в Redis hash, стримит прогресс в Pub/Sub.
- Воркер извлекает задания, вызывает `bot_service`, публикует чанки/статусы/прерывания/ошибки.
- Поддерживает heartbeats и watchdog для детекции зависших задач.

### 3.2 Конфигурация (services/task_queue/config.py)
- Redis:
  - `TASK_QUEUE_REDIS_URL` — строка подключения.
  - `TASK_QUEUE_QUEUE_KEY` — ключ списка (по умолчанию `agent:jobs`).
  - `TASK_QUEUE_STATUS_PREFIX` — префикс хэшей статусов (`agent:status:`).
  - `TASK_QUEUE_CHANNEL_PREFIX` — префикс Pub/Sub каналов (`agent:events:`).
  - `TASK_QUEUE_JOB_TTL_SECONDS` — TTL для статусов/результатов (6 ч).
  - `TASK_QUEUE_SSE_HEARTBEAT_SECONDS` — период heartbeat для SSE (10 c).
  - `TASK_QUEUE_WORKER_HEARTBEAT_SECONDS` — период heartbeat воркера (5 c).
  - `TASK_QUEUE_HEARTBEAT_STALE_AFTER_SECONDS` — порог устаревания (60 c).
  - `TASK_QUEUE_WATCHDOG_INTERVAL_SECONDS` — частота проверки устаревших (5 c).
- Стрим:
  - `TASK_QUEUE_CHUNK_CHAR_LIMIT` — размер чанка текста (600 символов).
- Вызовы bot_service:
  - `TASK_QUEUE_BOT_SERVICE_BASE_URL`, `TASK_QUEUE_BOT_REQUEST_TIMEOUT_SECONDS` (soft timeout), `TASK_QUEUE_BOT_CONNECT_TIMEOUT_SECONDS`.
- Ожидание в proxy:
  - `TASK_QUEUE_COMPLETION_WAIT_TIMEOUT_SECONDS` — таймаут ожидания финального события в non-stream (210 c).

### 3.3 Структуры данных (services/task_queue/models.py)
- `EnqueuePayload` (кладётся в Redis list, читает воркер):
  - `job_id`, `model` (agent_id), `conversation_id`, `user_id`, `user_role?`,
  - `text` (промпт), `raw_user_text?`, `attachments?` (list[dict]), `metadata?`.
- `QueueEvent` (Pub/Sub):
  - `job_id`, `type`: `status|chunk|completed|failed|heartbeat|interrupt`,
  - `status?` (`queued|running|streaming|completed|failed|interrupted`),
  - `content?` (чанк текста),
  - `metadata?` (результат/вложения/interrupt payload),
  - `usage?`, `error?`.

### 3.4 Redis-ключи и содержимое (services/task_queue/redis_queue.py)
- Очередь: `agent:jobs` (list). Элемент — JSON `EnqueuePayload`.
- Статус: `agent:status:<job_id>` (hash). Поля:
  - `status`, `created_at`, `updated_at`, `last_heartbeat` (float ts),
  - `conversation_id`, `model`, `user_id`,
  - `result` (JSON) при завершении,
  - `error` (string) при фейле,
  - `metadata` (если передано в `mark_status`).
  - TTL = `job_ttl_seconds`.
- Активные: ZSET `agent:status:active_jobs` — score = `last_heartbeat`, для watchdog.
- Каналы: `agent:events:<job_id>` — Pub/Sub поток `QueueEvent` в JSON.

### 3.5 Логика очереди (RedisTaskQueue)
- `enqueue(payload)`: пишет hash `status=queued`, TTL, кладёт JSON в list, публикует событие `status`.
- `publish_event(event)`: publish в канал.
- `mark_status(job_id, status, extra)`: обновляет hash и TTL, записывает `last_heartbeat`.
- `store_result` / `store_failure`: проставляют финальный статус, кладут `result`/`error`, очищают из `active_jobs`.
- `register_active_job` / `clear_active_job`: управляют ZSET.
- `update_heartbeat`: обновляет hash, ZSET и TTL.
- `fail_job_if_active`: если статус не финальный, ставит `failed`, публикует событие.
- `fail_stale_jobs`: находит в ZSET те, у кого `last_heartbeat < now - stale_after`, помечает как failed.
- `get_status(job_id)`: читает hash и декодирует JSON/float.
- `pop_job(timeout)`: `BLPOP` из списка, десериализация в `EnqueuePayload`.
- `iter_events(job_id, include_status_snapshot)`: подписка на канал, при `include_status_snapshot` сначала отдаёт `status` из hash, завершает после `completed|failed|interrupt`.
- `wait_for_completion(job_id, timeout)`: обёртка над `iter_events` с `asyncio.wait_for`.

### 3.6 Логика воркера (services/task_queue/worker.py)
- Запуск (`run`):
  - Инициализирует `RedisTaskQueue` и `BotServiceClient` (к `bot_service`).
  - Стартует `watchdog` и основной цикл `worker_loop`.
  - SIGINT/SIGTERM ставят `stop_event`.
- Watchdog:
  - Каждые `watchdog_interval_seconds` вызывает `fail_stale_jobs`.
  - Помечает зависшие задачи как failed и публикует событие `failed`.
- Основной цикл:
  - `BLPOP` из очереди с таймаутом 5 c.
  - Каждое задание передаёт в `_process_job`.

#### _process_job — последовательность
1) Логирование параметров (`job_id`, `conversation_id`, длина текста, количество вложений).
2) Статусы и heartbeat:
   - `mark_status(running)`, событие `status`.
   - `register_active_job`, `update_heartbeat`.
   - Запуск `_heartbeat_loop`: каждые `worker_heartbeat_seconds` публикует `heartbeat` и обновляет hash.
3) Soft-timeout: `bot_request_timeout_seconds` (если >0) — для логирования, не убивает задачу.
4) Создаёт задачу `client.send_message(...)` (POST `/api/conversations/{id}/messages` в bot_service) с `text`, `raw_user_text`, `attachments`, `metadata`.
5) Ожидание ответа:
   - `asyncio.wait_for` на интервал `interval=max(heartbeat,1)` секунд.
   - При `TimeoutError` — отправляет heartbeat, логирует длительность. Если превышен soft-timeout, логирует warning, но продолжает ждать.
6) После ответа:
   - `agent_message = response["agent_message"]`, `raw_text = agent_message.raw_text`, `agent_status = agent_message.metadata.agent_status`.
   - Вложения ответа извлекаются `_extract_response_attachments` (из `metadata.attachments` или из `content.parts` с типами вложений).
7) Ветка interrupt:
   - Если `agent_status == interrupted`: `mark_status(interrupted, result=metadata)`, событие `interrupt` (добавляет `content`, если его нет), `clear_active_job`, логирует и завершает без стрима.
8) Ветка стрима:
   - Если `raw_text` непустой: `mark_status(streaming)`, событие `status`, heartbeat; делит текст по `chunk_char_limit` и публикует `chunk` с heartbeat между чанками.
   - Если текста нет — стрим не публикуется.
9) Завершение:
   - `metadata = { conversation_id, content: raw_text, response, attachments? }`.
   - `store_result`, событие `completed`, `update_heartbeat`, `job_stage=completed`.
10) Ошибки:
   - Любое исключение → `error_message = "Agent invocation failed: <Exc>"`, `store_failure`, событие `failed`, heartbeat.
11) Финализация:
   - Отмена незавершённого future `send_message`, отмена heartbeat-задачи (с подавлением CancelledError).

### 3.7 Стейт-машина job_id
- `queued` (enqueue) → `running` (pop) → ветки:
  - `interrupted` (агент прервал) → финал.
  - `streaming` → `completed` → финал.
  - `failed` → финал.
- Heartbeat + watchdog гарантируют пометку зависших задач как `failed`.

---

## 4. Сквозные последовательности (детально)

### 4.1 Синхронный путь (прямой вызов bot_service)
Акторы: Клиент ↔ bot_service (без очереди).
1) Клиент зовёт `POST /api/conversations/{id}/messages` с заголовком `X-User-Id` (и опц. `X-User-Role`).
2) `bot_service`:
   - под блокировкой строкой читает разговор, валидирует пользователя;
   - проверяет готовность агента (`ensure_agent_ready`) — при неготовности 409;
   - прогоняет вложения через `process_attachments` → `attachment_text_segments`;
   - собирает `HumanMessage`, `RunnableConfig` (`user_id`, `user_role`, `thread_id`);
   - вызывает агента синхронно (`agent.invoke` или `Command(resume=...)`).
3) Возвращается результат агента:
   - если `__interrupt__` — формируется вопрос, статус разговора → `waiting_user`, `pending_interrupt` пишется в `metadata`;
   - иначе обычный ответ, статус → `active`.
4) В БД пишутся два сообщения (user/assistant) с нормализованным контентом и метаданными (`agent_status`, вложения, interrupt_payload).
5) Клиент получает `SendMessageResponse` сразу в этом же HTTP-ответе.

### 4.2 Асинхронный путь (через openai_proxy и Redis)
Акторы: Клиент ↔ openai_proxy ↔ Redis ↔ task_worker ↔ bot_service.
1) Клиент шлёт `POST /v1/chat/completions` в `openai_proxy` (может указать `stream=true/false`, `conversation_id` опционально, `user` опционально).
2) `openai_proxy`:
   - создает или получает разговор (POST /api/conversations/ или GET /api/conversations/{id}); при `pending` опрашивает до 30 с; при таймауте возвращает 503 (detail + Retry-After: 1).
   - из `messages` собирает промпт (`build_prompt`), сохраняет `raw_user_text` и вытаскивает вложения последнего user;
   - формирует `EnqueuePayload` (`job_id`, `model`, `conversation_id`, `user_id`, `user_role`, `text`, `raw_user_text`, `attachments`);
   - кладёт payload в Redis list `agent:jobs` и публикует `status=queued`.
3) `task_worker`:
   - `BLPOP` получает payload, отмечает `running`, запускает heartbeat;
   - вызывает `bot_service POST /api/conversations/{id}/messages` (с тем же `user_id/user_role`), передавая `text/raw_user_text/attachments`;
   - стримит ход выполнения в Pub/Sub `agent:events:<job_id>`: `status`, `heartbeat`, `chunk` (если есть текст), `completed` или `interrupt`/`failed`.
4) `openai_proxy`:
   - **stream=true:** подписывается на Pub/Sub, каждый `QueueEvent` конвертирует в SSE chunk (`data: {...}\n\n`), после терминального события отдаёт `[DONE]`.
   - **stream=false:** ждёт терминальное событие через `wait_for_completion`; по `completed`/`interrupt` формирует обычный `ChatCompletionResponse`, по `failed` возвращает 502.
5) Клиент:
   - при стриме получает статусные пульсы, текстовые чанки, финал (`completed/interrupt`) и `[DONE]`;
   - при non-stream — единовременный JSON-ответ или ошибку.

### 4.3 Обработка прерываний end-to-end
1) Агент во время вызова в `bot_service` возвращает `__interrupt__`:
   - `bot_service` пишет ассистентское сообщение с `agent_status=interrupted` и `interrupt_payload`,
   - статус разговора → `waiting_user`, `pending_interrupt` сохраняется в `metadata` разговора.
2) В асинхронном пути воркер:
   - получает ответ от `bot_service`, маркирует job `interrupted`,
   - публикует событие `interrupt` в канал `agent:events:<job_id>` (с вопросом/interrupt_id/артефактами).
3) Клиент (через SSE или обычный ответ) видит `agent_status=interrupted`, получает `interrupt_id/question`.
4) Клиент присылает новый запрос:
   - синхронный путь: снова `POST /api/conversations/{id}/messages`;
   - асинхронный путь: `POST /v1/chat/completions` с тем же `conversation_id`.
   В обоих случаях `raw_user_text` последнего user попадает в payload.
5) `bot_service` обнаруживает `pending_interrupt`, вызывает агента с `Command(resume=raw_user_text)`, очищает `pending_interrupt`; воркер при асинхронном пути публикует обычные `chunk/completed`.
6) Клиент получает финальный ответ (`completed`) и может продолжать диалог.

---

## 5. Требования к регистрации агентов (backend)
1) Добавить `AgentDefinition` в `bot_service/agent_registry.py` или обеспечить автогенерацию:
   - `id` (используется как `model` во всех сервисах и в Redis-ключах).
   - `factory(provider: ModelType) -> agent_instance` — потокобезопасная инициализация (исполняется в executor).
   - `default_provider` — `openai|yandex|mistral|gigachat`.
   - `supported_content_types` — `ContentType` для фильтрации вложений; неподдерживаемые будут попыткой преобразованы в текст.
   - `name/description` — используются в `/api/agents` и `/v1/models`.
2) Контракт `invoke`:
   - Обычный запуск: `agent.invoke({"messages":[HumanMessage(...)]}, config=RunnableConfig(...))` → `dict` или `Sequence[BaseMessage]` с `AIMessage`.
   - Резюме: поддерживать `Command(resume=raw_user_text)`.
   - Прерывания: возвращать ключ `__interrupt__` (list/dict). Если нет `interrupt_id`, `bot_service` сгенерирует.
3) Ответы агента:
   - `AIMessage.content` строка или сегменты; вложения можно класть в `content.parts` с типами `file|image|audio|video|attachment` или в `metadata.attachments`.
   - Можно добавлять `metadata.usage` для токен-статистики (используется воркером/proxy при наличии).
4) Производительность:
   - Укладываться в `BOT_SERVICE_REQUEST_TIMEOUT_SECONDS` (иначе HTTP-клиент/обвязка может оборвать).
   - Для долгих операций — использовать interrupts для запроса уточняющей информации и чтобы освободить клиентский цикл.

---

## 6. Быстрые ссылки
- bot_service: `bot_service/main.py`, `bot_service/api/conversations.py`, `bot_service/api/agents.py`, `bot_service/agent_registry.py`, `bot_service/attachments.py`, `bot_service/service.py`, `bot_service/schemas.py`, `bot_service/models.py`.
- openai_proxy: `openai_proxy/main.py`, `openai_proxy/schemas.py`, `openai_proxy/utils.py`, `openai_proxy/config.py`, `services/bot_client.py`.
- task_queue/worker: `services/task_queue/worker.py`, `services/task_queue/redis_queue.py`, `services/task_queue/models.py`, `services/task_queue/config.py`.
