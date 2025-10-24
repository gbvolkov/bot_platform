Платформа агентов-ботов

## Быстрый старт лёгкого веб‑чата

1. Установите зависимости проекта:

   ```bash
   uv pip install -r pyproject.toml
   ```

2. Подготовьте MySQL и создайте базу данных (по умолчанию `bot_platform`).  
   Пример строки подключения:

   ```
   BOT_SERVICE_DATABASE_URL=mysql+aiomysql://user:password@localhost:3306/bot_platform
   ```

   Переменную окружения можно сохранить в `.env` или задать перед запуском.

3. Запустите API:

   ```bash
   uvicorn bot_service.main:app --reload
   ```

   Таблицы будут созданы автоматически при старте сервера.

4. Для взаимодействия из терминала используйте клиент:

   ```bash
   # список агентов
   python chat_client.py list-agents

   # чат с конкретным агентом
   python chat_client.py chat find_job
   ```

   При необходимости можно передать собственный `X-User-Id` и `X-User-Role` через переменные
   окружения `BOT_SERVICE_USER_ID` и `BOT_SERVICE_USER_ROLE`.

5. REST‑эндпоинты доступны по пути `/api/*`. Примеры:

   * `GET /api/agents/` — список зарегистрированных агентов.
   * `POST /api/conversations` — создание диалога.
   * `POST /api/conversations/{id}/messages` — отправка сообщения и получение ответа агента.

## OpenAI-совместимый слой

Прокси-сервис `openai_proxy` эмулирует endpoint `POST /v1/chat/completions` и внутри обращается к `bot_service`.

1. Убедитесь, что `bot_service` запущен.
2. Запустите прокси:

   ```bash
   uvicorn openai_proxy.main:app --reload --port 8080
   ```

   При необходимости задайте `OPENAI_PROXY_BOT_SERVICE_BASE_URL` (по умолчанию `http://localhost:8000/api`).

3. Пример запроса:

   ```bash
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "find_job",
           "messages": [
             {"role": "system", "content": "Ты подбираешь вакансии."},
             {"role": "user", "content": "Найди позиции промпт-инженера."}
           ]
         }'
   ```

   В ответ вернётся объект в стиле OpenAI API c полем `conversation_id`. Используйте его в следующих запросах, чтобы продолжить диалог.
