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
