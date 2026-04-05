from __future__ import annotations

from agents.gaz_agent.prompts import get_prompt as get_gaz_prompt
from agents.gaz_agent.prompts import get_system_prompt as get_gaz_system_prompt


_STORE_ARTIFACT_RULE_EN = (
    "Use `store_artifact_tool` when the user explicitly asks to save, export, or persist the "
    "result to a file. The tool returns a Markdown link to the saved file. Do not modify that link."
)

_STORE_ARTIFACT_RULE_RU = (
    "Используйте `store_artifact_tool`, когда пользователь явно просит сохранить, экспортировать "
    "или зафиксировать результат в файле. Инструмент возвращает Markdown-ссылку на сохранённый файл. "
    "Не изменяйте эту ссылку."
)


def build_gaz_mycroft_system_prompt(
    *,
    locale: str = "en",
    pricing_subagent_name: str = "gaz_pricing_bi",
    web_tool_name: str = "web_search",
    store_tool_name: str = "store_artifact_tool",
    maps_search_tool_name: str | None = None,
    maps_route_tool_name: str | None = None,
    vin_decode_tool_name: str | None = None,
    recall_lookup_tool_name: str | None = None,
    gmail_draft_tool_name: str | None = None,
    gmail_send_tool_name: str | None = None,
    enable_web_search: bool = True,
) -> str:
    locale_key = "ru" if str(locale).strip().lower().startswith("ru") else "en"
    sections = [
        _adapt_gaz_system_section(locale_key, pricing_subagent_name),
        _adapt_sales_response_section(locale_key, pricing_subagent_name, web_tool_name),
        _adapt_tool_rules_section(locale_key, pricing_subagent_name, web_tool_name, enable_web_search),
        _adapt_pricing_section(locale_key, pricing_subagent_name),
    ]
    precedence_section = _mcp_precedence_section(
        locale_key,
        web_tool_name=web_tool_name,
        maps_search_tool_name=maps_search_tool_name,
        vin_decode_tool_name=vin_decode_tool_name,
        recall_lookup_tool_name=recall_lookup_tool_name,
    )
    if precedence_section:
        sections.append(precedence_section)
    maps_section = _maps_tools_section(
        locale_key,
        maps_search_tool_name=maps_search_tool_name,
        maps_route_tool_name=maps_route_tool_name,
    )
    if maps_section:
        sections.append(maps_section)
    nhtsa_section = _nhtsa_tools_section(
        locale_key,
        vin_decode_tool_name=vin_decode_tool_name,
        recall_lookup_tool_name=recall_lookup_tool_name,
    )
    if nhtsa_section:
        sections.append(nhtsa_section)
    gmail_section = _gmail_tools_section(
        locale_key,
        gmail_draft_tool_name=gmail_draft_tool_name,
        gmail_send_tool_name=gmail_send_tool_name,
    )
    if gmail_section:
        sections.append(gmail_section)
    if enable_web_search:
        sections.append(_adapt_web_search_section(locale_key, pricing_subagent_name, web_tool_name))
    sections.append(_store_artifact_section(locale_key, store_tool_name))
    return "\n\n".join(section for section in sections if section)


def build_system_prompt(*, enable_web_search: bool) -> str:
    return build_gaz_mycroft_system_prompt(enable_web_search=enable_web_search)


def _adapt_gaz_system_section(locale: str, pricing_subagent_name: str) -> str:
    base = get_gaz_system_prompt(locale)
    if locale == "ru":
        return (
            f"{base}\n"
            f"Используйте сабагент `{pricing_subagent_name}` как основной authoritative источник для "
            "цен, точных технических характеристик, комплектаций, опций, ТО, сервисных интервалов и "
            "стоимости обслуживания."
        )
    return (
        f"{base} Treat `{pricing_subagent_name}` as the authoritative source for prices, exact "
        "technical characteristics, trims, options, maintenance intervals, and service cost."
    )


def _adapt_sales_response_section(
    locale: str,
    pricing_subagent_name: str,
    web_tool_name: str,
) -> str:
    if locale == "ru":
        return (
            "Сначала отвечайте по сути. Любой вопрос о цене, стоимости, прайсе, всех моделях с ценами, "
            "комплектациях, опциях, ТО, сервисных интервалах, стоимости обслуживания, а также о точных "
            f"технических характеристиках сначала передавайте сабагенту `{pricing_subagent_name}`. "
            f"В сабагент передавайте только бизнес-вопрос. Только после ответа `{pricing_subagent_name}` "
            f"можно дополнять ответ через `{web_tool_name}`, если нужен внешний или актуальный публичный "
            "источник. Если сабагент вернул точный факт, считайте его authoritative BI fact. Если сабагент "
            "вернул неполный ответ или сообщил, что точного значения нет, не превращайте это в общий отказ: "
            "верните подтверждённую часть данных и явно перечислите, чего именно не хватает. Если сабагент "
            "вернул ошибку, честно скажите об ошибке поиска."
        )

    return (
        "Answer the ask first. Any question about price, cost, pricing, all models with prices, trims, "
        "options, maintenance intervals, service cost, or exact technical characteristics must be "
        f"delegated to `{pricing_subagent_name}` first. Pass only the business question to that subagent. "
        f"Only after `{pricing_subagent_name}` may you supplement the answer with `{web_tool_name}` when "
        "you need external or current public information. If the subagent returns a precise fact, treat it "
        "as the authoritative BI fact. If the subagent returns partial data or says an exact value is "
        "unavailable, do not turn that into a blanket refusal: return the confirmed part and state exactly "
        "what is still missing. If the subagent returns an error, state the lookup error honestly."
    )


def _adapt_tool_rules_section(
    locale: str,
    pricing_subagent_name: str,
    web_tool_name: str,
    enable_web_search: bool,
) -> str:
    _ = get_gaz_prompt(locale, "tool_rules")
    if locale == "ru":
        if enable_web_search:
            return (
                f"Любой вопрос о цене, точных технических характеристиках, комплектациях, опциях, ТО, "
                f"сервисных интервалах или стоимости обслуживания сначала направляйте в `{pricing_subagent_name}`. "
                f"Для остальных вопросов используйте `{web_tool_name}` только тогда, когда действительно нужен "
                "внешний источник, актуальная публичная информация или независимая проверка. Если на вопрос уже "
                f"надёжно ответил `{pricing_subagent_name}`, не вызывайте `{web_tool_name}`. При использовании "
                f"`{web_tool_name}` в финальном ответе обязательно добавляйте ссылку на источник."
            )
        return (
            f"Любой вопрос о цене, точных технических характеристиках, комплектациях, опциях, ТО, сервисных "
            f"интервалах или стоимости обслуживания сначала направляйте в `{pricing_subagent_name}`. Не "
            "заявляйте о доступе к интернету и не ссылайтесь на внешний поиск, если web tool недоступен."
        )

    if enable_web_search:
        return (
            f"Any question about price, exact technical characteristics, trims, options, maintenance intervals, "
            f"or service cost must go to `{pricing_subagent_name}` first. Use `{web_tool_name}` only when you "
            "need an external source, current public information, or independent verification. Do not use "
            f"`{web_tool_name}` when `{pricing_subagent_name}` already answers the question reliably. If you use "
            f"`{web_tool_name}` in the final answer, include source links."
        )
    return (
        f"Any question about price, exact technical characteristics, trims, options, maintenance intervals, "
        f"or service cost must go to `{pricing_subagent_name}` first. Do not claim internet access when the "
        "web tool is not available."
    )


def _adapt_pricing_section(locale: str, pricing_subagent_name: str) -> str:
    _ = get_gaz_prompt(locale, "tool_pricing_bi")
    if locale == "ru":
        return (
            f"Используйте сабагент `{pricing_subagent_name}` для любых вопросов о цене, стоимости, прайсе, "
            "списках моделей с ценами, точных технических характеристиках, комплектациях, опциях, ТО, "
            "сервисных интервалах и стоимости обслуживания. Это относится и к узким вопросам, и к широким "
            "запросам вида «дай всё». Передавайте только бизнес-вопрос без упоминания таблиц, полей и "
            "структуры БД. Если сабагент вернул частичный ответ, используйте подтверждённую часть и явно "
            "перечислите, что осталось неподтверждённым."
        )
    return (
        f"Use `{pricing_subagent_name}` for any question about price, cost, pricing, lists of models with "
        "prices, exact technical characteristics, trims, options, maintenance intervals, or service cost. "
        "This includes narrow questions and broad asks such as 'give me everything you have'. Pass only a "
        "business question without mentioning tables, fields, or database structure. If the subagent returns "
        "partial data, use the confirmed part and explicitly list what is still missing."
    )


def _adapt_web_search_section(locale: str, pricing_subagent_name: str, web_tool_name: str) -> str:
    _ = get_gaz_prompt(locale, "tool_web_search")
    if locale == "ru":
        return (
            f"Используйте `{web_tool_name}`, когда ответа `{pricing_subagent_name}` недостаточно или когда "
            "нужны внешний источник, актуальная публичная информация или независимая проверка. Не используйте "
            f"`{web_tool_name}`, если `{pricing_subagent_name}` уже уверенно отвечает на вопрос. Если вы "
            f"используете данные из `{web_tool_name}` в финальном ответе, обязательно добавляйте ссылку на источник."
        )
    return (
        f"Use `{web_tool_name}` when `{pricing_subagent_name}` is insufficient, or when you need an external "
        "source, current public information, or independent verification. Do not use it when "
        f"`{pricing_subagent_name}` already answers the question reliably. If you use `{web_tool_name}` "
        "results in the final answer, include source links."
    )


def _store_artifact_section(locale: str, store_tool_name: str) -> str:
    if locale == "ru":
        return _STORE_ARTIFACT_RULE_RU.replace("`store_artifact_tool`", f"`{store_tool_name}`")
    return _STORE_ARTIFACT_RULE_EN.replace("`store_artifact_tool`", f"`{store_tool_name}`")


def _mcp_precedence_section(
    locale: str,
    *,
    web_tool_name: str,
    maps_search_tool_name: str | None,
    vin_decode_tool_name: str | None,
    recall_lookup_tool_name: str | None,
) -> str:
    dedicated_tools = [name for name in (maps_search_tool_name, vin_decode_tool_name, recall_lookup_tool_name) if name]
    if not dedicated_tools:
        return ""

    tools_text = ", ".join(f"`{name}`" for name in dedicated_tools)
    if locale == "ru":
        return (
            f"Предпочитайте специализированные инструменты перед `{web_tool_name}`. Используйте {tools_text} "
            "там, где они подходят по задаче. Внешний `web_search` применяйте только тогда, когда запрос не "
            "закрывается pricing BI, картами или NHTSA, либо когда нужна независимая публичная проверка."
        )
    return (
        f"Prefer dedicated tools over `{web_tool_name}`. Use {tools_text} when they match the request. "
        f"Use `{web_tool_name}` only when the request is outside pricing BI, Maps, and NHTSA scope, or when "
        "you need independent public-source verification."
    )


def _maps_tools_section(
    locale: str,
    *,
    maps_search_tool_name: str | None,
    maps_route_tool_name: str | None,
) -> str:
    if not maps_search_tool_name and not maps_route_tool_name:
        return ""

    if locale == "ru":
        parts: list[str] = []
        if maps_search_tool_name:
            parts.append(
                f"Используйте `{maps_search_tool_name}` для поиска ближайших дилеров и сервисных центров."
            )
        if maps_route_tool_name:
            parts.append(
                f"Используйте `{maps_route_tool_name}` только после того, как известны исходная точка клиента и хотя бы один кандидат назначения."
            )
        parts.append(
            "Если автомобиль уже известен из предыдущих сообщений, а пользователь спрашивает, где ближайший сервисный центр, не спрашивайте модель повторно. Запрашивайте только недостающую географию: город, адрес или геолокацию."
        )
        parts.append("В финальном ответе по картам добавляйте найденные адреса и ссылки на карты.")
        return " ".join(parts)

    parts = []
    if maps_search_tool_name:
        parts.append(
            f"Use `{maps_search_tool_name}` for nearby dealer and service-center lookups."
        )
    if maps_route_tool_name:
        parts.append(
            f"Use `{maps_route_tool_name}` only after you know the customer's origin and at least one destination candidate."
        )
    parts.append(
        "If the vehicle is already known from prior turns and the user asks for the nearest service center, do not ask for the model again. Ask only for the missing location: city, address, or geolocation."
    )
    parts.append("Include addresses and map links when answering from Maps results.")
    return " ".join(parts)


def _nhtsa_tools_section(
    locale: str,
    *,
    vin_decode_tool_name: str | None,
    recall_lookup_tool_name: str | None,
) -> str:
    if not vin_decode_tool_name and not recall_lookup_tool_name:
        return ""

    if locale == "ru":
        parts: list[str] = []
        if vin_decode_tool_name:
            parts.append(f"Используйте `{vin_decode_tool_name}` для расшифровки VIN и базовых фактов о машине.")
        if recall_lookup_tool_name:
            parts.append(
                f"Используйте `{recall_lookup_tool_name}` для отзывов и контекста по безопасности. Если VIN уже есть, предпочитайте VIN-основанный запрос."
            )
        return " ".join(parts)

    parts = []
    if vin_decode_tool_name:
        parts.append(f"Use `{vin_decode_tool_name}` for VIN decoding and vehicle facts.")
    if recall_lookup_tool_name:
        parts.append(
            f"Use `{recall_lookup_tool_name}` for recalls and safety context. If a VIN is available, prefer a VIN-based lookup."
        )
    return " ".join(parts)


def _gmail_tools_section(
    locale: str,
    *,
    gmail_draft_tool_name: str | None,
    gmail_send_tool_name: str | None,
) -> str:
    if not gmail_draft_tool_name and not gmail_send_tool_name:
        return ""

    if locale == "ru":
        parts: list[str] = []
        if gmail_draft_tool_name:
            parts.append(f"Используйте `{gmail_draft_tool_name}` для подготовки черновиков follow-up писем.")
        if gmail_send_tool_name:
            parts.append(
                f"Используйте `{gmail_send_tool_name}` только когда пользователь явно просит отправить письмо. Отправка может быть остановлена на approval runtime-механизмом."
            )
        return " ".join(parts)

    parts = []
    if gmail_draft_tool_name:
        parts.append(f"Use `{gmail_draft_tool_name}` to draft follow-up emails.")
    if gmail_send_tool_name:
        parts.append(
            f"Use `{gmail_send_tool_name}` only when the user explicitly asks to send an email. Sending may be approval-gated by the runtime."
        )
    return " ".join(parts)


DEFAULT_SYSTEM_PROMPT = build_gaz_mycroft_system_prompt(enable_web_search=True)
DEFAULT_SYSTEM_PROMPT_WITHOUT_WEB_SEARCH = build_gaz_mycroft_system_prompt(enable_web_search=False)
