from __future__ import annotations

from copy import copy
import inspect
from importlib.util import find_spec
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config

from .context import GuardrailContext, build_guardrail_context, privacy_scope_key


DEFAULT_PALIMPSEST_ENTITIES = [
    "RU_PERSON",
    "CREDIT_CARD",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "URL",
    "RU_PASSPORT",
    "SNILS",
    "INN",
    "RU_BANK_ACC",
    "TICKET_NUMBER",
]
DEFAULT_PALIMPSEST_ENTITY_TABLE = {
    "RU_PERSON": {"placeholder": "PERSON"},
    "CREDIT_CARD": {"placeholder": "CREDIT_CARD"},
    "PHONE_NUMBER": {"placeholder": "PHONE"},
    "IP_ADDRESS": {"placeholder": "IP_ADDRESS"},
    "URL": {"placeholder": "URL"},
    "RU_PASSPORT": {"placeholder": "PASSPORT"},
    "SNILS": {"placeholder": "SNILS"},
    "INN": {"placeholder": "INN"},
    "RU_BANK_ACC": {"placeholder": "BANK_ACCOUNT"},
    "TICKET_NUMBER": {"placeholder": "TICKET"},
}

_DEFAULT_SESSION_ID = "__manual__"
_TEXT_KEYS = ("text", "content", "input", "title", "caption", "markdown", "explanation")
_PALIMPSEST_SPACY_MODELS_BY_LOCALE = {
    "ru": "ru_core_news_sm",
    "ru-ru": "ru_core_news_sm",
}
_METHOD_ALIASES = {
    "anonymize": ("anonymize", "anonimize"),
    "deanonymize": ("deanonymize", "deanonimize"),
}


def _call_text_transform(target: Any, method_name: str, text: str) -> str:
    for candidate in _METHOD_ALIASES.get(method_name, (method_name,)):
        method = getattr(target, candidate, None)
        if callable(method):
            return method(text)
    raise AttributeError(f"{target!r} has no {method_name!r} text transform")


def _supports_var_kwargs(callable_obj: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _unsupported_kwargs(callable_obj: Callable[..., Any], kwargs: Mapping[str, Any]) -> list[str]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return []
    if _supports_var_kwargs(callable_obj):
        return []
    supported = set(signature.parameters)
    return [key for key in kwargs if key not in supported]


def _require_supported_kwargs(
    callable_obj: Callable[..., Any],
    kwargs: Mapping[str, Any],
    *,
    api_name: str,
) -> None:
    unsupported = _unsupported_kwargs(callable_obj, kwargs)
    if unsupported:
        raise RuntimeError(
            f"{api_name} does not support required Palimpsest option(s): "
            f"{', '.join(sorted(unsupported))}. Upgrade Palimpsest; platform privacy "
            "configuration does not fall back to fake-name anonymization."
        )


def _normalise_session_id(value: Any) -> str:
    text = str(value or "").strip()
    return text or _DEFAULT_SESSION_ID


def _entity_table_enabled(value: Any) -> bool:
    if value is False:
        return False
    if isinstance(value, Mapping):
        return bool(value.get("enabled", True))
    return True


def entity_types_from_table(entity_table: Any) -> list[str]:
    """Extract Palimpsest entity names from a platform entity table."""
    if entity_table is None:
        return []
    if isinstance(entity_table, Mapping):
        return [
            str(entity_type)
            for entity_type, config in entity_table.items()
            if str(entity_type).strip() and _entity_table_enabled(config)
        ]
    if isinstance(entity_table, str):
        return [entity_table] if entity_table.strip() else []
    result: list[str] = []
    for item in entity_table:
        if isinstance(item, str):
            entity_type = item
            enabled = True
        elif isinstance(item, Mapping):
            entity_type = (
                item.get("entity_type")
                or item.get("type")
                or item.get("name")
                or item.get("id")
            )
            enabled = _entity_table_enabled(item)
        else:
            continue
        text = str(entity_type or "").strip()
        if text and enabled:
            result.append(text)
    return result


def _normalise_run_entities(
    run_entities: Iterable[str] | None,
    entity_table: Any = None,
) -> list[str]:
    if run_entities is not None:
        return [str(entity) for entity in run_entities]
    table_entities = entity_types_from_table(entity_table)
    return table_entities or list(DEFAULT_PALIMPSEST_ENTITIES)


def _palimpsest_constructor_kwargs(
    *,
    locale: str,
    run_entities: Iterable[str] | None,
    entity_table: Any,
    typed_placeholders: bool | None,
    verbose: bool,
    palimpsest_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "verbose": verbose,
        "run_entities": _normalise_run_entities(run_entities, entity_table),
        "locale": locale,
    }
    if entity_table is not None:
        kwargs["entity_table"] = entity_table
    if typed_placeholders is not None:
        kwargs["typed_placeholders"] = typed_placeholders
    if palimpsest_options:
        kwargs.update(dict(palimpsest_options))
    return kwargs


def _palimpsest_session_kwargs(
    *,
    entity_table: Any,
    typed_placeholders: bool | None,
    palimpsest_session_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if entity_table is not None:
        kwargs["entity_table"] = entity_table
    if typed_placeholders is not None:
        kwargs["typed_placeholders"] = typed_placeholders
    if palimpsest_session_options:
        kwargs.update(dict(palimpsest_session_options))
    return kwargs


def _ensure_palimpsest_dependencies(locale: str) -> None:
    model_name = _PALIMPSEST_SPACY_MODELS_BY_LOCALE.get((locale or "").lower())
    if not model_name or find_spec(model_name) is not None:
        return
    raise RuntimeError(
        "Palimpsest privacy guardrails require the spaCy model "
        f"{model_name!r} for locale {locale!r}. Install project dependencies with "
        "`uv sync` or run `python -m spacy download ru_core_news_sm` in the active "
        "environment."
    )


def thread_id_from_config(config: RunnableConfig | Mapping[str, Any] | None) -> str | None:
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None
    thread_id = configurable.get("thread_id")
    if thread_id is None:
        return None
    return str(thread_id)


def thread_id_from_runtime(runtime: Any) -> str | None:
    execution_info = getattr(runtime, "execution_info", None)
    thread_id = getattr(execution_info, "thread_id", None)
    if thread_id:
        return str(thread_id)

    config = getattr(runtime, "config", None)
    thread_id = thread_id_from_config(config)
    if thread_id:
        return thread_id

    try:
        thread_id = thread_id_from_config(get_config())
    except Exception:
        thread_id = None
    return thread_id


def content_is_reset(content: Any) -> bool:
    if isinstance(content, str):
        return content.strip().lower() == "reset"
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    return isinstance(first, dict) and first.get("type") == "reset"


def state_has_reset_message(state: Dict[str, Any]) -> bool:
    messages = list(state.get("messages") or [])
    if not messages:
        return False
    last = messages[-1]
    if getattr(last, "type", None) != "human":
        return False
    return content_is_reset(getattr(last, "content", None))


class PalimpsestSessionManager:
    """Owns one PalimpsestSession per explicit session id."""

    def __init__(
        self,
        processor: Any,
        *,
        default_session_id: str = _DEFAULT_SESSION_ID,
        create_session_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._processor = processor
        self._default_session_id = _normalise_session_id(default_session_id)
        self._create_session_kwargs = dict(create_session_kwargs or {})
        self._sessions: Dict[str, Any] = {}
        self._lock = RLock()

    def get_session(self, session_id: Any = None) -> Any:
        key = _normalise_session_id(session_id or self._default_session_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None or bool(getattr(session, "closed", False)):
                session_kwargs = {"session_id": key, **self._create_session_kwargs}
                _require_supported_kwargs(
                    self._processor.create_session,
                    session_kwargs,
                    api_name="Palimpsest.create_session",
                )
                session = self._processor.create_session(**session_kwargs)
                self._sessions[key] = session
            return session

    def session_for_config(self, config: RunnableConfig | None) -> Any:
        return self.get_session(thread_id_from_config(config))

    def session_for_runtime(self, runtime: Any) -> Any:
        return self.get_session(thread_id_from_runtime(runtime))

    def reset_session(self, session_id: Any = None) -> None:
        key = _normalise_session_id(session_id or self._default_session_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None or bool(getattr(session, "closed", False)):
                return
            reset = getattr(session, "reset", None)
            if callable(reset):
                reset()

    def reset_from_config(self, config: RunnableConfig | None) -> None:
        self.reset_session(thread_id_from_config(config))

    def anonymize(self, text: str, *, session_id: Any = None) -> str:
        return _call_text_transform(self.get_session(session_id), "anonymize", text)

    def deanonymize(self, text: str, *, session_id: Any = None) -> str:
        return _call_text_transform(self.get_session(session_id), "deanonymize", text)


class PrivacyRail:
    """Context-aware wrapper around Palimpsest reversible anonymization."""

    def __init__(
        self,
        processor: Any | None = None,
        *,
        session_manager: PalimpsestSessionManager | None = None,
    ) -> None:
        if session_manager is None and processor is None:
            raise ValueError("PrivacyRail requires either processor or session_manager.")
        self._sessions = session_manager or PalimpsestSessionManager(processor)

    @classmethod
    def from_palimpsest(
        cls,
        *,
        locale: str = "ru-RU",
        run_entities: Iterable[str] | None = None,
        entity_table: Any = None,
        typed_placeholders: bool | None = None,
        palimpsest_options: Mapping[str, Any] | None = None,
        palimpsest_session_options: Mapping[str, Any] | None = None,
        verbose: bool = False,
    ) -> "PrivacyRail":
        _ensure_palimpsest_dependencies(locale)

        from palimpsest import Palimpsest

        constructor_kwargs = _palimpsest_constructor_kwargs(
            locale=locale,
            run_entities=run_entities,
            entity_table=entity_table,
            typed_placeholders=typed_placeholders,
            verbose=verbose,
            palimpsest_options=palimpsest_options,
        )
        _require_supported_kwargs(
            Palimpsest,
            constructor_kwargs,
            api_name="Palimpsest",
        )
        processor = Palimpsest(**constructor_kwargs)
        session_kwargs = _palimpsest_session_kwargs(
            entity_table=entity_table,
            typed_placeholders=typed_placeholders,
            palimpsest_session_options=palimpsest_session_options,
        )
        return cls(
            session_manager=PalimpsestSessionManager(
                processor,
                create_session_kwargs=session_kwargs,
            )
        )

    @property
    def sessions(self) -> PalimpsestSessionManager:
        return self._sessions

    def session_id_for_context(self, context: GuardrailContext | Mapping[str, Any]) -> str:
        return privacy_scope_key(context)

    def anonymize_text(self, text: str, context: GuardrailContext, boundary: str = "unknown") -> str:
        if not text:
            return text
        return self._sessions.anonymize(text, session_id=self.session_id_for_context(context))

    def deanonymize_text(self, text: str, context: GuardrailContext, boundary: str = "unknown") -> str:
        if not text:
            return text
        if not context.get("allow_deanonymization", True):
            return text
        return self._sessions.deanonymize(text, session_id=self.session_id_for_context(context))

    def reset_context(self, context: GuardrailContext) -> None:
        self._sessions.reset_session(self.session_id_for_context(context))

    def context_from_runtime(
        self,
        runtime: Any,
        *,
        state: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        route: str | None = None,
        model: str | None = None,
        tool_name: str | None = None,
    ) -> GuardrailContext:
        return build_guardrail_context(
            runtime=runtime,
            state=state,
            agent_name=agent_name,
            route=route,
            model=model,
            tool_name=tool_name,
        )


def anonymize_with_session(anonymizer: Any, text: str) -> str:
    """Best-effort adapter for optional legacy tool hooks."""
    if isinstance(anonymizer, PalimpsestSessionManager):
        return anonymizer.anonymize(text)
    if isinstance(anonymizer, PrivacyRail):
        context = build_guardrail_context(agent_name="legacy_tool")
        return anonymizer.anonymize_text(text, context, boundary="legacy_tool")
    get_session = getattr(anonymizer, "get_session", None)
    if callable(get_session):
        return _call_text_transform(get_session(), "anonymize", text)
    create_session = getattr(anonymizer, "create_session", None)
    if callable(create_session):
        return _call_text_transform(create_session(session_id=_DEFAULT_SESSION_ID), "anonymize", text)
    return _call_text_transform(anonymizer, "anonymize", text)


def map_strings(value: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {key: map_strings(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [map_strings(item, transform) for item in value]
    return value


def transform_content(content: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(content, str):
        return transform(content)
    if isinstance(content, list):
        result: List[Any] = []
        for item in content:
            if isinstance(item, dict):
                part = dict(item)
                for key in _TEXT_KEYS:
                    if isinstance(part.get(key), str):
                        part[key] = transform(part[key])
                result.append(part)
            else:
                result.append(item)
        return result
    return content


def clone_message_with_transform(message: BaseMessage, transform: Callable[[str], str]) -> BaseMessage:
    cloned = copy(message)
    cloned.content = transform_content(message.content, transform)
    return cloned


def transform_messages(messages: Iterable[BaseMessage], transform: Callable[[str], str]) -> List[BaseMessage]:
    return [clone_message_with_transform(message, transform) for message in messages]


__all__ = [
    "DEFAULT_PALIMPSEST_ENTITIES",
    "DEFAULT_PALIMPSEST_ENTITY_TABLE",
    "PalimpsestSessionManager",
    "PrivacyRail",
    "_call_text_transform",
    "anonymize_with_session",
    "clone_message_with_transform",
    "content_is_reset",
    "entity_types_from_table",
    "map_strings",
    "state_has_reset_message",
    "thread_id_from_config",
    "thread_id_from_runtime",
    "transform_content",
    "transform_messages",
]
