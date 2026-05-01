from __future__ import annotations

from copy import copy
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


def _normalise_session_id(value: Any) -> str:
    text = str(value or "").strip()
    return text or _DEFAULT_SESSION_ID


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

    def __init__(self, processor: Any, *, default_session_id: str = _DEFAULT_SESSION_ID) -> None:
        self._processor = processor
        self._default_session_id = _normalise_session_id(default_session_id)
        self._sessions: Dict[str, Any] = {}
        self._lock = RLock()

    def get_session(self, session_id: Any = None) -> Any:
        key = _normalise_session_id(session_id or self._default_session_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None or bool(getattr(session, "closed", False)):
                session = self._processor.create_session(session_id=key)
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
        run_entities: list[str] | None = None,
        verbose: bool = False,
    ) -> "PrivacyRail":
        _ensure_palimpsest_dependencies(locale)

        from palimpsest import Palimpsest

        return cls(
            Palimpsest(
                verbose=verbose,
                run_entities=run_entities or DEFAULT_PALIMPSEST_ENTITIES,
                locale=locale,
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
    "PalimpsestSessionManager",
    "PrivacyRail",
    "_call_text_transform",
    "anonymize_with_session",
    "clone_message_with_transform",
    "content_is_reset",
    "map_strings",
    "state_has_reset_message",
    "thread_id_from_config",
    "thread_id_from_runtime",
    "transform_content",
    "transform_messages",
]
