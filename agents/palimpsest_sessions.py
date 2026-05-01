from __future__ import annotations

from copy import copy
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List

from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config


_DEFAULT_SESSION_ID = "__manual__"
_TEXT_KEYS = ("text", "content", "input", "title", "caption", "markdown", "explanation")
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


def thread_id_from_config(config: RunnableConfig | None) -> str | None:
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
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
    """Owns one PalimpsestSession per LangGraph thread for an initialized agent."""

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


def anonymize_with_session(anonymizer: Any, text: str) -> str:
    """Best-effort adapter for optional legacy tool hooks."""
    if isinstance(anonymizer, PalimpsestSessionManager):
        return anonymizer.anonymize(text)
    get_session = getattr(anonymizer, "get_session", None)
    if callable(get_session):
        return _call_text_transform(get_session(), "anonymize", text)
    create_session = getattr(anonymizer, "create_session", None)
    if callable(create_session):
        return _call_text_transform(create_session(session_id=_DEFAULT_SESSION_ID), "anonymize", text)
    return _call_text_transform(anonymizer, "anonymize", text)


def _map_strings(value: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_strings(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings(item, transform) for item in value]
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


class PalimpsestSessionMiddleware(AgentMiddleware):
    def __init__(
        self,
        sessions: PalimpsestSessionManager,
        *,
        anonymize_tool_results: bool = True,
        log_path: str | None = None,
    ) -> None:
        super().__init__()
        self._sessions = sessions
        self._anonymize_tool_results = anonymize_tool_results
        self._log_path = log_path

    def before_agent(self, state, runtime) -> Dict[str, Any] | None:
        if state_has_reset_message(state):
            self._sessions.reset_session(thread_id_from_runtime(runtime))
        return None

    async def abefore_agent(self, state, runtime) -> Dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def _transform_messages(self, messages: Iterable[BaseMessage], session: Any) -> List[BaseMessage]:
        transformed: List[BaseMessage] = []
        log_rows: List[tuple[Any, Any]] = []
        anonymize = lambda text: _call_text_transform(session, "anonymize", text)
        for message in messages:
            updated = clone_message_with_transform(message, anonymize)
            transformed.append(updated)
            if self._log_path:
                log_rows.append((getattr(message, "content", None), getattr(updated, "content", None)))
        if log_rows:
            with open(self._log_path, "a", encoding="utf-8") as log_file:
                for before, after in log_rows:
                    log_file.write(f"BEFORE ANONIMIZATION:\n{before}\n")
                    log_file.write(f"AFTER ANONIMIZATION:\n{after}\n\n")
        return transformed

    def _deanonymize_ai_message(self, message: BaseMessage, session: Any) -> BaseMessage:
        if not isinstance(message, AIMessage):
            return message
        deanonymize = lambda text: _call_text_transform(session, "deanonymize", text)
        updated = clone_message_with_transform(message, deanonymize)
        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"BEFORE DEANONIMIZATION:\n{message.content}\n")
                log_file.write(f"AFTER DEANONIMIZATION:\n{updated.content}\n\n")
        return updated

    def _deanonymize_model_result(self, result: Any, session: Any) -> Any:
        if isinstance(result, AIMessage):
            return self._deanonymize_ai_message(result, session)
        if isinstance(result, ModelResponse):
            return ModelResponse(
                result=[self._deanonymize_ai_message(message, session) for message in result.result],
                structured_response=result.structured_response,
            )
        if isinstance(result, ExtendedModelResponse):
            return ExtendedModelResponse(
                model_response=self._deanonymize_model_result(result.model_response, session),
                command=result.command,
            )
        return result

    def wrap_model_call(self, request, handler):
        session = self._sessions.session_for_runtime(request.runtime)
        updated_request = request.override(messages=self._transform_messages(request.messages, session))
        return self._deanonymize_model_result(handler(updated_request), session)

    async def awrap_model_call(self, request, handler):
        session = self._sessions.session_for_runtime(request.runtime)
        updated_request = request.override(messages=self._transform_messages(request.messages, session))
        return self._deanonymize_model_result(await handler(updated_request), session)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        session = self._sessions.session_for_runtime(request.runtime)
        tool_call = dict(request.tool_call)
        tool_call["args"] = _map_strings(
            tool_call.get("args", {}),
            lambda text: _call_text_transform(session, "deanonymize", text),
        )
        result = handler(request.override(tool_call=tool_call))
        return self._anonymize_tool_result(result, session)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        session = self._sessions.session_for_runtime(request.runtime)
        tool_call = dict(request.tool_call)
        tool_call["args"] = _map_strings(
            tool_call.get("args", {}),
            lambda text: _call_text_transform(session, "deanonymize", text),
        )
        result = await handler(request.override(tool_call=tool_call))
        return self._anonymize_tool_result(result, session)

    def _anonymize_tool_result(self, result: Any, session: Any) -> Any:
        if not self._anonymize_tool_results:
            return result
        if isinstance(result, ToolMessage):
            updated = copy(result)
            updated.content = transform_content(
                result.content,
                lambda text: _call_text_transform(session, "anonymize", text),
            )
            return updated
        return result
