from __future__ import annotations

from copy import copy
from typing import Any, Callable, Dict, List, TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:  # pragma: no cover
    from palimpsest import Palimpsest as _Palimpsest


def _map_strings(value: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_strings(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_strings(item, transform) for item in value]
    return value


def _transform_content(content: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(content, str):
        return transform(content)
    if isinstance(content, list):
        result: List[Any] = []
        for item in content:
            if isinstance(item, dict):
                part = dict(item)
                for key in ("text", "content", "input", "title", "caption", "markdown", "explanation"):
                    if isinstance(part.get(key), str):
                        part[key] = transform(part[key])
                result.append(part)
            else:
                result.append(item)
        return result
    return content


def _clone_message(message: BaseMessage, transform: Callable[[str], str]) -> BaseMessage:
    cloned = copy(message)
    cloned.content = _transform_content(message.content, transform)
    return cloned


class PalimpsestMiddleware(AgentMiddleware):
    def __init__(self, anonymizer: Any, *, anonymize_tool_results: bool = True) -> None:
        super().__init__()
        self._anonymizer = anonymizer
        self._anonymize_tool_results = anonymize_tool_results

    def before_model(self, state, runtime) -> Dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages:
            return None

        transformed: List[BaseMessage] = []
        any_changed = False
        for message in messages:
            if isinstance(message, (HumanMessage, ToolMessage)):
                transformed.append(_clone_message(message, self._anonymizer.anonimize))
                any_changed = True
            else:
                transformed.append(message)

        if not any_changed:
            return None
        return {"messages": transformed}

    def after_model(self, state, runtime) -> Dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AIMessage):
                updated = copy(message)
                updated.content = _transform_content(message.content, self._anonymizer.deanonimize)
                messages[index] = updated
                return {"messages": messages}
        return None

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_call = dict(request.tool_call)
        tool_call["args"] = _map_strings(tool_call.get("args", {}), self._anonymizer.deanonimize)
        result = handler(request.override(tool_call=tool_call))
        if not self._anonymize_tool_results:
            return result
        if isinstance(result, ToolMessage):
            updated = copy(result)
            updated.content = _transform_content(result.content, self._anonymizer.anonimize)
            return updated
        return result


def build_palimpsest_middleware(locale: str = "ru-RU") -> PalimpsestMiddleware:
    from palimpsest import Palimpsest

    anonymizer = Palimpsest(
        verbose=False,
        run_entities=[
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
        ],
        locale=locale,
    )
    return PalimpsestMiddleware(anonymizer)
