from __future__ import annotations

from copy import copy
import inspect
from uuid import uuid4
from typing import Any, Callable, Dict, Iterable, List

from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable

from .context import GuardrailContext, build_guardrail_context
from .decisions import GuardrailDecision, redact
from .injection import SECURITY_BLOCK_MESSAGE_RU, SECURITY_REVIEW_MESSAGE_RU
from .logging import GuardrailEventLogger
from .privacy import (
    PalimpsestSessionManager,
    PrivacyRail,
    _call_text_transform,
    clone_message_with_transform,
    map_strings,
    state_has_reset_message,
    thread_id_from_runtime,
    transform_content,
)
from .scanners import LLMGuardScannerRail, ScannerScanResult


class PalimpsestSessionMiddleware(AgentMiddleware):
    """Backward-compatible thread-scoped middleware used by legacy agents."""

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
        tool_call["args"] = map_strings(
            tool_call.get("args", {}),
            lambda text: _call_text_transform(session, "deanonymize", text),
        )
        result = handler(request.override(tool_call=tool_call))
        return self._anonymize_tool_result(result, session)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        session = self._sessions.session_for_runtime(request.runtime)
        tool_call = dict(request.tool_call)
        tool_call["args"] = map_strings(
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


_SCAN_TEXT_KEYS = ("text", "content", "input", "title", "caption", "markdown", "explanation")


def _message_is_untrusted_input(message: BaseMessage) -> bool:
    return isinstance(message, (HumanMessage, ToolMessage)) or getattr(message, "type", None) in {"human", "tool"}


def _message_is_model_output(message: BaseMessage) -> bool:
    return isinstance(message, AIMessage) or getattr(message, "type", None) == "ai"


def _decision_message(decision: GuardrailDecision) -> str:
    return SECURITY_REVIEW_MESSAGE_RU if decision["action"] == "review" else SECURITY_BLOCK_MESSAGE_RU


def _join_message_text(messages: Iterable[BaseMessage], *, system_prompt: str | None = None) -> str:
    parts: list[str] = []
    if system_prompt:
        parts.append(system_prompt)
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str) and item:
                    parts.append(item)
                elif isinstance(item, dict):
                    for key in _SCAN_TEXT_KEYS:
                        value = item.get(key)
                        if isinstance(value, str) and value:
                            parts.append(value)
    return "\n".join(parts)


def _message_text(message: BaseMessage) -> str:
    return _join_message_text([message])


def _message_role(message: BaseMessage) -> str:
    return str(getattr(message, "type", None) or type(message).__name__)


def _normalize_message_roles(message_roles: Iterable[str] | None) -> tuple[str, ...] | None:
    if message_roles is None:
        return None
    return tuple(str(role).lower() for role in message_roles)


def _latest_human_message_index(messages: Iterable[BaseMessage]) -> int | None:
    source_messages = list(messages)
    for index in range(len(source_messages) - 1, -1, -1):
        message = source_messages[index]
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            return index
    return None


def _command_with_update(command: Command, update: dict[str, Any]) -> Command:
    return Command(
        graph=command.graph,
        update=update,
        resume=command.resume,
        goto=command.goto,
    )


class SecurityScannerMiddleware(AgentMiddleware):
    """LLM Guard scanner middleware for model requests and responses."""

    def __init__(
        self,
        scanner_rail: LLMGuardScannerRail,
        *,
        agent_name: str = "unknown",
        event_logger: GuardrailEventLogger | None = None,
        event_log_path: str | None = None,
        scan_system_prompt: bool = False,
        scan_state_keys: Iterable[str] = (),
        composite_input_scanners: Iterable[str] | None = None,
        composite_recent_message_limit: int = 20,
        composite_message_roles: Iterable[str] | None = None,
        blocked_structured_response_factory: Callable[[GuardrailDecision], Any] | None = None,
    ) -> None:
        super().__init__()
        self._scanners = scanner_rail
        self._agent_name = agent_name
        self._events = event_logger or GuardrailEventLogger(event_log_path)
        self._scan_system_prompt = scan_system_prompt
        self._state_keys_to_scan = tuple(scan_state_keys)
        self._composite_input_scanners = (
            None if composite_input_scanners is None else tuple(composite_input_scanners)
        )
        self._composite_recent_message_limit = max(0, int(composite_recent_message_limit))
        self._composite_message_roles = _normalize_message_roles(composite_message_roles)
        self._blocked_structured_response_factory = blocked_structured_response_factory

    def _context(self, runtime: Any, state: Dict[str, Any] | None = None) -> GuardrailContext:
        return build_guardrail_context(
            runtime=runtime,
            state=state,
            agent_name=self._agent_name,
        )

    def _log_scan_result(self, result: ScannerScanResult) -> None:
        for decision in result.decisions:
            self._events.log_decision(decision)

    def before_agent(self, state, runtime) -> Dict[str, Any] | None:
        if state_has_reset_message(state):
            self._scanners.reset_context(self._context(runtime, state))
        return None

    async def abefore_agent(self, state, runtime) -> Dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def _blocked_model_response(
        self,
        decision: GuardrailDecision,
        *,
        remove_message_ids: Iterable[str] = (),
        replacement_messages: Iterable[BaseMessage] | None = None,
    ) -> ModelResponse | ExtendedModelResponse:
        structured_response = (
            self._blocked_structured_response_factory(decision)
            if self._blocked_structured_response_factory is not None
            else None
        )
        block_message = AIMessage(
            content=_decision_message(decision),
            id=f"guardrail-block-{uuid4().hex}",
        )
        model_response = ModelResponse(
            result=[block_message],
            structured_response=structured_response,
        )
        if replacement_messages is not None:
            return ExtendedModelResponse(
                model_response=model_response,
                command=Command(
                    update={
                        "messages": [
                            RemoveMessage(id=REMOVE_ALL_MESSAGES),
                            *replacement_messages,
                            block_message,
                        ]
                    }
                ),
            )
        removal_updates = [RemoveMessage(id=message_id) for message_id in remove_message_ids]
        if not removal_updates:
            return model_response
        return ExtendedModelResponse(
            model_response=model_response,
            command=Command(update={"messages": removal_updates}),
        )

    def _scan_content(
        self,
        content: Any,
        context: GuardrailContext,
        *,
        boundary: str,
        prompt: str = "",
        output: bool = False,
    ) -> tuple[Any, GuardrailDecision | None]:
        if isinstance(content, str):
            result = (
                self._scanners.scan_output_text(prompt, content, context, boundary=boundary)
                if output
                else self._scanners.scan_input_text(content, context, boundary=boundary)
            )
            self._log_scan_result(result)
            return result.text, result.blocked_decision

        if isinstance(content, list):
            updated: list[Any] = []
            for item in content:
                if isinstance(item, str):
                    scanned, decision = self._scan_content(
                        item,
                        context,
                        boundary=boundary,
                        prompt=prompt,
                        output=output,
                    )
                    if decision is not None:
                        return content, decision
                    updated.append(scanned)
                    continue
                if isinstance(item, dict):
                    part = dict(item)
                    for key in _SCAN_TEXT_KEYS:
                        if isinstance(part.get(key), str):
                            scanned, decision = self._scan_content(
                                part[key],
                                context,
                                boundary=boundary,
                                prompt=prompt,
                                output=output,
                            )
                            if decision is not None:
                                return content, decision
                            part[key] = scanned
                    updated.append(part)
                    continue
                updated.append(item)
            return updated, None

        return content, None

    def _scan_messages(
        self,
        messages: Iterable[BaseMessage],
        context: GuardrailContext,
        *,
        boundary: str,
        prompt: str = "",
        output: bool = False,
    ) -> tuple[list[BaseMessage], GuardrailDecision | None, list[str], list[BaseMessage] | None]:
        source_messages = list(messages)
        updated_messages: list[BaseMessage] = []
        for index, message in enumerate(source_messages):
            should_scan = _message_is_model_output(message) if output else _message_is_untrusted_input(message)
            if not should_scan:
                updated_messages.append(message)
                continue
            scanned_content, decision = self._scan_content(
                getattr(message, "content", None),
                context,
                boundary=boundary,
                prompt=prompt,
                output=output,
            )
            if decision is not None:
                message_id = getattr(message, "id", None)
                if isinstance(message_id, str) and message_id:
                    return source_messages, decision, [message_id], None
                return source_messages, decision, [], [*updated_messages, *source_messages[index + 1 :]]
            updated = copy(message)
            updated.content = scanned_content
            updated_messages.append(updated)
        return updated_messages, None, [], None

    def _scan_state_keys(
        self,
        state: Dict[str, Any],
        context: GuardrailContext,
        *,
        state_keys: Iterable[str] | None = None,
    ) -> tuple[Dict[str, Any], GuardrailDecision | None]:
        keys_to_scan = self._state_keys_to_scan if state_keys is None else tuple(state_keys)
        if not keys_to_scan:
            return state, None
        updated_state = dict(state)
        for key in keys_to_scan:
            value = updated_state.get(key)
            if not isinstance(value, str):
                continue
            result = self._scanners.scan_input_text(value, context, boundary=f"state.{key}")
            self._log_scan_result(result)
            if result.blocked_decision is not None:
                return state, result.blocked_decision
            updated_state[key] = result.text
        return updated_state, None

    def _composite_input_text(
        self,
        state: Dict[str, Any],
        messages: Iterable[BaseMessage],
        *,
        system_prompt: str | None = None,
        recent_message_limit: int | None = None,
        message_roles: Iterable[str] | None = None,
    ) -> str:
        parts: list[str] = []
        state_system_prompt = state.get("system_prompt")
        if isinstance(state_system_prompt, str) and state_system_prompt:
            parts.append("[STATE SYSTEM PROMPT]\n" + state_system_prompt)
        if system_prompt:
            parts.append("[RUNTIME SYSTEM PROMPT]\n" + system_prompt)

        source_messages = list(messages)
        limit = self._composite_recent_message_limit if recent_message_limit is None else max(0, int(recent_message_limit))
        if limit > 0:
            source_messages = source_messages[-limit:]
        allowed_roles = (
            self._composite_message_roles
            if message_roles is None
            else _normalize_message_roles(message_roles)
        )
        message_parts: list[str] = []
        for message in source_messages:
            role = _message_role(message).lower()
            if allowed_roles is not None and role not in allowed_roles:
                continue
            text = _message_text(message)
            if text:
                message_parts.append(f"[{role.upper()}]\n{text}")
        if message_parts:
            parts.append("[RECENT MESSAGES]\n" + "\n\n".join(message_parts))
        return "\n\n".join(parts)

    def _composite_block_cleanup(
        self,
        messages: Iterable[BaseMessage],
    ) -> tuple[list[str], list[BaseMessage] | None]:
        source_messages = list(messages)
        index = _latest_human_message_index(source_messages)
        if index is None:
            return [], None
        message = source_messages[index]
        message_id = getattr(message, "id", None)
        if isinstance(message_id, str) and message_id:
            return [message_id], None
        return [], [*source_messages[:index], *source_messages[index + 1 :]]

    def _scan_composite_input(
        self,
        state: Dict[str, Any],
        messages: Iterable[BaseMessage],
        context: GuardrailContext,
        *,
        system_prompt: str | None = None,
        boundary: str = "composite_model_request",
        scanner_names: Iterable[str] | None = None,
        recent_message_limit: int | None = None,
        message_roles: Iterable[str] | None = None,
    ) -> tuple[GuardrailDecision | None, list[str], list[BaseMessage] | None]:
        composite_text = self._composite_input_text(
            state,
            messages,
            system_prompt=system_prompt,
            recent_message_limit=recent_message_limit,
            message_roles=message_roles,
        )
        if not composite_text:
            return None, [], None
        result = self._scanners.scan_composite_input_text(
            composite_text,
            context,
            scanner_names=self._composite_input_scanners if scanner_names is None else scanner_names,
            boundary=boundary,
        )
        self._log_scan_result(result)
        if result.blocked_decision is None:
            return None, [], None
        remove_ids, replacement_messages = self._composite_block_cleanup(messages)
        return result.blocked_decision, remove_ids, replacement_messages

    def scan_node_state(
        self,
        state: Dict[str, Any],
        runtime: Any,
        *,
        boundary: str = "node_request",
        scan_state_keys: Iterable[str] | None = None,
        composite_input_scanners: Iterable[str] | None = None,
        composite_recent_message_limit: int | None = None,
        composite_message_roles: Iterable[str] | None = None,
    ) -> tuple[Dict[str, Any] | None, GuardrailDecision | None, list[str], list[BaseMessage] | None]:
        context = self._context(runtime, state)
        if state_has_reset_message(state):
            self._scanners.reset_context(context)

        source_state = dict(state or {})
        messages, decision, remove_message_ids, replacement_messages = self._scan_messages(
            source_state.get("messages") or [],
            context,
            boundary=boundary,
        )
        if decision is not None:
            return None, decision, remove_message_ids, replacement_messages
        source_state["messages"] = messages

        scanned_state, decision = self._scan_state_keys(source_state, context, state_keys=scan_state_keys)
        if decision is not None:
            return None, decision, [], None

        decision, remove_message_ids, replacement_messages = self._scan_composite_input(
            scanned_state,
            scanned_state.get("messages") or [],
            context,
            boundary=f"composite_{boundary}",
            scanner_names=composite_input_scanners,
            recent_message_limit=composite_recent_message_limit,
            message_roles=composite_message_roles,
        )
        if decision is not None:
            return None, decision, remove_message_ids, replacement_messages

        return scanned_state, None, [], None

    def _scan_model_request(self, request, context: GuardrailContext):
        messages, decision, remove_message_ids, replacement_messages = self._scan_messages(request.messages, context, boundary="model_request")
        if decision is not None:
            return None, decision, remove_message_ids, replacement_messages

        state, decision = self._scan_state_keys(dict(request.state or {}), context)
        if decision is not None:
            return None, decision, [], None

        system_prompt = getattr(request, "system_prompt", None)
        if self._scan_system_prompt and isinstance(system_prompt, str):
            result = self._scanners.scan_input_text(system_prompt, context, boundary="system_prompt")
            self._log_scan_result(result)
            if result.blocked_decision is not None:
                return None, result.blocked_decision, [], None
            system_prompt = result.text

        decision, remove_message_ids, replacement_messages = self._scan_composite_input(
            state,
            messages,
            context,
            system_prompt=system_prompt,
        )
        if decision is not None:
            return None, decision, remove_message_ids, replacement_messages

        return request.override(messages=messages, state=state, system_prompt=system_prompt), None, [], None

    def _scan_model_result(self, result: Any, context: GuardrailContext, *, prompt: str) -> Any:
        if isinstance(result, AIMessage):
            messages, decision, _remove_message_ids, _replacement_messages = self._scan_messages([result], context, boundary="model_response", prompt=prompt, output=True)
            if decision is not None:
                return self._blocked_model_response(decision)
            return messages[0]
        if isinstance(result, ModelResponse):
            messages, decision, _remove_message_ids, _replacement_messages = self._scan_messages(result.result, context, boundary="model_response", prompt=prompt, output=True)
            if decision is not None:
                return self._blocked_model_response(decision)
            return ModelResponse(result=messages, structured_response=result.structured_response)
        return result

    def wrap_model_call(self, request, handler):
        context = self._context(request.runtime, request.state)
        updated_request, decision, remove_message_ids, replacement_messages = self._scan_model_request(request, context)
        if decision is not None:
            return self._blocked_model_response(
                decision,
                remove_message_ids=remove_message_ids,
                replacement_messages=replacement_messages,
            )
        prompt = _join_message_text(updated_request.messages, system_prompt=getattr(updated_request, "system_prompt", None))
        return self._scan_model_result(handler(updated_request), context, prompt=prompt)

    async def awrap_model_call(self, request, handler):
        context = self._context(request.runtime, request.state)
        updated_request, decision, remove_message_ids, replacement_messages = self._scan_model_request(request, context)
        if decision is not None:
            return self._blocked_model_response(
                decision,
                remove_message_ids=remove_message_ids,
                replacement_messages=replacement_messages,
            )
        prompt = _join_message_text(updated_request.messages, system_prompt=getattr(updated_request, "system_prompt", None))
        return self._scan_model_result(await handler(updated_request), context, prompt=prompt)


class ToolContentScannerMiddleware(AgentMiddleware):
    """Scans model-generated tool arguments before tool execution."""

    def __init__(
        self,
        scanner_rail: LLMGuardScannerRail,
        *,
        agent_name: str = "unknown",
        event_logger: GuardrailEventLogger | None = None,
        event_log_path: str | None = None,
    ) -> None:
        super().__init__()
        self._scanners = scanner_rail
        self._agent_name = agent_name
        self._events = event_logger or GuardrailEventLogger(event_log_path)

    def _context(self, request: ToolCallRequest) -> GuardrailContext:
        tool_name = str(request.tool_call.get("name") or "")
        return build_guardrail_context(
            runtime=request.runtime,
            state=request.state,
            agent_name=self._agent_name,
            tool_name=tool_name,
        )

    def _log_scan_result(self, result: ScannerScanResult) -> None:
        for decision in result.decisions:
            self._events.log_decision(decision)

    def _scan_tool_value(
        self,
        value: Any,
        context: GuardrailContext,
        *,
        prompt: str = "",
    ) -> tuple[Any, GuardrailDecision | None]:
        if isinstance(value, str):
            result = self._scanners.scan_output_text(prompt, value, context, boundary="tool_arguments")
            self._log_scan_result(result)
            return result.text, result.blocked_decision
        if isinstance(value, dict):
            updated = {}
            for key, item in value.items():
                scanned, decision = self._scan_tool_value(item, context, prompt=prompt)
                if decision is not None:
                    return value, decision
                updated[key] = scanned
            return updated, None
        if isinstance(value, list):
            updated = []
            for item in value:
                scanned, decision = self._scan_tool_value(item, context, prompt=prompt)
                if decision is not None:
                    return value, decision
                updated.append(scanned)
            return updated, None
        return value, None

    def _tool_prompt(self, request: ToolCallRequest) -> str:
        if not isinstance(request.state, dict):
            return ""
        return _join_message_text(request.state.get("messages") or [])

    def _tool_call_message_ids(self, request: ToolCallRequest) -> list[str]:
        tool_call_id = request.tool_call.get("id")
        if not tool_call_id or not isinstance(request.state, dict):
            return []
        message_ids: list[str] = []
        for message in request.state.get("messages") or []:
            if not _message_is_model_output(message):
                continue
            for tool_call in getattr(message, "tool_calls", None) or []:
                if isinstance(tool_call, dict) and tool_call.get("id") == tool_call_id:
                    message_id = getattr(message, "id", None)
                    if isinstance(message_id, str) and message_id:
                        message_ids.append(message_id)
                    break
        return message_ids

    def _blocked_tool_result(self, request: ToolCallRequest, decision: GuardrailDecision) -> ToolMessage | Command:
        tool_call_id = request.tool_call.get("id")
        tool_message_id = f"guardrail-blocked-tool-{tool_call_id}" if isinstance(tool_call_id, str) and tool_call_id else None
        tool_message = ToolMessage(
            content=_decision_message(decision),
            tool_call_id=tool_call_id,
            id=tool_message_id,
        )
        removal_updates = [RemoveMessage(id=message_id) for message_id in self._tool_call_message_ids(request)]
        if not removal_updates:
            return tool_message
        tool_updates: list[BaseMessage] = [tool_message]
        if tool_message_id is not None:
            tool_updates.append(RemoveMessage(id=tool_message_id))
        return Command(
            update={
                "messages": [
                    *removal_updates,
                    *tool_updates,
                    AIMessage(
                        content=_decision_message(decision),
                        id=f"guardrail-block-{uuid4().hex}",
                    ),
                ]
            }
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        context = self._context(request)
        tool_call = dict(request.tool_call)
        scanned_args, decision = self._scan_tool_value(
            tool_call.get("args", {}),
            context,
            prompt=self._tool_prompt(request),
        )
        if decision is not None:
            return self._blocked_tool_result(request, decision)
        tool_call["args"] = scanned_args
        return handler(request.override(tool_call=tool_call))

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        context = self._context(request)
        tool_call = dict(request.tool_call)
        scanned_args, decision = self._scan_tool_value(
            tool_call.get("args", {}),
            context,
            prompt=self._tool_prompt(request),
        )
        if decision is not None:
            return self._blocked_tool_result(request, decision)
        tool_call["args"] = scanned_args
        return await handler(request.override(tool_call=tool_call))


class PrivacyModelRequestMiddleware(AgentMiddleware):
    """Context-aware privacy middleware for guarded agents."""

    def __init__(
        self,
        privacy_rail: PrivacyRail,
        *,
        agent_name: str = "unknown",
        anonymize_tool_results: bool = True,
        event_logger: GuardrailEventLogger | None = None,
        event_log_path: str | None = None,
    ) -> None:
        super().__init__()
        self._privacy = privacy_rail
        self._agent_name = agent_name
        self._anonymize_tool_results = anonymize_tool_results
        self._events = event_logger or GuardrailEventLogger(event_log_path)

    def _context(self, runtime: Any, state: Dict[str, Any] | None = None, *, tool_name: str | None = None) -> GuardrailContext:
        return build_guardrail_context(
            runtime=runtime,
            state=state,
            agent_name=self._agent_name,
            tool_name=tool_name,
        )

    def _log(self, context: GuardrailContext, *, boundary: str, reason: str, categories: list[str]) -> None:
        self._events.log_decision(
            redact(
                reason,
                categories=categories,
                metadata={
                    "boundary": boundary,
                    "rail": "privacy",
                    "agent_name": context["agent_name"],
                    "thread_id": context["thread_id"],
                    "user_role": context["user_role"],
                    "request_id": context["request_id"],
                    "tool_name": context.get("tool_name"),
                },
            )
        )

    def before_agent(self, state, runtime) -> Dict[str, Any] | None:
        if state_has_reset_message(state):
            self._privacy.reset_context(self._context(runtime, state))
        return None

    async def abefore_agent(self, state, runtime) -> Dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def _anonymize_messages(self, messages: Iterable[BaseMessage], context: GuardrailContext) -> List[BaseMessage]:
        transformed = [
            clone_message_with_transform(
                message,
                lambda text: self._privacy.anonymize_text(text, context, boundary="model_request"),
            )
            for message in messages
        ]
        self._log(context, boundary="model_request", reason="Model request text anonymized.", categories=["privacy", "pii"])
        return transformed

    def _anonymize_system_prompt(self, prompt: str | None, context: GuardrailContext) -> str | None:
        if prompt is None:
            return None
        updated = self._privacy.anonymize_text(prompt, context, boundary="system_prompt")
        self._log(context, boundary="system_prompt", reason="System prompt text anonymized.", categories=["privacy", "pii"])
        return updated

    def _deanonymize_ai_message(self, message: BaseMessage, context: GuardrailContext) -> BaseMessage:
        if not isinstance(message, AIMessage):
            return message
        if not context.get("allow_deanonymization", True):
            self._log(
                context,
                boundary="model_response",
                reason="Model response de-anonymization skipped by policy.",
                categories=["privacy", "policy"],
            )
            return message
        updated = clone_message_with_transform(
            message,
            lambda text: self._privacy.deanonymize_text(text, context, boundary="model_response"),
        )
        self._log(context, boundary="model_response", reason="Model response text de-anonymized.", categories=["privacy", "pii"])
        return updated

    def _deanonymize_model_result(self, result: Any, context: GuardrailContext) -> Any:
        if isinstance(result, AIMessage):
            return self._deanonymize_ai_message(result, context)
        if isinstance(result, ModelResponse):
            return ModelResponse(
                result=[self._deanonymize_ai_message(message, context) for message in result.result],
                structured_response=result.structured_response,
            )
        if isinstance(result, ExtendedModelResponse):
            return ExtendedModelResponse(
                model_response=self._deanonymize_model_result(result.model_response, context),
                command=result.command,
            )
        return result

    def _override_model_request(self, request, context: GuardrailContext):
        return request.override(
            messages=self._anonymize_messages(request.messages, context),
            system_prompt=self._anonymize_system_prompt(getattr(request, "system_prompt", None), context),
        )

    def _transform_node_messages(
        self,
        messages: Any,
        context: GuardrailContext,
        transform: Callable[[str], str],
    ) -> Any:
        if isinstance(messages, RemoveMessage):
            return messages
        if isinstance(messages, BaseMessage):
            return clone_message_with_transform(messages, transform)
        if isinstance(messages, list):
            return [
                self._transform_node_messages(message, context, transform)
                if isinstance(message, (BaseMessage, RemoveMessage))
                else message
                for message in messages
            ]
        return messages

    def anonymize_node_state(
        self,
        state: Dict[str, Any],
        runtime: Any,
        *,
        state_keys: Iterable[str] = ("system_prompt",),
    ) -> Dict[str, Any]:
        context = self._context(runtime, state)
        if state_has_reset_message(state):
            self._privacy.reset_context(context)
        updated = dict(state)
        if "messages" in updated:
            updated["messages"] = self._transform_node_messages(
                updated["messages"],
                context,
                lambda text: self._privacy.anonymize_text(text, context, boundary="node_input"),
            )
            self._log(context, boundary="node_input", reason="Node input messages anonymized.", categories=["privacy", "pii"])
        for key in state_keys:
            value = updated.get(key)
            if isinstance(value, str):
                updated[key] = self._privacy.anonymize_text(value, context, boundary=f"node_state.{key}")
                self._log(context, boundary=f"node_state.{key}", reason="Node state text anonymized.", categories=["privacy", "pii"])
        return updated

    def deanonymize_node_result(
        self,
        result: Any,
        runtime: Any,
        state: Dict[str, Any],
        *,
        state_keys: Iterable[str] = ("system_prompt",),
    ) -> Any:
        context = self._context(runtime, state)
        if not context.get("allow_deanonymization", True):
            self._log(
                context,
                boundary="node_output",
                reason="Node output de-anonymization skipped by policy.",
                categories=["privacy", "policy"],
            )
            return result

        def deanonymize(text: str) -> str:
            return self._privacy.deanonymize_text(text, context, boundary="node_output")

        def transform_update(update: Any) -> Any:
            if not isinstance(update, dict):
                return update
            updated = dict(update)
            if "messages" in updated:
                updated["messages"] = self._transform_node_messages(updated["messages"], context, deanonymize)
            for key in state_keys:
                value = updated.get(key)
                if isinstance(value, str):
                    updated[key] = deanonymize(value)
            return updated

        if isinstance(result, Command):
            update = transform_update(getattr(result, "update", None))
            if update is getattr(result, "update", None):
                return result
            self._log(context, boundary="node_output", reason="Node command output de-anonymized.", categories=["privacy", "pii"])
            return _command_with_update(result, update)
        if isinstance(result, dict):
            updated_result = transform_update(result)
            self._log(context, boundary="node_output", reason="Node output state de-anonymized.", categories=["privacy", "pii"])
            return updated_result
        return result

    def wrap_model_call(self, request, handler):
        context = self._context(request.runtime, request.state)
        updated_request = self._override_model_request(request, context)
        return self._deanonymize_model_result(handler(updated_request), context)

    async def awrap_model_call(self, request, handler):
        context = self._context(request.runtime, request.state)
        updated_request = self._override_model_request(request, context)
        return self._deanonymize_model_result(await handler(updated_request), context)

    def _anonymize_tool_messages(self, messages: Any, context: GuardrailContext) -> Any:
        if isinstance(messages, RemoveMessage):
            return messages
        if isinstance(messages, BaseMessage):
            return clone_message_with_transform(
                messages,
                lambda text: self._privacy.anonymize_text(text, context, boundary="tool_result"),
            )
        if isinstance(messages, list):
            return [
                self._anonymize_tool_messages(message, context)
                if isinstance(message, BaseMessage)
                else message
                for message in messages
            ]
        return messages

    def _anonymize_tool_result(self, result: Any, context: GuardrailContext) -> Any:
        if not self._anonymize_tool_results:
            return result
        if isinstance(result, ToolMessage):
            updated = copy(result)
            updated.content = transform_content(
                result.content,
                lambda text: self._privacy.anonymize_text(text, context, boundary="tool_result"),
            )
            self._log(context, boundary="tool_result", reason="Tool message content anonymized.", categories=["privacy", "pii"])
            return updated
        if isinstance(result, Command):
            update = getattr(result, "update", None)
            if isinstance(update, dict) and "messages" in update:
                updated_update = dict(update)
                updated_update["messages"] = self._anonymize_tool_messages(update["messages"], context)
                self._log(context, boundary="tool_result", reason="Command message updates anonymized.", categories=["privacy", "pii"])
                return Command(
                    graph=result.graph,
                    update=updated_update,
                    resume=result.resume,
                    goto=result.goto,
                )
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_name = str(request.tool_call.get("name") or "")
        context = self._context(request.runtime, request.state, tool_name=tool_name)
        tool_call = dict(request.tool_call)
        tool_call["args"] = map_strings(
            tool_call.get("args", {}),
            lambda text: self._privacy.deanonymize_text(text, context, boundary="tool_arguments"),
        )
        self._log(context, boundary="tool_arguments", reason="Tool arguments de-anonymized before execution.", categories=["privacy", "pii"])
        result = handler(request.override(tool_call=tool_call))
        return self._anonymize_tool_result(result, context)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        tool_name = str(request.tool_call.get("name") or "")
        context = self._context(request.runtime, request.state, tool_name=tool_name)
        tool_call = dict(request.tool_call)
        tool_call["args"] = map_strings(
            tool_call.get("args", {}),
            lambda text: self._privacy.deanonymize_text(text, context, boundary="tool_arguments"),
        )
        self._log(context, boundary="tool_arguments", reason="Tool arguments de-anonymized before execution.", categories=["privacy", "pii"])
        result = await handler(request.override(tool_call=tool_call))
        return self._anonymize_tool_result(result, context)


def _blocked_node_command(
    decision: GuardrailDecision,
    *,
    remove_message_ids: Iterable[str] = (),
    replacement_messages: Iterable[BaseMessage] | None = None,
) -> Command:
    block_message = AIMessage(
        content=_decision_message(decision),
        id=f"guardrail-block-{uuid4().hex}",
    )
    if replacement_messages is not None:
        return Command(
            update={
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *replacement_messages,
                    block_message,
                ]
            }
        )
    return Command(
        update={
            "messages": [
                *[RemoveMessage(id=message_id) for message_id in remove_message_ids],
                block_message,
            ]
        }
    )


def _call_graph_node(node: Callable[..., Any], state: Dict[str, Any], config: RunnableConfig, runtime: Any) -> Any:
    return node(state, config=config, runtime=runtime)


async def _acall_graph_node(node: Callable[..., Any], state: Dict[str, Any], config: RunnableConfig, runtime: Any) -> Any:
    result = _call_graph_node(node, state, config, runtime)
    if inspect.isawaitable(result):
        return await result
    return result


def guarded_node(
    node: Callable[..., Any],
    *,
    security_middleware: SecurityScannerMiddleware | None = None,
    privacy_middleware: PrivacyModelRequestMiddleware | None = None,
    scan_state_keys: Iterable[str] = ("system_prompt",),
    composite_input_scanners: Iterable[str] | None = ("PromptInjection",),
    composite_recent_message_limit: int = 20,
    composite_message_roles: Iterable[str] | None = None,
) -> RunnableCallable:
    """Wrap a LangGraph node with common scanner and privacy guardrails."""

    def _prepare_state(state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | Command:
        guarded_state = dict(state or {})
        if security_middleware is not None:
            scanned_state, decision, remove_message_ids, replacement_messages = security_middleware.scan_node_state(
                guarded_state,
                runtime,
                scan_state_keys=scan_state_keys,
                composite_input_scanners=composite_input_scanners,
                composite_recent_message_limit=composite_recent_message_limit,
                composite_message_roles=composite_message_roles,
            )
            if decision is not None:
                return _blocked_node_command(
                    decision,
                    remove_message_ids=remove_message_ids,
                    replacement_messages=replacement_messages,
                )
            guarded_state = scanned_state or guarded_state

        if privacy_middleware is not None:
            guarded_state = privacy_middleware.anonymize_node_state(
                guarded_state,
                runtime,
                state_keys=scan_state_keys,
            )
        return guarded_state

    def _finish_result(result: Any, runtime: Any, scanned_state: Dict[str, Any]) -> Any:
        if privacy_middleware is None:
            return result
        return privacy_middleware.deanonymize_node_result(
            result,
            runtime,
            scanned_state,
            state_keys=scan_state_keys,
        )

    def invoke(state: Dict[str, Any], config: RunnableConfig = None, runtime: Any = None) -> Any:
        prepared_state = _prepare_state(state, runtime)
        if isinstance(prepared_state, Command):
            return prepared_state
        result = _call_graph_node(node, prepared_state, config, runtime)
        if inspect.isawaitable(result):
            raise RuntimeError("guarded_node sync path received an awaitable node result.")
        return _finish_result(result, runtime, prepared_state)

    async def ainvoke(state: Dict[str, Any], config: RunnableConfig = None, runtime: Any = None) -> Any:
        prepared_state = _prepare_state(state, runtime)
        if isinstance(prepared_state, Command):
            return prepared_state
        result = await _acall_graph_node(node, prepared_state, config, runtime)
        return _finish_result(result, runtime, prepared_state)

    return RunnableCallable(invoke, ainvoke, trace=False)


__all__ = [
    "guarded_node",
    "PalimpsestSessionMiddleware",
    "PrivacyModelRequestMiddleware",
    "SecurityScannerMiddleware",
    "ToolContentScannerMiddleware",
]
