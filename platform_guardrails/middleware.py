from __future__ import annotations

from copy import copy
from typing import Any, Dict, Iterable, List

from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from .context import GuardrailContext, build_guardrail_context
from .decisions import redact
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

    def _anonymize_system_message(self, message: SystemMessage | None, context: GuardrailContext) -> SystemMessage | None:
        if message is None:
            return None
        updated = clone_message_with_transform(
            message,
            lambda text: self._privacy.anonymize_text(text, context, boundary="system_message"),
        )
        self._log(context, boundary="system_message", reason="System message text anonymized.", categories=["privacy", "pii"])
        return updated if isinstance(updated, SystemMessage) else SystemMessage(content=updated.content)

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
            system_message=self._anonymize_system_message(request.system_message, context),
        )

    def wrap_model_call(self, request, handler):
        context = self._context(request.runtime, request.state)
        updated_request = self._override_model_request(request, context)
        return self._deanonymize_model_result(handler(updated_request), context)

    async def awrap_model_call(self, request, handler):
        context = self._context(request.runtime, request.state)
        updated_request = self._override_model_request(request, context)
        return self._deanonymize_model_result(await handler(updated_request), context)

    def _anonymize_tool_messages(self, messages: Any, context: GuardrailContext) -> Any:
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


__all__ = [
    "PalimpsestSessionMiddleware",
    "PrivacyModelRequestMiddleware",
]
