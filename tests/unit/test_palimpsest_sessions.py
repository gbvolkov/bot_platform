from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.palimpsest_sessions import PalimpsestSessionManager, PalimpsestSessionMiddleware


class FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.closed = False
        self.reset_count = 0

    def anonymize(self, text: str) -> str:
        return f"anon[{self.session_id}]({text})"

    def deanonymize(self, text: str) -> str:
        return f"deanon[{self.session_id}]({text})"

    def reset(self) -> None:
        self.reset_count += 1


class FakeProcessor:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def create_session(self, session_id: str | None = None) -> FakeSession:
        session = FakeSession(session_id or "missing")
        self.sessions.append(session)
        return session


class LegacyFakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    def anonimize(self, text: str) -> str:
        return f"legacy-anon[{self.session_id}]({text})"

    def deanonimize(self, text: str) -> str:
        return f"legacy-deanon[{self.session_id}]({text})"


class LegacyFakeProcessor:
    def create_session(self, session_id: str | None = None) -> LegacyFakeSession:
        return LegacyFakeSession(session_id or "missing")


def test_session_manager_reuses_sessions_per_thread_and_isolates_threads():
    processor = FakeProcessor()
    manager = PalimpsestSessionManager(processor)

    assert manager.anonymize("first", session_id="thread-1") == "anon[thread-1](first)"
    assert manager.anonymize("second", session_id="thread-1") == "anon[thread-1](second)"
    assert manager.anonymize("third", session_id="thread-2") == "anon[thread-2](third)"

    assert [session.session_id for session in processor.sessions] == ["thread-1", "thread-2"]


def test_session_manager_supports_legacy_palimpsest_method_names():
    manager = PalimpsestSessionManager(LegacyFakeProcessor())

    assert manager.anonymize("first", session_id="thread-1") == "legacy-anon[thread-1](first)"
    assert manager.deanonymize("second", session_id="thread-1") == "legacy-deanon[thread-1](second)"


def test_session_manager_reset_only_resets_target_thread():
    processor = FakeProcessor()
    manager = PalimpsestSessionManager(processor)
    first = manager.get_session("thread-1")
    second = manager.get_session("thread-2")

    manager.reset_session("thread-1")

    assert first.reset_count == 1
    assert second.reset_count == 0


def test_middleware_uses_thread_session_for_model_messages_and_response():
    processor = FakeProcessor()
    manager = PalimpsestSessionManager(processor)
    middleware = PalimpsestSessionMiddleware(manager)
    runtime = SimpleNamespace(execution_info=SimpleNamespace(thread_id="conv-1"))
    request = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="hello")],
        state={"messages": []},
        runtime=runtime,
    )
    captured = {}

    def handler(updated_request):
        captured["messages"] = updated_request.messages
        return ModelResponse(result=[AIMessage(content="fake answer")])

    result = middleware.wrap_model_call(request, handler)

    assert captured["messages"][0].content == "anon[conv-1](hello)"
    assert result.result[0].content == "deanon[conv-1](fake answer)"
    assert [session.session_id for session in processor.sessions] == ["conv-1"]


def test_middleware_resets_thread_session_on_reset_message():
    processor = FakeProcessor()
    manager = PalimpsestSessionManager(processor)
    session = manager.get_session("conv-reset")
    middleware = PalimpsestSessionMiddleware(manager)
    runtime = SimpleNamespace(execution_info=SimpleNamespace(thread_id="conv-reset"))
    state = {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}

    middleware.before_agent(state, runtime)

    assert session.reset_count == 1


def test_async_tool_wrapper_uses_runtime_config_thread_session():
    processor = FakeProcessor()
    manager = PalimpsestSessionManager(processor)
    middleware = PalimpsestSessionMiddleware(manager)
    runtime = SimpleNamespace(
        execution_info=None,
        config={"configurable": {"thread_id": "conv-tool"}},
    )
    request = ToolCallRequest(
        tool_call={"name": "lookup", "args": {"query": "fake user"}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=runtime,
    )
    captured = {}

    async def handler(updated_request):
        captured["args"] = updated_request.tool_call["args"]
        return ToolMessage(content="real tool result", tool_call_id="call-1")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert captured["args"] == {"query": "deanon[conv-tool](fake user)"}
    assert result.content == "anon[conv-tool](real tool result)"
