from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from platform_guardrails.middleware import PrivacyModelRequestMiddleware
from platform_guardrails import privacy
from platform_guardrails.privacy import PalimpsestSessionManager, PrivacyRail


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


def _middleware(log_path=None, *, guard_tool_calls: bool = True):
    processor = FakeProcessor()
    rail = PrivacyRail(session_manager=PalimpsestSessionManager(processor))
    middleware = PrivacyModelRequestMiddleware(
        rail,
        agent_name="artifact_creator_agent.run",
        event_log_path=str(log_path) if log_path else None,
        guard_tool_calls=guard_tool_calls,
    )
    return middleware, processor


def _runtime(configurable=None):
    return SimpleNamespace(
        execution_info=None,
        config={"configurable": configurable or {}},
    )


def test_model_request_anonymizes_messages_system_and_deanonymizes_response():
    middleware, processor = _middleware()
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
        }
    )
    request = ModelRequest(
        model=object(),
        system_prompt="system prompt",
        messages=[HumanMessage(content=[{"type": "text", "text": "hello"}])],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=runtime,
    )
    captured = {}

    def handler(updated_request):
        captured["messages"] = updated_request.messages
        captured["system_prompt"] = updated_request.system_prompt
        return ModelResponse(result=[AIMessage(content="fake answer")])

    result = middleware.wrap_model_call(request, handler)

    scope = "tenant|user|thread"
    assert captured["messages"][0].content[0]["text"] == f"anon[{scope}](hello)"
    assert captured["system_prompt"] == f"anon[{scope}](system prompt)"
    assert result.result[0].content == f"deanon[{scope}](fake answer)"
    assert [session.session_id for session in processor.sessions] == [scope]


def test_model_request_anonymizes_tool_results_for_llm_context():
    middleware, _processor = _middleware(guard_tool_calls=False)
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
        }
    )
    tool_message = ToolMessage(
        content="raw tool result",
        tool_call_id="call-1",
    )
    request = ModelRequest(
        model=object(),
        system_prompt=None,
        messages=[
            HumanMessage(content="hello"),
            tool_message,
        ],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=runtime,
    )
    captured = {}

    def handler(updated_request):
        captured["messages"] = updated_request.messages
        return ModelResponse(result=[AIMessage(content="fake answer")])

    middleware.wrap_model_call(request, handler)

    scope = "tenant|user|thread"
    assert captured["messages"][0].content == f"anon[{scope}](hello)"
    assert captured["messages"][1].content == f"anon[{scope}](raw tool result)"


def test_model_response_deanonymizes_tool_call_arguments():
    middleware, _processor = _middleware()
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
        }
    )
    request = ModelRequest(
        model=object(),
        system_prompt=None,
        messages=[HumanMessage(content="hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=runtime,
    )

    def handler(_updated_request):
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lookup",
                            "args": {"query": "anon[tenant|user|thread](Ivan)"},
                            "id": "call-1",
                        }
                    ],
                )
            ]
        )

    result = middleware.wrap_model_call(request, handler)

    assert result.result[0].tool_calls[0]["args"] == {
        "query": "deanon[tenant|user|thread](anon[tenant|user|thread](Ivan))"
    }


def test_model_response_deanonymization_can_be_disabled_by_context_policy():
    middleware, _processor = _middleware()
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
            "allow_deanonymization": False,
        }
    )
    request = ModelRequest(
        model=object(),
        system_prompt=None,
        messages=[HumanMessage(content="hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=runtime,
    )

    def handler(_updated_request):
        return ModelResponse(result=[AIMessage(content="fake answer")])

    result = middleware.wrap_model_call(request, handler)

    assert result.result[0].content == "fake answer"


def test_async_tool_wrapper_deanonymizes_args_and_returns_raw_tool_message_result():
    middleware, _processor = _middleware()
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
        }
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

    scope = "tenant|user|thread"
    assert captured["args"] == {"query": f"deanon[{scope}](fake user)"}
    assert result.content == "real tool result"


def test_tool_wrapper_deanonymizes_args_and_returns_raw_command_message_updates():
    middleware, _processor = _middleware()
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
        }
    )
    request = ToolCallRequest(
        tool_call={"name": "commit", "args": {"final_text": "fake"}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=runtime,
    )

    def handler(_updated_request):
        return Command(
            update={
                "messages": [ToolMessage(content="Success for Ivan", tool_call_id="call-1")],
                "artifacts": {0: {"artifact_final_text": "Success for Ivan"}},
            }
        )

    result = middleware.wrap_tool_call(request, handler)

    scope = "tenant|user|thread"
    assert result.update["messages"][0].content == "Success for Ivan"
    assert result.update["artifacts"][0]["artifact_final_text"] == "Success for Ivan"


def test_tool_wrapper_can_defer_tool_guarding_to_execution_middleware():
    middleware, _processor = _middleware(guard_tool_calls=False)
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
        }
    )
    request = ToolCallRequest(
        tool_call={"name": "lookup", "args": {"query": "fake user"}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=runtime,
    )
    captured = {}

    def handler(updated_request):
        captured["args"] = updated_request.tool_call["args"]
        return ToolMessage(content="real tool result", tool_call_id="call-1")

    result = middleware.wrap_tool_call(request, handler)

    assert captured["args"] == {"query": "fake user"}
    assert result.content == "real tool result"


def test_middleware_resets_context_session_on_reset_message():
    middleware, processor = _middleware()
    scope = "tenant|user|thread"
    session = processor.create_session(scope)
    middleware._privacy.sessions._sessions[scope] = session
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
        }
    )
    state = {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}

    middleware.before_agent(state, runtime)

    assert session.reset_count == 1


def test_guardrail_event_log_does_not_include_raw_transformed_text(tmp_path):
    log_path = tmp_path / "guardrails.jsonl"
    middleware, _processor = _middleware(log_path)
    runtime = _runtime(
        {
            "tenant_id": "tenant",
            "user_id": "user",
            "thread_id": "thread",
        }
    )
    request = ModelRequest(
        model=object(),
        system_prompt=None,
        messages=[HumanMessage(content="hello raw pii")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=runtime,
    )

    def handler(_updated_request):
        return ModelResponse(result=[AIMessage(content="fake answer raw pii")])

    middleware.wrap_model_call(request, handler)

    text = log_path.read_text(encoding="utf-8")
    assert "hello raw pii" not in text
    assert "fake answer raw pii" not in text
    assert "guardrail_decision" in text


def test_palimpsest_preflight_reports_missing_ru_spacy_model(monkeypatch):
    monkeypatch.setattr(privacy, "find_spec", lambda name: None)

    try:
        PrivacyRail.from_palimpsest(locale="ru-RU")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing spaCy model preflight to fail")

    assert "ru_core_news_sm" in message
    assert "uv sync" in message


def test_palimpsest_entity_table_drives_run_entities_and_session_kwargs(monkeypatch):
    captured: dict[str, object] = {}

    class FakePalimpsest:
        def __init__(
            self,
            *,
            verbose=False,
            run_entities=None,
            locale="ru-RU",
            entity_table=None,
            typed_placeholders=None,
            placeholder_mode=None,
        ):
            captured["constructor"] = {
                "verbose": verbose,
                "run_entities": run_entities,
                "locale": locale,
                "entity_table": entity_table,
                "typed_placeholders": typed_placeholders,
                "placeholder_mode": placeholder_mode,
            }

        def create_session(
            self,
            *,
            session_id=None,
            entity_table=None,
            typed_placeholders=None,
            placeholder_style=None,
        ):
            captured["session"] = {
                "session_id": session_id,
                "entity_table": entity_table,
                "typed_placeholders": typed_placeholders,
                "placeholder_style": placeholder_style,
            }
            return FakeSession(session_id or "missing")

    entity_table = {
        "RU_PERSON": {"placeholder": "PERSON"},
        "URL": {"placeholder": "URL", "enabled": False},
        "PHONE_NUMBER": "PHONE",
    }
    monkeypatch.setitem(
        sys.modules,
        "palimpsest",
        SimpleNamespace(Palimpsest=FakePalimpsest),
    )

    rail = PrivacyRail.from_palimpsest(
        locale="en",
        entity_table=entity_table,
        typed_placeholders=True,
        palimpsest_options={"placeholder_mode": "typed"},
        palimpsest_session_options={"placeholder_style": "typed"},
    )
    rail.sessions.get_session("thread-1")

    assert captured["constructor"] == {
        "verbose": False,
        "run_entities": ["RU_PERSON", "PHONE_NUMBER"],
        "locale": "en",
        "entity_table": entity_table,
        "typed_placeholders": True,
        "placeholder_mode": "typed",
    }
    assert captured["session"] == {
        "session_id": "thread-1",
        "entity_table": entity_table,
        "typed_placeholders": True,
        "placeholder_style": "typed",
    }


def test_palimpsest_new_options_require_new_palimpsest_api(monkeypatch):
    captured: dict[str, object] = {}

    class FakeLegacyPalimpsest:
        def __init__(self, *, verbose=False, run_entities=None, locale="ru-RU"):
            captured["constructor"] = {
                "verbose": verbose,
                "run_entities": run_entities,
                "locale": locale,
            }

        def create_session(self, session_id=None):
            captured["session_id"] = session_id
            return FakeSession(session_id or "missing")

    monkeypatch.setitem(
        sys.modules,
        "palimpsest",
        SimpleNamespace(Palimpsest=FakeLegacyPalimpsest),
    )

    try:
        PrivacyRail.from_palimpsest(
            locale="en",
            entity_table={"RU_PERSON": {"placeholder": "PERSON"}},
            typed_placeholders=True,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected legacy Palimpsest API to be rejected.")

    assert captured == {}
    assert "Palimpsest" in message
    assert "entity_table" in message
    assert "does not fall back" in message
