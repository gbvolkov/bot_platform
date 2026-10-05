from __future__ import annotations

import asyncio
import pytest

from langchain_core.messages import AIMessage, AIMessageChunk

from bot_service.schemas import MessagePayload
from bot_service.service import build_human_message, invoke_agent_stream


class _FakeAgent:
    async def astream(self, initial_state, config=None, stream_mode=None, subgraphs=None):
        yield ("messages", (AIMessageChunk(content=[{"type": "text", "text": "Alpha "}]), {}))
        yield ("messages", (AIMessageChunk(content=[{"type": "text", "text": "Beta"}]), {}))
        yield ("messages", (AIMessage(content="Alpha Beta"), {}))
        yield ("values", {"messages": [AIMessage(content="Alpha Beta")]})


def test_build_human_message_assigns_message_id() -> None:
    message = build_human_message(MessagePayload(type="text", text="test"))

    assert message.id
    assert message.id.startswith("human-")


def test_invoke_agent_stream_does_not_replay_final_ai_message() -> None:
    async def _run() -> tuple[list[dict], str]:
        events, result_future = await invoke_agent_stream(
            agent=_FakeAgent(),
            payload=MessagePayload(type="text", text="test"),
            conversation_id="conv-1",
            agent_id="gaz_agent",
            user_id="user-1",
            user_role="user",
        )
        seen: list[dict] = []
        async for event in events:
            seen.append(event)
        result = await result_future
        return seen, result["ai"].content

    events, result_text = asyncio.run(_run())

    assert events == [
        {"type": "chunk", "content": "Alpha "},
        {"type": "chunk", "content": "Beta"},
    ]
    assert result_text == "Alpha Beta"


def test_invalid_stream_result_reports_error_without_hanging(monkeypatch):
    from agent_runtime import invocation
    def invalid_result(**kwargs):
        raise RuntimeError("Invalid final graph state")
    monkeypatch.setattr(invocation, "_build_agent_result_from_state", invalid_result)
    class EmptyAgent:
        async def astream(self, *args, **kwargs):
            if False:
                yield
    async def check():
        events, result = await invoke_agent_stream(EmptyAgent(), MessagePayload(text="hi"),
            "conversation", "gaz_agent", "user", "default")
        async for _ in events:
            pass
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(result, 1)
    asyncio.run(check())


def test_child_values_cannot_replace_root_terminal_state():
    class NestedAgent:
        async def astream(self, *args, **kwargs):
            yield (), "values", {"messages": [AIMessage(content="Root answer")]}
            yield ("child:task",), "values", {"messages": [AIMessage(content="Child answer")]}

    async def check():
        events, result = await invoke_agent_stream(NestedAgent(), MessagePayload(text="hi"),
            "conversation", "simple_agent", "user", "default")
        async for _ in events:
            pass
        assert (await result)["ai"].content == "Root answer"

    asyncio.run(check())
