from __future__ import annotations

import asyncio

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
