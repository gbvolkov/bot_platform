from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from agents.mycroft_agent.delegate_middleware import DelegateSubAgentMiddleware


class _StatefulFakeAgent:
    def __init__(self, name: str):
        self.name = name
        self.history_by_thread: dict[str, list[str]] = {}

    async def ainvoke(self, state, config):
        thread_id = config["configurable"]["thread_id"]
        message = state["messages"][-1]
        content = message.content if isinstance(message.content, str) else str(message.content)
        history = self.history_by_thread.setdefault(thread_id, [])
        history.append(content)
        return {
            "messages": [
                AIMessage(
                    content=f"{self.name} turn {len(history)}: {' | '.join(history)}"
                )
            ]
        }


def _runtime(thread_id: str, tool_call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        state={"messages": []},
        config={"configurable": {"thread_id": thread_id}},
        tool_call_id=tool_call_id,
    )


def test_delegate_tool_returns_subagent_answer_and_keeps_history_per_thread():
    ingolab = _StatefulFakeAgent("product_Инголаб")
    middleware = DelegateSubAgentMiddleware(
        default_model="openai:gpt-4o-mini",
        subagents=[
            {
                "name": "product_Инголаб",
                "description": "Инголаб expert.",
                "runnable": ingolab,
            }
        ],
        system_prompt="delegate system",
        task_description="delegate description",
        general_purpose_agent=False,
    )

    delegate_tool = middleware.tools[0]
    first = asyncio.run(
        delegate_tool.coroutine(
            description="Первый вопрос",
            subagent_type="product_Инголаб",
            runtime=_runtime("thread-1", "tc-1"),
        )
    )
    second = asyncio.run(
        delegate_tool.coroutine(
            description="Второй вопрос",
            subagent_type="product_Инголаб",
            runtime=_runtime("thread-1", "tc-2"),
        )
    )

    assert isinstance(first, Command)
    assert isinstance(second, Command)
    assert isinstance(first.update["messages"][0], ToolMessage)
    assert first.update["messages"][0].content == "product_Инголаб turn 1: Первый вопрос"
    assert second.update["messages"][0].content == (
        "product_Инголаб turn 2: Первый вопрос | Второй вопрос"
    )


def test_delegate_tool_does_not_mix_state_between_different_stateful_agents():
    ingolab = _StatefulFakeAgent("product_Инголаб")
    legal = _StatefulFakeAgent("product_Юридическая помощь")
    middleware = DelegateSubAgentMiddleware(
        default_model="openai:gpt-4o-mini",
        subagents=[
            {
                "name": "product_Инголаб",
                "description": "Инголаб expert.",
                "runnable": ingolab,
            },
            {
                "name": "product_Юридическая помощь",
                "description": "Legal help expert.",
                "runnable": legal,
            },
        ],
        system_prompt="delegate system",
        task_description="delegate description",
        general_purpose_agent=False,
    )

    delegate_tool = middleware.tools[0]
    ingolab_result = asyncio.run(
        delegate_tool.coroutine(
            description="Вопрос по Инголаб",
            subagent_type="product_Инголаб",
            runtime=_runtime("thread-shared", "tc-3"),
        )
    )
    legal_result = asyncio.run(
        delegate_tool.coroutine(
            description="Вопрос по юрпомощи",
            subagent_type="product_Юридическая помощь",
            runtime=_runtime("thread-shared", "tc-4"),
        )
    )

    assert isinstance(ingolab_result, Command)
    assert isinstance(legal_result, Command)
    assert ingolab_result.update["messages"][0].content == (
        "product_Инголаб turn 1: Вопрос по Инголаб"
    )
    assert legal_result.update["messages"][0].content == (
        "product_Юридическая помощь turn 1: Вопрос по юрпомощи"
    )
