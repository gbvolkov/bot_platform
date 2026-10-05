"""All active descriptors run through the same adapter using deterministic graphs.

Business-graph tests remain separate; these fixtures exercise protocol behavior
without replaying live tools, models, or historical side effects.
"""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from agent_runtime.runtime import AgentRuntime
from platform_contracts import AgentContext, RunRequest
from platform_contracts.catalog import AgentCatalog
from platform_sdk.context import current_context

CATALOG = AgentCatalog.load("config_defaults/agents.json")
ACTIVE = [a.id for a in CATALOG.agents.values() if a.active]


class Graph:
    def __init__(self):
        self.calls = []
    async def ainvoke(self, value, config):
        self.calls.append((value, config))
        return {"messages": [AIMessage(content="Hello world")]}
    async def aget_state(self, config):
        return SimpleNamespace(config={}, values={})
    async def astream(self, value, config, stream_mode, subgraphs):
        self.calls.append((value, config))
        for text in ("Hello ", "world"):
            yield "messages", (AIMessageChunk(content=text), {})
        yield "values", {"messages": [AIMessage(content="Hello world")]}


@pytest.mark.parametrize("agent_id", ACTIVE)
def test_local_and_worker_adapters_match(agent_id):
    async def run():
        runtime = AgentRuntime()
        graph = Graph()
        runtime.registry._instances[agent_id] = graph
        descriptor = runtime.describe(agent_id)
        context = AgentContext("user", "role", "unchanged-thread-id", "run")
        request = RunRequest("text", {"text": "hi"}, descriptor.revision, 1)
        local = await runtime.handle(agent_id).invoke(request, context)
        emitted = []
        async def emit(value):
            emitted.append(value)
        status, hosted = await runtime.execute({"id": "run", "agent_id": agent_id,
            "conversation_id": context.conversation_id, "user_id": context.user_id,
            "user_role": context.user_role, "input": request.input}, emit)
        assert status == "completed"
        assert hosted["agent_message"] == local.output["agent_message"]
        assert "".join(e["content"] for e in emitted if e["type"] == "chunk") == "Hello world"
        assert all(call[1]["configurable"]["thread_id"] == "unchanged-thread-id" for call in graph.calls)
        await runtime.handle(agent_id).resume("interrupt-id", "approved", context)
        assert graph.calls[-1][0].resume == "approved"
        await runtime.handle(agent_id).reset(context)
        assert graph.calls[-1][0]["messages"][0].content[0]["type"] == "reset"
        await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
def test_local_handle_preserves_context_and_actual_checkpoint(streaming):
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt

    observed = []

    async def approve(state):
        observed.append(current_context())
        answer = interrupt({"question": "Approve?"})
        return {"messages": [AIMessage(content=answer)]}

    async def run():
        runtime = AgentRuntime(capabilities={"models": object()})
        builder = StateGraph(MessagesState)
        builder.add_node("approve", approve)
        builder.add_edge(START, "approve")
        builder.add_edge("approve", END)
        graph = builder.compile(checkpointer=MemorySaver())
        runtime.registry._instances["simple_agent"] = graph
        handle = runtime.handle("simple_agent")
        context = AgentContext("user", "role", "local-thread", "local-run", "parent-run")
        request = RunRequest("text", {"text": "hi"}, runtime.describe("simple_agent").revision, 1)
        if streaming:
            events = []
            async for event in handle.stream(request, context):
                assert current_context() is None
                events.append(event)
            output = events[-1].payload
            pending = output["agent_message"]["metadata"]["interrupt_payload"]
            checkpoint = output["checkpoint_reference"]
        else:
            result = await handle.invoke(request, context)
            pending, checkpoint = result.interrupt, result.checkpoint
        state = await graph.aget_state({"configurable": {"thread_id": context.conversation_id}})
        assert checkpoint == state.config["configurable"]["checkpoint_id"]
        assert checkpoint != context.conversation_id
        result = await handle.resume(pending["interrupt_id"], "approved", context)
        assert result.output["agent_message"]["raw_text"] == "approved"
        assert result.checkpoint != checkpoint
        assert current_context() is None
        assert all(item.run_id == context.run_id and item.parent_run_id == "parent-run" for item in observed)
        assert all(item.trace["agent_id"] == "simple_agent" for item in observed)
        assert all(item.capabilities["models"] is runtime.capabilities["models"] for item in observed)
        await runtime.close()

    asyncio.run(run())
