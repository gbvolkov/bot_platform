from __future__ import annotations

import logging
import time
from typing import Any, List

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer


from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config

from agents.utils import ModelType, get_llm
from platform_guardrails.graph_compiler import PlatformGraphCompiler
from platform_guardrails.graph_spec import NodeGuardrailPolicy, PlatformStateGraph
from platform_guardrails.runtime import PlatformGuardrailRuntime
from platform_utils.llm_logger import JSONFileTracer

from .state import SimpleAgentContext, SimpleAgentState


from .prompts import (
    GREETINGS_WITH_PROMPT_RU
    , GREETINGS_WITHOUT_PROMPT_RU
    , SET_PROPMT_REQUEST_RU
    , PROMPT_CONFIRMATION_RU
    , DEFAULT_SYSTEM_PROMPT_RU
)

LOG = logging.getLogger(__name__)

def _safe_stream_writer():
    """Return a writer suitable for `stream_mode="custom"`; otherwise no-op."""
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None

class StreamWriterCallbackHandler(BaseCallbackHandler):
    """Forward LangChain callbacks (tool/chain lifecycle) into LangGraph custom stream."""

    def on_tool_start(self, serialized, input_str=None, **kwargs):
        writer = _safe_stream_writer()
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "tool"
        writer({"type": "tool_start", "name": name})

    def on_tool_end(self, output=None, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "tool_end"})

    def on_tool_error(self, error, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "tool_error", "error": str(error)})

    def on_chain_start(self, serialized, inputs, **kwargs):
        writer = _safe_stream_writer()
        name = (serialized or {}).get("name") or "chain"
        writer({"type": "chain_start", "name": name})

    def on_chain_end(self, outputs, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "chain_end"})


def route(state: SimpleAgentState) -> str:
    if not state.get("greeted"):
        return "greetings"

    if not state.get("phase") and not state.get("system_prompt"):
        return "set_prompt"
    
    return state.get("phase")

def create_greetings_node():

    def greetings_node(
        state: SimpleAgentState,
        config: RunnableConfig,
        runtime: Runtime[SimpleAgentContext],
    ) -> SimpleAgentState:
        system_prompt = None
        if runtime.context and runtime.context.get("system_prompt"):
            system_prompt = runtime.context["system_prompt"]
            state["system_prompt"] = system_prompt
            state["phase"] = "cleanup"

        if not state.get("greeted"):
            greet = (
                GREETINGS_WITH_PROMPT_RU
                if state.get("system_prompt")
                else GREETINGS_WITHOUT_PROMPT_RU
            )
            state["messages"] = (state.get("messages") or []) + [AIMessage(content=greet)]
            state["greeted"] = True
        return state

    return greetings_node


def create_set_prompt_node(model: BaseChatModel):

    def set_prompt_node(
        state: SimpleAgentState,
        config: RunnableConfig,
        runtime: Runtime[SimpleAgentContext],
    ) -> SimpleAgentState:
        #state["phase"] = "run"
        
        original_messages = state.get("messages", [])

        for msg in reversed(original_messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                last_user_msg = msg
                break
        #system_prompt = None
        if last_user_msg is None:
            response_text = SET_PROPMT_REQUEST_RU
            state["phase"] = "set_prompt"
            return state
        else:
            response_text = PROMPT_CONFIRMATION_RU
            #all_msg_ids = [m.id for m in state["messages"]]
            state["system_prompt"] = last_user_msg.content
            state["phase"] = "cleanup"
            # Returning RemoveMessage instances instructs the reducer to delete them
            #return {
            #    "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [AIMessage(content=response_text)],
            #    "system_prompt": system_prompt,
            #    "phase" : "cleanup"
            #}            
        state["messages"].append(AIMessage(content=response_text))
        return state
            

    return set_prompt_node


def cleanup_messages_node(
        state: SimpleAgentState,
        config: RunnableConfig,
        runtime: Runtime[SimpleAgentContext],
    ) -> SimpleAgentState:
    
    all_msg_ids = [m.id for m in state["messages"][:-1]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [state["messages"][-1]],
        "phase" : "run"
    }

def _build_run_agent(model: BaseChatModel, tools: List[Any] | None = None):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: SimpleAgentState = request.state
        system_prompt = state.get("system_prompt")
        return system_prompt or DEFAULT_SYSTEM_PROMPT_RU

    return create_agent(
        model=model,
        tools=tools or [],
        middleware=[build_prompt],
        state_schema=SimpleAgentState,
        context_schema=SimpleAgentContext,
    )


def create_run_node(model: BaseChatModel):
    def run_node(
        state: SimpleAgentState,
        config: RunnableConfig,
        runtime: Runtime[SimpleAgentContext],
    ) -> SimpleAgentState:
        tools = []
        if runtime.context:
            tools = runtime.context.get("tools") or []

        run_agent = _build_run_agent(model, tools)
        return run_agent.invoke(state, config=config, context=runtime.context)

    return run_node


def _build_callback_handlers(log_name: str) -> list[Any]:
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers: list[Any] = [StreamWriterCallbackHandler(), json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        _ = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        callback_handlers += [CallbackHandler()]
    return callback_handlers


def _build_dynamic_prompt():
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: SimpleAgentState = request.state
        system_prompt = state.get("system_prompt")
        return system_prompt or DEFAULT_SYSTEM_PROMPT_RU

    return build_prompt


def build_agent_graph(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "en",
    *,
    streaming: bool = True,
    tools: List[Any] | None = None,
):
    #set_locale(locale)
    #set_models_locale(locale)
    log_name = f"simple_agent_{time.strftime('%Y%m%d%H%M')}"
    callback_handlers = _build_callback_handlers(log_name)
    llm = get_llm(model="base", provider=provider.value, temperature=0.4, streaming=streaming)

    builder = PlatformStateGraph(SimpleAgentState)
    builder.add_node("greetings", create_greetings_node(), guardrails=False)
    builder.add_node(
        "set_prompt",
        create_set_prompt_node(llm),
        guardrails=NodeGuardrailPolicy(
            composite_message_roles=("human", "tool"),
        ),
    )
    builder.add_node("cleanup", cleanup_messages_node, guardrails=False)
    builder.add_agent_node(
        "run",
        model=llm,
        prompt=_build_dynamic_prompt(),
        tools_source="platform",
        tools=tools or [],
        state_schema=SimpleAgentState,
        context_schema=SimpleAgentContext,
    )

    builder.add_conditional_edges(
        START,
        route,
        {
            "greetings": "greetings",
            "set_prompt": "set_prompt",
            "cleanup": "cleanup",
            "run": "run"
        },
    )
    builder.add_edge("greetings", END)
    builder.add_edge("set_prompt", END)
    builder.add_edge("cleanup", "run")
    builder.add_edge("run", END)

    return builder.to_spec(callbacks=callback_handlers, compile_options={"debug": False})


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "en",
    checkpoint_saver=None,
    *,
    streaming: bool = True,
    tools: List[Any] | None = None,
):
    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    spec = build_agent_graph(
        provider=provider,
        use_platform_store=use_platform_store,
        locale=locale,
        streaming=streaming,
        tools=tools,
    )
    return PlatformGraphCompiler().compile(
        spec,
        guardrail_runtime=PlatformGuardrailRuntime.disabled(agent_id="simple_agent"),
        checkpointer=memory,
        tools=tools or [],
        tool_profiles={},
    )
