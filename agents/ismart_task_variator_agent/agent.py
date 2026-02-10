from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer


from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agents import state
import config

from utils.utils import is_valid_json_string
from agents.tools.think import ThinkTool
from agents.tools.yandex_search import YandexSearchTool as SearchTool

from agents.utils import ModelType, extract_text, get_llm
from platform_utils.llm_logger import JSONFileTracer

from .state import VariatorAgentContext, VariatorAgentState


from .prompts import (
    GREETINGS_PROMPT_RU
    , SYSTEM_PROMPT_RU
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


def route(
    state: VariatorAgentState,
    config: RunnableConfig,
    runtime: Runtime[VariatorAgentContext]
) -> str:
    if runtime.context and runtime.context.get("mode"):
        if runtime.context["mode"] == "auto":
            state["phase"] = "run"
            state["mode"] = runtime.context["mode"]
            return state["phase"]


    if not state.get("greeted"):
        return "greetings"
    
    original_messages = state.get("messages", [])
    for msg in reversed(original_messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            content = extract_text(msg)
            if content and content.startswith("/"):
                state["options_cnt"] = content[1:].strip().isdigit() and int(content[1:].strip()) or 3
                state["phase"] = "cleanup"
            break


    return state.get("phase")

def create_greetings_node():

    def greetings_node(
        state: VariatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[VariatorAgentContext],
    ) -> VariatorAgentState:
        if not state.get("greeted"):
            greet = (
                GREETINGS_PROMPT_RU
            )
            state["messages"] = (state.get("messages") or []) + [AIMessage(content=greet)]
            state["greeted"] = True
        state["phase"] = "cleanup"
        return state

    return greetings_node


def cleanup_messages_node(
        state: VariatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[VariatorAgentContext],
    ) -> VariatorAgentState:
    
    all_msg_ids = [m.id for m in state["messages"][:-1]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [state["messages"][-1]],
        "phase" : "run"
    }

def _build_run_agent(model: BaseChatModel):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: VariatorAgentState = request.state
        options_cnt = state.get("options_cnt", 3)
        system_prompt = SYSTEM_PROMPT_RU.format(number_of_options=options_cnt)
        return system_prompt

    return create_agent(
        model=model,
        #tools=[_think_tool],
        middleware=[build_prompt],
        system_prompt=SYSTEM_PROMPT_RU,
        state_schema=VariatorAgentState,
        context_schema=VariatorAgentContext,
    )


def create_run_node(model: BaseChatModel):
    return _build_run_agent(model)


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "en",
    checkpoint_saver=None,
    *,
    streaming: bool = True,
):
    #set_locale(locale)
    #set_models_locale(locale)
    log_name = f"simple_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    #callback_handlers = [json_handler]
    callback_handlers = [StreamWriterCallbackHandler(), json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        _ = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        callback_handlers += [CallbackHandler()]

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    llm = get_llm(model="base", provider=provider.value, temperature=0.4, streaming=streaming)
    
    builder = StateGraph(VariatorAgentState)
    builder.add_node("greetings", create_greetings_node())
    builder.add_node("cleanup", cleanup_messages_node)
    builder.add_node("run", create_run_node(llm))

    builder.add_conditional_edges(
        START,
        route,
        {
            "greetings": "greetings",
            "cleanup": "cleanup",
            "run": "run"
        },
    )
    builder.add_edge("greetings", END)
    builder.add_edge("cleanup", "run")
    builder.add_edge("run", END)

    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph
