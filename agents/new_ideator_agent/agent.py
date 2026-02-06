from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest, 
    dynamic_prompt, 
    SummarizationMiddleware, 
)
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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

from agents.utils import ModelType, get_llm, _extract_text
from platform_utils.llm_logger import JSONFileTracer

from .state import IdeatorAgentContext, IdeatorAgentState
from .tools import (
    commit_thematic_threads,
    commit_ideas,
    commit_final_docset
)


from .prompts import get_locale

LOG = logging.getLogger(__name__)

_LOCALE: Dict[str, Any] = {}
_AGENT_TEXT: Dict[str, str] = {}
_PROMPTS: Dict[str, str] = {}
_CURRENT_LOCALE = "ru"


def set_locale(locale: str = "ru") -> None:
    global _LOCALE, _AGENT_TEXT, _PROMPTS, _CURRENT_LOCALE
    _LOCALE = get_locale(locale)
    _AGENT_TEXT = _LOCALE["agent"]
    _PROMPTS = _LOCALE["prompts"]
    _CURRENT_LOCALE = locale


set_locale()


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


def route(state: IdeatorAgentState) -> str:
    if not state.get("greeted"):
        return "greetings"

    if not state.get("phase") and not state.get("scout_report"):
        return "set_report"
    
    return state.get("phase")

def create_greetings_node():

    def greetings_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        scout_report = None
        if runtime.context and runtime.context.get("scout_report"):
            scout_report = runtime.context["scout_report"]
            state["scout_report"] = scout_report
            state["phase"] = "run"

        if not state.get("greeted"):
            greet = (
                _AGENT_TEXT["greeting"]
            )
            state["messages"] = (state.get("messages") or []) + [AIMessage(content=greet)]
            state["greeted"] = True
        return state

    return greetings_node


def create_set_report_node(model: BaseChatModel):

    def set_report_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        #state["phase"] = "run"
        
        original_messages = state.get("messages", [])

        for msg in reversed(original_messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                last_user_msg = msg
                break
        #system_prompt = None
        if last_user_msg is None:
            response_text = _AGENT_TEXT["set_report_request"]
            state["phase"] = "set_report"
            return state
        else:
            response_text = _AGENT_TEXT["report_confirmation"]
            #all_msg_ids = [m.id for m in state["messages"]]
            state["set_report"] = last_user_msg.content
            state["phase"] = "run"
            # Returning RemoveMessage instances instructs the reducer to delete them
            #return {
            #    "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [AIMessage(content=response_text)],
            #    "system_prompt": system_prompt,
            #    "phase" : "cleanup"
            #}            
        state["messages"].append(AIMessage(content=response_text))
        return state
            

    return set_report_node


def cleanup_messages_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
    
    all_msg_ids = [m.id for m in state["messages"][:-1]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [state["messages"][-1]],
        "phase" : "run"
    }


def _build_run_agent(model: BaseChatModel, summarization_model: BaseChatModel = None):
    summarization_model = summarization_model or model
    return create_agent(
        model=model,
        tools=[commit_thematic_threads, commit_ideas, commit_final_docset],
        system_prompt=_PROMPTS["ideator_prompt"],
        middleware=[SummarizationMiddleware(
            model=summarization_model,
            trigger=("tokens", 80000),
            keep=("messages", 20),
            summary_prompt=_PROMPTS["summary_prompt"],
        )],
        state_schema=IdeatorAgentState,
        context_schema=IdeatorAgentContext,
    )

def create_run_node(model: BaseChatModel, summarization_model: BaseChatModel = None):
    _run_agent = _build_run_agent(model, summarization_model)
    def run_node(
        state: IdeatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[IdeatorAgentContext],
    ) -> IdeatorAgentState:
        
        result = _run_agent.invoke(state, config=config, context=runtime.context)
        return result

    return _run_agent


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "ru",
    checkpoint_saver=None,
    *,
    streaming: bool = True,
):
    set_locale(locale)
    #set_models_locale(locale)
    log_name = f"ideator_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
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
    summarization_llm = get_llm(model="mini", provider=provider.value, temperature=0.0)
    
    builder = StateGraph(IdeatorAgentState)
    builder.add_node("greetings", create_greetings_node())
    builder.add_node("set_report", create_set_report_node(llm))
    builder.add_node("run", create_run_node(llm, summarization_llm))

    builder.add_conditional_edges(
        START,
        route,
        {
            "greetings": "greetings",
            "set_report": "set_report",
            "run": "run"
        },
    )
    builder.add_edge("greetings", END)
    builder.add_edge("set_report", END)
    builder.add_edge("run", END)

    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph
