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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime


from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agents import state
import config

from utils.utils import is_valid_json_string
from agents.tools.think import ThinkTool
from agents.tools.yandex_search import YandexSearchTool as SearchTool

from agents.utils import ModelType, get_llm, _extract_text
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

def _build_run_agent(model: BaseChatModel):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: SimpleAgentState = request.state
        system_prompt = state.get("system_prompt")
        return system_prompt or DEFAULT_SYSTEM_PROMPT_RU

    return create_agent(
        model=model,
        #tools=[_think_tool],
        middleware=[build_prompt],
        state_schema=SimpleAgentState,
        context_schema=SimpleAgentContext,
    )


def create_run_node(model: BaseChatModel):
    _run_agent = _build_run_agent(model)
    def run_node(
        state: SimpleAgentState,
        config: RunnableConfig,
        runtime: Runtime[SimpleAgentContext],
    ) -> SimpleAgentState:
        
        result = _run_agent.invoke(state, config=config, context=runtime.context)
        return result

    return run_node


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "en",
):
    #set_locale(locale)
    #set_models_locale(locale)
    log_name = f"simple_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        lf_handler = CallbackHandler()
        callback_handlers += [lf_handler]

    memory = None if use_platform_store else MemorySaver()
    llm = get_llm(model="base", provider=provider.value, temperature=0.4)
    
    builder = StateGraph(SimpleAgentState)
    builder.add_node("greetings", create_greetings_node())
    builder.add_node("set_prompt", create_set_prompt_node(llm))
    builder.add_node("cleanup", cleanup_messages_node)
    builder.add_node("run", create_run_node(llm))

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

    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph
