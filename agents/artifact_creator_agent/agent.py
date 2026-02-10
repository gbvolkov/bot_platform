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
from langgraph.types import Command, interrupt


from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agents import state
import config

from utils.utils import is_valid_json_string

from agents.tools.think import ThinkTool
from agents.tools.yandex_search import YandexSearchTool as SearchTool
from agents.structured_prompt_utils import provider_then_tool
from agents.utils import (
    ModelType,
    extract_text,
    build_internal_invoke_config,
    get_llm,
)

from platform_utils.llm_logger import JSONFileTracer

from .state import (
    ArtifactCreatorAgentContext
    , ArtifactCreatorAgentState
    , ConfirmationAgentState
)
from .utils import UserConfirmation
from .tools import commit_artifact_final_text
from ..store_artifacts import store_artifacts

from .prompts import (
    ARTIFACT_STORE_ERROR_RU,
    GREETINGS_WITH_PROMPT_RU
    , GREETINGS_WITHOUT_PROMPT_RU
    , SET_PROPMT_REQUEST_RU
    , PROMPT_CONFIRMATION_RU
    , DEFAULT_SYSTEM_PROMPT_RU
    , СOMMIT_TOOL_PROMPT_RU
    , ARTIFACT_URL_PROMPT_RU
)

LOG = logging.getLogger(__name__)
_CONFIRMATION_STREAM_TAG = "internal_confirmation_agent"

def route(state: ArtifactCreatorAgentState) -> str:
    if not state.get("greeted"):
        return "greetings"

    if not state.get("phase") and not state.get("system_prompt"):
        return "set_prompt"
    
    return state.get("phase")

def create_greetings_node():

    def greetings_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
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
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
        #state["phase"] = "run"
        
        original_messages = state.get("messages", [])

        last_user_msg = None
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
            if isinstance(last_user_msg.content, str):
                state["system_prompt"] = last_user_msg.content
            else:
                state["system_prompt"] = extract_text(last_user_msg)
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
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
    
    all_msg_ids = [m.id for m in state["messages"][:-1]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [state["messages"][-1]],
        "phase" : "run"
    }

def _build_run_agent(model: BaseChatModel):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: ArtifactCreatorAgentState = request.state
        system_prompt = state.get("system_prompt")
        if system_prompt:
            if not isinstance(system_prompt, str):
                try:
                    system_prompt = extract_text(HumanMessage(content=system_prompt))
                except Exception:
                    system_prompt = str(system_prompt)
            return f"{system_prompt}\n\n{СOMMIT_TOOL_PROMPT_RU}"
        return DEFAULT_SYSTEM_PROMPT_RU

    return create_agent(
        model=model,
        tools=[commit_artifact_final_text],
        middleware=[build_prompt],
        state_schema=ArtifactCreatorAgentState,
        context_schema=ArtifactCreatorAgentContext,
    )
    
#def create_run_node(model: BaseChatModel):
#    _run_agent = _build_run_agent(model)
#    def run_node(
#        state: ArtifactCreatorAgentState,
#        config: RunnableConfig,
#        runtime: Runtime[ArtifactCreatorAgentContext],
#    ) -> ArtifactCreatorAgentState:
#        
#        result_state = _run_agent.invoke(state, config=config, context=runtime.context)
#        return result_state
#
#    return run_node


def _build_confirmation_agent(model: BaseChatModel):

    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: ConfirmationAgentState  = request.state
        artifact = state.get("artifact")
        if not isinstance(artifact, str):
            artifact = str(artifact or "")

        #response = state.get("last_user_answer")
        prompt = (
            "You have to analyze user's response to artifact.\n" 
            "=======================================\n"
            f"ARTIFACT TEXT:\n{artifact}\n"
            "END OF ARTIFACT TEXT\n"
            "=======================================\n"
            "Determine if user confirmed the text or requested change."        
            #f"User response is: {response}\n"
        )
        return prompt


    confirmation_agent = create_agent(
        model=model,
        middleware=[build_prompt, provider_then_tool],
        response_format=UserConfirmation,
        state_schema=ConfirmationAgentState,
    )
    return confirmation_agent #.with_config({"tags": [_CONFIRMATION_STREAM_TAG]})

def create_confirmation_node(model: BaseChatModel):
    _confirmation_agent = _build_confirmation_agent(model)
    def confirmation_node(
        state: ArtifactCreatorAgentState, 
        config: RunnableConfig, 
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
        last_message = state["messages"][-1]
        artifact_id = state.get("artifact_id", 0)
        artifacts = state.get("artifacts")
        artifact = ""
        if artifacts and artifact_id in artifacts:
            artifact = artifacts[artifact_id].get("artifact_final_text", "")

        confirmation_state = {
            "messages": [last_message],  # ensure no chat history is provided
            "artifact": artifact,
        }
        confirm_config = build_internal_invoke_config(
            config,
            extra_tags=[_CONFIRMATION_STREAM_TAG],
        )

        result = _confirmation_agent.invoke(confirmation_state, config=confirm_config, context=runtime.context)
        structured = result.get("structured_response") or {}
        state["is_artifact_confirmed"] = structured.is_artifact_confirmed
        if structured.is_artifact_confirmed:
            try:
                store_url = store_artifacts(state["artifacts"] or [])
                link_text = f"[{ARTIFACT_URL_PROMPT_RU}]({store_url})"
                state["final_artifact_url"] = store_url
            except Exception as e:
                logging.error(f"Error occured at store_artifacts.\nException: {e}")
            #state["messages"].append(AIMessage(content=f"\n***Работа завершена***.\nСогласованный текст артефакта:\n\n{artifact}\n\n{link_text}\n"))
            #state["messages"].append(AIMessage(content=f"\n\n***Работа завершена***.\nСогласованный текст артефакта:\n\n{artifact}\n\n{link_text}\n"))
            
            state["phase"] = "ready"
        return state

    return confirmation_node

def final_print_node(
        state: ArtifactCreatorAgentState, 
        config: RunnableConfig, 
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
    store_url = state["final_artifact_url"]
    if store_url:            
        link_text = f"[{ARTIFACT_URL_PROMPT_RU}]({store_url})"
    else:
        link_text = ARTIFACT_STORE_ERROR_RU
        
    message_update = [AIMessage(content=f"\n\n***Работа завершена***.\n\n{link_text}\n")]    
    return Command(
        update={
            "messages": message_update,
            "phase": "ready",
        },
    )


def is_confirmed(
    state: ArtifactCreatorAgentState, 
) -> ArtifactCreatorAgentState:

    if not state["is_artifact_confirmed"]:
        return "run"
    else:
        #artifact_id = state.get("artifact_id", 0)
        #artifacts = state.get("artifacts")
        #artifact = ""
        #if artifacts and artifact_id in artifacts:
        #    artifact = artifacts[artifact_id].get("artifact_final_text", "")

        #state["messages"].append(AIMessage(content=f"\nРабота завершена.\nСогласованный текст артефакта:\n\n{artifact}"))
        return "ready"


def ready_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
    artifact_id = state.get("artifact_id", 0)
    artifacts = state.get("artifacts")
    artifact = ""
    if artifacts and artifact_id in artifacts:
        artifact = artifacts[artifact_id].get("artifact_final_text", "")

    store_url = state["final_artifact_url"]
    if store_url:            
        link_text = f"[{ARTIFACT_URL_PROMPT_RU}]({store_url})"
    else:
        link_text = ARTIFACT_STORE_ERROR_RU
        
    message_update = [AIMessage(content=f"\n\n***Работа завершена***.\nСогласованный текст артефакта:\n\n{artifact}\n\n{link_text}\n")]    
    return Command(
        update={
            "messages": message_update,
            "phase": "ready",
        },
    )
    

def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "en",
    checkpoint_saver=None,
):
    #set_locale(locale)
    #set_models_locale(locale)
    log_name = f"artifact_creator_agent_{time.strftime('%Y%m%d%H%M')}"
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

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    llm = get_llm(model="base", provider=provider.value, temperature=0.4)
    response_analyser_llm = get_llm(model="nano", provider=provider.value, temperature=0, streaming=False)
    
    builder = StateGraph(ArtifactCreatorAgentState)
    builder.add_node("greetings", create_greetings_node())
    builder.add_node("set_prompt", create_set_prompt_node(llm))
    builder.add_node("cleanup", cleanup_messages_node)
    builder.add_node("run", _build_run_agent(llm))
    builder.add_node("confirm", create_confirmation_node(response_analyser_llm))
    builder.add_node("final_print", final_print_node)
    builder.add_node("ready", ready_node)

    builder.add_conditional_edges(
        START,
        route,
        {
            "greetings": "greetings",
            "set_prompt": "set_prompt",
            "cleanup": "cleanup",
            "run": "run",
            "confirm": "confirm",
            "ready": "ready"
        },
    )

    builder.add_edge("greetings", END)
    builder.add_edge("set_prompt", END)
    builder.add_edge("cleanup", "run")
    builder.add_conditional_edges(
        "run",
        route,
        {
            "confirm": "confirm",
            "run": END,
            "ready": END,
            "greetings": END,
            "set_prompt": END,
            "cleanup": END,
        },
    )
    builder.add_edge("confirm", "final_print")
    builder.add_conditional_edges(
        "final_print",
        is_confirmed,
        {
            "run": "run",
            "ready": END
        },
    )
    builder.add_edge("ready", END)
    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph
