from __future__ import annotations

import logging
import time
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langgraph.runtime import Runtime

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config

from agents.utils import ModelType, get_llm
from platform_utils.llm_logger import JSONFileTracer

from .artifacts_defs import (
    ARTIFACTS, 
    #ArtifactDetails,
    #ArtifactOptions,
    ArtifactState,
    ArtifactAgentState,
    ArtifactAgentContext,
    #AftifactFinalText,
)
from .choice_agent import initialize_agent as build_choice_agent

def init_node(state: ArtifactAgentState, 
              config: RunnableConfig) -> ArtifactAgentState:
    if not state["messages"]:
        return state          
    last_user_msg = state["messages"][-1]
    # We only augment on real user turns
    if last_user_msg.type != "human":
        return state

    state["user_prompt"] = last_user_msg.content
    state["current_artifact_state"] = ArtifactState.INIT
    state["current_artifact_id"] = 0
    #state["user_info"] = config.
    return state


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
):
    log_name = f"theo_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL
        )
        lf_handler = CallbackHandler()
        callback_handlers += [lf_handler]

    memory = MemorySaver() # Always use memory for this agent

    builder = StateGraph(ArtifactAgentState)

    builder.add_node("init", init_node)
    prev_node = "init"

    builder.add_edge(START, "init")
    for artifact in ARTIFACTS:
        _choice_agent = build_choice_agent(provider=provider, role=role, use_platform_store=use_platform_store, notify_on_reload=notify_on_reload, artifact_id=artifact['id'], uset_parental_memory=True)
        next_node = f"choice_agent_{artifact['id']}"
        builder.add_node(next_node, _choice_agent)
        builder.add_edge(prev_node, next_node)
        prev_node = next_node
    builder.add_edge(prev_node, END)

    graph = builder.compile(
        checkpointer=memory,
        debug=False
    ).with_config({"callbacks": callback_handlers})

    return graph

if __name__ == "__main__":
    print("Initializing Theodor Agent...")
    agent_graph = initialize_agent()
    user_llm = get_llm(model="mini", provider="openai")
    logging.info("Simulation started")

    config = {"configurable": {"thread_id": "simulation_thread_1"}}

    next_input: Any = {
        "messages": [
            HumanMessage(
                content=(
                    "Привет! У меня есть идея стартапа: Uber для выгула собак. "
                    "Помоги мне проработать её."
                )
            )
        ]
    }

    max_steps = 50  # Safety limit
    #print("\n--- STARTING SIMULATION ---\n")
    logging.info("\n--- STARTING SIMULATION ---\n")

    ctx = ArtifactAgentContext(user_prompt = "Привет! У меня есть идея стартапа: Uber для выгула собак. Помоги мне проработать её.",
                               generated_artifacts = [])

    
    while True:
        result = agent_graph.invoke(next_input, config=config, context=ctx)
        #res = agent_graph.get_state(config, subgraphs=True)
        state_snapshot = agent_graph.get_state(config)
        current_idx = state_snapshot.values.get("current_step_index", 0)
        logging.info("Current step index: %s", current_idx)
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = getattr(interrupts[-1], "value", interrupts[-1])
            last_ai = payload.get("content", "")
            #print(f"Interrupt payload: {payload}")
            prompt = f"Ответ наставника:\n{last_ai}\n\nТвой ответ:"
            user_reply = input(prompt)
            if user_reply == "exit":
                break
            next_input = Command(resume=user_reply)
            continue
        else:
            last_ai = ""
            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        last_ai = msg.content
                        break
            if last_ai:
                #print(f"Bot: {last_ai}")
                logging.info("Bot: %s", last_ai)
        #break

    print("ФСЁ")
