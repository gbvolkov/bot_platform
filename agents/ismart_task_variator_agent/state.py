from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class VariatorAgentContext(TypedDict, total=False):
    """Runtime context for the system agent."""

    mode: str



class VariatorAgentState(AgentState[Dict[str, Any]]):
    """State schema for simple agent graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]

    system_prompt: NotRequired[str]

    # phases: init -> set_prompt -> run
    phase: NotRequired[str]
    greeted: NotRequired[bool]

    mode: NotRequired[str]
    
    options_cnt: NotRequired[int]
