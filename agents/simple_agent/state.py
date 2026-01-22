from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SimpleAgentContext(TypedDict, total=False):
    """Runtime context for the system agent."""

    system_prompt: str



class SimpleAgentState(AgentState[Dict[str, Any]]):
    """State schema for simple agent graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]

    system_prompt: NotRequired[str]

    # phases: init -> set_prompt -> run
    phase: NotRequired[str]
    greeted: NotRequired[bool]
