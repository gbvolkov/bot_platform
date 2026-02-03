from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ArtifactCreatorAgentContext(TypedDict, total=False):
    """Runtime context for the system agent."""

    system_prompt: str


class ArtifactCreatorAgentState(AgentState[Dict[str, Any]]):
    """State schema for simple agent graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]

    system_prompt: NotRequired[str]

    # phases: init -> set_prompt -> run -> confirm -> ready
    phase: NotRequired[str]
    greeted: NotRequired[bool]
    artifact: NotRequired[str]
    last_user_answer: NotRequired[str]
    is_artifact_confirmed: NotRequired[bool]

class ConfirmationAgentState(TypedDict, total=False):
    # Keep it minimal. Only what the confirmation agent needs.
    messages: Annotated[List[BaseMessage], add_messages]
    last_user_answer: str
    artifact: NotRequired[str]