from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SysAdminAgentContext(TypedDict, total=False):
    """Runtime context for the system agent."""

    server: str



class SysAdminAgentState(AgentState[Dict[str, Any]]):
    """State schema for simple agent graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]

    mcp_session_key: NotRequired[str]
    server: NotRequired[str]
    target_id: NotRequired[str]
    allowed_paths: NotRequired[List[str]]
    working_dir: NotRequired[str]
    execution_id: NotRequired[str]
    password_secret_id: NotRequired[str]

    # phases: init -> set_prompt -> run
    phase: NotRequired[str]
    greeted: NotRequired[bool]
