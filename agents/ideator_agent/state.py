from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .models import IdeatorReport


class IdeatorAgentContext(TypedDict, total=False):
    """Runtime context for the ideator agent."""

    report_path: str


def _keep_first(current: IdeatorReport | None, new: IdeatorReport | None) -> IdeatorReport | None:
    return current if current is not None else new

def _merge_dicts(current: Dict[str, Any] | None, new: Dict[str, Any] | None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if current:
        merged.update(current)
    if new:
        merged.update(new)
    return merged


class IdeatorAgentState(AgentState[Dict[str, Any]]):
    """State schema for ideator agent graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]
    report: Annotated[NotRequired[IdeatorReport | str], _keep_first]
    report_path: Annotated[NotRequired[str], _keep_first]
    sense_lines: Annotated[NotRequired[List[Dict[str, Any]]], _keep_first]
    selected_line_id: NotRequired[str]
    filtered_article_ids: NotRequired[List[int]]
    filtered_articles: NotRequired[List[Any]]
    ideas: NotRequired[List[Dict[str, Any]]]
    # phases: ready -> lines (choose/discuss lines) -> ideas (generate/discuss ideas) -> finish
    phase: NotRequired[str]
    ideas_cache: Annotated[NotRequired[Dict[str, List[Dict[str, Any]]]], _merge_dicts]
    active_idea_id: NotRequired[int]
    force_regen: NotRequired[bool]
    force_regen_lines: NotRequired[bool]
    greeted: NotRequired[bool]
