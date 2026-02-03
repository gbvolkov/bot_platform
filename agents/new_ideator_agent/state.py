from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ArticleRef(TypedDict):
    id: Annotated[int, "Id from the provided list"]
    title: Annotated[str, "Title of the article"]
    summary: Annotated[str, "Summary of the article"]

class SenseLineItem(TypedDict):
    id: Annotated[str, "Stable id like L1/L2"]
    short_title: Annotated[str, "Short name of the sense line"]
    description: Annotated[str, "1-2 sentences grounded in the provided articles"]
    #article_ids: Annotated[List[int], "Ids from the provided list only"]
    articles: Annotated[List[ArticleRef], "List of articles from the provided list only"]
    region_note: Annotated[str, "Region applicability note"]

class SenseLineResponse(TypedDict):
    sense_lines: List[SenseLineItem]


class IdeaItem(TypedDict):
    title: Annotated[str, "Idea headline (1 short sentence)"]
    summary: Annotated[str, "1-2 sentences grounded strictly in provided articles"]
    #article_ids: Annotated[List[int], "Ids from the provided list only"]
    articles: Annotated[List[ArticleRef], "List of articles from the provided list only"]
    region_note: Annotated[str, "Region applicability note"]
    importance_hint: Annotated[NotRequired[str], "High/Medium/Low or empty"]


class IdeaListResponse(TypedDict):
    ideas: Annotated[List[IdeaItem], "List of generated ideas."]


class IdeatorAgentContext(TypedDict, total=False):
    """Runtime context for the system agent."""

    scout_report: str

class IdeatorAgentState(AgentState[Dict[str, Any]]):
    """State schema for simple agent graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]

    scout_report: NotRequired[str]
    thematic_threads: NotRequired[str]
    ideas: NotRequired[str]
    final_thematic_threads: NotRequired[str]
    final_thematic_threads_struct: NotRequired[SenseLineResponse]
    final_ideas: NotRequired[str]
    final_ideas_struct: NotRequired[IdeaListResponse]
    final_doc_set: NotRequired[str]

    # phases: init -> set_prompt -> run
    phase: NotRequired[str]
    greeted: NotRequired[bool]
