from typing import Any, Dict, List, Annotated, TypedDict

from typing_extensions import NotRequired
from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .state import SenseLineResponse, IdeaListResponse


@tool("commit_thematic_theeds_struct")
def commit_thematic_threads_struct(
    final_thematic_threads: SenseLineResponse,
    runtime: ToolRuntime = None,
) -> Command:
    """
    Commits final list of thematic threads.
    Фиксирует окончательный список смысловых линий

    Args:
        final_thematic_threads (dict): A final list of thematic threads. Окончательный список смысловых линий.

    """
    tool_call_id = runtime.tool_call_id if runtime else None
    return Command(update={
        "final_thematic_threads_struct": final_thematic_threads,
        "messages": [
            ToolMessage(
                content="Success",
                tool_call_id=tool_call_id,
            )
        ],
    })

@tool("commit_thematic_threads")
def commit_thematic_threads(
    final_thematic_threads: str,
    runtime: ToolRuntime = None,
) -> Command:
    """
    Commits final text of thematic threads.
    Фиксирует окончательный текст смысловых линий

    Args:
        final_thematic_threads (str): The final text of thematic threads. Окончательный текст смысловых линий

    """
    tool_call_id = runtime.tool_call_id if runtime else None
    return Command(update={
        "final_thematic_threads": final_thematic_threads,
        "messages": [
            ToolMessage(
                content="Success",
                tool_call_id=tool_call_id,
            )
        ],
    })

@tool("commit_ideas_struct")
def commit_ideas_struct(
    final_ideas: IdeaListResponse,
    runtime: ToolRuntime = None,
) -> Command:
    """
    Commits final list of ideas
    Фиксирует финальный список идей

    Args:
        final_ideas (dict): List of ideas. Список идей.

    """
    tool_call_id = runtime.tool_call_id if runtime else None
    return Command(update={
        "final_ideas_struct": final_ideas,
        "messages": [
            ToolMessage(
                content="Success",
                tool_call_id=tool_call_id,
            )
        ],
    })


@tool("commit_ideas")
def commit_ideas(
    final_ideas: str,
    runtime: ToolRuntime = None,
) -> Command:
    """
    Commits list of ideas, the final text of the selected idea, artifacts created to develop the ideas
    Фиксирует список идей, итоговый текст выбранной идеи, артефакты, созданные для раскрытия идей

    Args:
        final_ideas (str): List of ideas, the final text of the selected idea, artifacts created to develop the ideas. Список идей, итоговый текст выбранной идеи, артефакты, созданные для раскрытия идей.

    """
    tool_call_id = runtime.tool_call_id if runtime else None
    return Command(update={
        "final_ideas": final_ideas,
        "messages": [
            ToolMessage(
                content="Success",
                tool_call_id=tool_call_id,
            )
        ],
    })


@tool("commit_final_docset")
def commit_final_docset(
    final_doc_set: str,
    runtime: ToolRuntime = None,
) -> Command:
    """
    Commits final set of aftifacts
    Фиксирует финальный пакет документов

    Args:
        final_doc_set (str): final set of aftifacts. финальный пакет документов.

    """
    tool_call_id = runtime.tool_call_id if runtime else None
    return Command(update={
        "final_doc_set": final_doc_set,
        "messages": [
            ToolMessage(
                content="Success",
                tool_call_id=tool_call_id,
            )
        ],
    })
