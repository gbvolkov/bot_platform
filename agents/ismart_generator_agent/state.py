from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IsmartGeneratorAgentContext(TypedDict, total=False):
    input: str
    input_url: str
    output: str
    task_id: str
    lesson_number: str
    max_generation_iterations: int
    max_package_repair_iterations: int
    max_reference_chars: int
    generation_target: str
    verbose: bool


class IsmartGeneratorAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    phase: NotRequired[str]
    payload: NotRequired[Any]
    tasks: NotRequired[list[dict[str, Any]]]
    results: NotRequired[list[dict[str, Any]]]
    output_text: NotRequired[str]
    error: NotRequired[str]
