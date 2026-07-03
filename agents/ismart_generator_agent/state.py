from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IsmartGeneratorAgentContext(TypedDict, total=False):
    input: str
    input_url: str
    output: str
    lesson_numbers: list[str]
    task_ids: list[str]
    task_id: str
    lesson_number: str
    from_lesson: int
    to_lesson: int
    limit: int
    max_generation_iterations: int
    max_package_repair_iterations: int
    max_reference_chars: int
    prompts_dir: str
    generation_target: str
    run_name: str
    preserve_source_index: bool
    stop_on_error: bool
    stop_on_failure: bool
    previous_lessons_context: list[dict[str, Any]]
    previous_lesson_context: list[dict[str, Any]]
    verbose: bool


class GeneratorRunRequest(TypedDict, total=False):
    input: str
    input_url: str
    output: str
    lesson_numbers: list[str]
    task_ids: list[str]
    lesson_number: str
    task_id: str
    from_lesson: int
    to_lesson: int
    limit: int
    generation_target: str
    max_generation_iterations: int
    max_package_repair_iterations: int
    max_reference_chars: int
    prompts_dir: str
    run_name: str
    preserve_source_index: bool
    stop_on_error: bool
    stop_on_failure: bool
    previous_lessons_context: list[dict[str, Any]]
    previous_lesson_context: list[dict[str, Any]]
    verbose: bool


class IsmartGeneratorAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    phase: NotRequired[str]
    payload: NotRequired[Any]
    tasks: NotRequired[list[dict[str, Any]]]
    task_records: NotRequired[list[dict[str, Any]]]
    batch_manifest_path: NotRequired[str]
    task_entries: NotRequired[list[dict[str, Any]]]
    results: NotRequired[list[dict[str, Any]]]
    output_text: NotRequired[str]
    error: NotRequired[str]
