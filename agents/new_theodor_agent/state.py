from __future__ import annotations

from typing import Annotated, Any, Dict, List, NotRequired, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ArtifactDetails(TypedDict, total=False):
    artifact_definition: Dict[str, Any]
    artifact_options_text: str
    selected_option_text: str
    artifact_final_text: str
    artifact_summary: str


class ArtifactStage:
    INIT = "INIT"
    OPTIONS_GENERATED = "OPTIONS_GENERATED"
    OPTIONS_CONFIRMED = "OPTIONS_CONFIRMED"
    OPTION_SELECTED = "OPTION_SELECTED"
    ARTIFACT_GENERATED = "ARTIFACT_GENERATED"
    ARTIFACT_CONFIRMED = "ARTIFACT_CONFIRMED"


class TheodorAgentContext(TypedDict, total=False):
    user_prompt: str


def _merge_latest(a, b):
    return b if b is not None else a


def _merge_artifacts(
    a: Dict[int, ArtifactDetails] | None,
    b: Dict[int, ArtifactDetails] | None,
):
    merged: Dict[int, ArtifactDetails] = dict(a or {})
    for key, value in (b or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


class TheodorAgentState(AgentState[Dict[str, Any]]):
    messages: Annotated[List[BaseMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]

    user_prompt: NotRequired[Annotated[str, _merge_latest]]
    artifacts: NotRequired[Annotated[Dict[int, ArtifactDetails], _merge_artifacts]]

    current_artifact_id: NotRequired[Annotated[int, _merge_latest]]
    current_artifact_state: NotRequired[Annotated[str, _merge_latest]]
    current_artifact_text: NotRequired[Annotated[str, _merge_latest]]
