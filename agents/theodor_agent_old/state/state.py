from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class ArtifactVersion(TypedDict):
    version: int
    content: str
    comment: str
    timestamp: float


class ArtifactState(TypedDict):
    id: int
    name: str
    status: Literal["PENDING", "ACTIVE", "READY_FOR_CONFIRM", "APPROVED"]
    current_content: str
    history: List[ArtifactVersion]


def update_artifacts(existing: Dict[int, ArtifactState], new_data: Dict[int, ArtifactState]) -> Dict[int, ArtifactState]:
    if existing is None:
        return new_data
    return {**existing, **new_data}


class TheodorAgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    current_step_index: int
    artifacts: Annotated[Dict[int, ArtifactState], update_artifacts]  # Key is step index (0-12)
    user_info: Dict[str, Any]
    # Flag to indicate if we are waiting for explicit confirmation
    waiting_for_confirmation: bool
