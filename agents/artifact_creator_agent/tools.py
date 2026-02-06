from __future__ import annotations

from typing import Any, Dict

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .state import ArtifactStage


@tool("commit_artifact_final_text")
def commit_artifact_final_text(
    final_text: str,
    runtime: ToolRuntime = None,
) -> Command:
    """Persist the final artifact text."""
    tool_call_id = runtime.tool_call_id if runtime else None
    artifact_id = runtime.state.get("current_artifact_id", 0)
    details = {
        "artifact_final_text": final_text,
    }
    return Command(
        update={
            "artifacts": {artifact_id: details},
            "phase": "confirm",
            "messages": [ToolMessage(content="Success", tool_call_id=tool_call_id)],
        },
    )
