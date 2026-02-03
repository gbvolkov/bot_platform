from __future__ import annotations

from typing import Any, Dict

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .artifacts_defs import ARTIFACTS


def _resolve_definition(artifact_id: int) -> Dict[str, Any]:
    if 0 <= artifact_id < len(ARTIFACTS):
        return ARTIFACTS[artifact_id]
    return {"id": artifact_id, "name": f"Artifact {artifact_id + 1}"}


@tool("commit_artifact_options")
def commit_artifact_options(
    artifact_id: int,
    options_text: str,
    runtime: ToolRuntime = None,
) -> Command:
    """Persist the generated options text for an artifact."""
    tool_call_id = runtime.tool_call_id if runtime else None
    details = {
        "artifact_definition": _resolve_definition(artifact_id),
        "artifact_options_text": options_text,
    }
    return Command(
        update={
            "artifacts": {artifact_id: details},
            "messages": [ToolMessage(content="Success", tool_call_id=tool_call_id)],
        }
    )


@tool("commit_artifact_selection")
def commit_artifact_selection(
    artifact_id: int,
    selection_text: str,
    runtime: ToolRuntime = None,
) -> Command:
    """Persist the user selection for an artifact."""
    tool_call_id = runtime.tool_call_id if runtime else None
    details = {"selected_option_text": selection_text}
    return Command(
        update={
            "artifacts": {artifact_id: details},
            "messages": [ToolMessage(content="Success", tool_call_id=tool_call_id)],
        }
    )


@tool("commit_artifact_final_text")
def commit_artifact_final_text(
    artifact_id: int,
    final_text: str,
    runtime: ToolRuntime = None,
) -> Command:
    """Persist the final artifact text (including criteria assessment)."""
    tool_call_id = runtime.tool_call_id if runtime else None
    details = {
        "artifact_definition": _resolve_definition(artifact_id),
        "artifact_final_text": final_text,
    }
    return Command(
        update={
            "artifacts": {artifact_id: details},
            "messages": [ToolMessage(content="Success", tool_call_id=tool_call_id)],
        }
    )
