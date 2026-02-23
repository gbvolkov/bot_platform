
from types import SimpleNamespace
from typing import Type

from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import ensure_config
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage


from agents.store_artifacts import store_chapters

@tool("store_artifact_tool")
def store_artifact_tool(
    title: str,
    artifact: str,
    runtime: ToolRuntime = None,
) -> str:
    """
    Use the tool to store some artifacts on user's request.
    The tool will receive an artifact Title, its content in Markdownv2 format.
    The tool store an artifact and returns a downloadable link.
    Use this tool any time user reqests for an artifact to be saved.

    
    Args:
        title (str): A title of an artifact.
        artifact (str): Text of the artifact in MarkdownV2 format,

    """
    #tool_call_id = runtime.tool_call_id if runtime else None
    locale = runtime.state["locale"]
    url = store_chapters([{"title": title, "body": artifact}], artifact)

    template = locale["save_confirmation"]

    return template.format(url=url)
