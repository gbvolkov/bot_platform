from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ThinkInput(BaseModel):
    thought: str = Field(..., description="A thought to think about.")


class ThinkTool(BaseTool):
    name: str = "think_tool"
    description: str = (
        "Use the tool to think about something.\n"
        "It will not obtain new information or change the database, but just append the thought to the log.\n"
        "Use it when complex reasoning or some cache memory or brainstorming or some critisism is needed.\n"
        "For example, if you need to brainstorm several unique business ideas or hypotesis, call this tool.\n"
        "For example, if you need to critisise or estimate user's proposals for artifact change or new ideas or their critisism, call this tool."
    )
    args_schema: Type[BaseModel] = ThinkInput

    def _run(self, thought: str) -> str:
        # Simple echo for traceability; caller should treat output as internal.
        return f"[think] {thought}"
