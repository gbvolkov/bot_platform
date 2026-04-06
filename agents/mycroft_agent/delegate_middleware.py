from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from deepagents.middleware.subagents import (
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
    _create_task_tool,
)
from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool


def _create_delegate_tool(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
    delegate_description: str | None = None,
) -> BaseTool:
    task_tool = _create_task_tool(
        default_model=default_model,
        default_tools=default_tools,
        default_middleware=default_middleware,
        default_interrupt_on=default_interrupt_on,
        subagents=subagents,
        general_purpose_agent=general_purpose_agent,
        task_description=delegate_description,
    )
    if not isinstance(task_tool, StructuredTool):
        raise TypeError(
            "DelegateSubAgentMiddleware expected _create_task_tool() to return StructuredTool."
        )

    return StructuredTool.from_function(
        func=task_tool.func,
        coroutine=task_tool.coroutine,
        name="delegate",
        description=task_tool.description,
        args_schema=task_tool.args_schema,
        return_direct=task_tool.return_direct,
        response_format=task_tool.response_format,
    )


class DelegateSubAgentMiddleware(SubAgentMiddleware):
    def __init__(
        self,
        *,
        default_model: str | BaseChatModel,
        default_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        default_middleware: list[AgentMiddleware] | None = None,
        default_interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = None,
        general_purpose_agent: bool = True,
        task_description: str | None = None,
    ) -> None:
        AgentMiddleware.__init__(self)
        self.system_prompt = system_prompt
        delegate_tool = _create_delegate_tool(
            default_model=default_model,
            default_tools=default_tools or [],
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=subagents or [],
            general_purpose_agent=general_purpose_agent,
            delegate_description=task_description,
        )
        self.tools = [delegate_tool]
