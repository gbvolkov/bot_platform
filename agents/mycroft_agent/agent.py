from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import config
from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver

from agents.llm_utils import get_llm
from agents.tools.yandex_search import YandexSearchTool
from agents.utils import ModelType

from .delegate_middleware import DelegateSubAgentMiddleware
from .prompts import build_system_prompt
from .prompts import build_delegate_system_prompt, build_delegate_tool_description


VALID_MODEL_SIZES = {"base", "mini", "nano"}


def build_web_search_tool(
    *,
    max_results: int = 5,
    summarize: bool = True,
) -> Any:
    missing: list[str] = []
    if not config.YA_API_KEY:
        missing.append("YA_API_KEY")
    if not config.YA_FOLDER_ID:
        missing.append("YA_FOLDER_ID")
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Mycroft web_search requires environment variables: {missing_text}"
        )

    return YandexSearchTool(
        api_key=config.YA_API_KEY,
        folder_id=config.YA_FOLDER_ID,
        max_results=max_results,
        summarize=summarize,
    )


def _has_web_search_tool(tools: Sequence[Any]) -> bool:
    for tool in tools:
        if getattr(tool, "name", None) == "web_search":
            return True
    return False


def _build_callback_handlers() -> list[Any]:
    if not config.LANGFUSE_URL:
        return []

    Langfuse(
        public_key=config.LANGFUSE_PUBLIC,
        secret_key=config.LANGFUSE_SECRET,
        host=config.LANGFUSE_URL,
    )
    return [LangfuseCallbackHandler()]


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    *,
    model_size: str = "base",
    temperature: float = 0.2,
    system_prompt: str | None = None,
    tools: Sequence[Any] | None = None,
    stateless_subagents: Sequence[CompiledSubAgent] | None = None,
    stateful_subagents: Sequence[CompiledSubAgent] | None = None,
    checkpoint_saver: Any | None = None,
    streaming: bool = False,
    reasoning: str | None = None,
    max_tool_calls: int | None = 12,
    interrupt_on: dict[str, bool | dict[str, Any]] | None = None,
) -> Any:
    if provider.value in {"openai", "openai_4", "openai_pers", "openai_think"} and not config.OPENAI_API_KEY:
        raise ValueError(
            f"Mycroft provider '{provider.value}' requires environment variable OPENAI_API_KEY"
        )

    callback_handlers = _build_callback_handlers()
    llm = get_llm(
        model=model_size,
        provider=provider.value,
        temperature=temperature,
        streaming=streaming,
        reasoning=reasoning,
        max_tool_calls=max_tool_calls,
        callbacks=callback_handlers,
    )

    resolved_tools = list(tools or [])
    resolved_stateless_subagents = list(stateless_subagents or [])
    resolved_stateful_subagents = list(stateful_subagents or [])
    resolved_system_prompt = system_prompt or build_system_prompt(
        enable_web_search=_has_web_search_tool(resolved_tools)
    )
    resolved_middleware: list[Any] = []
    if resolved_stateful_subagents:
        resolved_middleware.append(
            DelegateSubAgentMiddleware(
                default_model=llm,
                default_tools=resolved_tools,
                subagents=resolved_stateful_subagents,
                system_prompt=build_delegate_system_prompt(tool_name="delegate"),
                general_purpose_agent=False,
                task_description=build_delegate_tool_description(tool_name="delegate"),
            )
        )

    agent = create_deep_agent(
        name="mycroft",
        model=llm,
        tools=resolved_tools,
        subagents=resolved_stateless_subagents,
        middleware=resolved_middleware,
        system_prompt=resolved_system_prompt,
        checkpointer=checkpoint_saver or MemorySaver(),
        interrupt_on=interrupt_on,
    )
    if callback_handlers:
        return agent.with_config({"callbacks": callback_handlers})
    return agent
