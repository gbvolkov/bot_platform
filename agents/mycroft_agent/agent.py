from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import config

from agents.llm_utils import get_llm
from agents.utils import ModelType
from langgraph.checkpoint.memory import MemorySaver

from .prompts import build_system_prompt

if TYPE_CHECKING:
    from agents.tools.yandex_search import YandexSearchTool


VALID_MODEL_SIZES = {"base", "mini", "nano"}


def _import_create_deep_agent():
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The `deepagents` package is not installed. Install dependencies with "
            "`uv pip install -r pyproject.toml` or update and install `requirements.txt`."
        ) from exc
    return create_deep_agent


def build_web_search_tool(
    *,
    max_results: int = 5,
    summarize: bool = True,
) -> Any:
    if not config.YA_API_KEY or not config.YA_FOLDER_ID:
        raise RuntimeError(
            "Yandex web search requires YA_API_KEY and YA_FOLDER_ID in the environment."
        )

    from agents.tools.yandex_search import YandexSearchTool

    return YandexSearchTool(
        api_key=config.YA_API_KEY,
        folder_id=config.YA_FOLDER_ID,
        max_results=max_results,
        summarize=summarize,
    )


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    *,
    model_size: str = "base",
    temperature: float = 0.2,
    system_prompt: str | None = None,
    tools: Sequence[Any] | None = None,
    enable_web_search: bool = True,
    max_search_results: int = 5,
    summarize_search: bool = True,
    checkpoint_saver: Any | None = None,
    streaming: bool = False,
    reasoning: str | None = None,
    max_tool_calls: int | None = 12,
):
    if model_size not in VALID_MODEL_SIZES:
        choices = ", ".join(sorted(VALID_MODEL_SIZES))
        raise ValueError(f"Unknown model size '{model_size}'. Available values: {choices}")

    create_deep_agent = _import_create_deep_agent()

    llm = get_llm(
        model=model_size,
        provider=provider.value,
        temperature=temperature,
        streaming=streaming,
        reasoning=reasoning,
        max_tool_calls=max_tool_calls,
    )

    resolved_tools = list(tools or [])
    if enable_web_search and not any(
        getattr(tool, "name", None) == "web_search" for tool in resolved_tools
    ):
        resolved_tools.append(
            build_web_search_tool(
                max_results=max_search_results,
                summarize=summarize_search,
            )
        )

    create_kwargs: dict[str, Any] = {
        "name": "mycroft",
        "model": llm,
        "tools": resolved_tools,
        "system_prompt": system_prompt or build_system_prompt(enable_web_search=enable_web_search),
        "checkpointer": checkpoint_saver or MemorySaver(),
    }

    try:
        return create_deep_agent(**create_kwargs)
    except TypeError as exc:
        if "checkpointer" not in str(exc):
            raise

    create_kwargs.pop("checkpointer", None)
    return create_deep_agent(**create_kwargs)
