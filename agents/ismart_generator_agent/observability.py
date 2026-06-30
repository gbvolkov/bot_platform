from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config as root_config
from platform_utils.llm_logger import JSONFileTracer


LOG = logging.getLogger(__name__)


def build_callback_handlers(log_name: str) -> list[Any]:
    handlers: list[Any] = [JSONFileTracer(f"./logs/{log_name}")]
    if root_config.LANGFUSE_URL and len(root_config.LANGFUSE_URL) > 0:
        try:
            Langfuse(
                public_key=root_config.LANGFUSE_PUBLIC,
                secret_key=root_config.LANGFUSE_SECRET,
                host=root_config.LANGFUSE_URL,
            )
            handlers.append(CallbackHandler())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Langfuse initialisation failed: %s", exc)
    return handlers


def langchain_config_from_runnable(config: RunnableConfig | None) -> dict[str, Any]:
    base_config = ensure_config(config)
    result: dict[str, Any] = {}
    for key in ("callbacks", "tags", "metadata", "recursion_limit", "max_concurrency"):
        value = base_config.get(key)
        if value is not None:
            result[key] = value
    configurable = base_config.get("configurable")
    if isinstance(configurable, dict):
        result["configurable"] = configurable.copy()
    return result
