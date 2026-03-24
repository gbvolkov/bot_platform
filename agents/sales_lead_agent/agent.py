from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
import httpx
from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agents.llm_utils import get_llm
from agents.utils import ModelType

from .prompts import build_system_prompt
from .tools import ToolUserCorrectableError, build_sales_lead_tools

import config

VALID_MODEL_SIZES = {"base", "mini", "nano"}
logger = logging.getLogger(__name__)


def _is_retryable_http_status(exc: httpx.HTTPStatusError) -> bool:
    status_code = exc.response.status_code if exc.response is not None else None
    return status_code == 429 or (status_code is not None and 500 <= status_code < 600)


def _should_retry_tool_exception(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, PlaywrightTimeoutError)):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return _is_retryable_http_status(exc)
    return False


class ToolErrorJsonMiddleware(AgentMiddleware):
    """Return structured JSON only for LLM-fixable or exhausted transient tool failures."""

    @staticmethod
    def _build_tool_message(
        request: ToolCallRequest,
        *,
        payload: dict[str, Any],
        exc: Exception,
    ) -> ToolMessage:
        tool_name = request.tool_call["name"]
        logger.error(
            "sales_lead_agent tool failed: tool=%s args=%r",
            tool_name,
            request.tool_call.get("args"),
            exc_info=True,
        )
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=request.tool_call["id"],
            name=tool_name,
            artifact={
                "tool": tool_name,
                "args": request.tool_call.get("args"),
                "error_type": exc.__class__.__name__,
            },
            status="error",
        )

    @classmethod
    def _user_correctable_message(
        cls,
        request: ToolCallRequest,
        exc: ToolUserCorrectableError,
    ) -> ToolMessage:
        return cls._build_tool_message(
            request,
            payload={
                "ok": False,
                "error_code": exc.code,
                "message": str(exc),
                "retryable": True,
                "suggestion": exc.suggestion,
                "input_field": exc.input_field,
            },
            exc=exc,
        )

    @classmethod
    def _transient_failure_message(
        cls,
        request: ToolCallRequest,
        exc: Exception,
    ) -> ToolMessage:
        return cls._build_tool_message(
            request,
            payload={
                "ok": False,
                "error_code": "TRANSIENT_TOOL_FAILURE",
                "message": str(exc),
                "retryable": True,
                "suggestion": "Retry the same tool call again.",
                "input_field": None,
            },
            exc=exc,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        try:
            return handler(request)
        except ToolUserCorrectableError as exc:
            return self._user_correctable_message(request, exc)
        except httpx.HTTPStatusError as exc:
            if _is_retryable_http_status(exc):
                return self._transient_failure_message(request, exc)
            raise
        except (asyncio.TimeoutError, TimeoutError, PlaywrightTimeoutError, httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
            return self._transient_failure_message(request, exc)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        try:
            return await handler(request)
        except ToolUserCorrectableError as exc:
            return self._user_correctable_message(request, exc)
        except httpx.HTTPStatusError as exc:
            if _is_retryable_http_status(exc):
                return self._transient_failure_message(request, exc)
            raise
        except (asyncio.TimeoutError, TimeoutError, PlaywrightTimeoutError, httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
            return self._transient_failure_message(request, exc)


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    *,
    model_size: str = "base",
    temperature: float = 0.1,
    system_prompt: str | None = None,
    tools: Sequence[Any] | None = None,
    checkpoint_saver: Any | None = None,
    streaming: bool = False,
    reasoning: str | None = None,
    max_tool_calls: int | None = 12,
):
    """Initialize the minimal sales lead agent."""
    if model_size not in VALID_MODEL_SIZES:
        choices = ", ".join(sorted(VALID_MODEL_SIZES))
        raise ValueError(f"Unknown model size '{model_size}'. Available values: {choices}")

    callback_handlers = []
    if config.LANGFUSE_URL:
        _ = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        callback_handlers.append(CallbackHandler())

    llm = get_llm(
        model=model_size,
        provider=provider.value,
        temperature=temperature,
        streaming=streaming,
        reasoning=reasoning,
        max_tool_calls=max_tool_calls,
    )
    tools = list(build_sales_lead_tools() if tools is None else tools)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=build_system_prompt() if system_prompt is None else system_prompt,
        middleware=[
            ToolErrorJsonMiddleware(),
            ToolRetryMiddleware(max_retries=2, retry_on=_should_retry_tool_exception, on_failure="error"),
        ],
        checkpointer=MemorySaver() if checkpoint_saver is None else checkpoint_saver,
        name="sales_lead_agent",
    ).with_config({
        "callbacks": callback_handlers,
    })
