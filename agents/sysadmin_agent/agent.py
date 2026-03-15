from __future__ import annotations

from contextlib import AsyncExitStack
import httpx
import json
import logging
import threading
import time
from typing import Any, Dict, List, MutableMapping

from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer


from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config

from agents.utils import ModelType, get_llm
from platform_utils.llm_logger import JSONFileTracer

from .mcp_utils import (
    apply_mcp_context_to_state,
    create_mcp_session_agent,
    merge_state_into_mcp_context,
)
from .state import SysAdminAgentContext, SysAdminAgentState


from .prompts import (
    GREETINGS_RU,
    SET_SERVER_REQUEST_RU,
    SERVER_CONFIRMATION_RU,
    DEFAULT_SYSTEM_PROMPT_RU
)

LOG = logging.getLogger(__name__)


def _walk_exceptions(error: BaseException) -> list[BaseException]:
    nested = getattr(error, "exceptions", None)
    if not nested:
        return [error]

    flat: list[BaseException] = []
    for item in nested:
        if isinstance(item, BaseException):
            flat.extend(_walk_exceptions(item))
    return flat


def _has_http_status(error: BaseException, status_code: int) -> bool:
    for item in _walk_exceptions(error):
        if isinstance(item, httpx.HTTPStatusError) and item.response is not None:
            if item.response.status_code == status_code:
                return True
    return False


def _is_connection_closed_error(error: BaseException) -> bool:
    for item in _walk_exceptions(error):
        if "Connection closed" in str(item):
            return True
    return False


def _session_context_http_status(
    session_context: MutableMapping[str, Any] | None,
) -> int | None:
    if session_context is None:
        return None
    value = session_context.get("_last_mcp_http_status")
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _session_context_auth_retry_attempted(
    session_context: MutableMapping[str, Any] | None,
) -> bool:
    if session_context is None:
        return False
    return bool(session_context.get("_mcp_auth_retry_attempted"))


def _is_mcp_401_failure(
    error: BaseException,
    session_context: MutableMapping[str, Any] | None,
) -> bool:
    if _has_http_status(error, 401):
        return True
    return _session_context_http_status(session_context) == 401 and _is_connection_closed_error(error)


def _first_http_status_error(
    error: BaseException,
    status_code: int,
) -> httpx.HTTPStatusError | None:
    for item in _walk_exceptions(error):
        if isinstance(item, httpx.HTTPStatusError) and item.response is not None:
            if item.response.status_code == status_code:
                return item
    return None


def _debug_print_mcp_401(
    *,
    session_key: str,
    session_context: MutableMapping[str, Any] | None,
    error: BaseException,
) -> None:
    status_error = _first_http_status_error(error, 401)
    print(f"[sysadmin-agent][debug] MCP returned 401 for session '{session_key}'")
    if status_error is not None:
        print(f"[sysadmin-agent][debug] 401 error: {status_error}")
    else:
        print(f"[sysadmin-agent][debug] 401 error: {error}")

    if session_context is None:
        print("[sysadmin-agent][debug] MCP request parameters: <unavailable>")
        return

    request_debug = session_context.get("_last_mcp_request")
    if request_debug is None:
        print("[sysadmin-agent][debug] MCP request parameters: <unavailable>")
        return

    try:
        formatted = json.dumps(request_debug, ensure_ascii=False, indent=2, default=str)
    except Exception:
        formatted = str(request_debug)
    print("[sysadmin-agent][debug] MCP request parameters:")
    print(formatted)


def _debug_session_context_from_error(error: BaseException) -> MutableMapping[str, Any] | None:
    for item in _walk_exceptions(error):
        session_context = getattr(item, "_sysadmin_session_context", None)
        if session_context is not None:
            return session_context
    session_context = getattr(error, "_sysadmin_session_context", None)
    if session_context is not None:
        return session_context
    return None

def _safe_stream_writer():
    """Return a writer suitable for `stream_mode="custom"`; otherwise no-op."""
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None

class StreamWriterCallbackHandler(BaseCallbackHandler):
    """Forward LangChain callbacks (tool/chain lifecycle) into LangGraph custom stream."""

    def on_tool_start(self, serialized, input_str=None, **kwargs):
        writer = _safe_stream_writer()
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "tool"
        writer({"type": "tool_start", "name": name})

    def on_tool_end(self, output=None, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "tool_end"})

    def on_tool_error(self, error, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "tool_error", "error": str(error)})

    def on_chain_start(self, serialized, inputs, **kwargs):
        writer = _safe_stream_writer()
        name = (serialized or {}).get("name") or "chain"
        writer({"type": "chain_start", "name": name})

    def on_chain_end(self, outputs, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "chain_end"})


def route(state: SysAdminAgentState) -> str:
    if not state.get("greeted"):
        return "greetings"

    return "run"

def create_greetings_node():

    def greetings_node(
        state: SysAdminAgentState,
        config: RunnableConfig,
        runtime: Runtime[SysAdminAgentContext],
    ) -> SysAdminAgentState:
        greet = GREETINGS_RU
        state["messages"] = (state.get("messages") or []) + [AIMessage(content=greet)]
        state["greeted"] = True
        return state

    return greetings_node


def create_set_server_node(model: BaseChatModel):

    def set_server_node(
        state: SysAdminAgentState,
        config: RunnableConfig,
        runtime: Runtime[SysAdminAgentContext],
    ) -> SysAdminAgentState:
        #state["phase"] = "run"
        last_user_msg = None
        original_messages = state.get("messages", [])

        for msg in reversed(original_messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                last_user_msg = msg
                break
        if last_user_msg is None:
            response_text = SET_SERVER_REQUEST_RU
            state["phase"] = "set_server"
            return state
        else:
            response_text = SERVER_CONFIRMATION_RU
            #all_msg_ids = [m.id for m in state["messages"]]
            state["server"] = last_user_msg.content
            state["phase"] = "cleanup"
            # Returning RemoveMessage instances instructs the reducer to delete them
            #return {
            #    "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [AIMessage(content=response_text)],
            #    "system_prompt": system_prompt,
            #    "phase" : "cleanup"
            #}            
        state["messages"].append(AIMessage(content=response_text))
        return state
            

    return set_server_node


def _build_run_agent(model: BaseChatModel):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: SysAdminAgentState = request.state
        notes: List[str] = [DEFAULT_SYSTEM_PROMPT_RU.strip()]

        if state.get("target_id"):
            notes.append(f"Current target_id: {state['target_id']}")
        if state.get("allowed_paths"):
            notes.append(f"Allowed file paths: {', '.join(state['allowed_paths'])}")
        if state.get("working_dir"):
            notes.append(f"Current working directory: {state['working_dir']}")

        notes.append(
            "Use list_targets first if target_id is unknown. "
            "For browse_files, read_file, and run_command always provide target_id. "
            "Prefer browse_files/read_file for filesystem inspection inside allowed paths only. "
            "If the requested path is outside the allowlist, explain that direct file access is blocked and inspect it via run_command instead. "
            "If the user asks to search the whole system or find compose files anywhere on disk, use run_command with find/grep rather than browse_files. "
            "Do not guess arbitrary directories like /opt/app/config when searching for docker compose files system-wide. "
            "If browse_files/read_file reports that a path does not exist, switch to another path or use run_command instead of stopping. "
            "When the user asks for directories or files, use browse_files and filter the returned entries instead of shell find/ls."
        )
        return "\n\n".join(notes)

    cache_lock = threading.Lock()
    session_entries: Dict[str, tuple[AsyncExitStack, Any, MutableMapping[str, Any]]] = {}

    async def create_thread_agent(
        session_context: MutableMapping[str, Any],
    ) -> tuple[AsyncExitStack, Any]:
        return await create_mcp_session_agent(
            model=model,
            middleware=[build_prompt],
            state_schema=SysAdminAgentState,
            context_schema=SysAdminAgentContext,
            session_context=session_context,
        )

    async def get_thread_agent(session_key: str) -> tuple[Any, MutableMapping[str, Any]]:
        with cache_lock:
            existing_entry = session_entries.get(session_key)
        if existing_entry is not None:
            return existing_entry[1], existing_entry[2]

        new_context: MutableMapping[str, Any] = {}
        try:
            new_stack, new_agent = await create_thread_agent(new_context)
        except Exception as exc:
            setattr(exc, "_sysadmin_session_context", new_context)
            raise
        duplicate_stack: AsyncExitStack | None = None
        with cache_lock:
            existing_entry = session_entries.get(session_key)
            if existing_entry is None:
                session_entries[session_key] = (new_stack, new_agent, new_context)
                return new_agent, new_context
            duplicate_stack = new_stack
        if duplicate_stack is not None:
            await duplicate_stack.aclose()
        return existing_entry[1], existing_entry[2]

    async def run_node(
        state: SysAdminAgentState,
        config: RunnableConfig,
        runtime: Runtime[SysAdminAgentContext],
    ) -> SysAdminAgentState:
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = str(configurable.get("thread_id") or "default")
        session_key = str(state.get("mcp_session_key") or thread_id)

        if state.get("mcp_session_key") != session_key:
            state = dict(state)
            state["mcp_session_key"] = session_key

        session_context: MutableMapping[str, Any] | None = None

        async def invoke_once() -> tuple[SysAdminAgentState, MutableMapping[str, Any]]:
            nonlocal session_context
            agent, session_context = await get_thread_agent(session_key)
            merge_state_into_mcp_context(state, session_context)
            result = await agent.ainvoke(state, config=config, context=runtime.context)
            return result, session_context

        try:
            result, session_context = await invoke_once()
        except Exception as exc:
            if not _is_mcp_401_failure(exc, session_context or _debug_session_context_from_error(exc)):
                raise

            _debug_print_mcp_401(
                session_key=session_key,
                session_context=session_context or _debug_session_context_from_error(exc),
                error=exc,
            )
            effective_context = session_context or _debug_session_context_from_error(exc)
            if _session_context_auth_retry_attempted(effective_context):
                raise RuntimeError(
                    "Sysadmin MCP authentication failed after refreshing the access token. "
                    "Check MCP OAuth credentials or MCP server authorization."
                ) from exc
            raise RuntimeError(
                "Sysadmin MCP authentication failed. "
                "Check the MCP bearer token or MCP OAuth credentials."
            ) from exc

        if isinstance(result, dict) and result.get("mcp_session_key") != session_key:
            result = apply_mcp_context_to_state(result, session_context)
            result = dict(result)
            result["mcp_session_key"] = session_key
        elif isinstance(result, dict):
            result = apply_mcp_context_to_state(result, session_context)
        return result

    return run_node


def create_run_node(model: BaseChatModel):
    return _build_run_agent(model)


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "en",
    checkpoint_saver=None,
    *,
    streaming: bool = True,
):
    #set_locale(locale)
    #set_models_locale(locale)
    log_name = f"simple_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    #callback_handlers = [json_handler]
    callback_handlers = [StreamWriterCallbackHandler(), json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        _ = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        callback_handlers += [CallbackHandler()]

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    llm = get_llm(model="base", provider=provider.value, temperature=0.4, streaming=streaming)
    
    builder = StateGraph(SysAdminAgentState)
    builder.add_node("greetings", create_greetings_node())
    builder.add_node("set_server", create_set_server_node(llm))

    builder.add_node("run", create_run_node(llm))

    builder.add_conditional_edges(
        START,
        route,
        {
            "greetings": "greetings",
            "run": "run"
        },
    )
    builder.add_edge("run", END)

    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph
