from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agents.artifact_creator_agent.prompts import ARTIFACT_URL_PROMPT_RU
import config
from agents.utils import ModelType
from platform_utils.llm_logger import JSONFileTracer

from .artifacts_defs import ARTIFACTS
from .locales import DEFAULT_LOCALE, resolve_locale, set_locale as set_global_locale
from ..store_artifacts import store_artifacts

from .choice_agent import initialize_agent as build_choice_agent
from .state import ArtifactStage, TheodorAgentContext, TheodorAgentState

LOG = logging.getLogger(__name__)

_PROGRESS_BANNER_BORDER = "════════════════════════════════"

_LOCALE_TEXT = {
    "en": {
        "final_report": "Final report: {url}",
        "progress_label": "PROGRESS: {bar} ({current}/{total})",
        "current_label": "CURRENT: Artifact {number} - {name}",
        "next_label": "NEXT: Artifact {number} - {name}",
        "next_finish": "NEXT: finish",
    },
    "ru": {
        "final_report": "Финальный отчет: {url}",
        "progress_label": "ПРОГРЕСС: {bar} ({current}/{total})",
        "current_label": "ТЕКУЩИЙ: Артефакт {number} — {name}",
        "next_label": "СЛЕДУЮЩИЙ: Артефакт {number} — {name}",
        "next_finish": "СЛЕДУЮЩИЙ: завершение",
    },
}

_CURRENT_LOCALE = DEFAULT_LOCALE


def _safe_stream_writer():
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None


class StreamWriterCallbackHandler(BaseCallbackHandler):
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


def _format_progress_banner(
    *,
    completed_count: int,
    total_count: int,
    current_artifact_number: int,
    current_artifact_name: str,
    next_artifact_number: Optional[int],
    next_artifact_name: Optional[str],
) -> str:
    completed_count = max(0, min(completed_count, total_count))
    total_count = max(0, total_count)
    bar = ("■" * completed_count) + ("□" * max(total_count - completed_count, 0))
    text = _LOCALE_TEXT[_CURRENT_LOCALE]

    progress_line = text["progress_label"].format(
        bar=bar,
        current=completed_count,
        total=total_count,
    )
    current_line = text["current_label"].format(
        number=current_artifact_number,
        name=current_artifact_name,
    )
    if next_artifact_number is not None and next_artifact_name:
        next_line = text["next_label"].format(
            number=next_artifact_number,
            name=next_artifact_name,
        )
    else:
        next_line = text["next_finish"]

    return "\n".join(
        [
            _PROGRESS_BANNER_BORDER,
            progress_line,
            current_line,
            next_line,
            _PROGRESS_BANNER_BORDER,
            "\n\n",
        ]
    )


def create_progress_node(artifact_id: int):
    def node(
        state: TheodorAgentState,
        config: RunnableConfig,
        runtime: Runtime[TheodorAgentContext],
    ) -> TheodorAgentState:
        total_artifacts = len(ARTIFACTS)
        current_def = ARTIFACTS[artifact_id]
        current_number = artifact_id + 1
        current_name = str(current_def.get("name") or f"Artifact {current_number}")

        next_def = ARTIFACTS[artifact_id + 1] if artifact_id + 1 < total_artifacts else None
        next_number = int(next_def["id"]) + 1 if next_def else None
        next_name = str(next_def.get("name") or f"Artifact {next_def['id'] + 1}") if next_def else None

        banner = _format_progress_banner(
            completed_count=artifact_id,
            total_count=total_artifacts,
            current_artifact_number=current_number,
            current_artifact_name=current_name,
            next_artifact_number=next_number,
            next_artifact_name=next_name,
        )
        return {"messages": [AIMessage(content=banner)]}

    return node


def init_node(
    state: TheodorAgentState,
    config: RunnableConfig,
    runtime: Runtime[TheodorAgentContext],
) -> TheodorAgentState:
    if state.get("current_artifact_id") is None:
        state["current_artifact_id"] = 0
    if state.get("current_artifact_state") is None:
        state["current_artifact_state"] = ArtifactStage.INIT

    if not state.get("user_prompt"):
        prompt = None
        if runtime and runtime.context and runtime.context.get("user_prompt"):
            prompt = runtime.context["user_prompt"]
        if not prompt:
            for msg in reversed(state.get("messages") or []):
                if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                    content = msg.content
                    prompt = content if isinstance(content, str) else str(content)
                    break
        if prompt:
            state["user_prompt"] = prompt
    return state


def _route_by_artifact(state: TheodorAgentState) -> str:
    artifact_id = state.get("current_artifact_id", 0)
    if artifact_id is None:
        artifact_id = 0
    artifact_id = max(0, min(int(artifact_id), len(ARTIFACTS) - 1))
    return f"progress_{artifact_id}"


def _after_choice_route(state: TheodorAgentState) -> str:
    artifact_id = state.get("current_artifact_id", 0)
    stage = state.get("current_artifact_state")
    if stage == ArtifactStage.ARTIFACT_CONFIRMED:
        if artifact_id >= len(ARTIFACTS) - 1:
            return "final_output"
        return "advance"
    return "end"


def advance_node(
    state: TheodorAgentState,
    config: RunnableConfig,
    runtime: Runtime[TheodorAgentContext],
) -> TheodorAgentState:
    current_id = state.get("current_artifact_id", 0)
    next_id = min(current_id + 1, len(ARTIFACTS) - 1)
    return {
        "current_artifact_id": next_id,
        "current_artifact_state": ArtifactStage.INIT,
    }


def final_output_node(
    state: TheodorAgentState,
    config: RunnableConfig,
    runtime: Runtime[TheodorAgentContext],
) -> TheodorAgentState:
    if state.get("artifacts"):
        out_path = store_artifacts(state.get("artifacts") or {})
        template = _LOCALE_TEXT[_CURRENT_LOCALE]["final_report"]
        return {"messages": [AIMessage(content=template.format(url=out_path))]}
    else:
        return {"messages": [AIMessage(content=_LOCALE_TEXT[_CURRENT_LOCALE]["store_report_error"])]}


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
    locale: str = DEFAULT_LOCALE,
    checkpoint_saver=None,
    *,
    streaming: bool = True,
):
    global _CURRENT_LOCALE
    _CURRENT_LOCALE = resolve_locale(locale)
    set_global_locale(_CURRENT_LOCALE)
    log_name = f"theodor_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [StreamWriterCallbackHandler(), json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        _ = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        callback_handlers += [CallbackHandler()]

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()

    builder = StateGraph(TheodorAgentState)
    builder.add_node("init", init_node)

    total_artifacts = len(ARTIFACTS)
    for artifact in ARTIFACTS:
        artifact_id = int(artifact["id"])
        progress_node_name = f"progress_{artifact_id}"
        choice_node_name = f"choice_agent_{artifact_id}"
        choice_agent = build_choice_agent(
            provider=provider,
            role=role,
            use_platform_store=use_platform_store,
            notify_on_reload=notify_on_reload,
            artifact_id=artifact_id,
            locale=locale,
            streaming=streaming,
        )
        builder.add_node(progress_node_name, create_progress_node(artifact_id))
        builder.add_node(choice_node_name, choice_agent)
        builder.add_edge(progress_node_name, choice_node_name)
        builder.add_conditional_edges(
            choice_node_name,
            _after_choice_route,
            {
                "advance": "advance",
                "final_output": "final_output",
                "end": END,
            },
        )

    builder.add_node("advance", advance_node)
    builder.add_node("final_output", final_output_node)

    builder.add_edge(START, "init")
    builder.add_conditional_edges(
        "init",
        _route_by_artifact,
        {f"progress_{idx}": f"progress_{idx}" for idx in range(total_artifacts)},
    )
    builder.add_conditional_edges(
        "advance",
        _route_by_artifact,
        {f"progress_{idx}": f"progress_{idx}" for idx in range(total_artifacts)},
    )
    builder.add_edge("final_output", END)

    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph
