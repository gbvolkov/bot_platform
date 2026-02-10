from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config
from agents.utils import ModelType, get_llm, extract_text
from platform_utils.llm_logger import JSONFileTracer

from ..tools.yandex_search import YandexSearchTool as SearchTool


from .artifacts_defs import ARTIFACTS
from .locales import DEFAULT_LOCALE, resolve_locale, set_locale as set_global_locale

from .prompts import get_generation_prompt, get_summary_prompt
from .state import ArtifactStage, TheodorAgentContext, TheodorAgentState
from .tools import commit_artifact_final_text

LOG = logging.getLogger(__name__)

_yandex_tool = SearchTool(
    api_key=config.YA_API_KEY,
    folder_id=config.YA_FOLDER_ID,
    max_results=3,
    summarize=True
)

_LOCALE_TEXT = {
    "en": {
        "options_confirm_question": "Confirm the options with one word \"confirm\" or describe changes.",
        "select_question": "Which option do you choose? Reply with A/B/C.",
        "confirm_question": "Confirm this artifact or describe changes.",
        "no_previous_artifacts": "No previous artifacts yet.",
        "not_finalized": "(not finalized)",
        "edit_keywords": [
            "edit",
            "change",
            "update",
            "revise",
            "adjust",
            "modify",
            "fix",
            "redo",
            "rework",
            "not right",
            "not correct",
        ],
        "confirm_keywords": [
            "confirm",
            "confirmed",
            "yes",
            "ok",
            "okay",
            "approve",
            "next",
            "go on",
            "continue",
            "looks good",
            "all good",
            "fine",
        ],
        "option_label": "Option {label}",
    },
    "ru": {
        "options_confirm_question": "Подтвердите варианты одним словом «подтверждаю» или напишите замечания.",
        "select_question": "Какой вариант выбираете? Ответьте A/B/C.",
        "confirm_question": "Подтвердите артефакт или скажите, что нужно изменить.",
        "no_previous_artifacts": "Предыдущих артефактов пока нет.",
        "not_finalized": "(не завершено)",
        "edit_keywords": [
            "исправ",
            "поправ",
            "поменя",
            "измен",
            "не так",
            "не то",
            "добав",
            "передел",
        ],
        "confirm_keywords": [
            "подтверждаю",
            "подтверждаюю",
            "подтвердить",
            "да",
            "ок",
            "окей",
            "согласен",
            "approve",
            "дальше",
            "верно",
            "хорошо",
            "норм",
        ],
        "option_label": "Вариант {label}",
    },
}

_CURRENT_LOCALE = DEFAULT_LOCALE


def _safe_stream_writer():
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None


class StreamWriterCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        if not token:
            return
        writer = _safe_stream_writer()
        writer({"type": "user_delta", "text": token})

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


def _artifact_definition(artifact_id: int) -> Dict[str, Any]:
    if 0 <= artifact_id < len(ARTIFACTS):
        return ARTIFACTS[artifact_id]
    return {"id": artifact_id, "name": f"Artifact {artifact_id + 1}"}


def _build_context_str(artifacts: Dict[int, Any]) -> str:
    if not artifacts:
        return _text("no_previous_artifacts")
    parts: List[str] = []
    for artifact_id in sorted(artifacts):
        details = artifacts.get(artifact_id) or {}
        definition = details.get("artifact_definition") or _artifact_definition(artifact_id)
        name = definition.get("name") or f"Artifact {artifact_id + 1}"
        text = (details.get("artifact_final_text") or "").strip()
        if not text:
            text = _text("not_finalized")
        parts.append(f"{artifact_id + 1}. {name}\n{text}")
    return "\n\n---\n\n".join(parts)


def _text(key: str) -> str:
    return _LOCALE_TEXT[_CURRENT_LOCALE][key]

def _build_artifact_agent(
        model: BaseChatModel, 
        summarization_model: BaseChatModel, 
        artifact_id: int,
    ):
    summarization_model = summarization_model or model
    _summarizator = SummarizationMiddleware(
            model=summarization_model,
            trigger=("tokens", 80000),
            keep=("messages", 20),
            summary_prompt=get_summary_prompt(_CURRENT_LOCALE),
            )
    _artifact_id = artifact_id
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state = request.state
        artifacts = state.get("artifacts") or {}
        definition = _artifact_definition(_artifact_id)
        previous_options = (artifacts.get(_artifact_id) or {}).get("artifact_options_text", "")
        components = "\n".join(f"- {item}" for item in definition.get("components", []) or [])
        criteria = "\n".join(f"- {item}" for item in definition.get("criteria", []) or [])
        return get_generation_prompt(
            artifact_id=_artifact_id,
            artifact_name=str(definition.get("name") or f"Artifact {_artifact_id + 1}"),
            goal=str(definition.get("goal") or ""),
            methodology=str(definition.get("methodology") or ""),
            components=components,
            criteria=criteria,
            data_source=str(definition.get("data_source") or ""),
            context_str=_build_context_str(artifacts),
            user_prompt=str(state.get("user_prompt") or ""),
            previous_options_text=str(previous_options or ""),
            locale=_CURRENT_LOCALE,
        )


    return create_agent(
        model=model,
        tools=[
            commit_artifact_final_text, 
            _yandex_tool
        ],
        middleware=[
            _summarizator,
            build_prompt,        
        ],
        state_schema=TheodorAgentState,
        context_schema=TheodorAgentContext,
    )

def create_generate_artifact_node(
        model: BaseChatModel, 
        summarization_model: BaseChatModel,
        artifact_id: int,
    ):
    _run_agent = _build_artifact_agent(
        model=model, 
        summarization_model=summarization_model, 
        artifact_id=artifact_id
    )
    def run_node(
        state: TheodorAgentState,
        config: RunnableConfig,
        runtime: Runtime[TheodorAgentContext],
    ) -> TheodorAgentState:
        
        result = _run_agent.invoke(state, config=config, context=runtime.context)
        return result

    return _run_agent

def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
    artifact_id: int = 0,
    locale: str = DEFAULT_LOCALE,
    checkpoint_saver=None,
    *,
    streaming: bool = True,
):
    global _CURRENT_LOCALE
    _CURRENT_LOCALE = resolve_locale(locale)
    set_global_locale(_CURRENT_LOCALE)
    log_name = f"new_theodor_choice_{time.strftime('%Y%m%d%H%M')}"
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
    llm = get_llm(model="base", provider=provider.value, temperature=0.4, streaming=streaming)
    summary_llm = get_llm(model="mini", provider=provider.value, temperature=0.0, streaming=False)

    builder = StateGraph(TheodorAgentState)
    builder.add_node("generate_aftifact", create_generate_artifact_node(
        model=llm, 
        summarization_model=summary_llm, 
        artifact_id=artifact_id))

    builder.add_edge(START, "generate_aftifact")
    builder.add_edge("generate_aftifact", END)

    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph
