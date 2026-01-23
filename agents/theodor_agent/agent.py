from __future__ import annotations

import logging
import time
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langgraph.runtime import Runtime

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config

from agents.utils import ModelType, get_llm
from platform_utils.llm_logger import JSONFileTracer

from .artifacts_defs import (
    ARTIFACTS, 
    #ArtifactDetails,
    #ArtifactOptions,
    ArtifactState,
    ArtifactAgentState,
    ArtifactAgentContext,
    #AftifactFinalText,
)
from .choice_agent import initialize_agent as build_choice_agent
from .locales import DEFAULT_LOCALE, resolve_locale, set_locale as set_global_locale
from .context_reduction import (
    ARTIFACT_SUMMARY_TAG,
    build_pruned_history,
    extract_text,
    last_summary_index,
    summarize_artifact_discussion,
)
from .store_artifacts import store_artifacts


_CONFIRMED_BANNER_BORDER = "────────────────────────────────"
_PROGRESS_BANNER_BORDER = "════════════════════════════════"


def _format_confirmed_banner(
    *,
    artifact_number: int,
    artifact_name: str,
    next_artifact_number: Optional[int],
    next_artifact_name: Optional[str],
    locale: str = DEFAULT_LOCALE,
) -> str:
    locale_key = resolve_locale(locale)
    if locale_key == "en":
        if next_artifact_number is not None and next_artifact_name:
            next_line = f"Next: Artifact {next_artifact_number} - {next_artifact_name}"
        else:
            next_line = "Next: finish"
        confirmed_line = f"✅ ARTIFACT CONFIRMED: Artifact {artifact_number} - {artifact_name}"
    else:
        if next_artifact_number is not None and next_artifact_name:
            next_line = f"Перейти дальше: к Артефакту {next_artifact_number} — {next_artifact_name}"
        else:
            next_line = "Перейти дальше: завершение"
        confirmed_line = f"✅ АРТЕФАКТ ПОДТВЕРЖДЁН: Артефакт {artifact_number} — {artifact_name}"

    return "\n".join(
        [
            _CONFIRMED_BANNER_BORDER,
            confirmed_line,
            next_line,
            _CONFIRMED_BANNER_BORDER,
        ]
    )

def _format_progress_banner(
    *,
    completed_count: int,
    total_count: int,
    current_artifact_number: int,
    current_artifact_name: str,
    next_artifact_number: Optional[int],
    next_artifact_name: Optional[str],
    locale: str = DEFAULT_LOCALE,
) -> str:
    completed_count = max(0, min(completed_count, total_count))
    total_count = max(0, total_count)
    bar = ("■" * completed_count) + ("□" * max(total_count - completed_count, 0))

    locale_key = resolve_locale(locale)
    if locale_key == "en":
        if next_artifact_number is not None and next_artifact_name:
            next_line = f"NEXT: Artifact {next_artifact_number} - {next_artifact_name}"
        else:
            next_line = "NEXT: finish"
        progress_line = f"PROGRESS: {bar} ({completed_count}/{total_count})"
        current_line = f"CURRENT: Artifact {current_artifact_number} - {current_artifact_name}"
    else:
        if next_artifact_number is not None and next_artifact_name:
            next_line = f"СЛЕДУЮЩИЙ: Артефакт {next_artifact_number} — {next_artifact_name}"
        else:
            next_line = "СЛЕДУЮЩИЙ: завершение"
        progress_line = f"ПРОГРЕСС: {bar} ({completed_count}/{total_count})"
        current_line = f"ТЕКУЩИЙ: Артефакт {current_artifact_number} — {current_artifact_name}"

    return "\n".join(
        [
            _PROGRESS_BANNER_BORDER,
            progress_line,
            current_line,
            next_line,
            _PROGRESS_BANNER_BORDER,
        ]
    )

def create_append_banner_node(*, content: str):
    def _node(state: ArtifactAgentState, config: RunnableConfig) -> ArtifactAgentState:
        return Command(update={"messages": [AIMessage(content=content)]})

    return _node


def init_node(state: ArtifactAgentState, 
              config: RunnableConfig) -> ArtifactAgentState:
    if not state["messages"]:
        return state          
    last_user_msg = state["messages"][-1]
    # We only augment on real user turns
    if last_user_msg.type != "human":
        return state
    content = last_user_msg.content
    #content = last_user_msg.content if isinstance(last_user_msg.content, str) else (last_user_msg.content.get("text", str(last_user_msg.content))) last_user_msg.content.get("text", str(last_user_msg.content))
    state["user_prompt"] = content if isinstance(content, str) else (content[0].get("text", str(content[0])) if isinstance(content, list) else str(content))
    state["current_artifact_state"] = ArtifactState.INIT
    state["current_artifact_id"] = 0
    #state["user_info"] = config.
    return state


def create_post_choice_cleanup_node(
    *,
    artifact_id: int,
    artifact_name: str,
    summary_model: BaseChatModel,
    keep_first_user_messages: int = 5,
    keep_last_messages: int = 5,
):
    def _node(
            state: ArtifactAgentState,
            config: RunnableConfig,
            runtime: Runtime[ArtifactAgentContext],
        ) -> ArtifactAgentState:
        messages = state.get("messages") or []
        artifacts = state.get("artifacts") or {}
        details = artifacts.get(artifact_id) or {}

        user_prompt = (state.get("user_prompt") or "")
        if not isinstance(user_prompt, str):
            user_prompt = str(user_prompt)
        user_prompt = user_prompt.strip()
        if not user_prompt and runtime is not None and runtime.context is not None:
            user_prompt = str(runtime.context.get("user_prompt") or "").strip()

        selected_idx = details.get("selected_option", -1)
        selected_label = ""
        selected_text = ""
        try:
            if selected_idx is not None and int(selected_idx) >= 0:
                selected_idx = int(selected_idx)
                selected_label = chr(ord("A") + selected_idx)
                options = details.get("artifact_options") or []
                if 0 <= selected_idx < len(options):
                    selected_text = str((options[selected_idx] or {}).get("artifact_option") or "").strip()
        except Exception:  # noqa: BLE001
            selected_label = ""
            selected_text = ""

        window_start = last_summary_index(messages, exclude_artifact_id=artifact_id)
        window = messages[window_start + 1 :] if window_start >= 0 else list(messages)
        user_notes: List[str] = []
        for msg in window:
            if getattr(msg, "type", "") != "human":
                continue
            note = extract_text(getattr(msg, "content", ""))
            if not note:
                continue
            if user_prompt and note.strip() == user_prompt:
                continue
            if len(note) > 4000:
                note = f"{note[:3999].rstrip()}…"
            user_notes.append(note)

        summary_text = summarize_artifact_discussion(
                    model=summary_model,
                    artifact_id=artifact_id,
                    artifact_name=artifact_name,
                    user_prompt=user_prompt,
                    selected_option_label=selected_label,
                    selected_option_text=selected_text,
                    user_notes=user_notes,
                ).strip() or "- Artifact completed; no additional notes captured."

        summary_message = SystemMessage(
            content=f"Artifact {artifact_id} — {artifact_name}\n{summary_text}",
            additional_kwargs={
                "type": ARTIFACT_SUMMARY_TAG,
                "artifact_id": artifact_id,
                "artifact_name": artifact_name,
            },
        )

        pruned_history = build_pruned_history(
            messages=messages,
            keep_first_user_messages=keep_first_user_messages,
            keep_last_messages=keep_last_messages,
            drop_summary_for_artifact_id=artifact_id,
        )

        updated_details = dict(details)
        updated_details["artifact_summary"] = summary_text

        return Command(
            update={
                "messages": [RemoveMessage(id="__remove_all__"), *pruned_history, summary_message],
                "artifacts": {artifact_id: updated_details},
            }
        )

    return _node        



def generate_final_output_node(
    state: ArtifactAgentState,
    config: RunnableConfig,
    runtime: Runtime[ArtifactAgentContext],
) -> ArtifactAgentState:
    out_path = store_artifacts(state["artifacts"] or {})
    locale_key = resolve_locale()
    link_text = "Final report available here" if locale_key == "en" else "Финальный отчет доступен здесь"
    return {"messages": [AIMessage(content=f"[{link_text}]({out_path})")]}


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
    locale: str = DEFAULT_LOCALE,
    checkpoint_saver=None,
):
    locale_key = set_global_locale(locale)
    log_name = f"theo_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL
        )
        lf_handler = CallbackHandler()
        callback_handlers += [lf_handler]

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()

    summary_llm = get_llm(model="mini", provider=provider.value, temperature=0)

    builder = StateGraph(ArtifactAgentState)

    builder.add_node("init", init_node)
    prev_node = "init"

    builder.add_edge(START, "init")
    total_artifacts = len(ARTIFACTS)

    for artifact in ARTIFACTS:
        artifact_id = int(artifact["id"])
        artifact_number = artifact_id + 1
        artifact_name = str(artifact.get("name") or f"artifact_{artifact_id}")

        next_def = ARTIFACTS[artifact_id + 1] if artifact_id + 1 < total_artifacts else None
        next_number = int(next_def["id"]) + 1 if next_def else None
        next_name = str(next_def.get("name") or f"artifact_{next_def['id']}") if next_def else None

        progress_node = f"progress_banner_{artifact_id}"
        _choice_agent = build_choice_agent(
            provider=provider,
            role=role,
            use_platform_store=use_platform_store,
            notify_on_reload=notify_on_reload,
            artifact_id=artifact["id"],
            uset_parental_memory=True,
            locale=locale_key,
        )
        choice_node = f"choice_agent_{artifact['id']}"
        cleanup_node = f"cleanup_{artifact['id']}"
        confirmed_node = f"confirmed_banner_{artifact_id}"

        builder.add_node(
            progress_node,
            create_append_banner_node(
                content=_format_progress_banner(
                    completed_count=artifact_id,
                    total_count=total_artifacts,
                    current_artifact_number=artifact_number,
                    current_artifact_name=artifact_name,
                    next_artifact_number=next_number,
                    next_artifact_name=next_name,
                    locale=locale_key,
                )
            ),
        )

        builder.add_node(choice_node, _choice_agent)
        builder.add_node(
            cleanup_node,
            create_post_choice_cleanup_node(
                artifact_id=artifact["id"],
                artifact_name=artifact.get("name", f"artifact_{artifact['id']}"),
                summary_model=summary_llm,
                keep_first_user_messages=5,
                keep_last_messages=5,
            ),
        )
        builder.add_node(
            confirmed_node,
            create_append_banner_node(
                content=_format_confirmed_banner(
                    artifact_number=artifact_number,
                    artifact_name=artifact_name,
                    next_artifact_number=next_number,
                    next_artifact_name=next_name,
                    locale=locale_key,
                )
            ),
        )

        builder.add_edge(prev_node, progress_node)
        builder.add_edge(progress_node, choice_node)
        builder.add_edge(choice_node, cleanup_node)
        builder.add_edge(cleanup_node, confirmed_node)
        prev_node = confirmed_node

    builder.add_node("final_output", generate_final_output_node)

    builder.add_edge(prev_node, "final_output")
    builder.add_edge("final_output", END)

    graph = builder.compile(
        checkpointer=memory,
        debug=False
    ).with_config({"callbacks": callback_handlers})

    return graph

if __name__ == "__main__":
    print("Initializing Theodor Agent...")
    agent_graph = initialize_agent()
    user_llm = get_llm(model="mini", provider="openai")
    logging.info("Simulation started")

    config = {"configurable": {"thread_id": "simulation_thread_1"}}

    next_input: Any = {
        "messages": [
            HumanMessage(
                content=(
                    "Привет! У меня есть идея стартапа: Uber для выгула собак. "
                    "Помоги мне проработать её."
                )
            )
        ]
    }

    max_steps = 50  # Safety limit
    #print("\n--- STARTING SIMULATION ---\n")
    logging.info("\n--- STARTING SIMULATION ---\n")

    ctx = ArtifactAgentContext(user_prompt = "Привет! У меня есть идея стартапа: Uber для выгула собак. Помоги мне проработать её.",
                               generated_artifacts = [])


    while True:
        result = agent_graph.invoke(next_input, config=config, context=ctx)
        #res = agent_graph.get_state(config, subgraphs=True)
        state_snapshot = agent_graph.get_state(config)
        current_idx = state_snapshot.values.get("current_step_index", 0)
        logging.info("Current step index: %s", current_idx)
        if interrupts := result.get("__interrupt__"):
            payload = getattr(interrupts[-1], "value", interrupts[-1])
            last_ai = payload.get("content", "")
            #print(f"Interrupt payload: {payload}")
            prompt = f"Ответ наставника:\n{last_ai}\n\nТвой ответ:"
            user_reply = input(prompt)
            if user_reply == "exit":
                break
            next_input = Command(resume=user_reply)
            continue
        else:
            last_ai = ""
            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        last_ai = msg.content
                        break
            if last_ai:
                #print(f"Bot: {last_ai}")
                logging.info("Bot: %s", last_ai)
            #break

    print("ФСЁ")
