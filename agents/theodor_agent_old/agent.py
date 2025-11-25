from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt, Command

from agents.utils import ModelType, get_llm
from .retrievers.retriever_utils import get_search_tool
from platform_utils.llm_logger import JSONFileTracer

from .prompts.prompts import ARTIFACTS, PROGRESS_BAR_TEMPLATE, SYSTEM_PROMPT
from .state.state import ArtifactState, TheodorAgentState

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _get_progress_bar(current_index: int) -> str:
    total = len(ARTIFACTS)
    filled = "■" * current_index
    empty = "□" * (total - current_index)
    
    current_name = ARTIFACTS[current_index]["name"] if current_index < total else "Финиш"
    next_name = ARTIFACTS[current_index + 1]["name"] if current_index + 1 < total else "Нет"
    
    return PROGRESS_BAR_TEMPLATE.format(
        progress_bar=filled + empty,
        current=current_index,
        total=total,
        current_name=current_name,
        next_name=next_name
    )


def _is_confirmation(text: str) -> bool:
    """
    Lightweight LLM-based classifier to judge approval/confirmation.
    Falls back to keyword match if the LLM call fails.
    """
    normalized = text.lower().strip().strip(".,!?;")
    try:
        clf = get_llm(model="mini", provider="openai", temperature=0)
        system = SystemMessage(
            content=(
                "You are a classifier. Determine whether the user's reply explicitly approves/"
                "confirms moving forward. Respond with one token: APPROVE or REJECT."
            )
        )
        user = HumanMessage(content=f"User reply: {text}")
        result = clf.invoke([system, user]).content.strip().upper()
        return "APPROVE" in result or "CONFIRM" in result
    except Exception as exc:
        logging.warning("Confirmation classifier failed, using fallback. Error: %s", exc)
        approved_inputs = {"подтверждаю", "да", "дальше", "approve", "ок", "ok", "+", "true"}
        return normalized in approved_inputs


def _initialize_state(state: TheodorAgentState) -> Dict[str, Any]:
    """Initialize artifacts if not present."""
    if not state.get("artifacts"):
        artifacts = {}
        for idx, art_def in enumerate(ARTIFACTS):
            artifacts[idx] = {  # Use index as key, not art_def["id"]
                "id": art_def["id"],
                "name": art_def["name"],
                "status": "PENDING",
                "current_content": "",
                "history": []
            }
        return {
            "artifacts": artifacts,
            "current_step_index": 0,
            "waiting_for_confirmation": False
        }
    return {}


def route_step(state: TheodorAgentState) -> Literal["generate_artifact", "human_review", "finalize_step"]:
    """Decide the next step based on state."""
    idx = state.get("current_step_index", 0)
    if idx >= len(ARTIFACTS):
        return "finalize_step"  # Or some end state

    artifacts = state.get("artifacts", {})
    current_artifact = artifacts.get(idx)
    
    if not current_artifact:
        return "generate_artifact" # Should not happen if initialized

    status = current_artifact["status"]
    
    if status == "PENDING":
        return "generate_artifact"
    
    if status == "ACTIVE":
        # Check if user provided input that looks like confirmation
        # But strictly, we want the LLM to decide if it's ready for confirmation
        # For now, let's go to generate/refine
        return "generate_artifact"

    if status == "READY_FOR_CONFIRM":
        return "human_review"

    return "generate_artifact"


def generate_artifact(state: TheodorAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate or refine the artifact content."""
    idx = state.get("current_step_index", 0)
    if idx >= len(ARTIFACTS):
        # Nothing to generate; stop gracefully.
        return Command(
            goto=END,
            update={"messages": [AIMessage(content="Все артефакты уже подготовлены.")]}
        )

    artifacts = state["artifacts"]
    current_artifact = artifacts[idx]
    art_def = ARTIFACTS[idx]
    
    # Prepare context
    messages = state["messages"]
    
    # If it's the very first interaction for this artifact
    if current_artifact["status"] == "PENDING":
        # Generate initial options
        prompt = (
            f"Мы переходим к этапу {idx + 1}: {art_def['name']}.\n"
            f"Цель: {art_def['goal']}\n"
            f"Методология: {art_def['methodology']}\n"
            f"Критерии: {', '.join(art_def['criteria'])}\n\n"
            "Предложи 2-3 варианта (A/B/C) для этого артефакта, основываясь на предыдущем контексте."
        )
        # Update status to ACTIVE
        new_status = "ACTIVE"
    else:
        # Refine based on user input
        prompt = (
            f"Мы работаем над этапом {idx + 1}: {art_def['name']}.\n"
            "Пользователь прислал ответ. Если это выбор варианта или правки — обнови артефакт.\n"
            "Если пользователь пишет 'подтверждаю', 'ок', 'дальше' — проверь, готов ли артефакт.\n"
            "Если готов — выведи финальную версию и спроси явное подтверждение (переведи статус в READY_FOR_CONFIRM).\n"
            "Если не готов — продолжи уточнение."
        )
        new_status = "ACTIVE" # Default, might change logic below

    # Call LLM (bind search tool for KB lookups)
    #search_kb = get_search_tool()
    llm = get_llm(model="base", provider="openai")#.bind_tools([search_kb])
    
    system_msg = SystemMessage(content=SYSTEM_PROMPT + "\n\n" + _get_progress_bar(idx))
    base_messages = [system_msg] + messages + [HumanMessage(content=prompt)]

    message_history = list(base_messages)
    response = llm.invoke(message_history)

    # Detect if LLM is asking for confirmation
    needs_confirmation = "подтверждаете" in response.content.lower() or "подтверди" in response.content.lower()

    if needs_confirmation:
        new_status = "READY_FOR_CONFIRM"

    updated_artifact = current_artifact.copy()
    updated_artifact["status"] = new_status
    updated_artifact["current_content"] = response.content

    updates: Dict[str, Any] = {
        "messages": [response],
        "artifacts": {idx: updated_artifact},
    }

    # Move to dedicated user-interaction node so the LLM call doesn't repeat on resume
    return Command(goto="await_user_input", update=updates)


def await_user_input(state: TheodorAgentState) -> Command:
    """Handle all user interaction (approval or feedback) via a single interrupt."""
    idx = state.get("current_step_index", 0)
    artifacts = state.get("artifacts", {})
    current_artifact = artifacts.get(idx)
    if current_artifact is None:
        return Command(goto="generate_artifact")

    awaiting_confirmation = current_artifact["status"] == "READY_FOR_CONFIRM"

    interrupt_payload = {
        "type": "confirmation" if awaiting_confirmation else "feedback",
        "artifact_id": idx,
        "artifact_name": current_artifact["name"],
        "content": current_artifact["current_content"],
        "question": "Подтвердите артефакт или внесите правки" if awaiting_confirmation else "Добавьте правки или выберите вариант",
    }

    user_response_raw = interrupt(interrupt_payload)

    # Always record the human message so context is preserved
    message_update = [HumanMessage(content=str(user_response_raw))]

    is_approval = _is_confirmation(str(user_response_raw)) if awaiting_confirmation else False

    if awaiting_confirmation and is_approval:
        return Command(
            goto="finalize_step",
            update={
                "messages": message_update,
            },
        )

    # Treat any non-approval as a request for refinement: resume generate loop
    updated_artifact = current_artifact.copy()
    if awaiting_confirmation:
        updated_artifact["status"] = "ACTIVE"

    return Command(
        goto="generate_artifact",
        update={
            "messages": message_update,
            "artifacts": {idx: updated_artifact},
        },
    )


def finalize_step(state: TheodorAgentState) -> Dict[str, Any]:
    """Mark step as approved and move to next."""
    idx = state["current_step_index"]
    artifacts = state["artifacts"]
    current_artifact = artifacts[idx]
    
    updated_artifact = current_artifact.copy()
    updated_artifact["status"] = "APPROVED"
    
    next_idx = idx + 1
    
    msg = AIMessage(content=f"✅ Артефакт '{current_artifact['name']}' подтверждён. Переходим к следующему этапу.")
    
    return {
        "current_step_index": next_idx,
        "artifacts": {idx: updated_artifact},
        "messages": [msg],
    }


def should_continue(state: TheodorAgentState) -> Literal["human_review", "finalize_step", "wait_for_user"]:
    idx = state.get("current_step_index", 0)
    artifacts = state.get("artifacts", {})
    current_artifact = artifacts.get(idx)
    
    if not current_artifact:
        return "wait_for_user"

    # If we just came from human_review and user confirmed (waiting_for_confirmation is False)
    # But wait, human_review logic above is a bit tricky.
    
    # Let's simplify:
    # If status is READY_FOR_CONFIRM, we want to INTERRUPT.
    # After interrupt, we check user input.
    
    messages = state["messages"]
    print(f"DEBUG: Messages count: {len(messages)}")
    for i, m in enumerate(messages):
        print(f"DEBUG: Msg {i}: {type(m).__name__} - {repr(m.content[:50])}")
    
    if messages:
        last_msg = messages[-1]
        print(f"DEBUG: Last message type: {type(last_msg)}, langchain_type: {last_msg.type}, content: {repr(last_msg.content)}")
        print(f"DEBUG: Is HumanMessage? {isinstance(last_msg, HumanMessage)}")
    
    # Check if last message is from human (using type check as fallback)
    if messages and (isinstance(messages[-1], HumanMessage) or messages[-1].type == "human"):
        user_text = messages[-1].content.lower().strip().strip(".,!?;")
        print(f"DEBUG: Checking confirmation. User text: '{user_text}', Status: {current_artifact['status']}")
        if current_artifact["status"] == "READY_FOR_CONFIRM":
            if user_text in ["подтверждаю", "да", "дальше", "approve", "ок", "ok", "+"]:
                print("DEBUG: Confirmation MATCHED! Returning finalize_step")
                return "finalize_step"
            else:
                print("DEBUG: Confirmation NOT matched. Returning wait_for_user")
                # User rejected or asked for changes
                return "wait_for_user"
    
    if current_artifact["status"] == "READY_FOR_CONFIRM":
        print("DEBUG: Status is READY_FOR_CONFIRM. Returning human_review")
        return "human_review" # This will be the interrupt point
        
    print("DEBUG: Default case. Returning wait_for_user")
    return "wait_for_user"


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
):
    log_name = f"theodor_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]

    memory = MemorySaver() # Always use memory for this agent

    builder = StateGraph(TheodorAgentState)
    
    builder.add_node("initialize", _initialize_state)
    builder.add_node("generate_artifact", generate_artifact)
    builder.add_node("await_user_input", await_user_input)
    builder.add_node("finalize_step", finalize_step)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "generate_artifact")
    
    # generate_artifact routes to the user-interaction node
    builder.add_edge("generate_artifact", "await_user_input")
    builder.add_edge("generate_artifact", END)
    builder.add_edge("await_user_input", "generate_artifact")
    builder.add_edge("await_user_input", "finalize_step")

    # After finalizing, generate the next artifact (if any)
    builder.add_edge("finalize_step", "generate_artifact")

    # Compile with checkpointer (no static interrupts - using interrupt() function instead)
    graph = builder.compile(
        checkpointer=memory
    ).with_config({"callbacks": callback_handlers})

    return graph
