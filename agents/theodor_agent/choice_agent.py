from __future__ import annotations
import functools

import re
import logging
import time
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    ToolStrategy,
    StructuredOutputValidationError,
)
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

from .prompts.prompts import (
    get_format_options_prompt,
    get_system_prompt,
    get_tool_policy_prompt,
)
from .locales import DEFAULT_LOCALE, resolve_locale, set_locale as set_global_locale

from ..tools.yandex_search import YandexSearchTool as SearchTool
from ..tools.think import ThinkTool
from agents.llm_utils import (
    build_model_fallback_middleware,
    get_llm,
    with_llm_fallbacks,
)
from agents.utils import ModelType
from agents.prettifier import prettify
from platform_utils.llm_logger import JSONFileTracer
from .artifacts_defs import (
    ARTIFACTS, 
    ArtifactDetails,
    ArtifactOptions,
    ArtifactState,
    ArtifactAgentState,
    ArtifactAgentContext,
    AftifactFinalText,
    get_artifact_schemas,
)
from agents.structured_prompt_utils import provider_then_tool

DEBUG = False
def debug_log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG:
            print(f"DEBUG: Enter the function {func.__name__}")
        try:
            return func(*args, **kwargs)
        finally:
            if DEBUG:
                print(f"DEBUG: Exit the function {func.__name__}")
    return wrapper

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from prompts.prompts import FORMAT_INSTRUCTION_EN, FORMAT_INSTRUCTION_RU

CHOICE_LOCALES = {
    "ru": {
        "artifact_placeholder": "В работе",
        "data_source_default": "Ответы пользователя",
        "select_option_question": "Выберите один из предложенных вариантов или скажите, что нужно поправить.",
        "confirm_question": "Подтвердите артефакт \"{artifact_name}\" или скажите, что нужно изменить.",
        "format_instruction": FORMAT_INSTRUCTION_RU,
        "options_prompt": (
            "Мы прорабатываем: {user_prompt}\n\n"
            "====================================================================================\n#Артефакты:\n"
            "{context_str}\n"
            "====================================================================================\n#Конец артефактов.\n\n"
            "Мы находимся на этапе {stage}.\n"
            "Цель этапа: {stage_goal}.\n"
            "Мы переходим к артефакту {artifact_number}: {artifact_name}.\n"
            "Цель: {goal}\n"
            "Компоненты: {components}\n"
            "Методология: {methodology}\n"
            "Критерии: {criteria}\n\n"
            "Реальные данные: {data_source}\n\n"
            "Предложи несколько вариантов для этого артефакта, основываясь на предыдущем контексте.\n\n"
            "Каждый вариант должен удовлетворять всем критериям.\n\n"
            "Для каждого варианта дай высокоуровневую оценку по каждому критерию.\n"
            "##ОГРАНИЧЕНИЯ:\n"
            "**ВАЖНО**: Ты работаешь ТОЛЬКО с артефактом {artifact_number}: {artifact_name}.\n\n"
            "**ВАЖНО**: Тебе **ЗАПРЕЩЕНО** формировать версии, готовить варианты или даже просто обсуждать любые другие артефакты, кроме {artifact_number}: {artifact_name}.\n"
        ),
        "final_prompt": (
            "Мы работаем над этапом {artifact_number}: {artifact_name}.\n"
            "Цель: {goal}\n"
            "Методология: {methodology}\n"
            "Реальные данные: {data_source}\n\n"
            "====================================================================================\n#Артефакты:\n"
            "{context_str}\n"
            "====================================================================================\n\n"
            "Пользователь прислал ответ.\n"
            "Вариант {artifact_number}: {selected_option}\n"
            "ВСЕГДА формируй **полную** финальную версию артефакта.\n"
            "Обязательно включай оценку по каждому из критериев.\n"
            "Список критериев: {criteria}\n"
            "ВАЖНО: В поле artifact_final_text указывай только финальный текст артефакта - без рассуждений и ответа пользователю!\n"
            "##ОГРАНИЧЕНИЯ:\n"
            "**ВАЖНО**: Ты работаешь ТОЛЬКО с артефактом {artifact_number}: {artifact_name}.\n\n"
            "**ВАЖНО**: Тебе **ЗАПРЕЩЕНО** формировать версии, готовить варианты или даже просто обсуждать любые другие артефакты, кроме {artifact_number}: {artifact_name}.\n"
        ),
        "keywords": {
            "confirm": [
                "подтверждаю",
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
            "edit": [
                "исправ",
                "поправ",
                "поменя",
                "измен",
                "не так",
                "не то",
                "добав",
                "передел",
            ],
        },
    },
    "en": {
        "artifact_placeholder": "In progress",
        "data_source_default": "User responses",
        "select_option_question": "Choose one of the proposed options or say what needs to be revised.",
        "confirm_question": "Confirm the artifact \"{artifact_name}\" or say what needs to change.",
        "format_instruction": FORMAT_INSTRUCTION_EN,
        "options_prompt": (
            "We are working on: {user_prompt}\n\n"
            "====================================================================================\n#Artifacts:\n"
            "{context_str}\n"
            "====================================================================================\n#End of artifacts.\n\n"
            "We are at stage {stage}.\n"
            "Stage goal: {stage_goal}.\n"
            "We move to artifact {artifact_number}: {artifact_name}.\n"
            "Goal: {goal}\n"
            "Components: {components}\n"
            "Methodology: {methodology}\n"
            "Criteria: {criteria}\n\n"
            "Real data: {data_source}\n\n"
            "Propose several options for this artifact based on the previous context.\n\n"
            "Each option must satisfy all criteria.\n\n"
            "For each option, provide a high-level assessment for each criterion.\n"
            "##CONSTRAINTS:\n"
            "**IMPORTANT**: You work ONLY on artifact {artifact_number}: {artifact_name}.\n\n"
            "**IMPORTANT**: You are **FORBIDDEN** to form versions, prepare options, or even discuss any other artifacts besides {artifact_number}: {artifact_name}.\n"
        ),
        "final_prompt": (
            "We are working on stage {artifact_number}: {artifact_name}.\n"
            "Goal: {goal}\n"
            "Methodology: {methodology}\n"
            "Real data: {data_source}\n\n"
            "====================================================================================\n#Artifacts:\n"
            "{context_str}\n"
            "====================================================================================\n\n"
            "The user provided a response.\n"
            "Option {artifact_number}: {selected_option}\n"
            "ALWAYS produce a **full** final version of the artifact.\n"
            "Include an assessment for each criterion.\n"
            "Criteria list: {criteria}\n"
            "IMPORTANT: In artifact_final_text include only the final artifact text — no reasoning and no user reply.\n"
            "##CONSTRAINTS:\n"
            "**IMPORTANT**: You work ONLY on artifact {artifact_number}: {artifact_name}.\n\n"
            "**IMPORTANT**: You are **FORBIDDEN** to form versions, prepare options, or even discuss any other artifacts besides {artifact_number}: {artifact_name}.\n"
        ),
        "keywords": {
            "confirm": [
                "confirm",
                "confirmed",
                "yes",
                "ok",
                "okay",
                "agree",
                "approve",
                "next",
                "go on",
                "continue",
                "looks good",
                "all good",
                "fine",
            ],
            "edit": [
                "fix",
                "edit",
                "change",
                "update",
                "revise",
                "adjust",
                "modify",
                "not right",
                "not correct",
                "add",
                "redo",
                "rework",
            ],
        },
    },
}

_CURRENT_LOCALE = DEFAULT_LOCALE
_CHOICE_TEXT: Dict[str, Any] = {}
_KEYWORDS: Dict[str, List[str]] = {}


def set_locale(locale: str = DEFAULT_LOCALE) -> None:
    global _CURRENT_LOCALE, _CHOICE_TEXT, _KEYWORDS
    locale_key = resolve_locale(locale)
    set_global_locale(locale_key)
    _CURRENT_LOCALE = locale_key
    _CHOICE_TEXT = CHOICE_LOCALES[locale_key]
    _KEYWORDS = _CHOICE_TEXT.get("keywords", {})


def _get_choice_text() -> Dict[str, Any]:
    return _CHOICE_TEXT or CHOICE_LOCALES[DEFAULT_LOCALE]


def _get_keywords(name: str) -> List[str]:
    if not _KEYWORDS:
        return CHOICE_LOCALES[DEFAULT_LOCALE]["keywords"].get(name, [])
    return _KEYWORDS.get(name, [])


set_locale()

_PRIMARY_RETRY_ATTEMPTS = 3

_llm = get_llm(model="base", provider="openai", temperature=0.9)
#_llm_fallback = get_llm(model="base", provider="openai", temperature=1)
_llm_fallback = get_llm(model="base", provider="openai_4", temperature=0.9)
_llm_fallback_middleware = build_model_fallback_middleware(
    primary_llm=_llm,
    alternative_llm=_llm_fallback,
    primary_retries=_PRIMARY_RETRY_ATTEMPTS,
)

_user_analyser_primary_llm = get_llm(model="mini", provider="openai", temperature=0)
_user_analyser_alternative_llm = get_llm(model="base", provider="openai", temperature=0)
_user_analyser_llm = with_llm_fallbacks(
    _user_analyser_primary_llm,
    alternative_llm=_user_analyser_alternative_llm,
    primary_retries=_PRIMARY_RETRY_ATTEMPTS,
)

_SUMMARY_MAX_TOKENS = 16000
_SUMMARY_KEEP_MESSAGES = 12
_summarization_middleware = SummarizationMiddleware(
    model=_user_analyser_llm,
    max_tokens_before_summary=_SUMMARY_MAX_TOKENS,
    messages_to_keep=_SUMMARY_KEEP_MESSAGES,
)


def create_init_node(artifact_id: int):
    @debug_log
    def init_node(state: ArtifactAgentState, 
                config: RunnableConfig,
                runtime: Runtime[ArtifactAgentContext],) -> ArtifactAgentState:
        original_messages = state.get("messages", [])
        remove_ops = _sanitize_messages(original_messages)
        
        if runtime is None or runtime.context is None or "user_prompt" not in runtime.context:
            if not original_messages:
                return state
            last_user_msg = None
            for msg in reversed(original_messages):
                if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                    last_user_msg = msg
                    break
            if last_user_msg is None:
                return state  # no human message to seed prompt
            #user_prompt = last_user_msg.content
            content = last_user_msg.content
            user_prompt = content if isinstance(content, str) else (content[0].get("text", str(content[0])) if isinstance(content, list) else str(content))
        else:
            user_prompt = runtime.context["user_prompt"]

        return Command(
            update={
                "messages": remove_ops,
                "user_prompt": user_prompt,
                "current_artifact_state": ArtifactState.INIT,
                "current_artifact_id": artifact_id,
            }
        )
    return init_node

"""
#Текущий артефакт
class ArtifactDetails:
    artifact_definition: ArtifactDefinition
    artifact_options: List[ArtifactOption]
    selected_option: int  # 0-based, -1 if not chosen yet
    artifact_final_text: str  # rendered final text + estimation

state:
class ArtifactAgentState(AgentState[ArtifactOptions]):
    user_info: NotRequired[Dict[str, Any]]
    user_prompt: str #Определяем на этапе инициализации
    artifacts: NotRequired[Dict[int, ArtifactDetails]]   # При входе в uзе init - пусто;
                                                    # При входе в generate_options_node 
                                                    # Или добавляем новый артефакт или изменяем существующий
                                                    # - заполняем artifact_definition (копируем по ID)
                                                    # artifact_final_text - ""
                                                    # При выходе в generate_options_node 
                                                    # - заполняем artifact_options - список вариантов
                                                    # - selected_option - пусто (или -1)
                                                    # artifact_final_text - ""
                                                    # При выходе из select_option_node:
                                                    # - заполняем selected_option (если пользователь выбрал)

    current_artifact_id: NotRequired[int]           #Заполняем при входе в generate_options_node
    current_artifact_state: NotRequired[ArtifactState] # меняем на шагах процесса

"""


class UserChangeRequest(BaseModel):
    """User change request.
    Запрос пользователя на изменение.
    """
    model_config = ConfigDict(extra="forbid")

    is_change_requested: Annotated[
        bool,
        Field(description="Did user request changed. True or False."),
    ]

    # NotRequired[str] -> optional field in Pydantic: give it a default
    change_request: Annotated[
        str | None,
        Field(default=None, description="Request to change."),
    ] = None

    @model_validator(mode="after")
    def validate_change_request(self) -> "UserChangeRequest":
        # Cross-field validation (v2 replacement for most root_validator use cases)
        if self.is_change_requested and not (self.change_request and self.change_request.strip()):
            raise ValueError("change_request must be provided when is_change_requested=True")
        return self


class UserSelectedOption(BaseModel):
    """User selected option or change request.
    Один из вариантов, который выбрал пользователь или запрос на изменение.
    """
    model_config = ConfigDict(extra="forbid")

    selected_option_number: Annotated[
        int | None,
        Field(default=None, ge=1, description="Number of selected option started 1."),
    ] = None

    change_request: Annotated[
        UserChangeRequest,
        Field(description="Request to change answer."),
    ]

#class UserChangeRequest(TypedDict):
#    """User change request.
#    Запрос пользователя на изменение.
#    """
#    is_change_requested: Annotated[bool, ..., "Did user request changed. True or False."]
#    change_request: Annotated[NotRequired[str], ..., "Request to change."]

#class UserSelectedOption(TypedDict):
#    """User selected option or change request.
#    Один из вариантов, который выбрал пользователь или запрос на изменение.
#    """
#    selected_option_number: Annotated[NotRequired[int], ..., "Number of selected option started 1."]
#    change_request: Annotated[UserChangeRequest, ..., "Request to change asnwer."]


def _user_select_option(text: str, options_text: str):
    """
    Lightweight LLM-based classifier to judge approval/confirmation.
    Falls back to keyword match if the LLM call fails.
    """
    #normalized = text.lower().strip().strip(".,!?;")
    clf = _user_analyser_llm.with_structured_output(UserSelectedOption)
    system = SystemMessage(
        content=(
            "User provided response to select option:\n" 
            f"{options_text}\n"
            "You have to analise user's response.\n" 
            "Determine which option user selected (use number started 1) or collect user's request to change."
        )
    )
    user = HumanMessage(content=f"User reply: {text}")
    try:
        result = clf.invoke([system, user])
        if isinstance(result, dict):
            return result
    except Exception as e:
        logging.warning("User option classifier failed, falling back. Error: %s", e)

    return _heuristic_user_selected_option(text)


def _resolve_selected_option_index(
    user_text: str,
    estimated: Optional[UserSelectedOption],
    num_options: int,
) -> Optional[int]:
    """Resolve user's chosen option to a 0-based index."""
    if num_options <= 0:
        return None

    if estimated:
        raw_num = estimated.get("selected_option_number")
        try:
            if raw_num is not None:
                idx = int(raw_num) - 1
                if 0 <= idx < num_options:
                    return idx
        except (TypeError, ValueError):
            pass

    text = user_text.strip()

    # Match single-letter choices like "A", "B", "C" (latin) or "А", "Б", "В" (cyrillic).
    m = re.search(r"\b([ABCАБВ])\b", text, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        letter_map = {"A": 0, "B": 1, "C": 2, "А": 0, "Б": 1, "В": 2}
        idx = letter_map.get(letter)
        if idx is not None and idx < num_options:
            return idx

    # Match explicit numbers.
    m = re.search(r"\b(\d+)\b", text)
    if m:
        try:
            idx = int(m.group(1)) - 1
            if 0 <= idx < num_options:
                return idx
        except ValueError:
            pass

    return None


def _heuristic_user_selected_option(text: str) -> UserSelectedOption:
    """Heuristic fallback for option selection."""
    normalized = (text or "").lower()

    change_markers = _get_keywords("edit")
    if any(marker in normalized for marker in change_markers):
        return {
            "change_request": {
                "is_change_requested": True,
                "change_request": text,
            }
        }

    # Letters A/B/C (latin) or А/Б/В (cyrillic).
    m = re.search(r"\b([abcабв])\b", normalized, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        letter_map = {"A": 1, "B": 2, "C": 3, "А": 1, "Б": 2, "В": 3}
        num = letter_map.get(letter)
        if num is not None:
            return {
                "selected_option_number": num,
                "change_request": {"is_change_requested": False},
            }

    # Digits 1/2/3.
    m = re.search(r"\b([1-3])\b", normalized)
    if m:
        return {
            "selected_option_number": int(m.group(1)),
            "change_request": {"is_change_requested": False},
        }

    # Unclear answer: let the caller re-ask for clarification.
    return {"change_request": {"is_change_requested": False}}

def _is_user_confirmed(text: str, artifact_text: str):
    """
    Lightweight LLM-based classifier to judge approval/confirmation.
    Falls back to keyword match if the LLM call fails.
    """
    #normalized = text.lower().strip().strip(".,!?;")
    clf = _user_analyser_llm.with_structured_output(UserChangeRequest)
    system = SystemMessage(
        content=(
            "User provided response to generated artifact:\n" 
            f"{artifact_text}\n"
            "You have to analise user's response.\n" 
            "Determine if user confirmed the text or requested change."
        )
    )
    user = HumanMessage(content=f"User reply: {text}")
    try:
        result = clf.invoke([system, user])
        if isinstance(result, dict):
            return result
        elif isinstance(result, UserChangeRequest):
            return result.model_dump()
    except Exception as e:
        logging.warning("User confirmation classifier failed, falling back. Error: %s", e)

    return _heuristic_user_change_request(text)


def _heuristic_user_change_request(text: str) -> UserChangeRequest:
    """Heuristic fallback for confirmation/change detection."""
    normalized = (text or "").lower()

    change_markers = _get_keywords("edit")
    if any(marker in normalized for marker in change_markers):
        return {"is_change_requested": True, "change_request": text}

    confirm_markers = _get_keywords("confirm")
    if any(marker in normalized for marker in confirm_markers):
        return {"is_change_requested": False}

    # Default to change request when unclear to enforce explicit confirmation.
    return {"is_change_requested": True, "change_request": text}

def _format_artifact_options_text(structured_response: Dict[str, Any]) -> str:
    """Build user-facing text from the structured options response."""
    general = structured_response.get("general_considerations") or ""
    artifact_options = structured_response.get("artifact_options") or []

    lines: List[str] = []
    if general:
        lines.append(str(general).strip())
    if general and artifact_options:
        lines.append("")  # blank line between general info and options

    for idx, option in enumerate(artifact_options):
        option_text = option.get("artifact_option", "")
        option_letter = chr(ord("A") + idx)
        lines.append(f"- {option_letter}. {option_text}")
        for estimation in option.get("criteris_estimations") or []:
            criteria_text = estimation.get("criteria_estimation", "")
            if criteria_text:
                lines.append(f"        {criteria_text}")

    return "\n".join(lines)

def _format_artifact_final_text(structured_response: Dict[str, Any]) -> str:
    """Build user-facing text from the final text generation response."""
    artifact_estimation = structured_response.get("artifact_estimation") or ""
    artifact_final_text = structured_response.get("artifact_final_text") or ""

    parts: List[str] = []
    if str(artifact_estimation).strip():
        parts.append(str(artifact_estimation).strip())
    if str(artifact_final_text).strip():
        parts.append(str(artifact_final_text).strip())

    return "\n\n".join(parts)


def _update_last_ai_message_content(messages: List[Any], text: str) -> List[Any]:
    """Replace the last AI message content/text fields with the provided text."""
    if not messages or not text:
        return messages

    updated_messages = list(messages)
    for idx in range(len(updated_messages) - 1, -1, -1):
        msg = updated_messages[idx]
        if isinstance(msg, AIMessage):
            updated_kwargs = {**(getattr(msg, "additional_kwargs", {}) or {}), "text": text}
            updated_messages[idx] = AIMessage(
                content=text,
                additional_kwargs=updated_kwargs,
                response_metadata=getattr(msg, "response_metadata", {}),
                id=getattr(msg, "id", None),
                name=getattr(msg, "name", None),
                tool_calls=getattr(msg, "tool_calls", None),
                invalid_tool_calls=getattr(msg, "invalid_tool_calls", None),
                usage_metadata=getattr(msg, "usage_metadata", None),
            )
            break

    return updated_messages


def _sanitize_messages(messages: List[Any]) -> List[RemoveMessage]:
    """Decide which messages to remove while leaving all others untouched."""
    remove_ops: List[RemoveMessage] = []
    for msg in messages or []:
        if isinstance(msg, ToolMessage):
            if getattr(msg, "id", None):
                remove_ops.append(RemoveMessage(id=msg.id))
            continue
        if isinstance(msg, AIMessage):
            has_tool_payload = bool(
                getattr(msg, "tool_calls", None)
                or getattr(msg, "invalid_tool_calls", None)
                or (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls")
            )
            if has_tool_payload and getattr(msg, "id", None):
                remove_ops.append(RemoveMessage(id=msg.id))
            continue
        # untouched messages are left in place
    return remove_ops


@debug_log
def select_option_node(state: ArtifactAgentState, 
              config: RunnableConfig,
              runtime: Runtime[ArtifactAgentContext],) -> ArtifactAgentState:

        #if not state["messages"]:
    #    return state          
    #last_user_msg = state["messages"][-1]
    # We only augment on real user turns
    #if last_user_msg.type != "human":
    #    return state

    #state["user_prompt"] = last_user_msg.content
    #state["current_artifact_state"] = ArtifactState.INIT
    #state["current_artifact_id"] = 0
    #state["user_info"] = config.
    #print(state.get("structured_response", {}).get("response", ""))


    #_SELECTED_OPTION = 0 #TODO: use interrupt to get option from user
    artifact_id = state["current_artifact_id"]
    artifacts = state.get("artifacts") or {}
    current_artifact = artifacts.get(artifact_id) or {}
    artifact_name = current_artifact.get("artifact_definition", {}).get("name", "")
    choice_text = _get_choice_text()

    interrupt_payload = {
        "type": "choice",
        "artifact_id": state["current_artifact_id"],
        "artifact_name": artifact_name,
        "current_artifact_state": ArtifactState.OPTIONS_GENERATED,
        "content": prettify(state["current_artifact_text"]),
        "question": choice_text["select_option_question"],
    }
    #print("DEBUG: before options_interrupt")
    user_response = interrupt(interrupt_payload)
    #print("DEBUG: after options_interrupt")
    message_update = [HumanMessage(content=str(user_response))]    

    #current_artifact["selected_option"] = _SELECTED_OPTION
    user_response_str = str(user_response)
    user_response_estimated: UserSelectedOption = _user_select_option(
        user_response_str, state["current_artifact_text"]
    )

    if not user_response_estimated.get("change_request", {}).get("is_change_requested", False):
        options = current_artifact.get("artifact_options") or []
        selected_idx = _resolve_selected_option_index(
            user_response_str, user_response_estimated, len(options)
        )
        if selected_idx is None:
            # Ask the user to clarify their choice.
            return Command(
                goto="select_option",
                update={"messages": message_update},
            )

        current_artifact["selected_option"] = selected_idx
        artifacts[artifact_id] = current_artifact

        return Command(
            goto="generate_aftifact",
            update={
                "messages": message_update,
                "current_artifact_state": ArtifactState.OPTION_SELECTED,
                "artifacts": artifacts,
            },
        )

    return Command(
        goto="generate_options",
        update={
            "messages": message_update,
        },
    )


@debug_log
def confirmation_node(state: ArtifactAgentState, 
              config: RunnableConfig,
              runtime: Runtime[ArtifactAgentContext],) -> ArtifactAgentState:
    
    artifacts = state.get("artifacts", {})
    current_artifact = artifacts.get(state["current_artifact_id"], {})
    artifact_name = current_artifact.get("artifact_definition", {}).get("name", "")
    choice_text = _get_choice_text()

    interrupt_payload = {
        "type": "choice",
        "artifact_id": state["current_artifact_id"],
        "artifact_name": artifact_name,
        "current_artifact_state": ArtifactState.ARTIFACT_GENERATED,
        #"content": prettify(current_artifact["artifact_estimation"] + "\n" + current_artifact["artifact_final_text"]),
        "content": current_artifact["artifact_final_text"],
        "question": choice_text["confirm_question"].format(artifact_name=artifact_name),
    }

    #print("DEBUG: before gererator_interrupt")
    user_response = interrupt(interrupt_payload)
    #print("DEBUG: after gererator_interrupt")
    message_update = [HumanMessage(content=str(user_response))]    

    user_response_estimated: UserChangeRequest = _is_user_confirmed(str(user_response), current_artifact["artifact_final_text"])
    if not user_response_estimated.get("is_change_requested", False):
        state["artifacts"][state["current_artifact_id"]]["artifact_final_text"] = prettify(current_artifact["artifact_final_text"])
        state["current_artifact_state"] = ArtifactState.ARTIFACT_CONFIRMED
        return Command(
            goto=END,
            update={
                "messages": message_update,
                "current_artifact_state": ArtifactState.ARTIFACT_CONFIRMED,
            },
        )

    return Command(
        goto="generate_aftifact",
        update={
            "messages": message_update,
        },
    )

response_format = AutoStrategy(schema=ArtifactOptions)

_yandex_tool = SearchTool(
    api_key=config.YA_API_KEY,
    folder_id=config.YA_FOLDER_ID,
    max_results=3,
    summarize=True
)
_think_tool = ThinkTool()


def create_options_generator_node(
    model: BaseChatModel,
    artifact_id: int,
    locale: str | None = None,
):
    """Creates options generation agent."""
    _artifact_id = artifact_id
    _artifact_def = ARTIFACTS[_artifact_id]
    locale_key = resolve_locale(locale)
    schemas = get_artifact_schemas(locale_key)

    @dynamic_prompt
    def build_agent_prompt(request: ModelRequest) -> str:
        agent_state = request.state
        artifacts_in_state = agent_state.get("artifacts") or {}
        choice_text = _get_choice_text()
        context_str = "\n------------------------------------------------------------------------\n\n".join(
            f"##{a['artifact_definition']['id'] + 1} {a['artifact_definition']['name']}:\n"
            f"{a['artifact_final_text'] or choice_text['artifact_placeholder']}"
            for a in artifacts_in_state.values()
        )
        components = ",\n-".join(_artifact_def["components"])
        criteria = ",\n-".join(_artifact_def["criteria"])
        data_source = _artifact_def.get("data_source") or choice_text["data_source_default"]
        prompt = choice_text["options_prompt"].format(
            user_prompt=agent_state.get("user_prompt", ""),
            context_str=context_str,
            stage=_artifact_def["stage"],
            stage_goal=_artifact_def["stage_goal"],
            artifact_number=_artifact_id + 1,
            artifact_name=_artifact_def["name"],
            goal=_artifact_def["goal"],
            components=components,
            methodology=_artifact_def["methodology"],
            criteria=criteria,
            data_source=data_source,
        )
        #print(f"DEBUG: {prompt}")
        return (
            get_system_prompt(locale_key)
            + "\n\n"
            + prompt
            + "\n\n"
            + choice_text["format_instruction"]
            + "\n\n"
            + get_format_options_prompt(locale_key)
            + "\n\n"
            + get_tool_policy_prompt(locale_key)
        )

    _agent = create_agent(
        model=model,
        tools=[_think_tool, _yandex_tool], # Includes internal scratchpad and search
        #system_prompt=SYSTEM_PROMPT + "\n\n" + prompt,
        middleware=[
            #_summarization_middleware,
            build_agent_prompt,
            provider_then_tool,
            _llm_fallback_middleware,
        ],
        response_format=schemas["options"],
        state_schema=ArtifactAgentState,
        context_schema=ArtifactAgentContext,
    )
    @debug_log
    def generate_options_node(state: ArtifactAgentState, 
              config: RunnableConfig,
              runtime: Runtime[ArtifactAgentContext],) -> ArtifactAgentState:
        if not state["messages"]:
            return state                       # safety guard

        #last_user_msg = state["messages"][-1]
        # We only augment on real user turns
        #if last_user_msg.type != "human":
        #    return state        
        state["current_artifact_id"] = _artifact_id

        # ensure artifacts list exists
        artifacts: List[ArtifactDetails] = state.get("artifacts") or {}

        # 3) ???????? ?????????????????? ?? ?????????? id ?????? ??? ??????????????, ?????????? ??????????????????
        details = artifacts.get(_artifact_id)
        if details is None:
            details = {
                "artifact_definition": _artifact_def,  # ???????????????? ???? ID
                "artifact_final_text": "",             # ?????????? ???? ??????????
                "artifact_options": [],                # ?????????? ?????????????????? ???? ????????????
                "selected_option": -1,                 # ???????????? ??????????
            }
            artifacts[artifact_id] = details
        else:
            details["artifact_definition"] = _artifact_def
            details["artifact_final_text"] = ""       # ???????????????????? ???? ??????????


        #state.pop("structured_response", {})
        result: ArtifactAgentState = _agent.invoke(
            state,
            config=config,
            context=runtime.context,
        )
        structured_response = result.pop("structured_response", {}) or {}
        choice_text = _get_choice_text()
        formatted_options_text = (
            _format_artifact_options_text(structured_response)
            + "\n\n"
            + choice_text["select_option_question"]
        )
        # on exit: enforce spec
        result_artifacts = result.get("artifacts") or {}
        result["artifacts"] = result_artifacts
        result["current_artifact_text"] = formatted_options_text
        result_details = result_artifacts.get(_artifact_id) or details
        result_details["artifact_options"] = structured_response.get("artifact_options", [])   
        result_details["selected_option"] = -1
        result_artifacts[_artifact_id] = result_details
        #result["messages"] = _update_last_ai_message_content(
        #    result.get("messages") or [], formatted_options_text
        #)

        result["current_artifact_state"] = ArtifactState.OPTIONS_GENERATED
        return result #Command(update=result)

    return generate_options_node

def create_generation_agent(
    model: BaseChatModel,
    artifact_id: int,
    locale: str | None = None,
):
    """Creates the artifact generation agent."""
    _artifact_id = artifact_id
    _artifact_def = ARTIFACTS[_artifact_id]
    locale_key = resolve_locale(locale)
    schemas = get_artifact_schemas(locale_key)
    @dynamic_prompt
    def build_agent_prompt(request: ModelRequest) -> str:
        agent_state: ArtifactAgentState = request.state
        artifacts_in_state = agent_state.get("artifacts") or {}
        choice_text = _get_choice_text()
        context_str = "\n========================================\n\n".join(
            f"##{a['artifact_definition']['id']+1} {a['artifact_definition']['name']}:\n"
            f"{a['artifact_final_text'] or choice_text['artifact_placeholder']}"
            for a in artifacts_in_state.values()
        )

        current_artifact = artifacts_in_state.get(_artifact_id)
        options = current_artifact.get("artifact_options")
        selected_option = options[current_artifact["selected_option"]]
        data_source = _artifact_def.get("data_source") or choice_text["data_source_default"]
        criteria = ", ".join(_artifact_def["criteria"])
        prompt = choice_text["final_prompt"].format(
            artifact_number=_artifact_id + 1,
            artifact_name=_artifact_def["name"],
            goal=_artifact_def["goal"],
            methodology=_artifact_def["methodology"],
            data_source=data_source,
            context_str=context_str,
            selected_option=selected_option.get("artifact_option", ""),
            criteria=criteria,
        )
        #print(f"DEBUG: {prompt}")
        return (
            get_system_prompt(locale_key)
            + "\n\n"
            + prompt
            + "\n\n"
            + choice_text["format_instruction"]
            + "\n\n"
            + get_tool_policy_prompt(locale_key)
        )

    
    _agent = create_agent(
        model=model,
        tools= [_think_tool, _yandex_tool], # Includes internal scratchpad and search
        middleware=[
            #_summarization_middleware,
            build_agent_prompt,
            provider_then_tool,
            _llm_fallback_middleware,
        ],
        response_format=schemas["final"],
        state_schema=ArtifactAgentState,
        context_schema=ArtifactAgentContext,
    )
    @debug_log
    def generate_artifact_node(state: ArtifactAgentState, 
              config: RunnableConfig,
              runtime: Runtime[ArtifactAgentContext],) -> ArtifactAgentState:
        if not state["messages"]:
            return state                       # safety guard

        last_user_msg = state["messages"][-1]
        # We only augment on real user turns
        #if last_user_msg.type != "human":
        #    return state        
        #state["current_artifact_id"] = _artifact_id
        _artifact_id = state["current_artifact_id"]
        # ensure artifacts list exists
        artifacts: List[ArtifactDetails] = state.get("artifacts") or {}

        result: ArtifactAgentState = _agent.invoke(
            state,
            config=config,
            context=runtime.context,
        )

        structured_response = result.get("structured_response", {}) or {}
        choice_text = _get_choice_text()
        artifact_name = str(_artifact_def.get("name") or "")
        formatted_final_text = (
            _format_artifact_final_text(structured_response)
            + "\n\n"
            + choice_text["confirm_question"].format(artifact_name=artifact_name)
        )

        # on exit: enforce spec
        result_artifacts = result.get("artifacts") or {}
        result["artifacts"] = result_artifacts

        result["current_artifact_text"] = result.get("structured_response", {}).get("artifact_final_text", "")   
        result_details = result_artifacts.get(_artifact_id) or {"artifact_definition": _artifact_def}
        result_details["artifact_final_text"] = (
            result.get("structured_response", {}).get("artifact_final_text", "")
            + "\n"
            + result.get("structured_response", {}).get("artifact_estimation", "")
        )
        result_artifacts[_artifact_id] = result_details
        result["messages"] = _update_last_ai_message_content(
            result.get("messages") or [], formatted_final_text
        )

        result["current_artifact_state"] = ArtifactState.ARTIFACT_GENERATED
        return result

    return generate_artifact_node

def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
    uset_parental_memory: bool = False,
    artifact_id: int = 0,
    locale: str = DEFAULT_LOCALE,
):

    set_locale(locale)
    log_name = f"choice_agent_{time.strftime('%Y%m%d%H%M')}"
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

    if uset_parental_memory:
        memory = None #memory_saver
    else:
        memory = MemorySaver() # Always use memory for this agent

    builder = StateGraph(ArtifactAgentState)
    

    builder.add_node("init", create_init_node(artifact_id=artifact_id))
    builder.add_node(
        "generate_options",
        create_options_generator_node(model=_llm, artifact_id=artifact_id, locale=locale),
    )
    builder.add_node("select_option", select_option_node)
    builder.add_node(
        "generate_aftifact",
        create_generation_agent(model=_llm, artifact_id=artifact_id, locale=locale),
    )
    builder.add_node("confirm", confirmation_node)

    builder.add_edge(START, "init")
    builder.add_edge("init", "generate_options")
    builder.add_edge("generate_options", "select_option")
    #builder.add_edge("select_option", "generate_aftifact")
    builder.add_edge("generate_aftifact", "confirm")
    #builder.add_edge("confirm", END)

    graph = builder.compile(
        checkpointer=memory,
        debug=False
    ).with_config({"callbacks": callback_handlers})

    return graph


if __name__ == "__main__":
    print("Initializing Choice Agent...")
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
    user_llm = get_llm(model="base", provider="openai")
    USER_SIMULATOR_PROMPT = """
    Ты — основатель стартапа "Uber для выгула собак".
    Ты общаешься с "Продуктовым Наставником" (AI), который ведет тебя по методологии создания продукта.

    Твоя задача:
    1. Отвечать на вопросы наставника кратко и по делу.
    2. Если наставник предлагает варианты (A, B, C), выбери один (например, "Выбираю вариант A") или попроси исправить вариант или попроси добавить дополнительный вариант.
    3. Если наставник спрашивает "Подтверждаете?", отвечай "Подтверждаю" или проси внести изменения в ответ наставника.
    4. Если окончательный вариант тебя полностью удовлетворяет, верни одно слово: "exit".
    5 Если разговор зашёл в тупик, верни одно слово: "exit".
    """
    SIMULATE=False
    while True:
        result = agent_graph.invoke(next_input, config=config, context=ctx)
        state_snapshot = agent_graph.get_state(config)
        if len(state_snapshot.next) == 0:
            break
        current_idx = state_snapshot.values.get("current_step_index", 0)
        logging.info("Current step index: %s", current_idx)
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = getattr(interrupts[-1], "value", interrupts[-1])
            last_ai = payload.get("content", "")
            #print(f"Interrupt payload: {payload}")
            if SIMULATE:
                prompt = f"Ответ наставника:\n{last_ai}\n\nТвой ответ:"
                print(prompt)
                sim_messages = [
                    SystemMessage(content=USER_SIMULATOR_PROMPT),
                    HumanMessage(content=prompt),
                ]

                user_reply: str = user_llm.invoke(sim_messages).content
                print(f"Your answer: {user_reply}")
            else:
                print(f"Ответ наставника:\n{last_ai}\n\n")
                user_reply = input("Твой ответ:")
            #user_reply = "не теперь exit"
            if "exit" in user_reply.lower():
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

    print("ФСЁ")
