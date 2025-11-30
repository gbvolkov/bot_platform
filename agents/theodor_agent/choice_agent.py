from __future__ import annotations
import functools

import logging
import time
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_model_call
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    ToolStrategy,
    StructuredOutputValidationError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langgraph.runtime import Runtime

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config

from .prompts.prompts import (
    SYSTEM_PROMPT,
    TOOL_POLICY_PROMPT,
    FORMAT_OPTIONS_PROMPT,
)

from ..tools.yandex_search import YandexSearchTool
from ..tools.think import ThinkTool
from agents.utils import ModelType, get_llm
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
)
from agents.structured_prompt_utils import build_json_prompt

DEBUG = True
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

_llm = get_llm(model="base", provider="openai", temperature=1)
_user_analyser_llm = get_llm(model="mini", provider="openai", temperature=0)


def create_init_node(artifact_id: int):
    @debug_log
    def init_node(state: ArtifactAgentState, 
                config: RunnableConfig,
                runtime: Runtime[ArtifactAgentContext],) -> ArtifactAgentState:
        
        if runtime is None or runtime.context is None or "user_prompt" not in runtime.context:
            if not state["messages"]:
                return state          
            last_user_msg = state["messages"][-1]
            # We only augment on real user turns
            if last_user_msg.type != "human":
                return state
            user_prompt = last_user_msg.content
        else:
            user_prompt = runtime.context["user_prompt"]

        state["user_prompt"] = user_prompt
        state["current_artifact_state"] = ArtifactState.INIT
        state["current_artifact_id"] = artifact_id
        #state["user_info"] = config.
        return state
    return init_node

"""
#Текущий артефакт
class ArtifactDetails:
    artifact_definition: ArtifactDefinition
    artifact_options: ArtifactOptions
    selected_option: int
    artifact_final_text: str

state:
class ArtifactAgentState(AgentState[ArtifactOptions]):
    user_info: NotRequired[Dict[str, Any]]
    user_prompt: str #Определяем на этапе инициализации
    artifacts: NotRequired[List[ArtifactDetails]]   # При входе в узе init - пусто;
                                                    # При входе в generate_options_node 
                                                    # Или добавляем новый артефакт или изменяем существующий
                                                    # - заполняем artifact_definition (копируем по ID)
                                                    # artifact_final_text - ""
                                                    # При выходе в generate_options_node 
                                                    # - заполняем artifact_options - пусто
                                                    # - selected_option - пусто (или -1)
                                                    # artifact_final_text - ""
                                                    # При выходе из select_option_node:
                                                    # - заполняем selected_option (если пользователь выбрал)

    current_artifact_id: NotRequired[int]           #Заполняем при входе в generate_options_node
    current_artifact_state: NotRequired[ArtifactState] # меняем на шагах процесса

"""

class UserChangeRequest(TypedDict):
    """User change request.
    Запрос пользователя на изменение.
    """
    is_change_requested: Annotated[bool, ..., "Did user request changed. True or False."]
    change_request: Annotated[NotRequired[str], ..., "Request to change."]

class UserSelectedOption(TypedDict):
    """User selected option or change request.
    Один из вариантов, который выбрал пользователь или запрос на изменение.
    """
    selected_option_number: Annotated[NotRequired[int], ..., "Number of selected option started 1."]
    change_request: Annotated[UserChangeRequest, ..., "Request to change asnwer."]


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
    result = clf.invoke([system, user])
    return result

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
    result = clf.invoke([system, user])
    return result

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
    artifact_estimation = structured_response.get("artifact_estimation")
    artifact_final_text = structured_response.get("artifact_final_text", "")
    final_text = artifact_estimation or "" + "\n" if artifact_estimation else "" + artifact_final_text

    return final_text


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
    artifacts = state.get("artifacts", {})
    current_artifact = artifacts.get(state["current_artifact_id"], {})
    artifact_name = current_artifact.get("artifact_definition", {}).get("name", "")

    interrupt_payload = {
        "type": "choice",
        "artifact_id": state["current_artifact_id"],
        "artifact_name": artifact_name,
        "content": prettify(state["current_artifact_text"]),
        "question": "Выберите один из предложенных вариантов или скажите, что нужно поправить.",
    }
    #print("DEBUG: before options_interrupt")
    user_response = interrupt(interrupt_payload)
    #print("DEBUG: after options_interrupt")
    message_update = [HumanMessage(content=str(user_response))]    

    #current_artifact["selected_option"] = _SELECTED_OPTION
    user_response_estimated: UserSelectedOption = _user_select_option(str(user_response), state["current_artifact_text"])

    if not user_response_estimated.get("change_request", {}).get("is_change_requested", False):
        state["current_artifact_state"] = ArtifactState.OPTION_SELECTED
        return Command(
            goto="generate_aftifact",
            update={
                "messages": message_update,
                "current_artifact_state": ArtifactState.OPTION_SELECTED
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
    
    artifacts = state.get("artifacts", {})
    current_artifact = artifacts.get(state["current_artifact_id"], {})
    artifact_name = current_artifact.get("artifact_definition", {}).get("name", "")

    #result = _is_user_confirmed("Ерунда какая-то", current_artifact["artifact_final_text"])
    #result = _is_user_confirmed("Подтверждаю", current_artifact["artifact_final_text"])
    #result = _is_user_confirmed("Норм. Едем дальше", current_artifact["artifact_final_text"])
    #result = _is_user_confirmed("Исправляй всё.", current_artifact["artifact_final_text"])

    interrupt_payload = {
        "type": "choice",
        "artifact_id": state["current_artifact_id"],
        "artifact_name": artifact_name,
        #"content": prettify(current_artifact["artifact_estimation"] + "\n" + current_artifact["artifact_final_text"]),
        "content": prettify(current_artifact["artifact_final_text"]),
        "question": f"Подтвердите артефакт \"{artifact_name}\" или скажите, что нужно изменить.",
    }

    #print("DEBUG: before gererator_interrupt")
    user_response = interrupt(interrupt_payload)
    #print("DEBUG: after gererator_interrupt")
    message_update = [HumanMessage(content=str(user_response))]    

    user_response_estimated: UserChangeRequest = _is_user_confirmed(str(user_response), current_artifact["artifact_final_text"])
    if not user_response_estimated.get("is_change_requested", False):
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

@wrap_model_call
def provider_then_tool(request: ModelRequest, handler):
    try:
        return handler(request)
    except (ValueError, StructuredOutputValidationError):
        rf = request.response_format
        if isinstance(rf, AutoStrategy):
            schema = rf.schema
        elif isinstance(rf, ProviderStrategy):
            schema = rf.schema
        else:
            raise  # already in ToolStrategy; bubble up
        # Retry using tool-based structured output
        return handler(request.override(response_format=ToolStrategy(schema=schema)))

_yandex_tool = YandexSearchTool(
    api_key=config.YA_API_KEY,
    folder_id=config.YA_FOLDER_ID,
    max_results=10
)
_think_tool = ThinkTool()


def create_options_generator_node(model: BaseChatModel, artifact_id: int):
    """Creates options generation agent."""
    _artifact_id = artifact_id
    _artifact_def = ARTIFACTS[_artifact_id]
    @dynamic_prompt
    def build_agent_prompt(request: ModelRequest) -> str:
        agent_state = request.state
        artifacts_in_state = agent_state.get("artifacts") or {}
        context_str = "\n".join(
            f"{a['artifact_definition']['id']} {a['artifact_definition']['name']}: {a['artifact_final_text']}"
            for a in artifacts_in_state.values()
        )
        #format_requirements = build_json_prompt(ArtifactOptions)
        prompt = (
            f"Мы прорабатываем {agent_state.get("user_prompt", "")}\n\n"
            f"{context_str}\n"
            f"Мы находимся на этапе {_artifact_def['stage']}.\n"
            f"Цель этапа: {_artifact_def['stage_goal']}.\n"
            f"Мы переходим к артефакту {_artifact_id + 1}: {_artifact_def['name']}.\n"
            f"Цель: {_artifact_def['goal']}\n"
            f"Компоненты: {',\n-'.join(_artifact_def['components'])}\n"
            f"Методология: {_artifact_def['methodology']}\n"
            f"Критерии: {',\n-'.join(_artifact_def['criteria'])}\n\n"
            f"Реальные данные: {_artifact_def['data_source'] if 'data_source' in _artifact_def else 'Ответы пользователя'}\n\n"
            "Предложи несколько вариантов для этого артефакта, основываясь на предыдущем контексте.\n\n"
            "Каждый вариант должен удовлетворять всем критериям.\n\n"
            "Для каждого варианта дай высокоуровневую оценку по каждому критерию."
        )
        #print(f"DEBUG: {prompt}")
        return SYSTEM_PROMPT + "\n\n" + FORMAT_OPTIONS_PROMPT + "\n\n" + TOOL_POLICY_PROMPT + "\n\n" + prompt

    _agent = create_agent(
        model=model,
        tools=[_think_tool, _yandex_tool], # Includes internal scratchpad and search
        #system_prompt=SYSTEM_PROMPT + "\n\n" + prompt,
        middleware=[build_agent_prompt, provider_then_tool],
        response_format=ArtifactOptions,
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

        # 3) если артефакта с таким id нет — создаём, иначе обновляем
        details = artifacts.get(_artifact_id)
        if details is None:
            details = {
                "artifact_definition": _artifact_def,  # копируем по ID
                "artifact_final_text": "",             # пусто на входе
                "artifact_options": [],                # можно заполнить на выходе
                "selected_option": -1,                 # пустой выбор
            }
            artifacts[artifact_id] = details
        else:
            details["artifact_definition"] = _artifact_def
            details["artifact_final_text"] = ""       # сбрасываем на входе


        #state.pop("structured_response", {})
        result: ArtifactAgentState = _agent.invoke(
            state,
            config=config,
            context=runtime.context,
        )
        structured_response = result.pop("structured_response", {}) or {}
        formatted_options_text = _format_artifact_options_text(structured_response) + "\n\nВыберите один из вариантов или скажите, что поправить."
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

def create_generation_agent(model: BaseChatModel, artifact_id: int):
    """Creates the artifact generation agent."""
    _artifact_id = artifact_id
    _artifact_def = ARTIFACTS[_artifact_id]
    @dynamic_prompt
    def build_agent_prompt(request: ModelRequest) -> str:
        agent_state: ArtifactAgentState = request.state
        artifacts_in_state = agent_state.get("artifacts") or {}
        context_str = "\n".join(
            f"{a['artifact_definition']['id']} {a['artifact_definition']['name']}: {a['artifact_final_text']}"
            for a in artifacts_in_state.values()
        )
        current_artifact = artifacts_in_state.get(_artifact_id)
        options = current_artifact.get("artifact_options")
        selected_option = options[current_artifact["selected_option"]]
        #prompt = (
        #    f"Мы прорабатываем {agent_state.get("user_prompt", "")}\n\n"
        #    f"{context_str}\n"
        #    f"Мы переходим к этапу {_artifact_id + 1}: {_artifact_def['name']}.\n"
        #    f"Цель: {_artifact_def['goal']}\n"
        #    f"Методология: {_artifact_def['methodology']}\n"
        #    f"Критерии: {', '.join(_artifact_def['criteria'])}\n\n"
        #    "Предложи 2-3 варианта для этого артефакта, основываясь на предыдущем контексте.\n\n"
        #    "Для каждого варианта дай высокоуровневую оценку по каждому критерию."
        #)
        prompt = (
            f"Мы работаем над этапом {_artifact_id + 1}: {_artifact_def['name']}.\n"
            f"Цель: {_artifact_def['goal']}\n"
            f"Методология: {_artifact_def['methodology']}\n"
            f"Реальные данные: {_artifact_def['data_source'] if 'data_source' in _artifact_def else 'Ответы пользователя'}\n\n"
            f"{context_str}\n"
            "Пользователь прислал ответ. \n"
            f"Вариант {_artifact_id + 1}: {selected_option["artifact_option"]}\n"
            "Cформируй финальную версию артефакта.\n"
            "Обязательно дай оценку по каждому из критериев."
            f"Список критериев: {', '.join(_artifact_def['criteria'])}\n"
        )
        #print(f"DEBUG: {prompt}")
        return SYSTEM_PROMPT + "\n\n" + TOOL_POLICY_PROMPT + "\n\n" + prompt    

    
    _agent = create_agent(
        model=model,
        tools= [_think_tool, _yandex_tool], # Includes internal scratchpad and search
        middleware=[build_agent_prompt, provider_then_tool],
        response_format=AftifactFinalText,
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
        formatted_final_text = _format_artifact_final_text(structured_response) + "\n\nПодтвердите артефакт или скажите, что поправить."

        # on exit: enforce spec
        result_artifacts = result.get("artifacts") or {}
        result["artifacts"] = result_artifacts

        result["current_artifact_text"] = result.get("structured_response", {}).get("artifact_final_text", "")   
        result_details = result_artifacts.get(_artifact_id)# or details
        result_details["artifact_final_text"] = result.get("structured_response", {}).get("artifact_final_text", "")  +  "\n" + result.get("structured_response", {}).get("artifact_estimation", "") 
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
    artifact_id: int = 0
):
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
    builder.add_node("generate_options", create_options_generator_node(model=_llm, artifact_id=artifact_id))
    builder.add_node("select_option", select_option_node)
    builder.add_node("generate_aftifact", create_generation_agent(model=_llm, artifact_id=artifact_id))
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
