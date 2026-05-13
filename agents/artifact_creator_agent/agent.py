from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, Dict, List, Literal, Mapping, NotRequired, Optional, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from langgraph.utils.runnable import RunnableCallable


from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agents import state
import config

from utils.utils import is_valid_json_string

from agents.tools.think import ThinkTool
from agents.tools.yandex_search import YandexSearchTool as SearchTool
from agents.structured_prompt_utils import provider_then_tool
from agents.utils import (
    ModelType,
    extract_text,
    build_internal_invoke_config,
    get_llm,
)

from platform_guardrails.logging import RedactingJSONFileTracer
from platform_guardrails.middleware import (
    PrivacyModelRequestMiddleware,
    SecurityScannerMiddleware,
    guarded_node,
)
from platform_guardrails.privacy import PrivacyRail
from platform_guardrails.scanners import (
    LLMGuardScannerProfile,
    LLMGuardScannerRail,
    ScannerFailurePolicy,
)
from platform_guardrails.tool_policy import ARTIFACT_CREATOR_TOOL_PROFILES, ToolSecurityProfile
from platform_guardrails.tool_registry import GuardedToolRegistry

from .state import (
    ArtifactCreatorAgentContext
    , ArtifactCreatorAgentState
    , ConfirmationAgentState
)
from .utils import UserConfirmation
from .tools import commit_artifact_final_text
from ..store_artifacts import store_artifacts

from .prompts import (
    ARTIFACT_STORE_ERROR_RU,
    GREETINGS_WITH_PROMPT_RU
    , GREETINGS_WITHOUT_PROMPT_RU
    , SET_PROPMT_REQUEST_RU
    , PROMPT_CONFIRMATION_RU
    , DEFAULT_SYSTEM_PROMPT_RU
    , СOMMIT_TOOL_PROMPT_RU
    , ARTIFACT_URL_PROMPT_RU
)

LOG = logging.getLogger(__name__)
_CONFIRMATION_STREAM_TAG = "internal_confirmation_agent"

def route(state: ArtifactCreatorAgentState) -> str:
    if not state.get("greeted"):
        return "greetings"

    if not state.get("phase") and not state.get("system_prompt"):
        return "set_prompt"
    
    return state.get("phase")

def create_greetings_node(initial_system_prompt: Any | None = None):

    def greetings_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
        system_prompt = initial_system_prompt
        if system_prompt is None and runtime.context and runtime.context.get("system_prompt"):
            system_prompt = runtime.context["system_prompt"]
        if system_prompt is not None and not state.get("system_prompt"):
            state["system_prompt"] = system_prompt
            state["phase"] = "cleanup"

        if not state.get("greeted"):
            greet = (
                GREETINGS_WITH_PROMPT_RU
                if state.get("system_prompt")
                else GREETINGS_WITHOUT_PROMPT_RU
            )
            state["messages"] = (state.get("messages") or []) + [AIMessage(content=greet)]
            state["greeted"] = True
        return state

    return greetings_node


def create_set_prompt_node(model: BaseChatModel):

    def set_prompt_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
        #state["phase"] = "run"
        
        original_messages = state.get("messages", [])

        last_user_msg = None
        for msg in reversed(original_messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                last_user_msg = msg
                break
        #system_prompt = None
        if last_user_msg is None:
            response_text = SET_PROPMT_REQUEST_RU
            state["phase"] = "set_prompt"
            return state
        else:
            response_text = PROMPT_CONFIRMATION_RU
            #all_msg_ids = [m.id for m in state["messages"]]
            if isinstance(last_user_msg.content, str):
                state["system_prompt"] = last_user_msg.content
            else:
                state["system_prompt"] = extract_text(last_user_msg)
            state["phase"] = "cleanup"
            # Returning RemoveMessage instances instructs the reducer to delete them
            #return {
            #    "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [AIMessage(content=response_text)],
            #    "system_prompt": system_prompt,
            #    "phase" : "cleanup"
            #}            
        state["messages"].append(AIMessage(content=response_text))
        return state
            

    return set_prompt_node


def cleanup_messages_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
    
    all_msg_ids = [m.id for m in state["messages"][:-1]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids] + [state["messages"][-1]],
        "phase" : "run"
    }


def _message_id(message: Any) -> str | None:
    message_id = getattr(message, "id", None)
    return message_id if isinstance(message_id, str) and message_id else None


def _message_updates_with_subgraph_removals(
    input_messages: List[Any] | None,
    result_messages: List[Any] | None,
) -> List[Any]:
    """Preserve message deletions that happened inside a nested agent graph."""
    input_messages = list(input_messages or [])
    result_messages = list(result_messages or [])
    if any(_message_id(message) is None for message in input_messages):
        idless_input_keys = {
            (getattr(message, "type", type(message).__name__), repr(getattr(message, "content", None)))
            for message in input_messages
            if _message_id(message) is None
        }
        safe_result_messages = [
            message
            for message in result_messages
            if (
                getattr(message, "type", type(message).__name__),
                repr(getattr(message, "content", None)),
            )
            not in idless_input_keys
        ]
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *safe_result_messages]
    result_ids = {
        message_id
        for message in result_messages
        if not isinstance(message, RemoveMessage)
        if (message_id := _message_id(message)) is not None
    }
    removals = [
        RemoveMessage(id=message_id)
        for message in input_messages
        if (message_id := _message_id(message)) is not None and message_id not in result_ids
    ]
    return [*removals, *result_messages]


def _state_update_with_subgraph_removals(
    input_state: Dict[str, Any],
    result_state: Any,
) -> Any:
    if not isinstance(result_state, dict) or "messages" not in result_state:
        return result_state
    updated = dict(result_state)
    updated["messages"] = _message_updates_with_subgraph_removals(
        list(input_state.get("messages") or []),
        list(result_state.get("messages") or []),
    )
    return updated


def _build_run_agent(
    model: BaseChatModel,
    tools: List[Any] | None = None,
    system_prompt: Any | None = None,
    security_middleware: Any | None = None,
    privacy_middleware: Any | None = None,
    tool_execution_middleware: Any | None = None,
):
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: ArtifactCreatorAgentState = request.state
        resolved_system_prompt = state.get("system_prompt", system_prompt)
        if resolved_system_prompt:
            if not isinstance(resolved_system_prompt, str):
                try:
                    resolved_system_prompt = extract_text(HumanMessage(content=resolved_system_prompt))
                except Exception:
                    resolved_system_prompt = str(resolved_system_prompt)
            return f"{resolved_system_prompt}\n\n{СOMMIT_TOOL_PROMPT_RU}"
        return DEFAULT_SYSTEM_PROMPT_RU

    middleware = [build_prompt]
    if security_middleware is not None:
        middleware.append(security_middleware)
    if privacy_middleware is not None:
        middleware.append(privacy_middleware)
    if tool_execution_middleware is not None:
        middleware.append(tool_execution_middleware)

    run_tools = tools if tools is not None else [commit_artifact_final_text]

    return create_agent(
        model=model,
        tools=run_tools,
        middleware=middleware,
        state_schema=ArtifactCreatorAgentState,
        context_schema=ArtifactCreatorAgentContext,
    )


def create_run_node(
    model: BaseChatModel,
    tools: List[Any] | None = None,
    system_prompt: Any | None = None,
    security_middleware: Any | None = None,
    privacy_middleware: Any | None = None,
    tool_execution_middleware: Any | None = None,
):
    run_agent = _build_run_agent(
        model,
        tools,
        system_prompt,
        security_middleware=security_middleware,
        privacy_middleware=privacy_middleware,
        tool_execution_middleware=tool_execution_middleware,
    )

    def run_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> Any:
        result = run_agent.invoke(state, config=config, context=runtime.context)
        return _state_update_with_subgraph_removals(state, result)

    async def arun_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> Any:
        result = await run_agent.ainvoke(state, config=config, context=runtime.context)
        return _state_update_with_subgraph_removals(state, result)

    return RunnableCallable(run_node, arun_node, trace=False)


#def create_run_node(model: BaseChatModel):
#    _run_agent = _build_run_agent(model)
#    def run_node(
#        state: ArtifactCreatorAgentState,
#        config: RunnableConfig,
#        runtime: Runtime[ArtifactCreatorAgentContext],
#    ) -> ArtifactCreatorAgentState:
#        
#        result_state = _run_agent.invoke(state, config=config, context=runtime.context)
#        return result_state
#
#    return run_node


def _build_confirmation_agent(
    model: BaseChatModel,
    security_middleware: Any | None = None,
    privacy_middleware: Any | None = None,
):

    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        state: ConfirmationAgentState  = request.state
        artifact = state.get("artifact")
        if not isinstance(artifact, str):
            artifact = str(artifact or "")

        #response = state.get("last_user_answer")
        prompt = (
            "You have to analyze user's response to artifact.\n" 
            "=======================================\n"
            f"ARTIFACT TEXT:\n{artifact}\n"
            "END OF ARTIFACT TEXT\n"
            "=======================================\n"
            "Determine if user confirmed the text or requested change."        
            #f"User response is: {response}\n"
        )
        return prompt


    middleware = [build_prompt, provider_then_tool]
    if security_middleware is not None:
        middleware.insert(1, security_middleware)
    if privacy_middleware is not None:
        middleware.append(privacy_middleware)

    confirmation_agent = create_agent(
        model=model,
        middleware=middleware,
        response_format=UserConfirmation,
        state_schema=ConfirmationAgentState,
    )
    return confirmation_agent #.with_config({"tags": [_CONFIRMATION_STREAM_TAG]})

def create_confirmation_node(
    model: BaseChatModel,
    security_middleware: Any | None = None,
    privacy_middleware: Any | None = None,
):
    _confirmation_agent = _build_confirmation_agent(
        model,
        security_middleware=security_middleware,
        privacy_middleware=privacy_middleware,
    )
    def confirmation_node(
        state: ArtifactCreatorAgentState, 
        config: RunnableConfig, 
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
        last_message = state["messages"][-1]
        artifact_id = state.get("artifact_id", 0)
        artifacts = state.get("artifacts")
        artifact = ""
        if artifacts and artifact_id in artifacts:
            artifact = artifacts[artifact_id].get("artifact_final_text", "")

        confirmation_state = {
            "messages": [last_message],  # ensure no chat history is provided
            "artifact": artifact,
        }
        confirm_config = build_internal_invoke_config(
            config,
            extra_tags=[_CONFIRMATION_STREAM_TAG],
        )

        result = _confirmation_agent.invoke(confirmation_state, config=confirm_config, context=runtime.context)
        structured = result.get("structured_response") or {}
        is_artifact_confirmed = bool(getattr(structured, "is_artifact_confirmed", False))
        update: Dict[str, Any] = {"is_artifact_confirmed": is_artifact_confirmed}

        result_messages = list(result.get("messages") or []) if isinstance(result, dict) else []
        last_message_id = _message_id(last_message)
        result_message_ids = {_message_id(message) for message in result_messages}
        if last_message_id and last_message_id not in result_message_ids:
            public_block_messages = [
                message
                for message in result_messages
                if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
            ]
            update["messages"] = [RemoveMessage(id=last_message_id), *public_block_messages]

        if is_artifact_confirmed:
            try:
                store_url = store_artifacts(state["artifacts"] or [])
                link_text = f"[{ARTIFACT_URL_PROMPT_RU}]({store_url})"
                update["final_artifact_url"] = store_url
            except Exception as e:
                logging.error(f"Error occured at store_artifacts.\nException: {e}")
                update["final_artifact_url"] = ""
            #state["messages"].append(AIMessage(content=f"\n***Работа завершена***.\nСогласованный текст артефакта:\n\n{artifact}\n\n{link_text}\n"))
            #state["messages"].append(AIMessage(content=f"\n\n***Работа завершена***.\nСогласованный текст артефакта:\n\n{artifact}\n\n{link_text}\n"))
            
            update["phase"] = "ready"
        return update

    return confirmation_node

def final_print_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
    if not state.get("is_artifact_confirmed", False):
        return Command(update={"phase": "run"})

    store_url = state.get("final_artifact_url", "")
    if store_url:
        link_text = f"[{ARTIFACT_URL_PROMPT_RU}]({store_url})"
    else:
        link_text = ARTIFACT_STORE_ERROR_RU
        
    message_update = [AIMessage(content=f"\n\n***Работа завершена***.\n\n{link_text}\n")]    
    return Command(
        update={
            "messages": message_update,
            "phase": "ready",
        },
    )


def is_confirmed(
    state: ArtifactCreatorAgentState,
) -> ArtifactCreatorAgentState:

    if not state.get("is_artifact_confirmed", False):
        return "run"
    else:
        #artifact_id = state.get("artifact_id", 0)
        #artifacts = state.get("artifacts")
        #artifact = ""
        #if artifacts and artifact_id in artifacts:
        #    artifact = artifacts[artifact_id].get("artifact_final_text", "")

        #state["messages"].append(AIMessage(content=f"\nРабота завершена.\nСогласованный текст артефакта:\n\n{artifact}"))
        return "ready"


def ready_node(
        state: ArtifactCreatorAgentState,
        config: RunnableConfig,
        runtime: Runtime[ArtifactCreatorAgentContext],
    ) -> ArtifactCreatorAgentState:
    artifact_id = state.get("artifact_id", 0)
    artifacts = state.get("artifacts")
    artifact = ""
    if artifacts and artifact_id in artifacts:
        artifact = artifacts[artifact_id].get("artifact_final_text", "")

    store_url = state.get("final_artifact_url", "")
    if store_url:            
        link_text = f"[{ARTIFACT_URL_PROMPT_RU}]({store_url})"
    else:
        link_text = ARTIFACT_STORE_ERROR_RU
        
    message_update = [AIMessage(content=f"\n\n***Работа завершена***.\nСогласованный текст артефакта:\n\n{artifact}\n\n{link_text}\n")]    
    return Command(
        update={
            "messages": message_update,
            "phase": "ready",
        },
    )


def _build_guarded_run_tool_bundle(
    extra_tools: List[Any] | None,
    *,
    scanner_rail: LLMGuardScannerRail | None,
    privacy_rail: PrivacyRail,
    guardrail_log_path: str,
    guardrail_tool_profiles: Mapping[str, ToolSecurityProfile | Mapping[str, Any]] | None,
    guardrail_unprofiled_tools: Literal["block", "allow_read_only"],
):
    profiles: dict[str, ToolSecurityProfile | Mapping[str, Any]] = dict(ARTIFACT_CREATOR_TOOL_PROFILES)
    profiles.update(dict(guardrail_tool_profiles or {}))

    run_tools = [commit_artifact_final_text, *(extra_tools or [])]
    registry = GuardedToolRegistry(unprofiled_tools=guardrail_unprofiled_tools)
    registry.register_many(run_tools, profiles)
    return registry.build_bundle(
        agent_name="artifact_creator_agent.run",
        scanner_rail=scanner_rail,
        privacy_rail=privacy_rail,
        event_log_path=guardrail_log_path,
    )


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = "en",
    checkpoint_saver=None,
    tools: List[Any] | None = None,
    system_prompt: Any | None = None,
    guardrails_enabled: bool = False,
    guardrails_locale: str = "ru-RU",
    guardrail_scanners_enabled: bool | None = None,
    guardrail_scanner_failure_policy: ScannerFailurePolicy = "fail_closed",
    guardrail_banned_topics: List[str] | None = None,
    guardrail_prompt_injection_model: str | Mapping[str, Any] | None = None,
    guardrail_prompt_injection_model_revision: str | None = None,
    guardrail_prompt_injection_threshold: float | None = None,
    guardrail_composite_input_scanners: tuple[str, ...] | None = None,
    guardrail_composite_recent_message_limit: int = 20,
    guardrail_tool_profiles: Mapping[str, ToolSecurityProfile | Mapping[str, Any]] | None = None,
    guardrail_unprofiled_tools: Literal["block", "allow_read_only"] = "block",
):
    #set_locale(locale)
    #set_models_locale(locale)
    log_name = f"artifact_creator_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = RedactingJSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]
    if config.LANGFUSE_URL and len(config.LANGFUSE_URL) > 0:
        langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        lf_handler = CallbackHandler()
        callback_handlers += [lf_handler]

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    llm = get_llm(model="base", provider=provider.value, temperature=0)
    response_analyser_llm = get_llm(model="nano", provider=provider.value, temperature=0, streaming=False)

    run_privacy_middleware = None
    confirmation_privacy_middleware = None
    set_prompt_privacy_middleware = None
    run_security_middleware = None
    confirmation_security_middleware = None
    set_prompt_security_middleware = None
    tool_execution_middleware = None
    scanner_rail = None
    run_tools = [commit_artifact_final_text, *(tools or [])]
    if guardrails_enabled:
        privacy_rail = PrivacyRail.from_palimpsest(locale=guardrails_locale)
        guardrail_log_path = f"./logs/{log_name}_guardrails.jsonl"
        set_prompt_privacy_middleware = PrivacyModelRequestMiddleware(
            privacy_rail,
            agent_name="artifact_creator_agent.set_prompt",
            event_log_path=guardrail_log_path,
        )
        run_privacy_middleware = PrivacyModelRequestMiddleware(
            privacy_rail,
            agent_name="artifact_creator_agent.run",
            guard_tool_calls=False,
            event_log_path=guardrail_log_path,
        )
        confirmation_privacy_middleware = PrivacyModelRequestMiddleware(
            privacy_rail,
            agent_name="artifact_creator_agent.confirm",
            event_log_path=guardrail_log_path,
        )

        if guardrail_scanners_enabled is None:
            guardrail_scanners_enabled = guardrails_enabled
        if guardrail_scanners_enabled:
            scanner_profile_kwargs: dict[str, Any] = {
                "banned_topics": guardrail_banned_topics,
                "failure_policy": guardrail_scanner_failure_policy,
            }
            if guardrail_prompt_injection_model is not None:
                scanner_profile_kwargs["prompt_injection_model"] = guardrail_prompt_injection_model
            if guardrail_prompt_injection_model_revision is not None:
                scanner_profile_kwargs["prompt_injection_model_revision"] = guardrail_prompt_injection_model_revision
            if guardrail_prompt_injection_threshold is not None:
                scanner_profile_kwargs["prompt_injection_threshold"] = guardrail_prompt_injection_threshold
            scanner_profile = LLMGuardScannerProfile.artifact_creator_default(
                **scanner_profile_kwargs,
            )
            scanner_rail = LLMGuardScannerRail(scanner_profile)
            set_prompt_security_middleware = SecurityScannerMiddleware(
                scanner_rail,
                agent_name="artifact_creator_agent.set_prompt",
                event_log_path=guardrail_log_path,
                scan_state_keys=("system_prompt",),
                composite_input_scanners=guardrail_composite_input_scanners,
                composite_recent_message_limit=guardrail_composite_recent_message_limit,
                composite_message_roles=("human", "tool"),
            )
            run_security_middleware = SecurityScannerMiddleware(
                scanner_rail,
                agent_name="artifact_creator_agent.run",
                event_log_path=guardrail_log_path,
                scan_system_prompt=True,
                scan_state_keys=("system_prompt",),
                composite_input_scanners=guardrail_composite_input_scanners,
                composite_recent_message_limit=guardrail_composite_recent_message_limit,
            )
            confirmation_security_middleware = SecurityScannerMiddleware(
                scanner_rail,
                agent_name="artifact_creator_agent.confirm",
                event_log_path=guardrail_log_path,
                composite_input_scanners=guardrail_composite_input_scanners,
                composite_recent_message_limit=guardrail_composite_recent_message_limit,
                blocked_structured_response_factory=lambda _decision: UserConfirmation(
                    is_artifact_confirmed=False
                ),
            )
        guarded_tool_bundle = _build_guarded_run_tool_bundle(
            tools,
            scanner_rail=scanner_rail,
            privacy_rail=privacy_rail,
            guardrail_log_path=guardrail_log_path,
            guardrail_tool_profiles=guardrail_tool_profiles,
            guardrail_unprofiled_tools=guardrail_unprofiled_tools,
        )
        run_tools = guarded_tool_bundle.tools
        tool_execution_middleware = guarded_tool_bundle.middleware

    builder = StateGraph(ArtifactCreatorAgentState)
    builder.add_node("greetings", create_greetings_node(system_prompt))
    set_prompt_node = create_set_prompt_node(llm)
    if guardrails_enabled:
        set_prompt_node = guarded_node(
            set_prompt_node,
            security_middleware=set_prompt_security_middleware,
            privacy_middleware=set_prompt_privacy_middleware,
            scan_state_keys=("system_prompt",),
            composite_input_scanners=guardrail_composite_input_scanners,
            composite_recent_message_limit=guardrail_composite_recent_message_limit,
            composite_message_roles=("human", "tool"),
        )
    builder.add_node("set_prompt", set_prompt_node)
    builder.add_node("cleanup", cleanup_messages_node)
    builder.add_node(
        "run",
        create_run_node(
            llm,
            run_tools,
            system_prompt,
            security_middleware=run_security_middleware,
            privacy_middleware=run_privacy_middleware,
            tool_execution_middleware=tool_execution_middleware,
        ),
    )
    builder.add_node(
        "confirm",
        create_confirmation_node(
            response_analyser_llm,
            security_middleware=confirmation_security_middleware,
            privacy_middleware=confirmation_privacy_middleware,
        ),
    )
    builder.add_node("final_print", final_print_node)
    builder.add_node("ready", ready_node)

    builder.add_conditional_edges(
        START,
        route,
        {
            "greetings": "greetings",
            "set_prompt": "set_prompt",
            "cleanup": "cleanup",
            "run": "run",
            "confirm": "confirm",
            "ready": "ready"
        },
    )

    builder.add_edge("greetings", END)
    builder.add_edge("set_prompt", END)
    builder.add_edge("cleanup", "run")
    builder.add_conditional_edges(
        "run",
        route,
        {
            "confirm": "confirm",
            "run": END,
            "ready": END,
            "greetings": END,
            "set_prompt": END,
            "cleanup": END,
        },
    )
    builder.add_edge("confirm", "final_print")
    builder.add_conditional_edges(
        "final_print",
        is_confirmed,
        {
            "run": "run",
            "ready": END
        },
    )
    builder.add_edge("ready", END)
    graph = builder.compile(checkpointer=memory, debug=False).with_config({"callbacks": callback_handlers})
    return graph
