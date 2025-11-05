
import uuid
import os
from typing import List, Any
from copy import copy

import config as cfg

#os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
#os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

#from langchain_openai import ChatOpenAI
#from langchain_mistralai import ChatMistralAI
#from langchain_gigachat import GigaChat
#from agents.assistants.yandex_tools.yandex_tooling import ChatYandexGPTWithTools as ChatYandexGPT

from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig, RunnableLambda
#from langchain_core.callbacks.file import FileCallbackHandler

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from platform_utils.llm_logger import JSONFileTracer
from services.kb_manager.notifications import KBReloadContext, register_reload_listener


from .tools.yandex_search import YandexSearchTool

from .classifier import classify_request, summarise_request
from .validate_answer import vadildate_AI_answer, CheckResult
from .state.state import SDAccAgentState
from ..state.state import ConfigSchema
from ..utils import create_tool_node_with_fallback, show_graph, _print_event, _print_response
from ..user_info import user_info
from ..utils import ModelType
from .tools.tools import get_term_and_defition_tools
from ..retrievers.retriever import get_search_tool, get_tickets_search_tool, reload_retrievers as reload_sd_retrievers
from ..llm_utils import get_llm

from .augment_query import get_terms_and_definitions

from palimpsest import Palimpsest

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


from .prompts.prompts import (
    default_prompt, 
    sm_prompt, 
    sd_prompt, 
    sv_prompt, 
    sd_agent_web_prompt, 
    sm_agent_web_prompt, 
    default_search_web_prompt)
import time

ANON_LOG_NAME = f"LLM_requests_log_{time.strftime("%Y%m%d%H%M")}"


def reset_or_run(state: SDAccAgentState, config: RunnableConfig) -> str:
    if state["messages"][-1].content[0].get("type") == "reset":
        return "reset_memory"
    else:
        return "augment_query"

def augment_query(state: SDAccAgentState, config: RunnableConfig) -> SDAccAgentState:
    """
    Retrieve relevant terms/definitions and append them as a SystemMessage,
    so downstream agents see the extra context.
    """
    if not state["messages"]:
        return state                       # safety guard

    last_user_msg = state["messages"][-1]
    # We only augment on real user turns
    if last_user_msg.type != "human":
        return state

    glossary = get_terms_and_definitions(
        last_user_msg.content[0]["text"]
    )

    # Add ONE additional system message
    new_msgs = (
        [AIMessage(content=glossary)] 
        if len(glossary.strip()) 
        else []) + state["messages"]
    return {"messages": new_msgs}

def route_request(state: SDAccAgentState, config: RunnableConfig) -> str:
    queries = []
    queries.extend(
        message.content[0]["text"]
        for message in state["messages"]
        if message.type == "human"
    )
    role = config["configurable"].get("user_role", "default")
    if role == "service_desk":
        role_name = "Сотрудник техподдержки"
    elif role == "sales_manager":
        role_name = "Сотрудник отдела продаж"
    else:
        role_name = "Сотрудник компании Интерлизинг"
    summary_query = f"{summarise_request(";".join(queries))}\n\nUser role: {role_name}"
    return classify_request(summary_query)

def reset_memory(state: SDAccAgentState) -> SDAccAgentState:
    """
    Delete every message currently stored in the thread’s state.
    """
    all_msg_ids = [m.id for m in state["messages"]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids],
        "verification_result": "",
        "verification_reason": ""
    }

roles = {
    "service_desk": "Provides answers to questions related to resolving problems with issues in Interleasing's systems and business processes.",
    "sales_manager": "Provides answers to questions related to sales activities, products, sales conditions, discounts provided to our clients, leasing agreements and so on. Consults sales managers for all sales related processes, including activities of underwrighting, risks, operations and so on.",
    "default": "Provides answers to questions internal rules and features provided to Employees. Consults Interleasing Employees on any HR related questions."
}

def anonymize_message_content(content: Any, anonymizer: Palimpsest) -> Any:
    # Content can be str OR a list of "content parts" dicts.
    if isinstance(content, str):
        return anonymizer.anonimize(content)
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                p = dict(part)
                for key in ("text", "content", "input", "title", "caption", "markdown", "explanation"):
                    if isinstance(p.get(key), str):
                        p[key] = anonymizer.anonimize(p[key])
                out.append(p)
            else:
                out.append(part)
        return out
    return content

def initialize_agent(provider: ModelType = ModelType.GPT, role: str = "default", use_platform_store: bool = False):
    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    print("SDAgent initialization started...")
    log_name = f"sd_ass_{time.strftime("%Y%m%d%H%M")}"
    #log_handler = FileCallbackHandler(f"./logs/{log_name}")
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]
    if cfg.LANGFUSE_URL and len(cfg.LANGFUSE_URL) > 0:
        langfuse = Langfuse(
            public_key=cfg.LANGFUSE_PUBLIC,
            secret_key=cfg.LANGFUSE_SECRET,
            host=cfg.LANGFUSE_URL
        )
        lf_handler = CallbackHandler()
        callback_handlers += [lf_handler]

    anonymizer = None
    if cfg.USE_ANONIMIZER:
        anon_entities = [
            "RU_PERSON"
            ,"CREDIT_CARD"
            ,"PHONE_NUMBER"
            ,"IP_ADDRESS"
            ,"URL"
            ,"RU_PASSPORT"
            ,"SNILS"
            ,"INN"
            ,"RU_BANK_ACC"
            ,"TICKET_NUMBER"
        ]
        anonymizer = Palimpsest(verbose=False, run_entities=anon_entities)
    memory = None if use_platform_store else MemorySaver()
    #team_llm = get_llm(cfg.TEAM_GPT_MODEL, temperature=1)
    team_llm = get_llm(model = cfg.TEAM_GPT_MODEL, provider = provider.value, temperature=0.4)
    
    search_kb = get_search_tool()
    (lookup_term, lookup_abbreviation) = get_term_and_defition_tools()
    search_tickets = get_tickets_search_tool()
    search_tools = [
        search_kb,
        lookup_term,
        lookup_abbreviation
    ]
    
    yandex_tool = YandexSearchTool(
        api_key=cfg.YA_API_KEY,
        folder_id=cfg.YA_FOLDER_ID,
        max_results=3
    )
    
    web_tools = [
        yandex_tool,
        lookup_term,
        lookup_abbreviation
    ]

    class SDAgentAnonymizationMiddleware:
        def __init__(self, anonymizer: Palimpsest):
            self._anonymizer = anonymizer

        def modify_model_request(self, request, state):
            anon_msgs: List[BaseMessage] = []

            with open(f"./logs/{ANON_LOG_NAME}", "a", encoding="utf-8") as f:
                for message in request.messages:
                    anon_msg = copy(message)
                    f.write(f"BEFORE ANONIMIZATION:\n{anon_msg.content}\n")
                    anon_msg.content = anonymize_message_content(
                        message.content, self._anonymizer
                    )
                    f.write(f"AFTER ANONIMIZATION:\n{anon_msg.content}\n\n")
                    anon_msgs.append(anon_msg)
            request.messages = anon_msgs
            return request

    def get_validator(agent: str):

        if agent == "sd_agent":
            search_web_prompt = sd_agent_web_prompt
        elif agent == "sm_agent":
            search_web_prompt = sm_agent_web_prompt
        else:
            search_web_prompt = default_search_web_prompt

        web_search_agent =      create_agent(
            model=team_llm, 
            tools=web_tools, 
            system_prompt=search_web_prompt, 
            name="search_web_sd", 
            middleware=middleware,
            #state_schema = State, 
            checkpointer=memory, 
            debug=cfg.DEBUG_WORKFLOW)

        def validate_answer(state: SDAccAgentState):
            queries = []
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.type != "ai" or len(last_message.tool_calls) > 0:
                return state

            ai_answer = "No asnwer."
            for message in messages:
                if message.type == "human":
                    queries.append(message.content[0]["text"])
            
            ai_answer = last_message.content
            
            if anonymizer:
                with open(f"./logs/{ANON_LOG_NAME}", "a", encoding="utf-8") as f:
                    f.write(f"BEFORE DEANONIMIZATION:\n{last_message.content}\n")
                    last_message.content = anonymizer.deanonimize(last_message.content)
                    f.write(f"AFTER DEANONIMIZATION:\n{last_message.content}\n\n")

            summary_query = summarise_request(";".join(queries))
            result = vadildate_AI_answer(summary_query, ai_answer)
            if result.result == "NO":
                search_result = web_search_agent.invoke({"messages": [HumanMessage(content=[{"type": "text", "text": summary_query}])]})
                web_answer = "⚡** Ответ получен из поисковой системы Яндекс **.\n\n" + search_result.get("messages", [])[-1].content
                new_messages = messages[:-1] + [AIMessage(content=web_answer)]
                return {"messages": new_messages,
                        "verification_result": result.result,
                        "verification_reason": result.reason}
            else:
                state.update({"verification_result": result.result,
                              "verification_reason": result.reason})
                return state

        return validate_answer

    middleware = [SDAgentAnonymizationMiddleware(anonymizer)] if anonymizer else []

    def with_validator(agent_runnable, validator):
        return agent_runnable | RunnableLambda(validator)

    sd_agent = with_validator(
        create_agent(
            model=team_llm,
            tools=search_tools + [search_tickets],
            system_prompt=sd_prompt,
            name="assistant_sd",
            state_schema=SDAccAgentState,
            checkpointer=memory,
            middleware=middleware,
            debug=cfg.DEBUG_WORKFLOW,
        ),
        get_validator("sd_agent"),
    )
    sm_agent = with_validator(
        create_agent(
            model=team_llm,
            tools=search_tools,
            system_prompt=sm_prompt,
            name="assistant_sm",
            state_schema=SDAccAgentState,
            checkpointer=memory,
            middleware=middleware,
            debug=cfg.DEBUG_WORKFLOW,
        ),
        get_validator("sm_agent"),
    )
    default_agent = with_validator(
        create_agent(
            model=team_llm,
            tools=search_tools,
            system_prompt=default_prompt,
            name="assistant_default",
            state_schema=SDAccAgentState,
            checkpointer=memory,
            middleware=middleware,
            debug=cfg.DEBUG_WORKFLOW,
        ),
        get_validator("default_agent"),
    )
    

    builder = StateGraph(SDAccAgentState, config_schema=ConfigSchema)
    # Define nodes
    builder.add_node("fetch_user_info", user_info)
    builder.add_node("reset_memory", reset_memory)
    builder.add_node("augment_query", augment_query)

    builder.add_node("sm_agent", sm_agent)
    builder.add_node("sd_agent", sd_agent)
    builder.add_node("default_agent", default_agent)

    # Define edges
    builder.add_edge(START, "fetch_user_info")
    builder.add_conditional_edges(
        "fetch_user_info",
        reset_or_run,
        {
            "reset_memory": "reset_memory",
            "augment_query": "augment_query",
        }
    )

    builder.add_conditional_edges(
        "augment_query",
        route_request,
        {
            "sm_agent": "sm_agent",
            "sd_agent": "sd_agent",
            "default_agent": "default_agent",
        }
    )
    builder.add_edge("reset_memory", END)
    agent = builder.compile(name="interleasing_qa_agent", checkpointer=memory).with_config({"callbacks": callback_handlers})

    agent_key = "service_desk"
    def _handle_kb_reload(context: KBReloadContext) -> None:
        logging.info("KB reload requested for %s: %s", agent_key, context.reason)
        reload_sd_retrievers(context)
    register_reload_listener(agent_key, _handle_kb_reload)

    logging.info("SDAgent initialized")
    return agent 


if __name__ == "__main__":
    assistant_graph = initialize_agent(model=ModelType.GPT)

    #show_graph(assistant_graph)
    from langchain_core.messages import HumanMessage

    # Let's create an example conversation a user might have with the assistant
    tutorial_questions = [
        "Кто такие кей юзеры?",
        "Не работает МФУ",
        "Как отресетить график?"
    ]

    thread_id = str(uuid.uuid4())

    cfg = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "user_info": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    assistant_graph.invoke({"messages": [HumanMessage(content=[{"type": "text", "text": "Кто ты?"}])]}, cfg)

    _printed = set()
    for question in tutorial_questions[:2]:
        events = assistant_graph.stream(
            {"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, cfg, stream_mode="values"
        )
        print("USER: ", question)
        print("-------------------")
        print("ASSISTANT:")
        for event in events:
            #_print_event(event, _printed)
            _print_response(event, _printed)
        print("===================")

    print("RESET")
    events = assistant_graph.invoke(
        {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, cfg, stream_mode="values"
    )
    #for event in events:
    #    _print_response(event, _printed)

    for question in tutorial_questions[2:]:
        events = assistant_graph.stream(
            {"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, cfg, stream_mode="values"
        )
        print("USER: ", question)
        print("-------------------")
        print("ASSISTANT:")
        for event in events:
            #_print_event(event, _printed)
            _print_response(event, _printed)
        print("===================")
