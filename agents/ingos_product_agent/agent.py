
import uuid
import os
from typing import List, Any

import config as cfg

#os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
#os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

#from langchain_openai import ChatOpenAI
#from langchain_mistralai import ChatMistralAI
#from langchain_gigachat import GigaChat
#from agents.assistants.yandex_tools.yandex_tooling import ChatYandexGPTWithTools as ChatYandexGPT

from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig, RunnableLambda
#from langchain_core.callbacks.file import FileCallbackHandler

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from platform_utils.llm_logger import JSONFileTracer
from services.kb_manager.notifications import KBReloadContext, register_reload_listener

from .state.state import ProductAgentState
from ..state.state import ConfigSchema
from ..utils import create_tool_node_with_fallback, show_graph, _print_event, _print_response
from ..user_info import user_info
from ..utils import ModelType
from ..llm_utils import get_llm
from .retrievers.retriever_utils import get_search_tool, reload_retrievers as reload_product_retrievers, get_chroma_vectore_store
from .retrievers.vector_store import VectorStore

from ..palimpsest_sessions import PalimpsestSessionManager, PalimpsestSessionMiddleware

import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from .prompts.prompts import (
    product_prompt)
import time

ANON_LOG_NAME = f"LLM_requests_log_{time.strftime("%Y%m%d%H%M")}"


def _content_parts(content: Any) -> List[dict[str, Any]]:
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    parts: List[str] = []
    for part in _content_parts(content):
        if part.get("type") != "text":
            continue
        text_value = part.get("text")
        if isinstance(text_value, str) and text_value.strip():
            parts.append(text_value.strip())
    return "\n".join(parts).strip()


def reset_or_run(state: ProductAgentState, config: RunnableConfig) -> str:
    last_parts = _content_parts(state["messages"][-1].content)
    if last_parts and last_parts[0].get("type") == "reset":
        return "reset_memory"
    if isinstance(state["messages"][-1].content, str) and state["messages"][-1].content.strip().lower() == "reset":
        return "reset_memory"
    else:
        return "default_agent"
    
def reset_memory(state: ProductAgentState) -> ProductAgentState:
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


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    product: str = "default",
    use_platform_store: bool = False,
    checkpoint_saver=None,
    *,
    prefetch_top_k: int = 3,
):
    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    print(f"ProductAgent initialization started for {product}...")
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

    palimpsest_sessions = None
    if cfg.USE_ANONIMIZER:
        from palimpsest import Palimpsest

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
        palimpsest_sessions = PalimpsestSessionManager(
            Palimpsest(verbose=False, run_entities=anon_entities)
        )
    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    team_llm = get_llm(model = cfg.TEAM_GPT_MODEL, provider = provider.value, temperature=0.4)
    
    search_kb = get_search_tool(product)
    vector_docs_path = os.getenv("INGOS_VECTOR_DOCS_PATH", "./data/docs")
    vector_store_path = os.getenv("INGOS_VECTOR_STORE_PATH", "./data/vector_store")
    vector_store = get_chroma_vectore_store(docs_path=vector_docs_path, vector_store_path=vector_store_path)
    search_tools = [
        search_kb,
    ]
    

    def get_validator(agent: str):

        def validate_answer(state: ProductAgentState, config: RunnableConfig | None = None):
            queries = []
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.type != "ai" or len(last_message.tool_calls) > 0:
                return state

            for message in messages:
                if message.type == "human":
                    query_text = _content_text(message.content)
                    if query_text:
                        queries.append(query_text)
            
            state.update({"verification_result": "OK",
                            "verification_reason": "OK"})

            return state

        return validate_answer

    middleware = (
        [PalimpsestSessionMiddleware(palimpsest_sessions, log_path=f"./logs/{ANON_LOG_NAME}")]
        if palimpsest_sessions
        else []
    )

    def with_validator(agent_runnable, validator):
        return agent_runnable | RunnableLambda(validator)

    def prefetch_context(state: ProductAgentState) -> ProductAgentState:
        if prefetch_top_k <= 0:
            return state

        messages = list(state["messages"])
        if not messages:
            return state

        # Remove any previously injected prefetch messages to avoid duplication.
        messages = [
            msg
            for msg in messages
            if not (
                isinstance(msg, SystemMessage)
                and isinstance(getattr(msg, "additional_kwargs", {}), dict)
                and msg.additional_kwargs.get("source") == "vector_prefetch"
            )
        ]

        last_user_idx = next(
            (idx for idx in range(len(messages) - 1, -1, -1) if messages[idx].type == "human"),
            None,
        )
        if last_user_idx is None:
            state["messages"] = messages
            return state

        last_user = messages[last_user_idx]
        query = _content_text(getattr(last_user, "content", []))
        if not query:
            query = f"information about product {product}"

        try:
            docs = vector_store.search(query=query, n_results=prefetch_top_k, product=product)
        except Exception as exc:
            logging.warning("Vector prefetch failed for product %s: %s", product, exc)
            state["messages"] = messages
            return state

        if not docs:
            state["messages"] = messages
            return state

        context_chunks = "\n\n".join(
            doc.page_content for doc in docs if getattr(doc, "page_content", "").strip()
        ).strip()
        if not context_chunks:
            state["messages"] = messages
            return state

        context_message = SystemMessage(
            content=f"Prefetched knowledge base context:\n{context_chunks}",
            additional_kwargs={"source": "vector_prefetch", "doc_count": len(docs)},
        )

        updated_messages = messages[:last_user_idx] + [context_message] + messages[last_user_idx:]
        state["messages"] = updated_messages
        logging.debug(
            "Prefetched %d documents for product=%s using query='%s'",
            len(docs),
            product,
            query[:120],
        )
        return state

    default_agent = with_validator(
        create_agent(
            model=team_llm,
            tools=search_tools,
            system_prompt=product_prompt,
            name="product_assistant",
            state_schema=ProductAgentState,
            checkpointer=memory,
            middleware=middleware,
            debug=cfg.DEBUG_WORKFLOW,
        ),
        get_validator("default_agent"),
    )
    

    builder = StateGraph(ProductAgentState, config_schema=ConfigSchema)
    # Define nodes
    builder.add_node("fetch_user_info", user_info)
    def reset_memory_node(state: ProductAgentState, config: RunnableConfig) -> ProductAgentState:
        if palimpsest_sessions:
            palimpsest_sessions.reset_from_config(config)
        return reset_memory(state)

    builder.add_node("reset_memory", reset_memory_node)
    builder.add_node("prefetch_context", prefetch_context)
    builder.add_node("default_agent", default_agent)

    # Define edges
    builder.add_edge(START, "fetch_user_info")
    builder.add_conditional_edges(
        "fetch_user_info",
        reset_or_run,
        {
            "reset_memory": "reset_memory",
            "default_agent": "prefetch_context",
        }
    )
    builder.add_edge("reset_memory", END)
    builder.add_edge("prefetch_context", "default_agent")
    agent = builder.compile(name="ingos_product_agent", checkpointer=memory).with_config({"callbacks": callback_handlers})

    agent_key = f"product_{product}"
    def _handle_kb_reload(context: KBReloadContext) -> None:
        logging.info("KB reload requested for %s: %s", agent_key, context.reason)
        reload_product_retrievers(context)
    register_reload_listener(agent_key, _handle_kb_reload)

    print(f"ProductAgent initialized for {product}")
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
