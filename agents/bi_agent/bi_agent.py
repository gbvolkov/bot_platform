from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.messages import AIMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig

from platform_utils.llm_logger import JSONFileTracer
from services.kb_manager.notifications import KBReloadContext, register_reload_listener

from ..sql_query_gen import get_response
from ..state.state import ConfigSchema
from ..utils import ModelType

import config as cfg


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

GraphicType = Literal["bar_chart", "line_chart", "scatter_plot", "pie"]


class TabularAttachment(TypedDict, total=False):
    filename: str
    data: str
    mime_type: str
    format: Literal["excel", "csv"]


class ImageAttachment(TypedDict, total=False):
    filename: str
    data: str
    mime_type: str


class BiAgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    user_info: Dict[str, Any]
    question: str
    answer: str
    data_attachment: Optional[TabularAttachment]
    image_attachment: Optional[ImageAttachment]
    graphic_type: Optional[GraphicType]


DEFAULT_DATA_SOURCES = [str(Path("data") / "data.csv")]
_GRAPHIC_TYPES: set[str] = {"bar_chart", "line_chart", "scatter_plot", "pie"}


def _is_reset_requested(state: BiAgentState) -> bool:
    messages = state.get("messages") or []
    if not messages:
        return False
    last = messages[-1]
    if last.type != "human":
        return False
    content = getattr(last, "content", [])
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return first.get("type") == "reset"
    return False


def reset_or_run(state: BiAgentState, config: RunnableConfig) -> str:
    return "reset_memory" if _is_reset_requested(state) else "generate_report"


def reset_memory(state: BiAgentState) -> BiAgentState:
    all_msg_ids = [m.id for m in state.get("messages", [])]
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids],
        "question": "",
        "answer": "",
        "data_attachment": None,
        "image_attachment": None,
        "graphic_type": None,
    }


def user_info(state: BiAgentState, config: RunnableConfig) -> Dict[str, Dict[str, Any]]:
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_id")
    user_role = configuration.get("user_role", "default")
    return {"user_info": {"user_id": user_id, "user_role": user_role}}


def _extract_question(messages: List[AnyMessage]) -> str:
    for message in reversed(messages):
        if message.type != "human":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            candidate = content.strip()
            if candidate:
                return candidate
        elif isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = (item.get("text") or "").strip()
                    if text_value:
                        parts.append(text_value)
                elif isinstance(item, str):
                    text_value = item.strip()
                    if text_value:
                        parts.append(text_value)
            if parts:
                return "\n".join(parts).strip()
    return ""


def _safe_remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        logging.debug("BI agent could not delete temporary file %s", path)


def _encode_tabular_file(file_path: Optional[str]) -> Optional[TabularAttachment]:
    if not file_path:
        return None
    path = Path(file_path)
    if not path.exists():
        logging.warning("BI agent expected data file %s but it was missing.", path)
        return None
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logging.warning("BI agent failed to read data file %s: %s", path, exc)
        return None
    finally:
        _safe_remove(path)
    ext = path.suffix.lower()
    fmt: Literal["excel", "csv"] = "csv" if ext == ".csv" else "excel"
    mime_type = "text/csv" if fmt == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return {
        "filename": path.name,
        "data": payload,
        "mime_type": mime_type,
        "format": fmt,
    }


def _encode_image_file(file_path: Optional[str]) -> Optional[ImageAttachment]:
    if not file_path:
        return None
    path = Path(file_path)
    if not path.exists():
        logging.warning("BI agent expected image file %s but it was missing.", path)
        return None
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logging.warning("BI agent failed to read image %s: %s", path, exc)
        return None
    finally:
        _safe_remove(path)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif ext == ".gif":
        mime_type = "image/gif"
    else:
        mime_type = "image/png"
    return {
        "filename": path.name,
        "data": payload,
        "mime_type": mime_type,
    }


def _normalise_graph_type(value: Any) -> Optional[GraphicType]:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned if cleaned in _GRAPHIC_TYPES else None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip().lower()
                if cleaned in _GRAPHIC_TYPES:
                    return cleaned  # type: ignore[return-value]
    return None


def generate_report(state: BiAgentState, config: Optional[RunnableConfig] = None) -> BiAgentState:
    question = _extract_question(state.get("messages", []))
    if not question:
        return {
            "question": "",
            "answer": "Укажите вопрос, чтобы я мог построить отчёт и подготовить данные.",
            "data_attachment": None,
            "image_attachment": None,
            "graphic_type": None,
        }

    try:
        raw_response = get_response(question=question, data_paths=DEFAULT_DATA_SOURCES)
    except Exception as exc:  # noqa: BLE001
        logging.exception("BI agent failed to build report: %s", exc)
        return {
            "question": question,
            "answer": f"Не удалось построить отчёт: {exc}",
            "data_attachment": None,
            "image_attachment": None,
            "graphic_type": None,
        }

    data_attachment = _encode_tabular_file(raw_response.get("data"))
    image_attachment = _encode_image_file(raw_response.get("image"))
    graphic_type = _normalise_graph_type(raw_response.get("graph_type"))
    answer_text = raw_response.get("answer") or "Мне не удалось сформировать текстовый ответ."

    return {
        "question": question,
        "answer": answer_text,
        "data_attachment": data_attachment,
        "image_attachment": image_attachment,
        "graphic_type": graphic_type,
    }


def respond_with_report(state: BiAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, List[AIMessage]]:
    answer_text = state.get("answer") or "Ответ недоступен."
    parts: List[Dict[str, Any]] = [{"type": "text", "text": answer_text}]

    data_attachment = state.get("data_attachment")
    if data_attachment:
        parts.append(
            {
                "type": "file",
                "format": data_attachment.get("format"),
                "filename": data_attachment.get("filename"),
                "mime_type": data_attachment.get("mime_type"),
                "data": data_attachment.get("data"),
            }
        )

    image_attachment = state.get("image_attachment")
    if image_attachment:
        image_part: Dict[str, Any] = {
            "type": "image",
            "filename": image_attachment.get("filename"),
            "mime_type": image_attachment.get("mime_type"),
            "data": image_attachment.get("data"),
        }
        if state.get("graphic_type"):
            image_part["graphic_type"] = state["graphic_type"]
        parts.append(image_part)
    #elif state.get("graphic_type"):
        #parts.append(
        #    {
        #        "type": "text",
        #        "text": f"Рекомендованный тип графика: {state['graphic_type']}",
        #    }
        #)

    return {"messages": [AIMessage(content=parts)]}


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
):
    log_name = f"bi_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [json_handler]

    if cfg.LANGFUSE_URL:
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler as LangfuseHandler

            Langfuse(
                public_key=cfg.LANGFUSE_PUBLIC,
                secret_key=cfg.LANGFUSE_SECRET,
                host=cfg.LANGFUSE_URL,
            )
            callback_handlers.append(LangfuseHandler())
        except Exception as exc:  # noqa: BLE001
            logging.warning("Langfuse initialisation failed: %s", exc)

    memory = None if use_platform_store else MemorySaver()

    builder = StateGraph(BiAgentState, config_schema=ConfigSchema)
    builder.add_node("fetch_user_info", user_info)
    builder.add_node("reset_memory", reset_memory)
    builder.add_node("generate_report", generate_report)
    builder.add_node("respond", respond_with_report)

    builder.add_edge(START, "fetch_user_info")
    builder.add_conditional_edges(
        "fetch_user_info",
        reset_or_run,
        {
            "reset_memory": "reset_memory",
            "generate_report": "generate_report",
        },
    )
    builder.add_edge("reset_memory", END)
    builder.add_edge("generate_report", "respond")
    builder.add_edge("respond", END)

    graph = builder.compile(name="bi_agent", checkpointer=memory).with_config({"callbacks": callback_handlers})

    agent_key = "bi_agent"

    def _handle_kb_reload(context: KBReloadContext) -> None:
        logging.info("KB reload requested for %s (reason=%s); no KB-backed retrievers configured.", agent_key, context.reason)

    #if notify_on_reload:
    register_reload_listener(agent_key, _handle_kb_reload)

    return graph
