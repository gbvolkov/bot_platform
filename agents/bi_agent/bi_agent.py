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

from ..sql_query_gen import SQLQueryExecutionError, get_response
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
    query: str
    row_count: Optional[int]
    data_attachment: Optional[TabularAttachment]
    image_attachment: Optional[ImageAttachment]
    graphic_type: Optional[GraphicType]
    notes: str


DEFAULT_DATA_SOURCES = [str(Path("data") / "data.csv")]
_GRAPHIC_TYPES: set[str] = {"bar_chart", "line_chart", "scatter_plot", "pie"}


def _coerce_optional_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_optional_int(value: Any, default: int, field_name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if resolved < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return resolved


def _coerce_row_limit(value: Any, default: int, field_name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


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
        "query": "",
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


def create_generate_report_node(init_context: Optional[Dict[str, Any]] = None):
    defaults = dict(init_context or {})
    default_database_url = defaults.get("database_url")
    default_database_prompt_context = defaults.get("database_prompt_context")
    default_return_files = _coerce_optional_bool(defaults.get("return_files"), True)
    default_return_images = _coerce_optional_bool(defaults.get("return_images"), True)
    default_max_string_length = _coerce_optional_int(
        defaults.get("max_string_length"),
        300,
        "init_context.max_string_length",
    )
    default_answer_row_limit = _coerce_row_limit(
        defaults.get("answer_row_limit"),
        0,
        "init_context.answer_row_limit",
    )

    def generate_report(state: BiAgentState, config: Optional[RunnableConfig] = None) -> BiAgentState:
        question = _extract_question(state.get("messages", []))
        if not question:
            return {
                "question": "",
                "answer": "Укажите вопрос, чтобы я мог построить отчёт и подготовить данные.",
                "query": "",
                "row_count": None,
                "data_attachment": None,
                "image_attachment": None,
                "graphic_type": None,
            }

        configuration = (config or {}).get("configurable", {})
        database_url = configuration.get("database_url")
        if database_url is None:
            database_url = default_database_url
        database_prompt_context = configuration.get("database_prompt_context")
        if database_prompt_context is None:
            database_prompt_context = default_database_prompt_context
        return_files = _coerce_optional_bool(configuration.get("return_files"), default_return_files)
        return_images = _coerce_optional_bool(configuration.get("return_images"), default_return_images)
        max_string_length = _coerce_optional_int(
            configuration.get("max_string_length"),
            default_max_string_length,
            "configurable.max_string_length",
        )
        answer_row_limit = _coerce_row_limit(
            configuration.get("answer_row_limit"),
            default_answer_row_limit,
            "configurable.answer_row_limit",
        )
        data_paths = None if database_url else DEFAULT_DATA_SOURCES

        try:
            raw_response = get_response(
                question=question,
                data_paths=data_paths,
                database_url=database_url,
                database_prompt_context=database_prompt_context,
                return_files=return_files,
                return_images=return_images,
                max_string_length=max_string_length,
                answer_row_limit=answer_row_limit,
            )
        except SQLQueryExecutionError as exc:
            logging.exception("BI agent failed to execute generated report query: %s", exc)
            return {
                "question": question,
                "answer": f"Не удалось построить отчёт: {exc}",
                "query": exc.query,
                "row_count": None,
                "data_attachment": None,
                "image_attachment": None,
                "graphic_type": None,
                "notes": exc.error,
            }
        except Exception as exc:  # noqa: BLE001
            logging.exception("BI agent failed to build report: %s", exc)
            return {
                "question": question,
                "answer": f"Не удалось построить отчёт: {exc}",
                "query": "",
                "row_count": None,
                "data_attachment": None,
                "image_attachment": None,
                "graphic_type": None,
                "notes": str(exc),
            }

        data_attachment = _encode_tabular_file(raw_response.get("data")) if return_files else None
        image_attachment = _encode_image_file(raw_response.get("image")) if return_images else None
        if not return_files:
            data_path = raw_response.get("data")
            if data_path:
                _safe_remove(Path(data_path))
        if not return_images:
            image_path = raw_response.get("image")
            if image_path:
                _safe_remove(Path(image_path))
        graphic_type = _normalise_graph_type(raw_response.get("graph_type")) if image_attachment else None
        answer_text = raw_response.get("answer") or "Мне не удалось сформировать текстовый ответ."
        notes = raw_response.get("notes", "")
        return {
            "question": question,
            "answer": answer_text,
            "query": raw_response.get("query", ""),
            "row_count": raw_response.get("row_count"),
            "data_attachment": data_attachment,
            "image_attachment": image_attachment,
            "graphic_type": graphic_type,
            "notes": notes,
        }

    return generate_report


def respond_with_report(state: BiAgentState, config: Optional[RunnableConfig] = None) -> Dict[str, List[AIMessage]]:
    answer_text = state.get("answer") or "Ответ недоступен."
    if state.get("notes"):
        answer_text += f"\n\n{state['notes']}"
    parts: List[Dict[str, Any]] = [
        {"type": "text", "text": answer_text},
    ]

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
    # elif state.get("graphic_type"):
    #     parts.append(
    #         {
    #             "type": "text",
    #             "text": f"Рекомендованный тип графика: {state['graphic_type']}",
    #         }
    #     )

    return {"messages": [AIMessage(content=parts)]}


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
    checkpoint_saver=None,
    *,
    init_context: Optional[Dict[str, Any]] = None,
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

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()

    builder = StateGraph(BiAgentState, config_schema=ConfigSchema)
    builder.add_node("fetch_user_info", user_info)
    builder.add_node("reset_memory", reset_memory)
    builder.add_node("generate_report", create_generate_report_node(init_context))
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

    # if notify_on_reload:
    register_reload_listener(agent_key, _handle_kb_reload)

    return graph
