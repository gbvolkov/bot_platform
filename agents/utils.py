import os
import config
from enum import Enum
from typing import List, Sequence
import re
import unicodedata


from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import BaseCallbackManager
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langgraph.constants import TAG_NOSTREAM
from langgraph.prebuilt import ToolNode

from langchain_core.messages import HumanMessage
#from langchain_openai import ChatOpenAI

import telegramify_markdown
import telegramify_markdown.customize as customize

from .llm_utils import get_llm

customize.strict_markdown = False

try:
    from langgraph.pregel._messages import StreamMessagesHandler as _StreamMessagesHandler
except Exception:  # pragma: no cover - safety for version drift
    _StreamMessagesHandler = None

class ModelType(Enum):
    GPT = ("openai", "GPT")
    YA = ("yandex", "YandexGPT")
    SBER = ("gigachat", "Sber")
    #LOCAL = ("local", "Local")
    MISTRAL = ("mistral", "MistralAI")
    #GGUF = ("gguf", "GGUF")
    GPT4 = ("openai_4", "GPT4")
    GPT_PERS = ("openai_pers", "GPT_PERS")
    GPT_THINK = ("openai_think", "GPT_THINK")

    def __init__(self, value, display_name):
        self._value_ = value
        self.display_name = display_name

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _is_public_stream_messages_handler(handler: BaseCallbackHandler) -> bool:
    if _StreamMessagesHandler is not None and isinstance(handler, _StreamMessagesHandler):
        return True
    handler_cls = handler.__class__
    return (
        handler_cls.__module__ == "langgraph.pregel._messages"
        and handler_cls.__name__ == "StreamMessagesHandler"
    )


def _strip_public_stream_callbacks(callbacks):
    if callbacks is None:
        return []

    if isinstance(callbacks, list):
        return [h for h in callbacks if not _is_public_stream_messages_handler(h)]

    if isinstance(callbacks, BaseCallbackManager):
        manager = callbacks.copy()
        manager.handlers = [
            h for h in (getattr(manager, "handlers", []) or [])
            if not _is_public_stream_messages_handler(h)
        ]
        manager.inheritable_handlers = [
            h for h in (getattr(manager, "inheritable_handlers", []) or [])
            if not _is_public_stream_messages_handler(h)
        ]
        return manager

    return callbacks


def build_internal_invoke_config(
    parent_config: RunnableConfig | None,
    *,
    extra_tags: Sequence[str] | None = None,
) -> RunnableConfig:
    """Build config for nested/internal invokes while keeping monitoring callbacks."""
    base_config = ensure_config(parent_config)
    parent_tags = list(base_config.get("tags") or [])
    tags = list(dict.fromkeys([*parent_tags, TAG_NOSTREAM, *(extra_tags or [])]))

    callbacks = _strip_public_stream_callbacks(base_config.get("callbacks"))
    internal_config: RunnableConfig = {
        "callbacks": callbacks,
        "tags": tags,
    }

    configurable = base_config.get("configurable")
    if isinstance(configurable, dict):
        internal_config["configurable"] = configurable.copy()

    metadata = base_config.get("metadata")
    if isinstance(metadata, dict):
        internal_config["metadata"] = metadata.copy()

    recursion_limit = base_config.get("recursion_limit")
    if isinstance(recursion_limit, int):
        internal_config["recursion_limit"] = recursion_limit

    max_concurrency = base_config.get("max_concurrency")
    if isinstance(max_concurrency, int):
        internal_config["max_concurrency"] = max_concurrency

    return internal_config


def _print_event(event: dict, _printed: set, max_length=1500):
    if current_state := event.get("dialog_state"):
        print("Currently in: ", current_state[-1])
    if message := event.get("messages"):
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def _print_response(event: dict, _printed: set, max_length=1500):
    if current_state := event.get("dialog_state"):
        print("Currently in: ", current_state[-1])
    if message := event.get("messages"):
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            if message.type == "ai" and message.content.strip() != "":
                msg_repr = message.content.strip()
                if len(msg_repr) > max_length:
                    msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
                print(msg_repr)
            _printed.add(message.id)



# JSON allows: space, tab, LF, CR as whitespace outside strings.
# Many real-world sources include other Unicode spaces, esp. NBSP (U+00A0).
UNICODE_SPACES = {
    "\u00A0": " ",  # NBSP
    "\u2007": " ",  # figure space
    "\u202F": " ",  # narrow NBSP
    "\u200B": "",   # zero-width space (often best removed)
    "\uFEFF": "",   # BOM/zero-width no-break space
}

# Remove ASCII control chars that are illegal in JSON text (outside strings)
# Note: if your JSON contains literal control characters inside string values,
# that JSON is invalid anyway; removing them is usually the best salvage strategy.
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def clean_text(s: str) -> str:
    # Normalize unicode to reduce weird variants
    s = unicodedata.normalize("NFC", s)

    # Translate common problematic unicode whitespace
    s = s.translate(str.maketrans(UNICODE_SPACES))

    # Remove illegal control characters
    s = CONTROL_CHARS_RE.sub("", s)

    return s


def extract_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return clean_text(content).strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    if text := clean_text((item.get("text") or "")).strip():
                        parts.append(text)
            elif isinstance(item, str):
                if text := clean_text(item).strip():
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()

def send_text_element(chat_id, element_content, bot, usr_msg = None):
    chunks = [element_content[i:i+3800] for i in range(0, len(element_content), 3800)]
    for chunk in chunks:
        try:
            formatted = telegramify_markdown.markdownify(chunk)
            if usr_msg:
                bot.reply_to(usr_msg, formatted, parse_mode='MarkdownV2')
            else:
                bot.send_message(chat_id, formatted, parse_mode='MarkdownV2')
        except Exception as e:
            bot.send_message(chat_id, chunk)


def _send_response(event: dict, _printed: set, thread, bot, usr_msg=None, max_length=0) -> str:
    if current_state := event.get("dialog_state"):
        print("Currently in: ", current_state[-1])
    if not (message := event.get("messages")):
        return ""
    answers = []
    if isinstance(message, list):
        message = message[-1]
    if message.id not in _printed:
        if message.type == "ai" and message.content.strip() != "":
            msg_repr = message.content.strip()
            if max_length > 0 and len(msg_repr) > max_length:
                msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
            answers.append(msg_repr)
            send_text_element(thread.chat_id, msg_repr, bot, usr_msg)
        _printed.add(message.id)
    return "\n".join(answers)

def _send_response_full(event: dict, _printed: set, thread, bot, usr_msg=None, max_length=0):
    if current_state := event.get("dialog_state"):
        print("Currently in: ", current_state[-1])
    if messages := event.get("messages"):
        if not isinstance(messages, list):
            messages = [messages]
        for message in messages:
            if message.id not in _printed:
                if message.type == "ai" and message.content.strip() != "":
                    msg_repr = message.content.strip()
                    if max_length > 0 and len(msg_repr) > max_length:
                        msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
                    send_text_element(thread.chat_id, msg_repr, bot, usr_msg)
                _printed.add(message.id)

def show_graph(graph):
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        # Write the PNG data to a file
        output_filename = "langgraph_visualization.png"
        with open(output_filename, "wb") as f:
            f.write(png_data)
        os.startfile(output_filename)
    except Exception as e:
        print(f"Error showing graph: {e}")


def image_to_uri(image_data: str) -> str:
    return f"data:image/jpeg;base64,{image_data}"

def summarise_image(image_uri: str) -> str:
    model = get_llm("nano")  # ensure this resolves to a vision-capable OpenAI model

    # Ask for a compact, consistent output
    prompt = (
        "Сгенерируй до четырёх ключевых слов, описывающих изображение. "
        "Ответь по-русски, в виде списка слов, разделённых запятыми."
    )

    parts = [{"type": "text", "text": prompt}]
    if image_uri.startswith("data:"):
        # Already a valid data URL → pass through
        parts.append({"type": "image_url", "image_url": {"url": image_uri}})
    elif image_uri.startswith(("http://", "https://")):
        # Standard URL → pass through
        parts.append({"type": "image_url", "image_url": {"url": image_uri}})
    else:
        # Fallback: assume raw base64 (no data: prefix). Wrap as a PNG data URL.
        data_url = f"data:image/png;base64,{image_uri}"
        parts.append({"type": "image_url", "image_url": {"url": data_url}})

    message = HumanMessage(content=parts)
    response = model.invoke([message])  # synchronous on purpose
    return response.content


