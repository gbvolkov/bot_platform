# aiogram_notify_stream.py
import asyncio
import contextlib
from typing import Optional, List
from aiogram import Bot, types
from aiogram.enums import ChatAction

from telegramify_markdown import markdownify

# ----- Chat actions: immediate + pulsing every 4s -----

async def _pulse_actions(bot: Bot, chat_id: int, action: ChatAction):
    with contextlib.suppress(asyncio.CancelledError):
        while True:
            await bot.send_chat_action(chat_id, action)
            await asyncio.sleep(4)

async def start_show_typing(bot: Bot, chat_id: int, action: ChatAction = ChatAction.TYPING) -> asyncio.Task:
    """
    Send an immediate action frame, then pulse it every 4 seconds.
    Returns the task so you can cancel it in finally.
    """
    await bot.send_chat_action(chat_id, action)
    return asyncio.create_task(_pulse_actions(bot, chat_id, action))


# ----- Determine which upload action to show during file work -----

def determine_upload_action(message: types.Message) -> Optional[ChatAction]:
    """
    Returns:
      - UPLOAD_VOICE for voice
      - UPLOAD_PHOTO for photo or document with image/* mime
      - UPLOAD_DOCUMENT for other documents
      - None for pure-text
    """
    if getattr(message, "voice", None):
        return ChatAction.UPLOAD_VOICE
    if getattr(message, "photo", None):
        return ChatAction.UPLOAD_PHOTO
    if getattr(message, "document", None):
        mt = (message.document.mime_type or "").lower()
        if mt.startswith("image/"):
            return ChatAction.UPLOAD_PHOTO
        return ChatAction.UPLOAD_DOCUMENT
    return None

def vision_part_from_uri(uri: str):
    if uri.startswith("data:"):
        header, b64 = uri.split(",", 1)
        mime = (header.split(";")[0][5:] or "image/png") if header.startswith("data:") else "image/png"
        return {"type": "input_image", "image": {"data": b64, "mime_type": mime}}
    return {"type": "image_url", "image_url": {"url": uri}}


# ----- Streaming: run blocking .stream() off the event loop -----

async def collect_final_text_from_stream(assistant, payload_msg, cfg) -> str:
    """
    Offloads synchronous assistant.stream(...) to a thread
    and returns the concatenated final assistant text.
    """
    def _run_sync():
        final_parts: List[str] = []
        printed_ids = set()
        events = assistant.stream({"messages": [payload_msg]}, cfg, stream_mode="values")
        for event in events:
            msg = event.get("messages") if isinstance(event, dict) else None
            if not msg:
                continue
            if isinstance(msg, list):
                msg = msg[-1]
            mid = getattr(msg, "id", None)
            if mid in printed_ids:
                continue
            if getattr(msg, "type", "") == "ai":
                text = (getattr(msg, "content", "") or "").strip()
                if text:
                    final_parts.append(text)
                    printed_ids.add(mid)
        return "\n".join(final_parts).strip() or "—"

    return await asyncio.to_thread(_run_sync)


# ----- Safe text sending identical to your working behavior -----

async def send_text_element(
    bot: Bot,
    chat_id: int,
    element_content: str,
    usr_msg: Optional[types.Message] = None,
) -> None:
    """
    Split into ~3800 chars, convert to MarkdownV2 with telegramify_markdown,
    fallback to raw text (parse_mode=None) on conversion error.
    """
    chunks = [element_content[i: i + 3800] for i in range(0, len(element_content), 3800)]
    for chunk in chunks:
        try:
            formatted = markdownify(chunk)
            if usr_msg:
                await bot.send_message(
                    chat_id=usr_msg.chat.id,
                    text=formatted,
                    parse_mode="MarkdownV2",
                    reply_to_message_id=usr_msg.message_id,
                )
            else:
                await bot.send_message(chat_id=chat_id, text=formatted, parse_mode="MarkdownV2")
        except Exception:
            if usr_msg:
                await bot.send_message(
                    chat_id=usr_msg.chat.id,
                    text=chunk,
                    reply_to_message_id=usr_msg.message_id,
                    parse_mode=None,
                )
            else:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=None)


# ----- Placeholder finalization helpers -----

def chunk_text(s: str, n: int = 4000):
    for i in range(0, len(s), n):
        yield s[i:i+n]

async def finalize_placeholder_or_fallback(
    bot: Bot,
    placeholder: types.Message,
    chat_id: int,
    final_text: str,
):
    chunks = list(chunk_text(final_text))
    # First chunk: edit placeholder (safe path)
    try:
        safe_first = markdownify(chunks[0])
        await bot.edit_message_text(
            chat_id=placeholder.chat.id,
            message_id=placeholder.message_id,
            text=safe_first,
            parse_mode="MarkdownV2"
        )
    except Exception:
        # Fallback: send as new message with no parsing
        await bot.send_message(chat_id, chunks[0], parse_mode=None)

    # Remaining chunks as follow-ups
    for extra in chunks[1:]:
        await send_text_element(bot, chat_id, extra)
