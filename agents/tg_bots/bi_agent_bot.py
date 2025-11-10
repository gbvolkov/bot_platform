"""
bi_agent_aiogram_bot.py
-----------------------
Telegram bot wrapper around the BI LangGraph agent.
"""

import asyncio
import base64
import contextlib
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, BufferedInputFile

from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

import config

if config.NO_CUDA == "True":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .bi_thread_settings import BIThreadSettings
from agents.utils import summarise_image, image_to_uri, ModelType
from langchain_core.messages import HumanMessage, AIMessage
from store_managers.google_sheets_man import GoogleSheetsManager

from vrecog.vrecog import recognise_text

from .bot_helpers import (
    start_show_typing,
    determine_upload_action,
    finalize_placeholder_or_fallback,
)

# --- webhook config reused from service desk bot ---
BOT_MODE = (getattr(config, "BOT_MODE", "polling")).lower()
WEBAPP_HOST = getattr(config, "WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(getattr(config, "WEBAPP_PORT", "8080"))
WEBHOOK_BASE = getattr(config, "WEBHOOK_BASE", "https://0.0.0.0:88")
WEBHOOK_PATH = getattr(config, "WEBHOOK_PATH", "/tg-webhook")
WEBHOOK_URL = (WEBHOOK_BASE or "").rstrip("/") + WEBHOOK_PATH if WEBHOOK_BASE else None
WEBHOOK_SECRET = getattr(config, "WEBHOOK_SECRET", None)

GRAPH_LABELS = {
    "bar_chart": "Bar chart",
    "line_chart": "Line chart",
    "scatter_plot": "Scatter plot",
    "pie": "Pie chart",
}


def create_rating_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Needs improvements", callback_data="rate_1")],
            [InlineKeyboardButton(text="Great result!", callback_data="rate_3")],
        ]
    )


def _extract_bi_response(messages: List[AIMessage]) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    last_ai: Optional[AIMessage] = None
    for message in reversed(messages or []):
        if getattr(message, "type", "") == "ai":
            last_ai = message
            break

    if last_ai is None:
        return "", None, None, None

    text_parts: List[str] = []
    tabular_part: Optional[Dict[str, Any]] = None
    image_part: Optional[Dict[str, Any]] = None
    graphic_type: Optional[str] = None

    content = getattr(last_ai, "content", "")
    if isinstance(content, str):
        cleaned = content.strip()
        if cleaned:
            text_parts.append(cleaned)
    elif isinstance(content, list):
        for piece in content:
            if not isinstance(piece, dict):
                continue
            if piece.get("type") == "text":
                text_value = (piece.get("text") or "").strip()
                if text_value:
                    text_parts.append(text_value)
            elif piece.get("type") == "file":
                tabular_part = piece
            elif piece.get("type") == "image":
                image_part = piece
                if piece.get("graphic_type"):
                    graphic_type = piece["graphic_type"]

    if not graphic_type and getattr(last_ai, "additional_kwargs", None):
        graphic_type = (last_ai.additional_kwargs or {}).get("graphic_type")

    return "\n\n".join(text_parts).strip(), tabular_part, image_part, graphic_type


async def _send_tabular_attachment(
    bot: Bot,
    chat_id: int,
    attachment: Dict[str, Any],
    reply_to: Optional[int] = None,
) -> None:
    data_b64 = attachment.get("data")
    if not data_b64:
        return
    try:
        raw = base64.b64decode(data_b64)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to decode table attachment: %s", exc)
        await bot.send_message(chat_id, "Could not decode report table.", reply_to_message_id=reply_to, parse_mode=None)
        return

    filename = attachment.get("filename") or f"report_{int(time.time()*1000)}.xlsx"
    caption = attachment.get("caption") or "Report table"

    await bot.send_document(
        chat_id,
        BufferedInputFile(raw, filename),
        caption=caption,
        reply_to_message_id=reply_to,
        parse_mode=None,
    )


async def _send_image_attachment(
    bot: Bot,
    chat_id: int,
    attachment: Dict[str, Any],
    graphic_type: Optional[str],
    reply_to: Optional[int] = None,
) -> None:
    data_b64 = attachment.get("data")
    if not data_b64:
        return

    try:
        raw = base64.b64decode(data_b64)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to decode image attachment: %s", exc)
        await bot.send_message(chat_id, "Could not decode visualization.", reply_to_message_id=reply_to, parse_mode=None)
        return

    filename = attachment.get("filename") or f"chart_{int(time.time()*1000)}.png"
    chart_label = GRAPH_LABELS.get((graphic_type or "").lower(), graphic_type or "Visualization")
    caption = f"{chart_label}"

    await bot.send_photo(
        chat_id,
        BufferedInputFile(raw, filename),
        caption=caption,
        reply_to_message_id=reply_to,
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    token = getattr(config, "TELEGRAM_BI_BOT_TOKEN", None)
    if not token:
        raise RuntimeError("TELEGRAM_BI_BOT_TOKEN is not configured")

    bot = Bot(
        token,
        default=DefaultBotProperties(parse_mode="MarkdownV2"),
    )
    dp = Dispatcher()

    chats: Dict[int, BIThreadSettings] = {}

    try:
        sheets_manager = GoogleSheetsManager(config.GOOGLE_SHEETS_CRED, config.FEEDBACK_SHEET_ID)
    except Exception as exc:  # noqa: BLE001
        logging.error("Error initializing Google Sheets manager: %s", exc)
        sheets_manager = None

    class ThrottlingMiddleware:
        def __init__(self, rate: float = 3.0):
            self.rate = rate
            self.last_called: Dict[Tuple[int, int], float] = {}

        async def __call__(self, handler, event, data):
            from_user = getattr(event, "from_user", None)
            chat = getattr(event, "chat", None)
            if from_user and chat:
                key = (from_user.id, chat.id)
                now = time.monotonic()
                last = self.last_called.get(key, 0.0)
                if now - last < self.rate:
                    await asyncio.sleep(self.rate - (now - last))
                self.last_called[key] = time.monotonic()
            return await handler(event, data)

    dp.message.middleware(ThrottlingMiddleware(rate=3.0))

    @dp.errors()
    async def global_error_handler(event, exception, **kwargs):
        logging.exception("Unhandled exception: %s", exception)
        return True

    @dp.message(Command("start"))
    async def cmd_start(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username or f"id:{message.from_user.id}"
        chats[chat_id] = BIThreadSettings(user_id=user_id, chat_id=chat_id)

        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
            chats[chat_id].get_config(),
            stream_mode="values",
        )

        onboarding_text = (
            "Привет! Я интеллектуальный помощник в бизнес-аналитике.\n"
            "Я могу отвечать на вопросы о предоставленных данных."
        )
        placeholder = await message.reply("Preparing welcome message...", parse_mode=None)
        typing_task = await start_show_typing(bot, chat_id, ChatAction.TYPING)
        try:
            await finalize_placeholder_or_fallback(bot, placeholder, chat_id, onboarding_text)
        finally:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

    def _set_model(message: types.Message, model: ModelType) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username or f"id:{message.from_user.id}"
        if chat_id not in chats:
            chats[chat_id] = BIThreadSettings(user_id=user_id, chat_id=chat_id)
        chats[chat_id].model = model
        chats[chat_id].assistant = None

    @dp.message(Command("sber"))
    async def cmd_sber(message: types.Message) -> None:
        _set_model(message, ModelType.SBER)

    @dp.message(Command("gpt"))
    async def cmd_gpt(message: types.Message) -> None:
        _set_model(message, ModelType.GPT)

    @dp.message(Command("reset"))
    async def cmd_reset(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username or f"id:{message.from_user.id}"
        chats[chat_id] = BIThreadSettings(user_id=user_id, chat_id=chat_id)

        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
            chats[chat_id].get_config(),
            stream_mode="values",
        )
        await bot.send_message(chat_id, "State cleared for this chat.", parse_mode=None)

    @dp.message(Command("help"))
    async def cmd_help(message: types.Message) -> None:
        chat_id = message.chat.id
        with open("./help/help.md", encoding="utf-8") as file:
            help_text = file.read()
        await bot.send_message(chat_id, help_text, parse_mode="MarkdownV2")

    @dp.message(Command("reload"))
    async def cmd_reload(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username or f"id:{message.from_user.id}"
        if chat_id not in chats:
            chats[chat_id] = BIThreadSettings(user_id=user_id, chat_id=chat_id)
        if chats[chat_id].is_admin():
            chats[chat_id].assistant = None
            await bot.send_message(chat_id, "BI agent will reinitialize on the next request.", parse_mode=None)

    @dp.message(Command("users"))
    async def cmd_users(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username or f"id:{message.from_user.id}"
        if chat_id not in chats:
            chats[chat_id] = BIThreadSettings(user_id=user_id, chat_id=chat_id)
        if chats[chat_id].reload_users():
            await bot.send_message(chat_id, "User rights cache reloaded.", parse_mode=None)

    @dp.message(lambda m: m.content_type in {"text", "voice", "photo", "document"})
    async def handle_message(message: types.Message) -> None:
        placeholder: Optional[types.Message] = None
        typing_task: Optional[asyncio.Task] = None
        upload_task: Optional[asyncio.Task] = None

        try:
            chat_id = message.chat.id
            user_id = message.from_user.username or f"id:{message.from_user.id}"

            if chat_id not in chats:
                chats[chat_id] = BIThreadSettings(user_id=user_id, chat_id=chat_id)
            thread = chats[chat_id]

            if not thread.is_allowed():
                await bot.send_message(chat_id, "Access denied. Please contact the administrator.", parse_mode=None)
                return

            query = (message.text or getattr(message, "any_text", None) or (message.caption or "")).strip()

            upload_action = determine_upload_action(message)
            if upload_action:
                upload_task = await start_show_typing(bot, chat_id, upload_action)
                try:
                    if message.content_type == "voice":
                        file_id = message.voice.file_id
                        file_info = await bot.get_file(file_id)
                        voice_data = await bot.download_file(file_info.file_path)
                        raw = voice_data.getvalue() if hasattr(voice_data, "getvalue") else voice_data
                        tmp_path = f"voice_{user_id}_{int(time.time()*1000)}.ogg"
                        with open(tmp_path, "wb") as temp_file:
                            temp_file.write(raw)
                        query = recognise_text(tmp_path)
                        with contextlib.suppress(Exception):
                            os.remove(tmp_path)
                        if not query:
                            await bot.send_message(chat_id, "Voice message could not be transcribed.", parse_mode=None)
                            return
                    elif message.content_type in {"photo", "document"} and not message.voice:
                        if message.photo:
                            file_id = message.photo[-1].file_id
                        else:
                            file_id = message.document.file_id
                        file_info = await bot.get_file(file_id)
                        file_io = await bot.download_file(file_info.file_path)
                        raw_bytes = file_io.getvalue() if hasattr(file_io, "getvalue") else file_io
                        uri = image_to_uri(base64.b64encode(raw_bytes).decode())
                        summary = await asyncio.to_thread(summarise_image, uri)
                        if summary:
                            query = (query + "\n\n" + summary).strip()
                finally:
                    if upload_task:
                        upload_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await upload_task
                        upload_task = None

            if not query:
                await bot.send_message(chat_id, "Please send a text question for the BI agent.", parse_mode=None)
                return

            assistant = thread.assistant

            if not message.reply_to_message:
                assistant.invoke(
                    {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
                    thread.get_config(),
                    stream_mode="values",
                )

            placeholder = await message.reply("Working on the report...", parse_mode=None)
            typing_task = await start_show_typing(bot, chat_id, ChatAction.TYPING)
            payload_msg = HumanMessage(content=[{"type": "text", "text": query}])

            def _invoke():
                return assistant.invoke({"messages": [payload_msg]}, thread.get_config())

            try:
                result = await asyncio.to_thread(_invoke)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Error while invoking BI agent: %s", exc)
                await bot.edit_message_text(
                    chat_id=placeholder.chat.id,
                    message_id=placeholder.message_id,
                    text="Failed to build the report.",
                )
                return

            final_state = result if isinstance(result, dict) else {}
            messages = final_state.get("messages", [])
            text_answer, tabular_part, image_part, graphic_type = _extract_bi_response(messages)
            final_text = text_answer or "The BI agent produced results without a textual summary."
            await finalize_placeholder_or_fallback(bot, placeholder, chat_id, final_text)

            thread.question = query
            thread.answer = final_text
            thread.context = f"table={bool(tabular_part)};image={bool(image_part)}"

            reply_to_id = message.message_id
            if tabular_part:
                await _send_tabular_attachment(bot, chat_id, tabular_part, reply_to=reply_to_id)
            if image_part:
                await _send_image_attachment(bot, chat_id, image_part, graphic_type, reply_to=reply_to_id)

            await bot.send_message(
                chat_id,
                "Please rate this BI answer.",
                reply_markup=create_rating_keyboard(),
                parse_mode=None,
            )

        except Exception as exc:  # noqa: BLE001
            logging.exception("Unexpected error in BI handler: %s", exc)
            if placeholder:
                with contextlib.suppress(Exception):
                    await bot.edit_message_text(
                        chat_id=placeholder.chat.id,
                        message_id=placeholder.message_id,
                        text="Unexpected error while preparing the report.",
                    )
            else:
                with contextlib.suppress(Exception):
                    await bot.send_message(message.chat.id, "Unexpected error while preparing the report.", parse_mode=None)
        finally:
            if typing_task:
                typing_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing_task
            if upload_task:
                upload_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await upload_task

    @dp.callback_query(lambda c: c.data and c.data.startswith("rate_"))
    async def callback_rating(call: types.CallbackQuery):
        rating = call.data.split("_")[1]
        chat_id = call.message.chat.id
        user_id = call.from_user.id
        username = call.from_user.username or f"id:{user_id}"

        thread = chats.get(chat_id)
        user_question = getattr(thread, "question", "-") if thread else "-"
        bot_response = getattr(thread, "answer", "-") if thread else "-"
        model = getattr(thread.model, "value", "-") if thread and getattr(thread, "model", None) else "-"
        context = getattr(thread, "context", "") if thread else ""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if sheets_manager:
            with contextlib.suppress(Exception):
                sheets_manager.append_row(
                    [
                        timestamp,
                        username,
                        user_question,
                        bot_response,
                        rating,
                        model,
                        (context or "")[:32000],
                    ]
                )

        with contextlib.suppress(Exception):
            await call.answer(f"Thanks! Rating recorded: {rating}")
        with contextlib.suppress(Exception):
            await bot.edit_message_reply_markup(chat_id=chat_id, message_id=call.message.message_id, reply_markup=None)

    hook_mode = BOT_MODE in {"hook", "@hook@", "webhook"}
    if hook_mode:
        if not WEBHOOK_URL:
            logging.error("WEBHOOK_BASE is not configured for webhook mode.")
            return

        app = web.Application()
        SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=WEBHOOK_SECRET).register(app, path=WEBHOOK_PATH)
        setup_application(app, dp, bot=bot)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=WEBAPP_HOST, port=WEBAPP_PORT)

        try:
            await bot.set_webhook(url=WEBHOOK_URL, secret_token=WEBHOOK_SECRET)
            await site.start()
            logging.info("Webhook set: %s (%s:%s)", WEBHOOK_URL, WEBAPP_HOST, WEBAPP_PORT)
            await asyncio.Event().wait()
        finally:
            with contextlib.suppress(Exception):
                await bot.delete_webhook(drop_pending_updates=False)
            with contextlib.suppress(Exception):
                await runner.cleanup()
    else:
        await dp.start_polling(bot)


if __name__ == "__main__":
    pid = os.getpid()
    with open(".process", "w") as proc_file:
        proc_file.write(f"{pid}")
    asyncio.run(main())
