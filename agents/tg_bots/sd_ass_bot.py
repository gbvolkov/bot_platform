"""
sd_ass_aiogram_bot.py
---------------------
aiogram-based Telegram bot.

Key features carried over:
 • Per-user/per-chat throttling middleware
 • Robust chat actions: show UPLOAD_* during file work, then TYPING while streaming
 • Placeholder -> stream (no partials) -> final edit, with chunking
 • Voice transcription (.ogg) via recognise_text
 • Image pipeline: download -> base64 -> image_to_uri -> summarise_image (off-thread)
 • Periodic KB refresh with pause/resume around heavy work
 • Rating flow stored to Google Sheets via GoogleSheetsManager
"""

import asyncio
import contextlib
import base64
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

import os
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums import ChatAction

# --- webhook imports (added) ---
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

import config

if config.NO_CUDA == "True":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .thread_settings import ThreadSettings
from agents.utils import summarise_image, image_to_uri, ModelType  # noqa: F401 (ModelType kept for parity)
from langchain_core.messages import HumanMessage
from agents.retrievers.utils.load_common_retrievers import refresh_indexes
from store_managers.google_sheets_man import GoogleSheetsManager

# Voice recognition
from vrecog.vrecog import recognise_text

import telegramify_markdown

# Periodic KB updater (threaded)
from utils.periodic_task import PeriodicTask

from bot_helpers import (
    start_show_typing,
    determine_upload_action,
    collect_final_text_from_stream,
    send_text_element,
    finalize_placeholder_or_fallback,
    vision_part_from_uri,
)

# --- webhook config (added) ---
BOT_MODE = (getattr(config, "BOT_MODE", "polling")).lower()
WEBAPP_HOST = getattr(config, "WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(getattr(config, "WEBAPP_PORT", "8080"))
WEBHOOK_BASE = getattr(config, "WEBHOOK_BASE", "https://0.0.0.0:88")  # e.g. https://bot.example.com
WEBHOOK_PATH = getattr(config, "WEBHOOK_PATH", "/tg-webhook")
WEBHOOK_URL = (WEBHOOK_BASE or "").rstrip("/") + WEBHOOK_PATH if WEBHOOK_BASE else None
WEBHOOK_SECRET = getattr(config, "WEBHOOK_SECRET", None)  # optional

def create_rating_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Промазал:(", callback_data="rate_1")],
            # [InlineKeyboardButton(text="Похоже на правду.", callback_data="rate_2")],
            [
                InlineKeyboardButton(
                    text="В точку! Спасибо!", callback_data="rate_3"
                )
            ],
        ]
    )


# -------------------------------------------------------------
# Main bot
# -------------------------------------------------------------

async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Aiogram ≥3.7.0: set default parse mode here
    bot = Bot(
        config.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="MarkdownV2"),
    )
    dp = Dispatcher()

    # Thread state (per chat_id, matching sd_ass_bot.py behavior)
    chats: Dict[int, ThreadSettings] = defaultdict(ThreadSettings)

    # Sheets manager (optional)
    try:
        sheets_manager = GoogleSheetsManager(config.GOOGLE_SHEETS_CRED, config.FEEDBACK_SHEET_ID)
    except Exception as e:
        logging.error(f"Error initializing Google Sheets manager: {e}")
        sheets_manager = None

    # Throttling middleware (simple, per user per chat)
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
                elapsed = now - last
                if elapsed < self.rate:
                    await asyncio.sleep(self.rate - elapsed)
                self.last_called[key] = time.monotonic()
            return await handler(event, data)

    dp.message.middleware(ThrottlingMiddleware(rate=3.0))

    # Global error handler
    @dp.errors()
    async def global_error_handler(event, exception, **kwargs):
        logging.exception(f"Unhandled exception occured: {exception}")
        return True

    # /start — resets Ai agent's memory, greet
    @dp.message(Command("start"))
    async def cmd_start(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

        # Reset memory
        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
            chats[chat_id].get_config(),
            stream_mode="values",
        )

        # Greet via streaming final text only
        query = "Привет! Представься пожалуйста и расскажи о себе."
        payload = HumanMessage(content=[{"type": "text", "text": query}])

        typing_task = await start_show_typing(bot, chat_id, ChatAction.TYPING)
        try:
            final_text = await collect_final_text_from_stream(
                assistant, payload, chats[chat_id].get_config()
            )
            await send_text_element(bot, chat_id, final_text, usr_msg=message)
        finally:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

    def _set_model(message: types.Message, model: ModelType) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        if chat_id not in chats:
            chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)
        chats[chat_id].model = model
        chats[chat_id].assistant = None

    # /sber — set model to Sber GIGAChat
    @dp.message(Command("sber"))
    async def cmd_sber(message: types.Message) -> None:
        _set_model(message, ModelType.SBER)

    # /gpt — set model to OpenAI GPT model
    @dp.message(Command("gpt"))
    async def cmd_gpt(message: types.Message) -> None:
        _set_model(message, ModelType.GPT)

    # /reset — resets AI agent's memory only
    @dp.message(Command("reset"))
    async def cmd_reset(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
            chats[chat_id].get_config(),
            stream_mode="values",
        )
        await bot.send_message(chat_id, "Память бота очищена.", parse_mode=None)

    @dp.message(Command("help"))
    async def cmd_help(message: types.Message) -> None:
        chat_id = message.chat.id
        with open("./help/help.md", encoding="utf-8") as f:
            help_str = f.read()
        await bot.send_message(chat_id, help_str, parse_mode="MarkdownV2")

    # /reload — refreshes indexes
    @dp.message(Command("reload"))
    async def cmd_reload(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        if chat_id not in chats:
            chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)
        if chats[chat_id].is_admin():
            chat_id = message.chat.id
            kb_update_thread.pause()
            refresh_indexes()
            kb_update_thread.resume()
            await bot.send_message(chat_id, "База знаний обновлена.", parse_mode=None)

    # /users — refreshes users rights
    @dp.message(Command("users"))
    async def cmd_users(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        if chat_id not in chats:
            chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)
        if chats[chat_id].reload_users():
            await bot.send_message(chat_id, "База пользователей обновлена.", parse_mode=None)

    # Main message handler: text, voice, photo, document
    @dp.message(lambda m: m.content_type in {"text", "voice", "photo", "document"})
    async def handle_message(message: types.Message) -> None:
        kb_update_thread.pause()
        placeholder: Optional[types.Message] = None
        typing_task: Optional[asyncio.Task] = None

        try:
            chat_id = message.chat.id
            user_id = message.from_user.username
            image_uri_payload: list[dict] = []

            # Ensure thread settings exist
            if chat_id not in chats:
                chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

            if not chats[chat_id].is_allowed():
                await bot.send_message(
                    chat_id,
                    'К сожалению, мне не разрешено помогать вам. Пожалуйста, обратитесь к администратору бота.',
                    parse_mode=None,
                )
                return

            # Build query from text/caption/voice
            query = message.text or getattr(message, "any_text", None) or (message.caption or "")

            upload_action = determine_upload_action(message)
            if upload_action is not None:
                upload_task = await start_show_typing(bot, chat_id, upload_action)
                try:
                    # Voice handling
                    if message.content_type == "voice":
                        # keep upload voice while saving/transcribing
                        file_id = message.voice.file_id
                        file_info = await bot.get_file(file_id)
                        voice_io = await bot.download_file(file_info.file_path)
                        raw = voice_io.getvalue() if hasattr(voice_io, "getvalue") else voice_io

                        # Save to temp ogg
                        tmp_path = f"voice_{user_id}_{int(time.time()*1000)}.ogg"
                        with open(tmp_path, "wb") as f:
                            f.write(raw)

                        # Transcribe
                        query = recognise_text(tmp_path)
                        with contextlib.suppress(Exception):
                            os.remove(tmp_path)

                        if not query:
                            await bot.send_message(
                                chat_id,
                                "Не удалось распознать голосовое сообщение. "
                                "Пожалуйста, отправьте текст вручную или попробуйте снова.",
                                parse_mode=None,
                            )
                            return

                    # Photo or document handling (treated as image for summary)
                    if message.content_type in {"photo", "document"} and not message.voice:
                        if message.photo:
                            file_id = message.photo[-1].file_id
                        else:
                            file_id = message.document.file_id

                        file_info = await bot.get_file(file_id)
                        file_io = await bot.download_file(file_info.file_path)
                        raw_bytes = file_io.getvalue() if hasattr(file_io, "getvalue") else file_io
                        uri = image_to_uri(base64.b64encode(raw_bytes).decode())

                        # Summarise (off the loop)
                        summary = await asyncio.to_thread(summarise_image, uri)
                        if summary:
                            query = (query + "\n\n" + summary).strip()
                        image_uri_payload = [vision_part_from_uri(uri)]
                finally:
                    if upload_task:
                        upload_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await upload_task
                    upload_task = None  # will be re-started for TYPING

            # Access the assistant from the thread state
            assistant = chats[chat_id].assistant

            # Reset memory if not replying in-thread
            if not message.reply_to_message:
                assistant.invoke(
                    {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
                    chats[chat_id].get_config(),
                    stream_mode="values",
                )

            # Post a placeholder and start TYPING while we stream
            placeholder = await message.reply("⌛ Обрабатываю запрос...", parse_mode=None)
            typing_task = await start_show_typing(bot, chat_id, ChatAction.TYPING)

            # Build the message for the assistant
            payload_msg = HumanMessage(content=[{"type": "text", "text": query}] + image_uri_payload)

            try:
                final_answer = await collect_final_text_from_stream(
                    assistant, payload_msg, chats[chat_id].get_config()
                )
                await finalize_placeholder_or_fallback(bot, placeholder, chat_id, final_answer)

                # Persist Q/A for rating flow
                chats[chat_id].question = query
                chats[chat_id].answer = final_answer

                # Ask for rating
                await bot.send_message(chat_id, "Пожалуйста, оцените ответ.", reply_markup=create_rating_keyboard(), parse_mode=None)

            except Exception as e:
                logging.exception("Error while streaming/answering")
                # Surface error into placeholder if possible
                with contextlib.suppress(Exception):
                    if placeholder:
                        await bot.edit_message_text(
                            chat_id=placeholder.chat.id,
                            message_id=placeholder.message_id,
                            text="❌ Ошибка при обработке ответа",
                        )
                    else:
                        await bot.send_message(chat_id, "❌ Ошибка при обработке ответа", parse_mode=None)
            finally:
                if typing_task:
                    typing_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await typing_task

        except Exception as e:
            logging.exception("Unexpected error in handle_message")
            if placeholder:
                with contextlib.suppress(Exception):
                    await bot.edit_message_text(
                        chat_id=placeholder.chat.id,
                        message_id=placeholder.message_id,
                        text="❌ Ошибка при обработке ответа",
                    )
            else:
                with contextlib.suppress(Exception):
                    await bot.send_message(message.chat.id, "❌ Ошибка при обработке ответа", parse_mode=None)
        finally:
            kb_update_thread.resume()

    # Rating callback handler
    @dp.callback_query(lambda c: c.data and c.data.startswith("rate_"))
    async def callback_rating(call: types.CallbackQuery):
        rating = call.data.split("_")[1]
        chat_id = call.message.chat.id
        user_id = call.from_user.id
        username = call.from_user.username or f"id:{user_id}"

        # Collect data from the thread state
        user_question = getattr(chats[chat_id], "question", "-")
        bot_response = getattr(chats[chat_id], "answer", "-")
        model = getattr(chats[chat_id].model, "value", "-") if getattr(chats[chat_id], "model", None) else "-"
        context = getattr(chats[chat_id], "context", "") or ""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if sheets_manager:
            with contextlib.suppress(Exception):
                sheets_manager.append_row([
                    timestamp,
                    username,
                    user_question,
                    bot_response,
                    rating,
                    model,
                    context[:32000],
                ])

        # Acknowledge & remove keyboard
        with contextlib.suppress(Exception):
            await call.answer(f"Вы поставили рейтинг: {rating}")
        with contextlib.suppress(Exception):
            await bot.edit_message_reply_markup(chat_id=chat_id, message_id=call.message.message_id, reply_markup=None)

    # Periodic knowledge base update (threaded, paused during heavy work)
    def periodic_kb_update():
        refresh_indexes()

    kb_update_thread = PeriodicTask(periodic_kb_update)
    kb_update_thread.start()

    # --- MODE SWITCH: polling vs webhook (added) ---
    hook_mode = BOT_MODE in {"hook", "@hook@", "webhook"}
    if hook_mode:
        if not WEBHOOK_URL:
            logging.error("WEBHOOK_BASE is not set. Set config.WEBHOOK_BASE or env WEBHOOK_BASE to run in @hook@ mode.")
            return

        app = web.Application()
        # register webhook handler on WEBHOOK_PATH
        SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=WEBHOOK_SECRET).register(app, path=WEBHOOK_PATH)
        setup_application(app, dp, bot=bot)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=WEBAPP_HOST, port=WEBAPP_PORT)

        try:
            await bot.set_webhook(url=WEBHOOK_URL, secret_token=WEBHOOK_SECRET)
            await site.start()
            logging.info(f"Webhook set: {WEBHOOK_URL} (listening on {WEBAPP_HOST}:{WEBAPP_PORT})")
            # keep running
            await asyncio.Event().wait()
        finally:
            with contextlib.suppress(Exception):
                await bot.delete_webhook(drop_pending_updates=False)
            with contextlib.suppress(Exception):
                await runner.cleanup()
    else:
        # Start polling (existing behavior)
        await dp.start_polling(bot)


if __name__ == "__main__":
    pid = os.getpid()
    with open(".process", "w") as f:
        f.write(f"{pid}")
    asyncio.run(main())
