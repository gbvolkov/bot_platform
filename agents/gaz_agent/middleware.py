from __future__ import annotations

from typing import Any

from agents.palimpsest_sessions import PalimpsestSessionManager, PalimpsestSessionMiddleware


_ANON_ENTITIES = [
    "RU_PERSON",
    "CREDIT_CARD",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "URL",
    "RU_PASSPORT",
    "SNILS",
    "INN",
    "RU_BANK_ACC",
    "TICKET_NUMBER",
]


class PalimpsestMiddleware(PalimpsestSessionMiddleware):
    pass


def build_palimpsest_session_manager(locale: str = "ru-RU") -> PalimpsestSessionManager:
    from palimpsest import Palimpsest

    return PalimpsestSessionManager(
        Palimpsest(
            verbose=False,
            run_entities=_ANON_ENTITIES,
            locale=locale,
        )
    )


def build_palimpsest_middleware(
    locale: str = "ru-RU",
    *,
    sessions: PalimpsestSessionManager | None = None,
    processor: Any = None,
) -> PalimpsestMiddleware:
    if sessions is None:
        if processor is None:
            sessions = build_palimpsest_session_manager(locale)
        else:
            sessions = PalimpsestSessionManager(processor)
    return PalimpsestMiddleware(sessions)
