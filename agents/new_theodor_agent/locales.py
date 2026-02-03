from __future__ import annotations

from typing import Optional

from .artifacts_defs import set_artifacts_locale

DEFAULT_LOCALE = "ru"
_CURRENT_LOCALE = DEFAULT_LOCALE


def normalize_locale(locale: Optional[str] = None) -> str:
    if locale in {"ru", "en"}:
        return str(locale)
    return DEFAULT_LOCALE


def resolve_locale(locale: Optional[str] = None) -> str:
    if locale in {"ru", "en"}:
        return str(locale)
    return _CURRENT_LOCALE


def set_locale(locale: str = DEFAULT_LOCALE) -> str:
    global _CURRENT_LOCALE
    _CURRENT_LOCALE = normalize_locale(locale)
    set_artifacts_locale(_CURRENT_LOCALE)
    return _CURRENT_LOCALE


def get_current_locale() -> str:
    return _CURRENT_LOCALE
