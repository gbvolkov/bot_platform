from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict


_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-zA-Z0-9а-яА-Я]+")


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    lowered = value.casefold().strip()
    lowered = _NON_WORD_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def compute_dedup_key(payload: Dict[str, Any]) -> str:
    inn = normalize_text(str(payload.get("inn") or ""))
    if inn:
        return f"inn:{inn}"

    source_type = normalize_text(str(payload.get("source_type") or ""))
    source_id = normalize_text(str(payload.get("source_id") or ""))
    source_url = normalize_text(str(payload.get("source_url") or ""))
    if source_type and (source_id or source_url):
        return f"source:{source_type}:{source_id or source_url}"

    company = normalize_text(str(payload.get("company_name") or "unknown"))
    event_type = normalize_text(str(payload.get("event_type") or "event"))
    event_date = payload.get("event_date")
    if isinstance(event_date, str):
        event_date = event_date[:10]
    elif isinstance(event_date, datetime):
        event_date = event_date.date().isoformat()
    else:
        event_date = "na"

    amount = payload.get("amount")
    try:
        amount_bucket = int(float(amount or 0) // 1_000_000)
    except Exception:  # noqa: BLE001
        amount_bucket = 0
    return f"soft:{company}:{event_type}:{event_date}:{amount_bucket}"
