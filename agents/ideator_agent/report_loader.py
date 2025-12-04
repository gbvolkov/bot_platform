from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .models import ArticleRecord, IdeatorReport


def _norm_importance(val: str) -> str:
    val = (val or "").lower().strip()
    if val in ("high", "medium", "low"):
        return val
    return "medium"


def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


def _as_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default


def load_report(path: str | Path) -> IdeatorReport:
    data: Dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))

    articles: List[ArticleRecord] = []
    for idx, raw in enumerate(data.get("articles", [])):
        articles.append(
            ArticleRecord(
                id=idx,
                title=_safe_str(raw.get("title", "")).strip(),
                summary=_safe_str(raw.get("summary", "")).strip(),
                url=_safe_str(raw.get("url", "")).strip(),
                importance=_norm_importance(raw.get("importance", "")),
                date=_safe_str(raw.get("date", ""))[:10],
                processed_at=_safe_str(raw.get("processed_at", "")),
                search_country=_safe_str(raw.get("search_country", "")).lower(),
                search_language=_safe_str(raw.get("search_language", "")).lower(),
                source_file=_safe_str(raw.get("source_file", "")),
                word_count=_as_int(raw.get("word_count", 0)),
                importance_reasoning=_safe_str(raw.get("importance_reasoning", "")),
                search_keywords=[_safe_str(k) for k in raw.get("search_keywords", [])],
                raw=raw,
            )
        )

    search_goal = ""
    search_prompt = ""
    if data.get("articles"):
        sample = data["articles"][0] or {}
        search_goal = _safe_str(sample.get("search_goal", ""))
        search_prompt = _safe_str(sample.get("search_prompt", ""))

    return IdeatorReport(
        generated_at=_safe_str(data.get("generated_at", "")),
        report_date=_safe_str(data.get("report_date", "")),
        total_articles=_as_int(data.get("total_articles", len(articles))),
        high_importance=_as_int(data.get("high_importance", 0)),
        medium_importance=_as_int(data.get("medium_importance", 0)),
        low_importance=_as_int(data.get("low_importance", 0)),
        search_goal=search_goal,
        search_prompt=search_prompt,
        articles=articles,
    )


__all__ = ["load_report"]
