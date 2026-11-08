"""Literal text fallback for clients that otherwise interpret agent prose as Markdown."""
from __future__ import annotations

import re
from typing import Any
from threading import RLock


# Installed Mistletoe renderers share process-local token registries. All repository
# parser contexts use this same lock; unrelated model/tool execution remains parallel.
MARKDOWN_PARSE_LOCK = RLock()


def literal_markdown(text: str) -> str:
    """Encode all text as one inert CommonMark block, including embedded fences/HTML/URLs."""
    if not text:
        return ""
    length = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, length + 1)
    return fence + "\n" + text + ("" if text.endswith("\n") else "\n") + fence


def public_text_piece(piece: dict[str, Any]) -> str:
    """Opt-in only; preserve the established behavior of unmarked text parts."""
    text = piece.get("text") or ""
    return literal_markdown(text) if piece.get("format") == "plain" else text
