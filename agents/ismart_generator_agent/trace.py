from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass
class TraceLogger:
    enabled: bool = False
    stream: TextIO | None = None

    def log(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = json.dumps(fields, ensure_ascii=False, default=str)
        stream = self.stream or sys.stdout
        line = f"[ismart-generator-agent] {event} {payload}"
        try:
            print(line, file=stream, flush=True)
        except UnicodeEncodeError:
            safe_line = line.encode(getattr(stream, "encoding", None) or "utf-8", errors="backslashreplace").decode(
                getattr(stream, "encoding", None) or "utf-8",
                errors="replace",
            )
            print(safe_line, file=stream, flush=True)
