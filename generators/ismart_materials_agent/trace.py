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
        print(f"[ismart-materials] {event} {payload}", file=self.stream or sys.stdout, flush=True)
