from __future__ import annotations

from typing import Any


def initialize_agent(*args: Any, **kwargs: Any):
    from .agent import initialize_agent as _initialize_agent

    return _initialize_agent(*args, **kwargs)


__all__ = ["initialize_agent"]
