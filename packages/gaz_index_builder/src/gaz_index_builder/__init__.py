"""Standalone GAZ index builder package."""

from __future__ import annotations

from typing import Any

__all__ = ["GazRuntimeService"]


def __getattr__(name: str) -> Any:
    if name == "GazRuntimeService":
        from .gaz_runtime import GazRuntimeService

        return GazRuntimeService
    raise AttributeError(name)
