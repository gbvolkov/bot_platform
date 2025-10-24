from __future__ import annotations

from fastapi import APIRouter

from . import agents, conversations

router = APIRouter()
router.include_router(agents.router)
router.include_router(conversations.router)

__all__ = ["router"]

