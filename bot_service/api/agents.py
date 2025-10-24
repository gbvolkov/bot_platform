from __future__ import annotations

from fastapi import APIRouter

from ..agent_registry import agent_registry
from ..schemas import AgentInfo

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/", response_model=list[AgentInfo])
async def list_agents() -> list[AgentInfo]:
    return agent_registry.list_agents()

