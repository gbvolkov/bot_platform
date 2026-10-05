from __future__ import annotations

import asyncio
from platform_contracts import AgentResolver

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

from .web_search_subagent import WEB_SEARCH_AGENT_ID, build_web_search_subagent


async def initialize_configured_subagents(
    agent_ids: tuple[str, ...],
    *,
    agent_resolver: AgentResolver,
    state_scope: str,
) -> list[SubAgent | CompiledSubAgent]:
    if state_scope not in {"stateless", "stateful"}:
        raise ValueError("Subagent state scope must be explicit")

    async def load_one(agent_id: str) -> SubAgent | CompiledSubAgent:
        if agent_id == WEB_SEARCH_AGENT_ID:
            return build_web_search_subagent()

        if agent_resolver is None:
            raise ValueError("Configured subagents require an injected agent resolver")
        definition = agent_resolver.describe(agent_id)
        instance = await agent_resolver.resolve(agent_id, state_scope=state_scope)
        return CompiledSubAgent(
            name=definition.id,
            description=f"{definition.name}. {definition.description}",
            runnable=instance,
        )

    return await asyncio.gather(*(load_one(agent_id) for agent_id in agent_ids))
