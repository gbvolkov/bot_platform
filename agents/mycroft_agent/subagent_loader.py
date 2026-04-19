from __future__ import annotations

import asyncio

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

from .web_search_subagent import WEB_SEARCH_AGENT_ID, build_web_search_subagent


async def initialize_configured_subagents(
    agent_ids: tuple[str, ...],
) -> list[SubAgent | CompiledSubAgent]:
    from bot_service.agent_registry import agent_registry

    async def load_one(agent_id: str) -> SubAgent | CompiledSubAgent:
        if agent_id == WEB_SEARCH_AGENT_ID:
            return build_web_search_subagent()

        definitions = getattr(agent_registry, "_definitions", {})
        definition = definitions.get(agent_id)
        if definition is None:
            available = ", ".join(sorted(definitions))
            raise ValueError(
                f"Unknown registry agent '{agent_id}'. Available agents: {available}"
            )

        while True:
            ready = await agent_registry.ensure_agent_ready(agent_id)
            if ready:
                break
            await asyncio.sleep(0.01)

        instance = agent_registry.get_agent(agent_id)
        return CompiledSubAgent(
            name=definition.id,
            description=f"{definition.name}. {definition.description}",
            runnable=instance,
        )

    return await asyncio.gather(*(load_one(agent_id) for agent_id in agent_ids))
