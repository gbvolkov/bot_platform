import asyncio
from types import SimpleNamespace

import pytest

from agents.mycroft_agent.subagent_loader import initialize_configured_subagents


def test_resolver_receives_explicit_state_scope():
    calls = []
    runnable = object()
    class Resolver:
        def describe(self, agent_id):
            return SimpleNamespace(id=agent_id, name="Child", description="Description")
        async def resolve(self, agent_id, *, state_scope):
            calls.append((agent_id, state_scope))
            return runnable
    result = asyncio.run(initialize_configured_subagents(("child",), agent_resolver=Resolver(), state_scope="stateful"))
    assert calls == [("child", "stateful")]
    assert result[0]["runnable"] is runnable


def test_missing_resolver_is_an_error():
    with pytest.raises(ValueError, match="injected agent resolver"):
        asyncio.run(initialize_configured_subagents(("child",), agent_resolver=None, state_scope="stateless"))
