"""Explicit runtime context available to compatibility factories and tools."""
from contextlib import contextmanager
from contextvars import ContextVar
from platform_contracts import AgentContext

_context: ContextVar[AgentContext | None] = ContextVar("agent_context", default=None)


def current_context() -> AgentContext | None:
    return _context.get()


@contextmanager
def use_context(context: AgentContext):
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)
