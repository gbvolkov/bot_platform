"""Shared local composition for CLIs retaining their existing arguments."""
import inspect
from typing import Any, Callable


def build_local_agent(factory: Callable[..., Any], **settings):
    signature = inspect.signature(factory)
    unsupported = set(settings) - set(signature.parameters)
    if unsupported:
        raise TypeError(f"Unsupported settings: {', '.join(sorted(unsupported))}")
    # bind rejects unsupported settings and missing required arguments without
    # deleting user configuration. Provider/checkpointer objects stay explicit.
    signature.bind(**settings)
    return factory(**settings)


def compile_local_graph(factory, *, guardrail_runtime, checkpointer, tools, tool_profiles, compiler=None, **settings):
    from platform_guardrails.graph_compiler import PlatformGraphCompiler
    spec = build_local_agent(factory, **settings)
    return (compiler or PlatformGraphCompiler()).compile(spec, guardrail_runtime=guardrail_runtime,
        checkpointer=checkpointer, tools=tools, tool_profiles=tool_profiles)
