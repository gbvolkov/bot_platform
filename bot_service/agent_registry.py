"""Compatibility import; implementation lives in agent_runtime.registry."""
from platform_sdk.compatibility import alias_module
alias_module(__name__, globals(), "agent_runtime.registry")
