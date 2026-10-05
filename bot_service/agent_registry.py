"""Compatibility import; implementation lives in agent_runtime.registry."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("agent_runtime.registry")
