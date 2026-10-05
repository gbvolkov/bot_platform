"""Compatibility import; implementation lives in agent_runtime.invocation."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("agent_runtime.invocation")
