"""Compatibility import; implementation lives in platform_capabilities.retrievers.teamly_retriever."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("platform_capabilities.retrievers.teamly_retriever")
