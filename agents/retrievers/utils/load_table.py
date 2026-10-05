"""Compatibility import; implementation lives in platform_capabilities.retrievers.utils.load_table."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("platform_capabilities.retrievers.utils.load_table")
