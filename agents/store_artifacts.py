"""Compatibility import; implementation lives in platform_capabilities.rendering."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("platform_capabilities.rendering")
