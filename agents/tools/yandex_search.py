"""Compatibility import; implementation lives in platform_capabilities.tools.yandex_search."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("platform_capabilities.tools.yandex_search")
