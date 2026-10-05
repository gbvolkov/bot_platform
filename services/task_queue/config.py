"""Compatibility import; implementation lives in platform_sdk.queue_settings."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("platform_sdk.queue_settings")
