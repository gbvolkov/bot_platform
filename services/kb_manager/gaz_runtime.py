"""Compatibility import for the canonical GAZ runtime."""
import sys
from gaz_index_builder import gaz_runtime
sys.modules[__name__] = gaz_runtime
