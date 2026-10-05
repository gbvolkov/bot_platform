"""Compatibility import; implementation lives in platform_capabilities.retrievers.cross_encoder_reranker_with_score."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("platform_capabilities.retrievers.cross_encoder_reranker_with_score")
