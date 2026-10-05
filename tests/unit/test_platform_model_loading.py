"""Privacy and retrieval constructors must not overlap global torch contexts."""
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
import sys
import threading

import pytest

from platform_capabilities.retrievers.utils import models_builder
from platform_guardrails.privacy import PrivacyRail


@pytest.mark.parametrize("getter,constructor,cache", [
    ("getEmbeddingModel", "HuggingFaceEmbeddings", "_embedding_model"),
    ("getRerankerModel", "HuggingFaceCrossEncoder", "_reranker_model"),
    ("scanner", None, None),
])
def test_privacy_and_retrieval_model_construction_is_serial(getter, constructor, cache, monkeypatch):
    privacy_entered, retrieval_waiting, retrieval_entered, release = (
        threading.Event() for _ in range(4))
    loaded = object()
    calls = []
    class PrivacyProcessor:
        def __init__(self, **kwargs):
            privacy_entered.set()
            assert release.wait(5)
    module = ModuleType("palimpsest")
    module.Palimpsest = PrivacyProcessor
    monkeypatch.setitem(sys.modules, "palimpsest", module)
    monkeypatch.setattr("platform_guardrails.privacy._ensure_palimpsest_dependencies", lambda locale: None)
    if cache is not None:
        monkeypatch.setattr(models_builder, cache, None)
    def retrieval_constructor(**kwargs):
        retrieval_entered.set()
        assert release.is_set(), "Retrieval entered a concurrent privacy model's tensor context"
        calls.append(kwargs)
        return loaded
    if getter == "scanner":
        from platform_guardrails.scanners import LLMGuardScannerRail, ScannerSpec
        rail = LLMGuardScannerRail(input_factory=lambda spec: retrieval_constructor())
        spec = ScannerSpec("PromptInjection")
        retrieve = lambda: rail._instance_for_spec(spec, "input")
    else:
        monkeypatch.setattr(models_builder, constructor, retrieval_constructor)
        retrieve = getattr(models_builder, getter)
    def retrieval():
        retrieval_waiting.set()
        return retrieve()
    with ThreadPoolExecutor(2) as executor:
        privacy = executor.submit(PrivacyRail.from_palimpsest)
        assert privacy_entered.wait(5)
        future = executor.submit(retrieval)
        try:
            assert retrieval_waiting.wait(5)
            assert not retrieval_entered.wait(.2), "Neural-model construction overlapped"
        finally:
            release.set()
        assert isinstance(privacy.result(5), PrivacyRail)
        assert future.result(5) is loaded
    assert retrieve() is loaded and len(calls) == 1
