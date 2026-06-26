"""Top-level package for in-house LangGraph agents."""

__all__ = ["get_llm", "ModelType"]


def __getattr__(name: str):
    if name == "get_llm":
        from .llm_utils import get_llm

        return get_llm
    if name == "ModelType":
        from .utils import ModelType

        return ModelType
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
