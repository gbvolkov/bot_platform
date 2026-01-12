import json
from typing import Any, Annotated, get_args, get_origin, get_type_hints

from typing_extensions import NotRequired, is_typeddict
from langchain.agents.middleware import ModelRequest, wrap_model_call
from langchain.agents.structured_output import (
    AutoStrategy,
    ProviderStrategy,
    ToolStrategy,
    StructuredOutputValidationError,
)


def _unwrap_annotated(tp: Any) -> Any:
    """Strip Annotated to get the underlying type."""
    if get_origin(tp) is Annotated:
        return get_args(tp)[0]
    return tp


def _describe_annotated(tp: Any) -> str | None:
    """Extract a description from Annotated metadata if present."""
    if get_origin(tp) is Annotated:
        meta = get_args(tp)[1:]
        for m in meta:
            if isinstance(m, str):
                return m
    return None


def _example_for_type(tp: Any) -> Any:
    """Build a minimal example value for a type hint."""
    tp = _unwrap_annotated(tp)
    origin = get_origin(tp)

    if origin is NotRequired:
        args = get_args(tp)
        inner = args[0] if args else Any
        return _example_for_type(inner)

    if is_typeddict(tp):
        return _example_from_typed_dict(tp)

    if origin is list:
        args = get_args(tp)
        inner = args[0] if args else str
        return [_example_for_type(inner)]

    if origin is dict:
        args = get_args(tp)
        value_type = args[1] if len(args) > 1 else str
        return {"key": _example_for_type(value_type)}

    if tp in (str,):
        return "<string>"
    if tp in (int, float):
        return 0

    return f"<{getattr(tp, '__name__', 'value')}>"


def _example_from_typed_dict(schema: Any) -> dict[str, Any]:
    """Recursively build an example dict from a TypedDict schema."""
    hints = get_type_hints(schema, include_extras=True)
    return {name: _example_for_type(tp) for name, tp in hints.items()}


def _collect_field_descriptions(schema: Any, prefix: str = "") -> list[str]:
    """Collect field descriptions (including nested TypedDicts)."""
    hints = get_type_hints(schema, include_extras=True)
    descriptions: list[str] = []
    for name, tp in hints.items():
        desc = _describe_annotated(tp)
        field_name = f"{prefix}{name}"
        if desc:
            descriptions.append(f"{field_name}: {desc}")
        base = _unwrap_annotated(tp)
        origin = get_origin(base)
        if origin is NotRequired:
            args = get_args(base)
            base = args[0] if args else Any
        if is_typeddict(base):
            descriptions.extend(_collect_field_descriptions(base, prefix=f"{field_name}."))
    return descriptions


def build_json_prompt(schema: Any) -> str:
    """
    Build an instruction snippet that forces the model to return JSON for a TypedDict schema.
    """
    if not is_typeddict(schema):
        raise TypeError("schema must be a TypedDict class")

    skeleton = _example_from_typed_dict(schema)
    descriptions = _collect_field_descriptions(schema)
    desc_block = ""
    if descriptions:
        desc_block = "Field descriptions:\n" + "\n".join(f"- {d}" for d in descriptions)

    return (
        "Return exactly one JSON object with this structure. "
        "No extra text, no code fences:\n"
        f"{json.dumps(skeleton, ensure_ascii=False, indent=2)}\n"
        "No additional keys."
        + ("\n" + desc_block if desc_block else "")
    )

@wrap_model_call
def provider_then_tool(request: ModelRequest, handler):
    """Retry structured output via tool strategy if provider strategy fails."""
    try:
        return handler(request)
    except (ValueError, StructuredOutputValidationError):
        rf = request.response_format
        if isinstance(rf, AutoStrategy):
            schema = rf.schema
        elif isinstance(rf, ProviderStrategy):
            schema = rf.schema
        else:
            raise  # already in ToolStrategy; bubble up
        # Retry using tool-based structured output
        return handler(request.override(response_format=ToolStrategy(schema=schema)))


__all__ = ["build_json_prompt", "provider_then_tool"]
