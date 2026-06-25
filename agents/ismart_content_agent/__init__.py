"""iSMART template-only content generation MVP."""

from .contracts import ContentItem, GenerationRequest, TemplateSpec
from .pipeline import (
    generate_run,
    mark_approved,
    mark_preview_passed,
    publish_run,
    validate_run,
)

__all__ = [
    "ContentItem",
    "GenerationRequest",
    "TemplateSpec",
    "generate_run",
    "mark_approved",
    "mark_preview_passed",
    "publish_run",
    "validate_run",
]
