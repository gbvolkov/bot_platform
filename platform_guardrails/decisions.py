from __future__ import annotations

from typing import Any, Literal, TypedDict


GuardrailAction = Literal[
    "allow",
    "block",
    "redact",
    "rewrite",
    "review",
    "fallback",
]


class GuardrailDecision(TypedDict):
    allowed: bool
    action: GuardrailAction
    reason: str
    risk_score: float
    categories: list[str]
    metadata: dict[str, Any]


def make_decision(
    *,
    allowed: bool,
    action: GuardrailAction,
    reason: str,
    risk_score: float = 0.0,
    categories: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuardrailDecision:
    return {
        "allowed": allowed,
        "action": action,
        "reason": reason,
        "risk_score": risk_score,
        "categories": list(categories or []),
        "metadata": dict(metadata or {}),
    }


def allow(
    reason: str = "Allowed.",
    *,
    categories: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuardrailDecision:
    return make_decision(
        allowed=True,
        action="allow",
        reason=reason,
        categories=categories,
        metadata=metadata,
    )


def redact(
    reason: str,
    *,
    risk_score: float = 0.35,
    categories: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuardrailDecision:
    return make_decision(
        allowed=True,
        action="redact",
        reason=reason,
        risk_score=risk_score,
        categories=categories,
        metadata=metadata,
    )


def rewrite(
    reason: str,
    *,
    risk_score: float = 0.45,
    categories: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuardrailDecision:
    return make_decision(
        allowed=True,
        action="rewrite",
        reason=reason,
        risk_score=risk_score,
        categories=categories,
        metadata=metadata,
    )


def review(
    reason: str,
    *,
    risk_score: float = 0.7,
    categories: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuardrailDecision:
    return make_decision(
        allowed=False,
        action="review",
        reason=reason,
        risk_score=risk_score,
        categories=categories,
        metadata=metadata,
    )


def fallback(
    reason: str,
    *,
    risk_score: float = 0.5,
    categories: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuardrailDecision:
    return make_decision(
        allowed=True,
        action="fallback",
        reason=reason,
        risk_score=risk_score,
        categories=categories,
        metadata=metadata,
    )


def block(
    reason: str,
    *,
    risk_score: float = 1.0,
    categories: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuardrailDecision:
    return make_decision(
        allowed=False,
        action="block",
        reason=reason,
        risk_score=risk_score,
        categories=categories,
        metadata=metadata,
    )


__all__ = [
    "GuardrailAction",
    "GuardrailDecision",
    "allow",
    "block",
    "fallback",
    "make_decision",
    "redact",
    "review",
    "rewrite",
]
