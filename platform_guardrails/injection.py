from __future__ import annotations

from typing import Any

from .decisions import GuardrailDecision, block, make_decision, redact, review


SECURITY_BLOCK_MESSAGE_RU = (
    "Запрос заблокирован системой безопасности. Измените запрос и попробуйте снова."
)
SECURITY_REVIEW_MESSAGE_RU = (
    "Запрос требует проверки системой безопасности. Измените запрос и попробуйте снова."
)


SCANNER_CATEGORY_BY_NAME = {
    "PromptInjection": "prompt_injection",
    "Secrets": "secret",
    "MaliciousURLs": "malicious_url",
    "Toxicity": "toxic_content",
    "BanTopics": "banned_topic",
    "Sensitive": "output_sensitivity",
    "TokenLimit": "token_limit",
}


def scanner_category(scanner_name: str) -> str:
    return SCANNER_CATEGORY_BY_NAME.get(scanner_name, "scanner")


def scanner_decision(
    *,
    scanner_name: str,
    is_valid: bool,
    risk_score: float,
    sanitized_changed: bool,
    stage: str,
    boundary: str,
    metadata: dict[str, Any] | None = None,
) -> GuardrailDecision:
    category = scanner_category(scanner_name)
    safe_metadata = {
        "rail": "llm_guard",
        "scanner": scanner_name,
        "stage": stage,
        "boundary": boundary,
        "sanitized_changed": sanitized_changed,
    }
    safe_metadata.update(metadata or {})

    normalized_score = max(0.0, min(1.0, risk_score if risk_score >= 0 else 0.0))

    if is_valid:
        return make_decision(
            allowed=True,
            action="allow",
            reason=f"{scanner_name} scanner allowed {stage} text.",
            risk_score=normalized_score,
            categories=[category],
            metadata=safe_metadata,
        )

    if scanner_name in {"Secrets", "Sensitive"} and sanitized_changed:
        return redact(
            f"{scanner_name} scanner redacted {stage} text.",
            risk_score=max(normalized_score, 0.35),
            categories=[category],
            metadata=safe_metadata,
        )

    if scanner_name in {"Toxicity", "BanTopics"}:
        return review(
            f"{scanner_name} scanner requires review for {stage} text.",
            risk_score=max(normalized_score, 0.7),
            categories=[category],
            metadata=safe_metadata,
        )

    return block(
        f"{scanner_name} scanner blocked {stage} text.",
        risk_score=max(normalized_score, 0.8),
        categories=[category],
        metadata=safe_metadata,
    )


__all__ = [
    "SCANNER_CATEGORY_BY_NAME",
    "SECURITY_BLOCK_MESSAGE_RU",
    "SECURITY_REVIEW_MESSAGE_RU",
    "scanner_category",
    "scanner_decision",
]
