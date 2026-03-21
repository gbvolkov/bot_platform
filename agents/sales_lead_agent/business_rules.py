from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import settings


LOG = logging.getLogger(__name__)


class RequestRules(BaseModel):
    stopwords: list[str] = Field(default_factory=list)
    region_aliases: dict[str, str] = Field(default_factory=dict)
    priority_aliases: dict[str, str] = Field(default_factory=dict)
    feedback_aliases: dict[str, str] = Field(default_factory=dict)
    result_type_triggers: dict[str, list[str]] = Field(default_factory=dict)
    source_priority_keywords: dict[str, list[str]] = Field(default_factory=dict)
    period_aliases: dict[str, list[str]] = Field(default_factory=dict)
    topic_patterns: list[str] = Field(default_factory=list)
    stop_word_patterns: list[str] = Field(default_factory=list)
    only_with_inn_markers: list[str] = Field(default_factory=list)
    only_with_contacts_markers: list[str] = Field(default_factory=list)


class ScoringRules(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)
    amount_thresholds: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    insufficient_data: dict[str, float] = Field(default_factory=dict)
    rationale_messages: dict[str, str] = Field(default_factory=dict)
    next_steps: dict[str, str] = Field(default_factory=dict)


class OpenSourceRule(BaseModel):
    host: str
    source_type: str = "open_source"
    event_type: str = "open_source"
    tags: list[str] = Field(default_factory=list)
    query_templates: list[str] = Field(default_factory=list)
    path_allow_patterns: list[str] = Field(default_factory=list)
    path_deny_patterns: list[str] = Field(default_factory=list)
    required_keywords: list[str] = Field(default_factory=list)
    blocked_keywords: list[str] = Field(default_factory=list)
    title_patterns: list[str] = Field(default_factory=list)
    summary_patterns: list[str] = Field(default_factory=list)
    company_patterns: list[str] = Field(default_factory=list)
    document_url_patterns: list[str] = Field(default_factory=list)


class OpenSourceRulesConfig(BaseModel):
    rules: list[OpenSourceRule] = Field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to load sales lead rules from %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache
def load_request_rules() -> RequestRules:
    return RequestRules.model_validate(_read_json(Path(settings.request_rules_path)))


@lru_cache
def load_scoring_rules() -> ScoringRules:
    return ScoringRules.model_validate(_read_json(Path(settings.scoring_rules_path)))


@lru_cache
def load_open_source_rules() -> OpenSourceRulesConfig:
    path = Path(settings.whitelist_path)
    if not path.exists():
        return OpenSourceRulesConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to load sales lead source rules from %s: %s", path, exc)
        return OpenSourceRulesConfig()
    if isinstance(payload, list):
        return OpenSourceRulesConfig(
            rules=[OpenSourceRule(host=str(item).strip()) for item in payload if isinstance(item, str) and str(item).strip()]
        )
    if isinstance(payload, dict):
        rules_payload = payload.get("rules")
        if isinstance(rules_payload, list):
            return OpenSourceRulesConfig.model_validate({"rules": rules_payload})
    return OpenSourceRulesConfig()
