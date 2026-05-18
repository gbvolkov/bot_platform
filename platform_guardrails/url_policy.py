from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import re
import unicodedata
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from .context import GuardrailContext
from .decisions import GuardrailDecision, block, make_decision


UrlPolicyMode = Literal["audit", "enforce"]

_URL_TOKEN_PATTERN = re.compile(r"[^\s<>()\"']+")
_DNS_DOT_TRANSLATION = str.maketrans(
    {
        "。": ".",
        "．": ".",
        "｡": ".",
    }
)
_TRAILING_PUNCTUATION = ".,;!?)]}"
_LEADING_PUNCTUATION = "([{"
_DOMAIN_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)

_CONFUSABLE_ASCII = str.maketrans(
    {
        "а": "a",
        "А": "a",
        "е": "e",
        "Е": "e",
        "о": "o",
        "О": "o",
        "р": "p",
        "Р": "p",
        "с": "c",
        "С": "c",
        "у": "y",
        "У": "y",
        "х": "x",
        "Х": "x",
        "і": "i",
        "І": "i",
        "ј": "j",
        "Ј": "j",
        "ѕ": "s",
        "Ѕ": "s",
        "α": "a",
        "Α": "a",
        "ο": "o",
        "Ο": "o",
        "ρ": "p",
        "Ρ": "p",
        "ν": "v",
        "Ν": "v",
    }
)


@dataclass(frozen=True)
class UrlReference:
    raw: str
    normalized: str
    host: str
    decoded_host: str
    port: int | None = None
    has_userinfo: bool = False


@dataclass(frozen=True)
class UrlPolicyConfig:
    mode: UrlPolicyMode
    blocked_domains: tuple[str, ...] = field(default_factory=tuple)
    allowed_domains: tuple[str, ...] = field(default_factory=tuple)
    protected_domains: tuple[str, ...] = field(default_factory=tuple)
    block_private_hosts: bool = False
    block_userinfo_urls: bool = False
    block_mixed_script_idn: bool = False
    lookalike_enabled: bool = False
    lookalike_max_distance: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"audit", "enforce"}:
            raise ValueError("URL policy mode must be 'audit' or 'enforce'.")
        object.__setattr__(
            self,
            "blocked_domains",
            tuple(_normalize_domain_pattern(value) for value in self.blocked_domains),
        )
        object.__setattr__(
            self,
            "allowed_domains",
            tuple(_normalize_domain_pattern(value) for value in self.allowed_domains),
        )
        object.__setattr__(
            self,
            "protected_domains",
            tuple(_normalize_domain_pattern(value) for value in self.protected_domains),
        )
        if self.lookalike_max_distance < 0:
            raise ValueError("URL policy lookalike max distance must be non-negative.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UrlPolicyConfig | None":
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("URL policy field 'enabled' must be a boolean.")
        if not enabled:
            return None
        if "mode" not in value:
            raise ValueError("Enabled URL policy requires a 'mode' field.")

        lookalike = value.get("lookalike") or {}
        if not isinstance(lookalike, Mapping):
            raise ValueError("URL policy field 'lookalike' must be an object.")

        return cls(
            mode=_require_mode(value["mode"]),
            blocked_domains=tuple(_require_str_list(value.get("blocked_domains", []), "blocked_domains")),
            allowed_domains=tuple(_require_str_list(value.get("allowed_domains", []), "allowed_domains")),
            protected_domains=tuple(_require_str_list(value.get("protected_domains", []), "protected_domains")),
            block_private_hosts=_require_bool(value.get("block_private_hosts", False), "block_private_hosts"),
            block_userinfo_urls=_require_bool(value.get("block_userinfo_urls", False), "block_userinfo_urls"),
            block_mixed_script_idn=_require_bool(
                value.get("block_mixed_script_idn", False),
                "block_mixed_script_idn",
            ),
            lookalike_enabled=_require_bool(lookalike.get("enabled", False), "lookalike.enabled"),
            lookalike_max_distance=int(lookalike.get("max_distance", 1)),
        )


def coerce_url_policy_config(value: UrlPolicyConfig | Mapping[str, Any] | None) -> UrlPolicyConfig | None:
    if value is None:
        return None
    if isinstance(value, UrlPolicyConfig):
        return value
    if isinstance(value, Mapping):
        return UrlPolicyConfig.from_mapping(value)
    raise TypeError("URL policy config must be an object.")


def scan_url_policy(
    text: str,
    context: GuardrailContext,
    config: UrlPolicyConfig,
    *,
    stage: str,
    boundary: str,
    prompt: str = "",
    source_urls: set[str] | None = None,
) -> list[GuardrailDecision]:
    source_url_keys = set(source_urls or set())
    source_url_keys.update(normalized_url_keys(prompt))
    decisions: list[GuardrailDecision] = []

    for reference in extract_url_references(text):
        decision = _decision_for_reference(
            reference,
            config,
            context,
            stage=stage,
            boundary=boundary,
            source_url=reference.normalized in source_url_keys,
        )
        if decision is not None:
            decisions.append(decision)
    return decisions


def extract_url_references(text: str) -> list[UrlReference]:
    references: list[UrlReference] = []
    seen: set[str] = set()
    for match in _URL_TOKEN_PATTERN.finditer(text or ""):
        raw = match.group(0).strip(_LEADING_PUNCTUATION).rstrip(_TRAILING_PUNCTUATION)
        reference = _parse_url_reference(raw)
        if reference is None or reference.normalized in seen:
            continue
        seen.add(reference.normalized)
        references.append(reference)
    return references


def normalized_url_keys(text: str) -> set[str]:
    return {reference.normalized for reference in extract_url_references(text)}


def _decision_for_reference(
    reference: UrlReference,
    config: UrlPolicyConfig,
    context: GuardrailContext,
    *,
    stage: str,
    boundary: str,
    source_url: bool,
) -> GuardrailDecision | None:
    if config.block_private_hosts and _is_private_host(reference.host):
        return _url_policy_decision(
            config,
            "URL policy detected a private or local host.",
            rule="private_host",
            reference=reference,
            context=context,
            stage=stage,
            boundary=boundary,
            source_url=source_url,
        )

    blocked_pattern = _matching_domain_pattern(reference.host, config.blocked_domains)
    if blocked_pattern is not None:
        return _url_policy_decision(
            config,
            "URL policy detected a blocked domain.",
            rule="blocked_domain",
            reference=reference,
            context=context,
            stage=stage,
            boundary=boundary,
            source_url=source_url,
            matched_pattern=blocked_pattern,
        )

    if config.block_userinfo_urls and reference.has_userinfo:
        return _url_policy_decision(
            config,
            "URL policy detected a URL with userinfo credentials.",
            rule="userinfo_url",
            reference=reference,
            context=context,
            stage=stage,
            boundary=boundary,
            source_url=source_url,
        )

    if config.block_mixed_script_idn and _has_mixed_script_idn(reference.decoded_host):
        return _url_policy_decision(
            config,
            "URL policy detected a mixed-script internationalized domain.",
            rule="mixed_script_idn",
            reference=reference,
            context=context,
            stage=stage,
            boundary=boundary,
            source_url=source_url,
        )

    if config.lookalike_enabled:
        matched_protected = _matching_domain_pattern(reference.host, config.protected_domains)
        if matched_protected is None:
            lookalike_target = _lookalike_target(reference, config)
            if lookalike_target is not None:
                return _url_policy_decision(
                    config,
                    "URL policy detected a protected-domain lookalike.",
                    rule="lookalike_domain",
                    reference=reference,
                    context=context,
                    stage=stage,
                    boundary=boundary,
                    source_url=source_url,
                    matched_pattern=lookalike_target,
                )

    allowed_pattern = _matching_domain_pattern(reference.host, config.allowed_domains)
    if allowed_pattern is not None:
        return make_decision(
            allowed=True,
            action="allow",
            reason="URL policy allowed a configured domain.",
            categories=["url_policy"],
            metadata=_url_policy_metadata(
                config,
                "allowed_domain",
                reference,
                context,
                stage=stage,
                boundary=boundary,
                source_url=source_url,
                matched_pattern=allowed_pattern,
            ),
        )

    return None


def _url_policy_decision(
    config: UrlPolicyConfig,
    reason: str,
    *,
    rule: str,
    reference: UrlReference,
    context: GuardrailContext,
    stage: str,
    boundary: str,
    source_url: bool,
    matched_pattern: str | None = None,
) -> GuardrailDecision:
    metadata = _url_policy_metadata(
        config,
        rule,
        reference,
        context,
        stage=stage,
        boundary=boundary,
        source_url=source_url,
        matched_pattern=matched_pattern,
    )
    if config.mode == "audit":
        return make_decision(
            allowed=True,
            action="allow",
            reason=f"{reason} Audit mode allowed the text.",
            risk_score=0.85,
            categories=["url_policy"],
            metadata=metadata,
        )
    return block(
        reason,
        risk_score=0.95,
        categories=["url_policy"],
        metadata=metadata,
    )


def _url_policy_metadata(
    config: UrlPolicyConfig,
    rule: str,
    reference: UrlReference,
    context: GuardrailContext,
    *,
    stage: str,
    boundary: str,
    source_url: bool,
    matched_pattern: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "rail": "url_policy",
        "mode": config.mode,
        "rule": rule,
        "stage": stage,
        "boundary": boundary,
        "host": reference.host,
        "normalized_url": reference.normalized,
        "source_url": source_url,
        "agent_name": context.get("agent_name"),
        "thread_id": context.get("thread_id"),
        "user_role": context.get("user_role"),
        "request_id": context.get("request_id"),
        "tool_name": context.get("tool_name"),
    }
    if matched_pattern is not None:
        metadata["matched_pattern"] = matched_pattern
    return metadata


def _parse_url_reference(raw: str) -> UrlReference | None:
    if not raw:
        return None
    has_scheme = "://" in raw
    if has_scheme and not raw.lower().startswith(("http://", "https://")):
        return None
    if not _could_be_url(raw, has_scheme=has_scheme):
        return None

    parseable = raw if has_scheme else f"http://{raw}"
    parsed = urlsplit(parseable)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.username and not has_scheme and "." not in parsed.username and ":" not in parsed.username:
        return None
    raw_host = parsed.hostname or ""
    host = _canonical_host(raw_host)
    if host is None or not _valid_host(host):
        return None
    try:
        port = parsed.port
    except ValueError:
        port = None
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    port_text = f":{port}" if port is not None else ""
    return UrlReference(
        raw=raw,
        normalized=f"{host}{port_text}{path}{query}",
        host=host,
        decoded_host=_decoded_host(host),
        port=port,
        has_userinfo=parsed.username is not None,
    )


def _could_be_url(raw: str, *, has_scheme: bool) -> bool:
    if has_scheme:
        return True
    normalized = _normalize_dns_separators(raw)
    lowered = raw.lower()
    return (
        "." in normalized
        or "@" in raw
        or lowered.startswith("localhost")
        or _looks_like_ipv4(raw.split("/", 1)[0].split(":", 1)[0])
    )


def _canonical_host(host: str) -> str | None:
    normalized = _normalize_dns_separators(host).strip().strip("[]").strip(".").lower()
    if not normalized:
        return None
    try:
        return str(ipaddress.ip_address(normalized))
    except ValueError:
        pass
    try:
        return normalized.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None


def _decoded_host(host: str) -> str:
    labels: list[str] = []
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                labels.append(label.encode("ascii").decode("idna"))
            except UnicodeError:
                labels.append(label)
        else:
            labels.append(label)
    return ".".join(labels)


def _valid_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    labels = host.split(".")
    if len(labels) < 2 or len(labels[-1]) < 2:
        return False
    return all(_DOMAIN_LABEL_PATTERN.match(label) for label in labels)


def _looks_like_ipv4(value: str) -> bool:
    parts = value.split(".")
    if len(parts) != 4:
        return False
    return all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


def _normalize_domain_pattern(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("URL policy domain entries must be non-empty strings.")
    text = _normalize_dns_separators(value).strip().lower()
    wildcard = text.startswith("*.")
    domain = text[2:] if wildcard else text
    if "/" in domain or ":" in domain or "@" in domain:
        raise ValueError("URL policy domain entries must not include scheme, path, port, or userinfo.")
    canonical = _canonical_host(domain)
    if canonical is None or not _valid_host(canonical):
        raise ValueError(f"Invalid URL policy domain entry: {value!r}.")
    return f"*.{canonical}" if wildcard else canonical


def _normalize_dns_separators(value: str) -> str:
    return value.translate(_DNS_DOT_TRANSLATION)


def _matching_domain_pattern(host: str, patterns: Sequence[str]) -> str | None:
    for pattern in patterns:
        if pattern.startswith("*."):
            suffix = pattern[2:]
            if host.endswith(f".{suffix}") and host != suffix:
                return pattern
        elif host == pattern:
            return pattern
    return None


def _is_private_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_reserved
        or parsed.is_multicast
        or parsed.is_unspecified
    )


def _has_mixed_script_idn(decoded_host: str) -> bool:
    scripts: set[str] = set()
    for char in decoded_host:
        if char.isascii() or not char.isalpha():
            continue
        script = _script_name(char)
        if script is not None:
            scripts.add(script)
    return len(scripts) > 1 or (bool(scripts) and any(char.isascii() and char.isalpha() for char in decoded_host))


def _script_name(char: str) -> str | None:
    name = unicodedata.name(char, "")
    for script in ("CYRILLIC", "GREEK", "LATIN"):
        if script in name:
            return script
    return None


def _lookalike_target(reference: UrlReference, config: UrlPolicyConfig) -> str | None:
    candidate = _domain_skeleton(reference.decoded_host)
    candidate_compact = candidate.replace(".", "")
    for protected in config.protected_domains:
        target = protected[2:] if protected.startswith("*.") else protected
        target_decoded = _decoded_host(target)
        target_skeleton = _domain_skeleton(target_decoded)
        if reference.host == target or candidate == target_skeleton:
            continue
        if _levenshtein(candidate_compact, target_skeleton.replace(".", "")) <= config.lookalike_max_distance:
            return protected
        candidate_parts = candidate.split(".")
        target_parts = target_skeleton.split(".")
        if (
            len(candidate_parts) == len(target_parts)
            and candidate_parts[-1] == target_parts[-1]
            and _levenshtein(candidate_parts[-2], target_parts[-2]) <= config.lookalike_max_distance
        ):
            return protected
    return None


def _domain_skeleton(value: str) -> str:
    return value.translate(_CONFUSABLE_ASCII).lower()


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _require_mode(value: Any) -> UrlPolicyMode:
    if value in {"audit", "enforce"}:
        return value
    raise ValueError("URL policy mode must be 'audit' or 'enforce'.")


def _require_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"URL policy field '{field_name}' must be a boolean.")


def _require_str_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"URL policy field '{field_name}' must be a list.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"URL policy field '{field_name}' must contain only non-empty strings.")
    return list(value)


__all__ = [
    "UrlPolicyConfig",
    "UrlPolicyMode",
    "UrlReference",
    "coerce_url_policy_config",
    "extract_url_references",
    "normalized_url_keys",
    "scan_url_policy",
]
