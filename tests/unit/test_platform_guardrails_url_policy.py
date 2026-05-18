from __future__ import annotations

import pytest

from platform_guardrails.url_policy import (
    UrlPolicyConfig,
    extract_url_references,
    normalized_url_keys,
    scan_url_policy,
)


def _context():
    return {
        "agent_name": "test",
        "thread_id": "thread",
        "user_role": "default",
        "request_id": "req",
        "tool_name": None,
    }


def test_url_policy_blocks_exact_and_wildcard_denylist_matches():
    config = UrlPolicyConfig(
        mode="enforce",
        blocked_domains=("bad.example", "*.evil.example"),
    )

    exact = scan_url_policy("Open https://bad.example/path", _context(), config, stage="output", boundary="model")
    wildcard = scan_url_policy(
        "Open https://a.evil.example/path",
        _context(),
        config,
        stage="output",
        boundary="model",
    )

    assert exact[0]["allowed"] is False
    assert exact[0]["metadata"]["matched_pattern"] == "bad.example"
    assert wildcard[0]["allowed"] is False
    assert wildcard[0]["metadata"]["matched_pattern"] == "*.evil.example"


def test_url_policy_allowlist_match_emits_allow_decision():
    config = UrlPolicyConfig(mode="audit", allowed_domains=("safe.example",))

    decisions = scan_url_policy("Open https://safe.example/path", _context(), config, stage="output", boundary="model")

    assert decisions[0]["allowed"] is True
    assert decisions[0]["metadata"]["rule"] == "allowed_domain"
    assert decisions[0]["metadata"]["matched_pattern"] == "safe.example"


def test_url_policy_allows_source_url_unless_it_is_denylisted():
    source_urls = normalized_url_keys("Source: https://source.example/path")
    allowed_config = UrlPolicyConfig(mode="enforce")
    blocked_config = UrlPolicyConfig(mode="enforce", blocked_domains=("source.example",))

    allowed = scan_url_policy(
        "Reuse https://source.example/path",
        _context(),
        allowed_config,
        stage="output",
        boundary="model",
        source_urls=source_urls,
    )
    blocked = scan_url_policy(
        "Reuse https://source.example/path",
        _context(),
        blocked_config,
        stage="output",
        boundary="model",
        source_urls=source_urls,
    )

    assert allowed == []
    assert blocked[0]["allowed"] is False
    assert blocked[0]["metadata"]["source_url"] is True


def test_url_policy_detects_private_hosts_and_userinfo_urls():
    config = UrlPolicyConfig(
        mode="enforce",
        block_private_hosts=True,
        block_userinfo_urls=True,
    )

    private = scan_url_policy("Call http://169.254.169.254/latest", _context(), config, stage="output", boundary="tool")
    userinfo = scan_url_policy("Open https://trusted.com@evil.example", _context(), config, stage="output", boundary="tool")

    assert private[0]["metadata"]["rule"] == "private_host"
    assert private[0]["allowed"] is False
    assert userinfo[0]["metadata"]["rule"] == "userinfo_url"
    assert userinfo[0]["allowed"] is False


def test_url_policy_idna_canonicalization_is_stable():
    references = extract_url_references("Open https://пример.рф/path")

    assert references[0].host == "xn--e1afmkfd.xn--p1ai"
    assert references[0].normalized == "xn--e1afmkfd.xn--p1ai/path"


def test_url_policy_matches_non_latin_configured_domains():
    config = UrlPolicyConfig(
        mode="enforce",
        blocked_domains=(
            "пример.рф",
            "*.тест.рф",
        ),
    )

    exact = scan_url_policy(
        "Open https://пример.рф/path",
        _context(),
        config,
        stage="output",
        boundary="model",
    )
    wildcard = scan_url_policy(
        "Open https://под.тест.рф/path",
        _context(),
        config,
        stage="output",
        boundary="model",
    )

    assert exact[0]["allowed"] is False
    assert exact[0]["metadata"]["host"] == "xn--e1afmkfd.xn--p1ai"
    assert wildcard[0]["allowed"] is False
    assert wildcard[0]["metadata"]["matched_pattern"] == "*.xn--e1aybc.xn--p1ai"


def test_url_policy_normalizes_unicode_dns_dot_separators():
    config = UrlPolicyConfig(
        mode="enforce",
        blocked_domains=("例え.テスト",),
    )

    decisions = scan_url_policy(
        "Open https://例え。テスト/path",
        _context(),
        config,
        stage="output",
        boundary="model",
    )

    assert decisions[0]["allowed"] is False
    assert decisions[0]["metadata"]["host"] == "xn--r8jz45g.xn--zckzah"


def test_url_policy_audit_mode_logs_violation_without_blocking():
    config = UrlPolicyConfig(mode="audit", blocked_domains=("bad.example",))

    decisions = scan_url_policy("Open https://bad.example", _context(), config, stage="output", boundary="model")

    assert decisions[0]["allowed"] is True
    assert decisions[0]["action"] == "allow"
    assert decisions[0]["metadata"]["rail"] == "url_policy"
    assert decisions[0]["metadata"]["mode"] == "audit"
    assert decisions[0]["metadata"]["rule"] == "blocked_domain"


def test_url_policy_enforce_mode_blocks_violation():
    config = UrlPolicyConfig(mode="enforce", blocked_domains=("bad.example",))

    decisions = scan_url_policy("Open https://bad.example", _context(), config, stage="output", boundary="model")

    assert decisions[0]["allowed"] is False
    assert decisions[0]["action"] == "block"


def test_url_policy_detects_mixed_script_idn_and_lookalikes():
    config = UrlPolicyConfig(
        mode="enforce",
        protected_domains=("paypal.com",),
        block_mixed_script_idn=True,
        lookalike_enabled=True,
        lookalike_max_distance=1,
    )

    mixed = scan_url_policy(
        "Open https://раураl.com",
        _context(),
        config,
        stage="output",
        boundary="model",
    )
    lookalike = scan_url_policy("Open https://paypa1.com", _context(), config, stage="output", boundary="model")

    assert mixed[0]["metadata"]["rule"] == "mixed_script_idn"
    assert mixed[0]["allowed"] is False
    assert lookalike[0]["metadata"]["rule"] == "lookalike_domain"
    assert lookalike[0]["allowed"] is False


def test_url_policy_rejects_invalid_mode_and_malformed_domain_lists():
    with pytest.raises(ValueError, match="mode"):
        UrlPolicyConfig(mode="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="scheme"):
        UrlPolicyConfig(mode="audit", blocked_domains=("https://bad.example/path",))
