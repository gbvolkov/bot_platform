from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from .context import GuardrailContext
from .decisions import GuardrailDecision, allow, block
from .injection import scanner_category, scanner_decision


ScannerFailurePolicy = Literal["fail_closed", "fail_open"]

FORBIDDEN_LLM_GUARD_SCANNERS = {"Anonymize", "Deanonymize"}
GENERIC_BANNED_TOPICS = [
    "self-harm instructions",
    "weapons construction",
    "illegal activity",
    "explicit sexual content",
]
URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://)?(?:(?:[a-z0-9-]+\.)+[a-z]{2,}|(?:\d{1,3}\.){3}\d{1,3})"
    r"(?::\d{1,5})?(?:/[^\s<>()\"']*)?"
)


@dataclass(frozen=True)
class ScannerSpec:
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    scanner: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.name in FORBIDDEN_LLM_GUARD_SCANNERS:
            raise ValueError(f"LLM Guard scanner {self.name!r} is not allowed in this platform path.")


@dataclass(frozen=True)
class ScannerScanResult:
    text: str
    decisions: list[GuardrailDecision]

    @property
    def blocked_decision(self) -> GuardrailDecision | None:
        for decision in self.decisions:
            if not decision["allowed"]:
                return decision
        return None


@dataclass(frozen=True)
class LLMGuardScannerProfile:
    input_scanners: Sequence[ScannerSpec] = field(default_factory=tuple)
    output_scanners: Sequence[ScannerSpec] = field(default_factory=tuple)
    composite_input_scanners: Sequence[ScannerSpec] = field(default_factory=tuple)
    failure_policy: ScannerFailurePolicy = "fail_closed"
    fail_fast: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_scanners", tuple(_coerce_spec(spec) for spec in self.input_scanners))
        object.__setattr__(self, "output_scanners", tuple(_coerce_spec(spec) for spec in self.output_scanners))
        object.__setattr__(
            self,
            "composite_input_scanners",
            tuple(_coerce_spec(spec) for spec in self.composite_input_scanners),
        )
        if self.failure_policy not in {"fail_closed", "fail_open"}:
            raise ValueError("failure_policy must be 'fail_closed' or 'fail_open'.")

    @classmethod
    def artifact_creator_default(
        cls,
        *,
        banned_topics: Sequence[str] | None = None,
        failure_policy: ScannerFailurePolicy = "fail_closed",
        token_limit: int = 8192,
    ) -> "LLMGuardScannerProfile":
        topics = list(GENERIC_BANNED_TOPICS if banned_topics is None else banned_topics)
        prompt_injection = ScannerSpec("PromptInjection", {"threshold": 0.92, "match_type": "sentence"})
        input_scanners = [
            ScannerSpec("TokenLimit", {"limit": token_limit}),
            ScannerSpec("Secrets", {"redact_mode": "all"}),
            prompt_injection,
            ScannerSpec("Toxicity", {"threshold": 0.8, "match_type": "sentence"}),
        ]
        output_scanners = [
            ScannerSpec("MaliciousURLs", {"threshold": 0.7}),
            ScannerSpec("Toxicity", {"threshold": 0.8}),
        ]
        if topics:
            topic_config = {"topics": topics, "threshold": 0.75}
            input_scanners.append(ScannerSpec("BanTopics", topic_config))
            output_scanners.append(ScannerSpec("BanTopics", topic_config))
        return cls(
            input_scanners=tuple(input_scanners),
            output_scanners=tuple(output_scanners),
            composite_input_scanners=(prompt_injection,),
            failure_policy=failure_policy,
        )


class LLMGuardScannerRail:
    def __init__(
        self,
        profile: LLMGuardScannerProfile | None = None,
        *,
        input_factory: Callable[[ScannerSpec], Any] | None = None,
        output_factory: Callable[[ScannerSpec], Any] | None = None,
    ) -> None:
        self._profile = profile or LLMGuardScannerProfile.artifact_creator_default()
        self._input_factory = input_factory or _build_input_scanner
        self._output_factory = output_factory or _build_output_scanner
        self._input_instances: dict[int, Any] = {}
        self._output_instances: dict[int, Any] = {}
        self._source_urls_by_scope: dict[tuple[str, str, str, str], set[str]] = {}

    @property
    def profile(self) -> LLMGuardScannerProfile:
        return self._profile

    def scan_input_text(
        self,
        text: str,
        context: GuardrailContext,
        *,
        boundary: str = "model_request",
    ) -> ScannerScanResult:
        result = _scan_text(
            text,
            context,
            specs=self._profile.input_scanners,
            stage="input",
            boundary=boundary,
            prompt="",
            source_urls=set(),
            failure_policy=self._profile.failure_policy,
            fail_fast=self._profile.fail_fast,
            instance_for_spec=lambda spec: self._instance_for_spec(spec, "input"),
            scan_one=lambda scanner, value: scanner.scan(value),
        )
        if result.blocked_decision is None:
            self._remember_source_urls(context, text)
        return result

    def scan_output_text(
        self,
        prompt: str,
        output: str,
        context: GuardrailContext,
        *,
        boundary: str = "model_response",
    ) -> ScannerScanResult:
        return _scan_text(
            output,
            context,
            specs=self._profile.output_scanners,
            stage="output",
            boundary=boundary,
            prompt=prompt,
            source_urls=self._source_urls_for_context(context),
            failure_policy=self._profile.failure_policy,
            fail_fast=self._profile.fail_fast,
            instance_for_spec=lambda spec: self._instance_for_spec(spec, "output"),
            scan_one=lambda scanner, value: scanner.scan(prompt, value),
        )

    def scan_composite_input_text(
        self,
        text: str,
        context: GuardrailContext,
        *,
        scanner_names: Iterable[str] | None = None,
        boundary: str = "composite_model_request",
    ) -> ScannerScanResult:
        specs = self._composite_specs(scanner_names)
        return _scan_text(
            text,
            context,
            specs=specs,
            stage="input",
            boundary=boundary,
            prompt="",
            source_urls=set(),
            failure_policy=self._profile.failure_policy,
            fail_fast=self._profile.fail_fast,
            instance_for_spec=lambda spec: self._instance_for_spec(spec, "input"),
            scan_one=lambda scanner, value: scanner.scan(value),
        )

    def _instance_for_spec(self, spec: ScannerSpec, stage: Literal["input", "output"]) -> Any:
        if spec.scanner is not None:
            return spec.scanner
        cache = self._input_instances if stage == "input" else self._output_instances
        key = id(spec)
        if key not in cache:
            factory = self._input_factory if stage == "input" else self._output_factory
            cache[key] = factory(spec)
        return cache[key]

    def reset_context(self, context: GuardrailContext) -> None:
        self._source_urls_by_scope.pop(_source_url_scope(context), None)

    def _remember_source_urls(self, context: GuardrailContext, text: str) -> None:
        urls = _normalized_urls(text)
        if not urls:
            return
        scope = _source_url_scope(context)
        self._source_urls_by_scope.setdefault(scope, set()).update(urls)

    def _source_urls_for_context(self, context: GuardrailContext) -> set[str]:
        return set(self._source_urls_by_scope.get(_source_url_scope(context), set()))

    def _composite_specs(self, scanner_names: Iterable[str] | None) -> tuple[ScannerSpec, ...]:
        if scanner_names is None:
            return tuple(self._profile.composite_input_scanners)
        selected: list[ScannerSpec] = []
        for name in scanner_names:
            name_text = str(name)
            matching = next(
                (
                    spec
                    for spec in (*self._profile.composite_input_scanners, *self._profile.input_scanners)
                    if spec.name == name_text
                ),
                None,
            )
            selected.append(matching or ScannerSpec(name_text))
        return tuple(selected)


def scan_input_text(
    rail: LLMGuardScannerRail,
    text: str,
    context: GuardrailContext,
    *,
    boundary: str = "model_request",
) -> ScannerScanResult:
    return rail.scan_input_text(text, context, boundary=boundary)


def scan_output_text(
    rail: LLMGuardScannerRail,
    prompt: str,
    output: str,
    context: GuardrailContext,
    *,
    boundary: str = "model_response",
) -> ScannerScanResult:
    return rail.scan_output_text(prompt, output, context, boundary=boundary)


def scan_composite_input_text(
    rail: LLMGuardScannerRail,
    text: str,
    context: GuardrailContext,
    *,
    scanner_names: Iterable[str] | None = None,
    boundary: str = "composite_model_request",
) -> ScannerScanResult:
    return rail.scan_composite_input_text(
        text,
        context,
        scanner_names=scanner_names,
        boundary=boundary,
    )


def _coerce_spec(value: ScannerSpec | Mapping[str, Any]) -> ScannerSpec:
    if isinstance(value, ScannerSpec):
        return value
    if isinstance(value, Mapping):
        return ScannerSpec(
            name=str(value.get("name") or ""),
            config=value.get("config") or {},
            scanner=value.get("scanner"),
        )
    if isinstance(value, str):
        return ScannerSpec(value)
    raise TypeError(f"Unsupported scanner spec: {value!r}")


def _scan_text(
    text: str,
    context: GuardrailContext,
    *,
    specs: Sequence[ScannerSpec],
    stage: Literal["input", "output"],
    boundary: str,
    prompt: str,
    source_urls: set[str],
    failure_policy: ScannerFailurePolicy,
    fail_fast: bool,
    instance_for_spec: Callable[[ScannerSpec], Any],
    scan_one: Callable[[Any, str], tuple[str, bool, float]],
) -> ScannerScanResult:
    current = text
    decisions: list[GuardrailDecision] = []
    if not current or not specs:
        return ScannerScanResult(text=current, decisions=decisions)

    for spec in specs:
        if (
            stage == "output"
            and spec.name == "MaliciousURLs"
            and _output_urls_are_sourced(prompt, current, source_urls)
        ):
            decisions.append(
                allow(
                    "MaliciousURLs scanner allowed output URLs already present in agent input.",
                    categories=[scanner_category(spec.name)],
                    metadata={
                        "rail": "llm_guard",
                        "scanner": spec.name,
                        "stage": stage,
                        "boundary": boundary,
                        "sanitized_changed": False,
                        "prompt_url_match": True,
                        **_context_metadata(context),
                    },
                )
            )
            continue

        try:
            scanner = instance_for_spec(spec)
            sanitized, is_valid, risk_score = scan_one(scanner, current)
        except Exception as exc:  # noqa: BLE001 - scanner errors are policy-controlled
            decision = _scanner_error_decision(
                spec,
                exc,
                stage=stage,
                boundary=boundary,
                failure_policy=failure_policy,
                context=context,
            )
            decisions.append(decision)
            if not decision["allowed"] or fail_fast:
                break
            continue

        sanitized_changed = sanitized != current
        decision = scanner_decision(
            scanner_name=spec.name,
            is_valid=is_valid,
            risk_score=risk_score,
            sanitized_changed=sanitized_changed,
            stage=stage,
            boundary=boundary,
            metadata=_context_metadata(context),
        )
        decisions.append(decision)

        if decision["allowed"]:
            current = sanitized
        if not decision["allowed"] or (fail_fast and not is_valid):
            break

    return ScannerScanResult(text=current, decisions=decisions)


def _normalized_urls(text: str) -> set[str]:
    urls: set[str] = set()
    for match in URL_PATTERN.finditer(text or ""):
        raw_url = match.group(0).strip().rstrip(".,;!?)]}")
        if not raw_url:
            continue
        parseable_url = raw_url if "://" in raw_url else f"http://{raw_url}"
        parsed = urlsplit(parseable_url)
        host = parsed.hostname.lower() if parsed.hostname else ""
        if not host:
            continue
        try:
            parsed_port = parsed.port
        except ValueError:
            parsed_port = None
        port = f":{parsed_port}" if parsed_port is not None else ""
        path = parsed.path.rstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        urls.add(f"{host}{port}{path}{query}")
    return urls


def _output_urls_are_prompt_sourced(prompt: str, output: str) -> bool:
    output_urls = _normalized_urls(output)
    if not output_urls:
        return False
    prompt_urls = _normalized_urls(prompt)
    return bool(prompt_urls) and output_urls.issubset(prompt_urls)


def _output_urls_are_sourced(prompt: str, output: str, source_urls: set[str]) -> bool:
    output_urls = _normalized_urls(output)
    if not output_urls:
        return False
    available_source_urls = set(source_urls)
    available_source_urls.update(_normalized_urls(prompt))
    return bool(available_source_urls) and output_urls.issubset(available_source_urls)


def _source_url_scope(context: GuardrailContext) -> tuple[str, str, str, str]:
    return (
        str(context.get("tenant_id") or ""),
        str(context.get("user_id") or ""),
        str(context.get("thread_id") or ""),
        str(context.get("agent_name") or ""),
    )


def _scanner_error_decision(
    spec: ScannerSpec,
    exc: Exception,
    *,
    stage: str,
    boundary: str,
    failure_policy: ScannerFailurePolicy,
    context: GuardrailContext,
) -> GuardrailDecision:
    metadata = {
        "rail": "llm_guard",
        "scanner": spec.name,
        "stage": stage,
        "boundary": boundary,
        "error_type": type(exc).__name__,
    }
    metadata.update(_context_metadata(context))
    if failure_policy == "fail_open":
        return allow(
            f"{spec.name} scanner failed open for {stage} text.",
            categories=["scanner_error"],
            metadata=metadata,
        )
    return block(
        f"{spec.name} scanner failed closed for {stage} text.",
        risk_score=1.0,
        categories=["scanner_error"],
        metadata=metadata,
    )


def _context_metadata(context: GuardrailContext) -> dict[str, Any]:
    return {
        "agent_name": context.get("agent_name"),
        "thread_id": context.get("thread_id"),
        "user_role": context.get("user_role"),
        "request_id": context.get("request_id"),
        "tool_name": context.get("tool_name"),
    }


def _build_input_scanner(spec: ScannerSpec) -> Any:
    if spec.name == "PromptInjection":
        from llm_guard.input_scanners import PromptInjection

        return PromptInjection(**dict(spec.config))
    if spec.name == "Secrets":
        from llm_guard.input_scanners import Secrets

        return Secrets(**dict(spec.config))
    if spec.name == "TokenLimit":
        from llm_guard.input_scanners import TokenLimit

        return TokenLimit(**dict(spec.config))
    if spec.name == "Toxicity":
        from llm_guard.input_scanners import Toxicity

        return Toxicity(**dict(spec.config))
    if spec.name == "BanTopics":
        from llm_guard.input_scanners import BanTopics

        return BanTopics(**dict(spec.config))
    raise ValueError(f"Unsupported LLM Guard input scanner: {spec.name!r}")


def _build_output_scanner(spec: ScannerSpec) -> Any:
    if spec.name == "Sensitive":
        from llm_guard.output_scanners import Sensitive

        return Sensitive(**dict(spec.config))
    if spec.name == "MaliciousURLs":
        from llm_guard.output_scanners import MaliciousURLs

        return MaliciousURLs(**dict(spec.config))
    if spec.name == "Toxicity":
        from llm_guard.output_scanners import Toxicity

        return Toxicity(**dict(spec.config))
    if spec.name == "BanTopics":
        from llm_guard.output_scanners import BanTopics

        return BanTopics(**dict(spec.config))
    raise ValueError(f"Unsupported LLM Guard output scanner: {spec.name!r}")


__all__ = [
    "FORBIDDEN_LLM_GUARD_SCANNERS",
    "GENERIC_BANNED_TOPICS",
    "LLMGuardScannerProfile",
    "LLMGuardScannerRail",
    "ScannerFailurePolicy",
    "ScannerScanResult",
    "ScannerSpec",
    "scan_composite_input_text",
    "scan_input_text",
    "scan_output_text",
]
