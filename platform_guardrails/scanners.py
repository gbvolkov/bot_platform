from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence
from uuid import uuid4

from .context import GuardrailContext
from .decisions import GuardrailDecision, allow, block, redact
from .injection import scanner_category, scanner_decision
from .url_policy import UrlPolicyConfig, coerce_url_policy_config, normalized_url_keys, scan_url_policy


ScannerFailurePolicy = Literal["fail_closed", "fail_open"]

FORBIDDEN_LLM_GUARD_SCANNERS = {"Anonymize", "Deanonymize"}
PROMPT_INJECTION_SENTENCE_PLACEHOLDER = "[guarded sentence removed]"
_TOOL_RESULT_PROMPT_INJECTION_NOISE_PATTERNS = (
    re.compile(r"\*\*\s*Ссылка на статью:\s*[^*\r\n]+?\s*\*\*"),
    re.compile(r"=+\s*END OF DOCUMENT\s*=+"),
    re.compile(re.escape(PROMPT_INJECTION_SENTENCE_PLACEHOLDER)),
)
GENERIC_BANNED_TOPICS = [
    "self-harm instructions",
    "weapons construction",
    "illegal activity",
    "explicit sexual content",
]
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
    url_policy: UrlPolicyConfig | Mapping[str, Any] | None = None
    tool_result_prompt_injection_threshold: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_scanners", tuple(_coerce_spec(spec) for spec in self.input_scanners))
        object.__setattr__(self, "output_scanners", tuple(_coerce_spec(spec) for spec in self.output_scanners))
        object.__setattr__(
            self,
            "composite_input_scanners",
            tuple(_coerce_spec(spec) for spec in self.composite_input_scanners),
        )
        object.__setattr__(self, "url_policy", coerce_url_policy_config(self.url_policy))
        if self.tool_result_prompt_injection_threshold is not None:
            object.__setattr__(
                self,
                "tool_result_prompt_injection_threshold",
                float(self.tool_result_prompt_injection_threshold),
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
        prompt_injection_threshold: float | None = None,
        tool_result_prompt_injection_threshold: float | None = None,
        prompt_injection_model: str | Mapping[str, Any] | None = None,
        prompt_injection_model_revision: str | None = None,
        url_policy: UrlPolicyConfig | Mapping[str, Any] | None = None,
    ) -> "LLMGuardScannerProfile":
        topics = list(GENERIC_BANNED_TOPICS if banned_topics is None else banned_topics)
        prompt_injection_config: dict[str, Any] = {
            "match_type": "sentence",
        }
        if prompt_injection_threshold is not None:
            prompt_injection_config["threshold"] = prompt_injection_threshold
        if prompt_injection_model is not None:
            prompt_injection_config["model"] = prompt_injection_model_config(
                prompt_injection_model,
                revision=prompt_injection_model_revision,
            )
        elif prompt_injection_model_revision is not None:
            raise ValueError("prompt_injection_model_revision requires prompt_injection_model.")
        prompt_injection = ScannerSpec("PromptInjection", prompt_injection_config)
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
            url_policy=url_policy,
            tool_result_prompt_injection_threshold=tool_result_prompt_injection_threshold,
        )


class LLMGuardScannerRail:
    def __init__(
        self,
        profile: LLMGuardScannerProfile | None = None,
        *,
        input_factory: Callable[[ScannerSpec], Any] | None = None,
        output_factory: Callable[[ScannerSpec], Any] | None = None,
        verbose_logging: bool = False,
    ) -> None:
        self._profile = profile or LLMGuardScannerProfile.artifact_creator_default()
        self._input_factory = input_factory or _build_input_scanner
        self._output_factory = output_factory or _build_output_scanner
        self._input_instances: dict[Any, Any] = {}
        self._output_instances: dict[Any, Any] = {}
        self._source_urls_by_scope: dict[tuple[str, str, str, str], set[str]] = {}
        self._verbose_logging = verbose_logging

    @property
    def profile(self) -> LLMGuardScannerProfile:
        return self._profile

    @property
    def verbose_logging(self) -> bool:
        return self._verbose_logging

    def scan_input_text(
        self,
        text: str,
        context: GuardrailContext,
        *,
        boundary: str = "model_request",
        excluded_scanner_names: Iterable[str] = (),
        prompt_injection_text_filter: Callable[[str], str] | None = None,
        prompt_injection_threshold: float | None = None,
    ) -> ScannerScanResult:
        excluded = {str(name) for name in excluded_scanner_names}
        specs = tuple(
            spec for spec in self._profile.input_scanners if spec.name not in excluded
        )
        prompt_injection_threshold = self._resolve_prompt_injection_threshold(
            boundary=boundary,
            prompt_injection_threshold=prompt_injection_threshold,
        )
        result = _scan_text(
            text,
            context,
            specs=specs,
            stage="input",
            boundary=boundary,
            prompt="",
            source_urls=set(),
            failure_policy=self._profile.failure_policy,
            fail_fast=self._profile.fail_fast,
            url_policy=None,
            instance_for_spec=lambda spec: self._input_instance_for_scan(
                spec,
                prompt_injection_threshold=prompt_injection_threshold,
            ),
            scan_one=lambda scanner, value: scanner.scan(value),
            verbose_log_injections=self._verbose_logging,
            scanner_input_filter=(
                _prompt_injection_scanner_input_filter(prompt_injection_text_filter)
                if prompt_injection_text_filter is not None
                else None
            ),
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
            url_policy=self._profile.url_policy,
            instance_for_spec=lambda spec: self._instance_for_spec(spec, "output"),
            scan_one=lambda scanner, value: scanner.scan(prompt, value),
            verbose_log_injections=self._verbose_logging,
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
            url_policy=None,
            instance_for_spec=lambda spec: self._instance_for_spec(spec, "input"),
            scan_one=lambda scanner, value: scanner.scan(value),
            verbose_log_injections=self._verbose_logging,
        )

    def redact_prompt_injection_sentences(
        self,
        text: str,
        context: GuardrailContext,
        *,
        boundary: str = "tool_result",
        placeholder: str = PROMPT_INJECTION_SENTENCE_PLACEHOLDER,
        prompt_injection_text_filter: Callable[[str], str] | None = None,
        prompt_injection_threshold: float | None = None,
    ) -> ScannerScanResult:
        current = text
        decisions: list[GuardrailDecision] = []
        redacted_count = 0
        highest_score = 0.0
        confirmed_injection_texts: list[str] = []
        prompt_injection_threshold = self._resolve_prompt_injection_threshold(
            boundary=boundary,
            prompt_injection_threshold=prompt_injection_threshold,
        )

        prompt_injection_specs = [
            spec for spec in self._profile.input_scanners if spec.name == "PromptInjection"
        ]
        for spec in prompt_injection_specs:
            try:
                scanner = self._input_instance_for_scan(
                    spec,
                    prompt_injection_threshold=prompt_injection_threshold,
                )
                sentences = _prompt_injection_sentence_inputs(scanner, current)
            except Exception as exc:  # noqa: BLE001 - scanner errors are policy-controlled
                decisions.append(
                    _scanner_error_decision(
                        spec,
                        exc,
                        stage="input",
                        boundary=boundary,
                        failure_policy=self._profile.failure_policy,
                        context=context,
                    )
                )
                if decisions[-1]["allowed"] is False or self._profile.fail_fast:
                    break
                continue

            for sentence in sentences:
                if not isinstance(sentence, str) or not sentence.strip():
                    continue
                try:
                    scanner_text = (
                        prompt_injection_text_filter(sentence)
                        if prompt_injection_text_filter is not None
                        else sentence
                    )
                    _sanitized, is_valid, risk_score = scanner.scan(scanner_text)
                except Exception as exc:  # noqa: BLE001 - scanner errors are policy-controlled
                    decisions.append(
                        _scanner_error_decision(
                            spec,
                            exc,
                            stage="input",
                            boundary=boundary,
                            failure_policy=self._profile.failure_policy,
                            context=context,
                        )
                    )
                    if decisions[-1]["allowed"] is False or self._profile.fail_fast:
                        break
                    continue
                if is_valid:
                    continue
                updated, replaced = _replace_first_occurrence(current, sentence, placeholder)
                if not replaced:
                    continue
                current = updated
                redacted_count += 1
                confirmed_injection_texts.append(sentence)
                highest_score = max(highest_score, risk_score if risk_score >= 0 else 0.0)

        if redacted_count:
            decisions.append(
                redact(
                    "PromptInjection scanner redacted tool result sentence(s).",
                    risk_score=max(0.8, min(1.0, highest_score)),
                    categories=[scanner_category("PromptInjection")],
                    metadata={
                        "rail": "llm_guard",
                        "scanner": "PromptInjection",
                        "stage": "input",
                        "boundary": boundary,
                        "sanitized_changed": current != text,
                        "redacted_sentences": redacted_count,
                        **_confirmed_injection_metadata(
                            confirmed_injection_texts,
                            verbose=self._verbose_logging,
                        ),
                        **_context_metadata(context),
                    },
                )
            )
        return ScannerScanResult(text=current, decisions=decisions)

    def _instance_for_spec(
        self,
        spec: ScannerSpec,
        stage: Literal["input", "output"],
        *,
        cache_key: Any | None = None,
    ) -> Any:
        if spec.scanner is not None:
            return spec.scanner
        cache = self._input_instances if stage == "input" else self._output_instances
        key = id(spec) if cache_key is None else cache_key
        if key not in cache:
            factory = self._input_factory if stage == "input" else self._output_factory
            cache[key] = factory(spec)
        return cache[key]

    def _input_instance_for_scan(
        self,
        spec: ScannerSpec,
        *,
        prompt_injection_threshold: float | None = None,
    ) -> Any:
        if spec.name != "PromptInjection" or prompt_injection_threshold is None:
            return self._instance_for_spec(spec, "input")
        override = _scanner_spec_with_threshold(spec, prompt_injection_threshold)
        return self._instance_for_spec(
            override,
            "input",
            cache_key=(id(spec), "prompt_injection_threshold", prompt_injection_threshold),
        )

    def _resolve_prompt_injection_threshold(
        self,
        *,
        boundary: str,
        prompt_injection_threshold: float | None,
    ) -> float | None:
        if prompt_injection_threshold is not None:
            return float(prompt_injection_threshold)
        if boundary == "tool_result":
            return self._profile.tool_result_prompt_injection_threshold
        return None

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
    prompt_injection_text_filter: Callable[[str], str] | None = None,
    prompt_injection_threshold: float | None = None,
) -> ScannerScanResult:
    return rail.scan_input_text(
        text,
        context,
        boundary=boundary,
        prompt_injection_text_filter=prompt_injection_text_filter,
        prompt_injection_threshold=prompt_injection_threshold,
    )


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


def prompt_injection_model_config(
    model: str | Mapping[str, Any],
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    if isinstance(model, str):
        config: dict[str, Any] = {"path": model}
    elif isinstance(model, Mapping):
        config = dict(model)
    else:
        raise TypeError(f"Unsupported prompt injection model config: {model!r}")

    if revision is not None:
        config["revision"] = revision
    if str(config.get("revision") or "").strip().lower() == "latest":
        config.pop("revision", None)
    return _normalize_llm_guard_model_config(config)


def _normalize_llm_guard_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(config)

    kwargs = dict(updated.get("kwargs") or {})
    for label_key in ("id2label", "label2id"):
        if label_key in updated:
            kwargs[label_key] = updated.pop(label_key)

    id2label = kwargs.get("id2label")
    if isinstance(id2label, Mapping):
        kwargs["id2label"] = {
            _coerce_int_key(key): value
            for key, value in id2label.items()
        }

    for section in ("kwargs", "pipeline_kwargs", "tokenizer_kwargs"):
        section_value = kwargs if section == "kwargs" else updated.get(section)
        if section_value is None:
            updated.pop(section, None)
        elif isinstance(section_value, Mapping):
            updated[section] = dict(section_value)
        else:
            raise TypeError(f"Prompt injection model {section} must be a mapping.")

    return updated


def _coerce_int_key(key: Any) -> Any:
    if isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            return key
    return key


def _prompt_injection_sentence_inputs(scanner: Any, text: str) -> list[str]:
    match_type = getattr(scanner, "_match_type", None)
    get_inputs = getattr(match_type, "get_inputs", None)
    if not callable(get_inputs):
        return []
    inputs = get_inputs(text)
    return list(inputs or [])


def filter_tool_result_prompt_injection_noise(text: str) -> str:
    """Replace known tool-result boilerplate before PromptInjection scoring."""

    filtered = text
    for pattern in _TOOL_RESULT_PROMPT_INJECTION_NOISE_PATTERNS:
        filtered = pattern.sub(lambda _match: str(uuid4()), filtered)
    return filtered


def _scanner_spec_with_threshold(spec: ScannerSpec, threshold: float) -> ScannerSpec:
    config = dict(spec.config)
    config["threshold"] = float(threshold)
    return ScannerSpec(spec.name, config, scanner=spec.scanner)


def _prompt_injection_scanner_input_filter(
    text_filter: Callable[[str], str],
) -> Callable[[ScannerSpec, str], str]:
    def _filter(spec: ScannerSpec, text: str) -> str:
        if spec.name != "PromptInjection":
            return text
        return text_filter(text)

    return _filter


def _replace_first_occurrence(text: str, target: str, replacement: str) -> tuple[str, bool]:
    index = text.find(target)
    if index < 0:
        return text, False
    return text[:index] + replacement + text[index + len(target) :], True


def _coerce_prompt_injection_config(config: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    model_config = updated.get("model")
    if model_config is None:
        return updated

    from llm_guard.model import Model

    if isinstance(model_config, Model):
        return updated
    updated["model"] = Model(**prompt_injection_model_config(model_config))
    return updated


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
    url_policy: UrlPolicyConfig | None,
    instance_for_spec: Callable[[ScannerSpec], Any],
    scan_one: Callable[[Any, str], tuple[str, bool, float]],
    verbose_log_injections: bool = False,
    scanner_input_filter: Callable[[ScannerSpec, str], str] | None = None,
) -> ScannerScanResult:
    current = text
    decisions: list[GuardrailDecision] = []
    if not current:
        return ScannerScanResult(text=current, decisions=decisions)

    if stage == "output" and url_policy is not None:
        url_policy_decisions = scan_url_policy(
            current,
            context,
            url_policy,
            stage=stage,
            boundary=boundary,
            prompt=prompt,
            source_urls=source_urls,
        )
        decisions.extend(url_policy_decisions)
        if any(not decision["allowed"] for decision in url_policy_decisions):
            return ScannerScanResult(text=current, decisions=decisions)

    if not specs:
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
            scanner_input = (
                scanner_input_filter(spec, current)
                if scanner_input_filter is not None
                else current
            )
            sanitized, is_valid, risk_score = scan_one(scanner, scanner_input)
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

        scanner_input_filtered = scanner_input != current
        sanitized_changed = sanitized != scanner_input
        metadata = _context_metadata(context)
        if scanner_input_filtered:
            metadata["scanner_input_filtered"] = True
        if verbose_log_injections and spec.name == "PromptInjection" and not is_valid:
            metadata.update(_confirmed_injection_metadata(current, verbose=True))
        decision = scanner_decision(
            scanner_name=spec.name,
            is_valid=is_valid,
            risk_score=risk_score,
            sanitized_changed=sanitized_changed,
            stage=stage,
            boundary=boundary,
            metadata=metadata,
        )
        decisions.append(decision)

        if decision["allowed"] and not scanner_input_filtered:
            current = sanitized
        if not decision["allowed"] or (fail_fast and not is_valid):
            break

    return ScannerScanResult(text=current, decisions=decisions)


def _normalized_urls(text: str) -> set[str]:
    return normalized_url_keys(text)


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


def _confirmed_injection_metadata(
    text: str | Sequence[str],
    *,
    verbose: bool,
) -> dict[str, Any]:
    if not verbose:
        return {}
    if isinstance(text, str):
        return {"confirmed_injection_text": text}
    values = [value for value in text if isinstance(value, str) and value]
    if not values:
        return {}
    metadata: dict[str, Any] = {"confirmed_injection_texts": values}
    if len(values) == 1:
        metadata["confirmed_injection_text"] = values[0]
    return metadata


def _build_input_scanner(spec: ScannerSpec) -> Any:
    if spec.name == "PromptInjection":
        from llm_guard.input_scanners import PromptInjection

        return PromptInjection(**_coerce_prompt_injection_config(spec.config))
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
    "PROMPT_INJECTION_SENTENCE_PLACEHOLDER",
    "ScannerFailurePolicy",
    "ScannerScanResult",
    "ScannerSpec",
    "filter_tool_result_prompt_injection_noise",
    "prompt_injection_model_config",
    "scan_composite_input_text",
    "scan_input_text",
    "scan_output_text",
]
