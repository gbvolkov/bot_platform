from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.structured_output import ProviderStrategy, StructuredOutputValidationError
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from typing_extensions import NotRequired, TypedDict

from agents.llm_utils import get_llm
from agents.utils import ModelType, build_internal_invoke_config

from ..schemas import (
    EnrichmentAssessmentResult,
    ProcurementRelevanceBatch,
    TaskUnderstandingResult,
)


class _ClassifierState(TypedDict):
    messages: list[Any]
    summary: NotRequired[str]
    hits: NotRequired[list[dict[str, Any]]]
    signals: NotRequired[dict[str, Any]]


class ClassifierExecutionError(RuntimeError):
    """Recoverable runtime failure while invoking the internal classifier."""


class ClassifierContractError(RuntimeError):
    """Non-recoverable structured-output contract violation in the internal classifier."""


def _dump(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return str(payload)


def _is_recoverable_classifier_exception(exc: Exception) -> bool:
    module_name = exc.__class__.__module__
    class_name = exc.__class__.__name__
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if module_name.startswith(
        (
            "openai",
            "anthropic",
            "httpx",
            "langchain_openai",
            "langchain_anthropic",
        )
    ):
        return True
    return class_name in {"APIError", "APIConnectionError", "APITimeoutError", "RateLimitError"}


class InternalClassifier:
    def __init__(self, *, provider: ModelType) -> None:
        model = get_llm(
            model="mini",
            provider=provider.value,
            temperature=0.0,
            streaming=False,
            reasoning="minimal",
        )
        self._intent_agent = self._build_intent_agent(model)
        self._assessment_agent = self._build_assessment_agent(model)
        self._relevance_agent = self._build_relevance_agent(model)

    def _build_intent_agent(self, model):
        @dynamic_prompt
        def build_prompt(request) -> str:
            state: _ClassifierState = request.state
            summary = state.get("summary") or ""
            return (
                "Classify the current user request for sales_lead_agent.\n"
                "Return only the structured schema.\n"
                "Field mapping is strict:\n"
                "- task_kind=procurement_search -> answer_type=lead_list\n"
                "- task_kind=procurement_analysis -> answer_type=lead_card\n"
                "- task_kind=fact_lookup -> answer_type=lead_card\n"
                "- task_kind=company_check -> answer_type=company_check\n"
                "- task_kind=comparison -> answer_type=comparison\n"
                "Never output procurement_search, procurement_analysis, or fact_lookup in answer_type.\n"
                "Use short contextual summaries, not a verbatim copy of the user text, "
                "for procurement search fields.\n"
                "If the task is procurement_search or procurement_analysis and the user did not provide "
                "a direct procurement URL or registry number, populate search_filters.query_text as short "
                "Russian zakupki.gov.ru search keywords suitable for a deterministic searchString builder "
                "and set needs_purchase_search=true. "
                "For procurement_analysis without a direct procurement identifier, do not stop at generic "
                "missing data: prepare a best-effort procurement search query from the business context so "
                "the runtime can acquire one relevant procurement first. "
                "Use search_filters.query_text for acquisition intent, and use document_questions for the "
                "specific facts that must be verified in the documents after acquisition. "
                "Use one compact Russian phrase with 2-5 meaningful words. "
                "Do not use OR, lists of alternatives, punctuation-heavy prose, or English sentences. "
                "Prefer concise domain phrases like 'услуг страхования грузов' or "
                "'страхование транспортных средств'.\n\n"
                f"Summary:\n{summary}"
            )

        return create_agent(
            model=model,
            middleware=[build_prompt],
            response_format=ProviderStrategy(schema=TaskUnderstandingResult),
            state_schema=_ClassifierState,
        )

    def _build_assessment_agent(self, model):
        @dynamic_prompt
        def build_prompt(request) -> str:
            state: _ClassifierState = request.state
            signals = state.get("signals") or {}
            return (
                "Assess enrichment and risk signals for sales_lead_agent.\n"
                "Return only the structured schema.\n"
                "Interpret semantics using only the provided signals.\n\n"
                f"Signals:\n{_dump(signals)}"
            )

        return create_agent(
            model=model,
            middleware=[build_prompt],
            response_format=ProviderStrategy(schema=EnrichmentAssessmentResult),
            state_schema=_ClassifierState,
        )

    def _build_relevance_agent(self, model):
        @dynamic_prompt
        def build_prompt(request) -> str:
            state: _ClassifierState = request.state
            summary = state.get("summary") or ""
            hits = state.get("hits") or []
            return (
                "Decide which procurement hits are semantically relevant for the search intent.\n"
                "Return only the structured schema.\n"
                "Lexical overlap is insufficient for relevance.\n\n"
                f"Intent summary:\n{summary}\n\n"
                f"Hits:\n{_dump(hits)}"
            )

        return create_agent(
            model=model,
            middleware=[build_prompt],
            response_format=ProviderStrategy(schema=ProcurementRelevanceBatch),
            state_schema=_ClassifierState,
        )

    def classify_intent(
        self,
        *,
        summary: str,
        invoke_config: dict[str, Any] | None = None,
    ) -> TaskUnderstandingResult:
        try:
            result = self._intent_agent.invoke(
                {"messages": [HumanMessage(content=summary)], "summary": summary},
                config=build_internal_invoke_config(invoke_config),
            )
        except StructuredOutputValidationError as exc:
            raise ClassifierContractError(
                "Intent classifier returned a schema-invalid structured response."
            ) from exc
        except Exception as exc:
            if not _is_recoverable_classifier_exception(exc):
                raise
            raise ClassifierExecutionError("Intent classifier invocation failed.") from exc
        structured = result.get("structured_response")
        if isinstance(structured, TaskUnderstandingResult):
            return structured
        if isinstance(structured, dict):
            try:
                return TaskUnderstandingResult.model_validate(structured)
            except ValidationError as exc:
                raise ClassifierContractError(
                    "Intent classifier returned a schema-invalid structured response."
                ) from exc
        raise ClassifierContractError("Internal classifier did not return TaskUnderstandingResult.")

    def assess_signals(
        self,
        *,
        signals: dict[str, Any],
        invoke_config: dict[str, Any] | None = None,
    ) -> EnrichmentAssessmentResult:
        try:
            result = self._assessment_agent.invoke(
                {"messages": [HumanMessage(content=_dump(signals))], "signals": signals},
                config=build_internal_invoke_config(invoke_config),
            )
        except StructuredOutputValidationError as exc:
            raise ClassifierContractError(
                "Assessment classifier returned a schema-invalid structured response."
            ) from exc
        except Exception as exc:
            if not _is_recoverable_classifier_exception(exc):
                raise
            raise ClassifierExecutionError("Assessment classifier invocation failed.") from exc
        structured = result.get("structured_response")
        if isinstance(structured, EnrichmentAssessmentResult):
            return structured
        if isinstance(structured, dict):
            try:
                return EnrichmentAssessmentResult.model_validate(structured)
            except ValidationError as exc:
                raise ClassifierContractError(
                    "Assessment classifier returned a schema-invalid structured response."
                ) from exc
        raise ClassifierContractError("Internal classifier did not return EnrichmentAssessmentResult.")

    def classify_procurement_hits(
        self,
        *,
        summary: str,
        hits: list[dict[str, Any]],
        invoke_config: dict[str, Any] | None = None,
    ) -> ProcurementRelevanceBatch:
        try:
            result = self._relevance_agent.invoke(
                {
                    "messages": [HumanMessage(content=summary)],
                    "summary": summary,
                    "hits": hits,
                },
                config=build_internal_invoke_config(invoke_config),
            )
        except StructuredOutputValidationError as exc:
            raise ClassifierContractError(
                "Procurement relevance classifier returned a schema-invalid structured response."
            ) from exc
        except Exception as exc:
            if not _is_recoverable_classifier_exception(exc):
                raise
            raise ClassifierExecutionError("Procurement relevance classifier invocation failed.") from exc
        structured = result.get("structured_response")
        if isinstance(structured, ProcurementRelevanceBatch):
            return structured
        if isinstance(structured, dict):
            try:
                return ProcurementRelevanceBatch.model_validate(structured)
            except ValidationError as exc:
                raise ClassifierContractError(
                    "Procurement relevance classifier returned a schema-invalid structured response."
                ) from exc
        raise ClassifierContractError("Internal classifier did not return ProcurementRelevanceBatch.")
