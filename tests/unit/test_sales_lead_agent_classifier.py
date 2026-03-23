from types import SimpleNamespace

import pytest
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import AIMessage

from agents.sales_lead_agent.services.classifier import (
    ClassifierContractError,
    InternalClassifier,
)
from agents.sales_lead_agent.schemas import TaskUnderstandingResult


def test_internal_classifier_reraises_structured_output_contract_errors():
    classifier = InternalClassifier.__new__(InternalClassifier)
    classifier._intent_agent = SimpleNamespace(
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(
            StructuredOutputValidationError(
                "TaskUnderstandingResult",
                ValueError("bad output"),
                AIMessage(content=""),
            )
        )
    )

    with pytest.raises(
        ClassifierContractError,
        match="Intent classifier returned a schema-invalid structured response",
    ):
        classifier.classify_intent(summary="find leads")


def test_internal_classifier_reraises_programmer_errors():
    classifier = InternalClassifier.__new__(InternalClassifier)
    classifier._intent_agent = SimpleNamespace(
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AttributeError("broken wiring"))
    )

    with pytest.raises(AttributeError, match="broken wiring"):
        classifier.classify_intent(summary="find leads")


def test_internal_classifier_reraises_invoke_time_validation_errors():
    classifier = InternalClassifier.__new__(InternalClassifier)
    classifier._intent_agent = SimpleNamespace(
        invoke=lambda *args, **kwargs: {
            "structured_response": {
                "answer_type": "procurement_analysis",
                "task_kind": "procurement_analysis",
                "search_url": None,
                "search_filters": {},
                "requested_company_inns": [],
                "comparison_targets": [],
                "document_questions": [],
                "needs_purchase_search": True,
                "needs_open_source": False,
                "needs_doc_search": True,
                "needs_enrichment": False,
                "missing_data": [],
            }
        }
    )

    with pytest.raises(
        ClassifierContractError,
        match="Intent classifier returned a schema-invalid structured response",
    ) as exc_info:
        classifier.classify_intent(summary="find leads")

    assert "answer_type" in str(exc_info.value.__cause__)
    assert "procurement_analysis" in str(exc_info.value.__cause__)
