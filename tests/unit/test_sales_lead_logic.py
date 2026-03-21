import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.sales_lead_agent.adapters import DocumentPipelineAdapter
from agents.sales_lead_agent.business_rules import RequestRules, ScoringRules
from agents.sales_lead_agent.logic import (
    build_export_attachment,
    build_leads_from_sources,
    deduplicate_leads,
    parse_request,
    score_leads,
)


def test_parse_request_extracts_free_form_filters():
    understanding = parse_request(
        "Собери выгрузку потенциальных лидов за неделю по строительству и грузоперевозкам в ЦФО, только с контактами и проверь ИНН 7701234567",
        {},
        [],
    )

    assert understanding["result_type"] == "export"
    assert understanding["filters"]["inn"] == "7701234567"
    assert understanding["filters"]["regions"] == ["ЦФО"]
    assert "строительству" in understanding["filters"]["topics"]
    assert "грузоперевозкам" in understanding["filters"]["topics"]
    assert understanding["filters"]["period_from"] is not None
    assert understanding["filters"]["only_with_contacts"] is True
    assert understanding["needs_export"] is True
    assert understanding["use_existing_store"] is True


def test_parse_request_detects_feedback_intent():
    understanding = parse_request(
        "Пометь лид 123e4567-e89b-12d3-a456-426614174000 как дубль",
        {},
        [],
    )

    assert understanding["result_type"] == "feedback"
    assert understanding["filters"]["lead_id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert understanding["filters"]["feedback_status"] == "duplicate"
    assert understanding["needs_feedback"] is True
    assert understanding["needs_source_collection"] is False


def test_parse_request_detects_index_lookup():
    understanding = parse_request(
        'Что сказано в документах по компании ООО "ТрансЛогистик" о страховании перевозок?',
        {},
        [],
    )

    assert understanding["result_type"] == "index_lookup"
    assert understanding["needs_index_search"] is True
    assert understanding["use_existing_store"] is True
    assert understanding["needs_source_collection"] is False


def test_parse_request_uses_externalized_request_rules(monkeypatch):
    monkeypatch.setattr(
        "agents.sales_lead_agent.logic.load_request_rules",
        lambda: RequestRules(
            stopwords=["собери"],
            region_aliases={},
            priority_aliases={"приоритет 1": "high"},
            feedback_aliases={},
            result_type_triggers={"shortlist": ["пакет продавцу"]},
            source_priority_keywords={"procurement": ["закуп"], "open_source": ["новост"]},
            period_aliases={},
            topic_patterns=[],
            stop_word_patterns=[],
        ),
    )

    understanding = parse_request("Собери пакет продавцу с приоритет 1", {}, [])

    assert understanding["result_type"] == "shortlist"
    assert understanding["filters"]["priority"] == "high"
    assert "собери" not in understanding["filters"]["keywords"]


def test_build_leads_from_sources_deduplicates_same_company():
    source_hits = [
        {
            "source_type": "procurement",
            "source_id": "123",
            "source_url": "https://zakupki.example/item/123",
            "title": 'Закупка ООО "ТрансЛогистик"',
            "summary": 'ООО "ТрансЛогистик" закупает страхование перевозок на 12 000 000 руб.',
            "company_name": 'ООО "ТрансЛогистик"',
            "inn": "7701234567",
            "region": "ЦФО",
            "documents": [
                {
                    "document_url": "file://spec1.pdf",
                    "file_name": "spec1.pdf",
                    "file_type": "pdf",
                    "parse_status": "parsed",
                    "index_status": "ready",
                    "source_reference": "file://spec1.pdf",
                    "extracted_excerpt": "Контакт: sales@trans.example, телефон +7 999 123 45 67",
                    "text": "Контакт: sales@trans.example, телефон +7 999 123 45 67",
                    "segments": [
                        {
                            "page_number": 1,
                            "position_start": 0,
                            "position_end": 57,
                            "text": "Контакт: sales@trans.example, телефон +7 999 123 45 67",
                            "metadata": {"page_number": 1},
                        }
                    ],
                }
            ],
        },
        {
            "source_type": "open_source",
            "source_id": "news-77",
            "source_url": "https://news.example/trans",
            "title": 'ООО "ТрансЛогистик" расширяет парк',
            "summary": "Компания ООО ТрансЛогистик открыла новый проект перевозок.",
            "company_name": 'ООО "ТрансЛогистик"',
            "inn": "7701234567",
            "region": "ЦФО",
            "documents": [],
        },
    ]

    leads = build_leads_from_sources(
        understanding={"result_type": "search", "filters": {"keywords": ["страхование"]}},
        source_hits=source_hits,
        documents=[],
    )

    assert len(leads) == 1
    lead = leads[0]
    assert lead["company_name"] == 'ООО "ТрансЛогистик"'
    assert lead["inn"] == "7701234567"
    assert lead["contacts"]
    assert len(lead["sources"]) == 2
    assert lead["documents"][0]["text"]


def test_score_leads_marks_manual_review_on_risk_signal():
    scored = score_leads(
        [
            {
                "company_name": 'ООО "ТрансЛогистик"',
                "inn": "7701234567",
                "event_summary": "страхование перевозок и крупный новый контракт",
                "amount": 12_000_000,
                "contacts": [{"contact_email": "sales@trans.example"}],
                "enrichments": [
                    {"provider": "api_scoring", "status": "ok", "payload": {"high_risk": True}},
                    {"provider": "api_fssp", "status": "ok", "payload": {"manual_review_required": True}},
                ],
                "missing_data": [],
                "documents": [],
            }
        ],
        ["страхование", "перевозок"],
    )

    assert scored[0]["lead_priority"] in {"medium", "high"}
    assert scored[0]["manual_review_required"] is True
    assert "ручн" in scored[0]["rationale"].lower()


def test_score_leads_uses_externalized_scoring_rules(monkeypatch):
    monkeypatch.setattr(
        "agents.sales_lead_agent.logic.load_scoring_rules",
        lambda: ScoringRules(
            weights={"keyword_match": 5, "identified_inn": 100},
            amount_thresholds={"large": 10000000, "medium": 1000000},
            thresholds={"high": 80, "medium": 10},
            insufficient_data={"missing_fields_min": 3, "score_max": 1},
            rationale_messages={"identified_inn": "кастомный ИНН фактор"},
            next_steps={"default": "кастомный следующий шаг"},
        ),
    )

    scored = score_leads(
        [
            {
                "company_name": "A",
                "inn": "7701234567",
                "event_summary": "ничего особенного",
                "contacts": [],
                "enrichments": [],
                "documents": [],
            }
        ],
        [],
    )

    assert scored[0]["lead_priority"] == "high"
    assert "кастомный ИНН фактор" in scored[0]["rationale"]
    assert scored[0]["facts"]["recommended_next_step"] == "кастомный следующий шаг"


def test_document_pipeline_adapter_reads_supported_file(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "agents.sales_lead_agent.adapters.load_single_document",
        lambda path: [
            SimpleNamespace(page_content="Страница 1. ООО Пример", metadata={"page_number": 1}),
            SimpleNamespace(page_content="Страница 2. ИНН 7701234567, email contact@example.com", metadata={"page_number": 2}),
        ],
    )

    adapter = DocumentPipelineAdapter()
    documents = asyncio.run(
        adapter.resolve(
            source_hits=[
                {
                    "documents": [
                        {
                            "file_name": "sample.docx",
                            "file_type": "docx",
                            "stored_path": str(file_path),
                            "document_url": str(file_path),
                        }
                    ]
                }
            ],
            attachments=[],
            require_index=True,
        )
    )

    assert len(documents) == 1
    assert documents[0]["parse_status"] == "parsed"
    assert documents[0]["index_status"] == "ready"
    assert "7701234567" in documents[0]["text"]
    assert [segment["page_number"] for segment in documents[0]["segments"]] == [1, 2]


def test_deduplicate_keeps_combined_contacts():
    deduped = deduplicate_leads(
        [
            {
                "company_name": "A",
                "inn": "7701234567",
                "dedup_key": "inn:7701234567",
                "contacts": [{"contact_email": "one@example.com"}],
                "documents": [],
                "sources": [],
                "facts": {},
            },
            {
                "company_name": "A",
                "inn": "7701234567",
                "dedup_key": "inn:7701234567",
                "contacts": [{"contact_phone": "+7 999 123 45 67"}],
                "documents": [],
                "sources": [],
                "facts": {},
            },
        ]
    )

    assert len(deduped) == 1
    assert len(deduped[0]["contacts"]) == 2


def test_build_export_attachment_supports_text(tmp_path):
    text_path = tmp_path / "summary.txt"
    text_path.write_text("summary", encoding="utf-8")

    attachment = build_export_attachment(str(text_path))

    assert attachment is not None
    assert attachment["format"] == "text"
    assert attachment["mime_type"] == "text/plain"
