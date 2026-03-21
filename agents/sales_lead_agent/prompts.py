from __future__ import annotations

STAGE_LABELS = {
    "request_understanding": "request_understanding",
    "task_planning": "task_planning",
    "source_collection": "source_collection",
    "document_resolution": "document_resolution",
    "index_lookup": "index_lookup",
    "fact_extraction": "fact_extraction",
    "normalization": "normalization",
    "enrichment": "enrichment",
    "rule_scoring": "rule_scoring",
    "persistence": "persistence",
    "feedback": "feedback",
    "response_composition": "response_composition",
}

NO_RESULTS_MESSAGE_RU = (
    "Подходящих лидов по текущему запросу не найдено. "
    "Можно уточнить фильтры, проверить другой источник или сформировать пустую выгрузку."
)
