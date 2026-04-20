"""Hybrid retrieval helpers for BOT_PROMPT_MODE=kpi_table."""
from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from utils.kpi_query_resolver import (
    build_kpi_search_query,
    is_methodology_follow_up,
    resolve_kpi_reference,
)

DialogMessage = tuple[str, str, str]
ScoredSearchFn = Callable[[str, int, Optional[Dict[str, str]]], List[Tuple[Document, float]]]

_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_QUOTED_TEXT_RE = re.compile(r"[«\"]([^«»\"\n]{3,200})[»\"]")
_EXPLAIN_RE = re.compile(
    r"\b(?:что\s+такое|что\s+за|объясни|объяснить|расшифруй|расшифровать|"
    r"расскажи|суть|смысл|значит|означает|как\s+считает(?:ся|ься)|"
    r"как\s+рассчитывает(?:ся|ься)|формул\w*|методик\w*|детализац\w*|"
    r"расч[её]т\w*|рассчит\w*|подробн\w*|нюанс\w*)\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"\b(?:какие|какой\s+список|перечисли|перечень|список|все|весь|"
    r"назови|покажи|оглас[аи]|каковы)\b",
    re.IGNORECASE,
)
_KPI_RE = re.compile(r"\bkpi\b|кпи|кпэ|показател", re.IGNORECASE)
_SHORT_DEICTIC_RE = re.compile(
    r"\b(?:здесь|тут|это|этот|эта|его|ее|её|их|тот|та|они|он|она|второй|"
    r"третий|четвертый|четвёртый|пятый|шестой|седьмой|восьмой|девятый|"
    r"десятый|\d{1,2})\b",
    re.IGNORECASE,
)
_FIELD_LABEL_RE = re.compile(r"^\s*([^:]{1,80}):\s*(.*)$")
_SPLIT_RE = re.compile(r"\s*>\s*")

_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "для",
    "у",
    "к",
    "ко",
    "с",
    "со",
    "из",
    "от",
    "до",
    "или",
    "ли",
    "не",
    "что",
    "это",
    "как",
    "какие",
    "какой",
    "какая",
    "какие",
    "каковы",
    "покажи",
    "перечисли",
    "перечень",
    "список",
    "все",
    "весь",
    "нужны",
    "нужно",
    "назови",
    "мне",
    "для",
    "этой",
    "этого",
    "этот",
    "эта",
    "эти",
    "kpi",
    "кпи",
    "кпэ",
    "показатель",
    "показатели",
    "показателей",
    "начальник",
    "начальника",
    "заместитель",
    "директор",
    "директора",
    "руководитель",
    "руководителя",
    "отдел",
    "отдела",
    "сектор",
    "сектора",
    "офис",
    "офиса",
    "работник",
    "работников",
    "исполнители",
    "исполнитель",
    "группа",
    "группы",
    "должность",
    "должностей",
    "должности",
    "подразделение",
    "подразделения",
    "центр",
    "ответственности",
    "работа",
    "работе",
    "продаж",
    "продажи",
    "филиал",
    "компания",
    "года",
    "год",
    "квартал",
}

_FIELD_WEIGHTS = {
    "department_path": 1.9,
    "center": 1.5,
    "worker_group": 1.35,
    "position_group": 1.2,
    "position": 1.8,
    "role_scope": 1.1,
}

_EXACT_BONUSES = {
    "department_path": 8.0,
    "center": 5.5,
    "worker_group": 4.5,
    "position_group": 3.8,
    "position": 6.5,
    "role_scope": 5.0,
}


def normalize_kpi_text(value: object) -> str:
    """Normalizes text for KPI matching."""
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").replace("Ё", "Е").replace("ё", "е")
    text = text.casefold()
    text = text.replace("«", '"').replace("»", '"')
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_kpi_text(value: object) -> List[str]:
    tokens = [token.casefold().replace("ё", "е") for token in _TOKEN_RE.findall(str(value or ""))]
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def count_kpi_tokens(value: object) -> int:
    return len(_TOKEN_RE.findall(str(value or "")))


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value == "-":
            continue
        key = normalize_kpi_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _join_unique(values: Iterable[str], sep: str = " | ") -> str:
    unique_values = _dedupe_preserve_order(values)
    return sep.join(unique_values)


def _extract_labeled_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    current_key: Optional[str] = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _FIELD_LABEL_RE.match(line)
        if match:
            current_key = match.group(1).strip()
            fields[current_key] = match.group(2).strip()
            continue
        if current_key:
            fields[current_key] = f"{fields[current_key]}\n{line}".strip()

    return fields


def _coerce_sheet_row(value: object) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return 0


def _build_scope_key(
    department_path: str,
    center: str,
    worker_group: str,
    position_group: str,
    position: str,
) -> str:
    return "||".join(
        [
            normalize_kpi_text(department_path),
            normalize_kpi_text(center),
            normalize_kpi_text(worker_group),
            normalize_kpi_text(position_group),
            normalize_kpi_text(position),
        ]
    )


def _human_scope_hint(scope: "KPIScopeEntry") -> str:
    dept_tail = [part.strip() for part in _SPLIT_RE.split(scope.department_path) if part.strip()]
    label_parts = []
    if dept_tail:
        label_parts.append(dept_tail[-1])
    elif scope.center:
        label_parts.append(scope.center)
    if scope.worker_group:
        label_parts.append(scope.worker_group)
    if scope.position_group:
        label_parts.append(scope.position_group)
    if scope.position:
        label_parts.append(scope.position)
    return " / ".join(label_parts) or scope.role_scope or "нужное подразделение"


def _drop_current_user_message(
    dialog_messages: Sequence[DialogMessage],
    current_query: str,
) -> List[DialogMessage]:
    remaining: List[DialogMessage] = []
    skipped = False
    current_query = (current_query or "").strip()

    for message_text, role, timestamp in dialog_messages:
        if not skipped and role == "user" and (message_text or "").strip() == current_query:
            skipped = True
            continue
        remaining.append((message_text, role, timestamp))

    return remaining


def _get_last_message(history: Sequence[DialogMessage], role: str) -> str | None:
    for message_text, message_role, _ in history:
        if message_role == role and (message_text or "").strip():
            return message_text
    return None


@dataclass
class KPIQueryAnalysis:
    intent: str
    raw_query: str
    scope_query: str
    semantic_query: str
    explicit_kpi_name: Optional[str]
    needs_methodology: bool


@dataclass
class KPIScopeEntry:
    scope_key: str
    department_path: str
    center: str
    worker_group: str
    position_group: str
    position: str
    role_scope: str
    search_text: str
    field_texts: Dict[str, str]
    field_tokens: Dict[str, set[str]]
    tokens: set[str]


@dataclass
class KPIAggregatedEntry:
    scope_key: str
    kpi_name: str
    kpi_name_norm: str
    document: Document
    min_sheet_row: int


@dataclass
class KPISlotValueEntry:
    slot_name: str
    canonical_value: str
    normalized_value: str
    scope_keys: Tuple[str, ...]
    aliases: Tuple[str, ...]
    normalized_aliases: Tuple[str, ...]


@dataclass
class KPICatalog:
    scopes: Dict[str, KPIScopeEntry]
    scope_documents: Dict[str, List[KPIAggregatedEntry]]
    aggregated_entries: Dict[Tuple[str, str], KPIAggregatedEntry]
    kpi_entries: Dict[str, List[KPIAggregatedEntry]]
    summary_documents: Dict[str, Document]
    token_idf: Dict[str, float]
    position_scope_counts: Dict[str, int]
    canonical_kpi_names: Dict[str, str]
    slot_values: Dict[str, List[str]]
    slot_value_index: Dict[str, Dict[str, Tuple[str, ...]]]
    slot_entries: Dict[str, List[KPISlotValueEntry]]
    scope_fingerprints: Dict[str, Tuple[str, ...]]


@dataclass
class KPIScopeCandidate:
    scope: KPIScopeEntry
    score: float
    matched_fields: set[str] = field(default_factory=set)
    exact_fields: set[str] = field(default_factory=set)
    matched_tokens: set[str] = field(default_factory=set)


@dataclass
class KPIHybridResult:
    documents: List[Document]
    extra_system_prompt: str = ""
    direct_answer: str = ""
    diagnostics: Dict[str, object] = field(default_factory=dict)


_SLOT_FIELD_MAP = {
    "department": "department_path",
    "center": "center",
    "worker_group": "worker_group",
    "position_group": "position_group",
    "position": "position",
}


def _build_scope_fingerprints(scope_documents: Dict[str, List[KPIAggregatedEntry]]) -> Dict[str, Tuple[str, ...]]:
    fingerprints: Dict[str, Tuple[str, ...]] = {}
    for scope_key, entries in scope_documents.items():
        fingerprints[scope_key] = tuple(
            sorted(
                entry.kpi_name_norm
                for entry in entries
                if entry.kpi_name_norm
            )
        )
    return fingerprints


def _slot_aliases(slot_name: str, value: str) -> Tuple[str, ...]:
    if not value:
        return ()
    aliases: List[str] = [value]
    if slot_name == "department":
        segments = [part.strip() for part in _SPLIT_RE.split(value) if part.strip()]
        aliases.extend(segments)
        if segments:
            aliases.append(segments[-1])
    seen: set[str] = set()
    result: List[str] = []
    for alias in aliases:
        normalized = normalize_kpi_text(alias)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(alias.strip())
    return tuple(result)


def _build_slot_indexes(scopes: Dict[str, KPIScopeEntry]) -> Tuple[
    Dict[str, List[str]],
    Dict[str, Dict[str, Tuple[str, ...]]],
    Dict[str, List[KPISlotValueEntry]],
]:
    slot_values: Dict[str, List[str]] = {}
    slot_value_index: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    slot_entries: Dict[str, List[KPISlotValueEntry]] = {}

    for slot_name, field_name in _SLOT_FIELD_MAP.items():
        by_value: Dict[str, Dict[str, object]] = {}
        for scope in scopes.values():
            raw_value = str(getattr(scope, field_name, "") or "").strip()
            normalized_value = normalize_kpi_text(raw_value)
            if not normalized_value:
                continue
            bucket = by_value.setdefault(
                normalized_value,
                {
                    "canonical_value": raw_value,
                    "scope_keys": set(),
                    "aliases": [],
                },
            )
            bucket["scope_keys"].add(scope.scope_key)
            bucket["aliases"].extend(_slot_aliases(slot_name, raw_value))

        ordered_entries: List[KPISlotValueEntry] = []
        ordered_values: List[str] = []
        value_index: Dict[str, Tuple[str, ...]] = {}
        for normalized_value, payload in by_value.items():
            aliases = _dedupe_preserve_order(payload["aliases"])
            scope_keys = tuple(sorted(payload["scope_keys"]))
            entry = KPISlotValueEntry(
                slot_name=slot_name,
                canonical_value=str(payload["canonical_value"]),
                normalized_value=normalized_value,
                scope_keys=scope_keys,
                aliases=tuple(aliases),
                normalized_aliases=tuple(normalize_kpi_text(alias) for alias in aliases if normalize_kpi_text(alias)),
            )
            ordered_entries.append(entry)
            ordered_values.append(entry.canonical_value)
            value_index[normalized_value] = scope_keys

        slot_entries[slot_name] = sorted(ordered_entries, key=lambda item: item.canonical_value)
        slot_values[slot_name] = ordered_values
        slot_value_index[slot_name] = value_index

    return slot_values, slot_value_index, slot_entries


def build_kpi_catalog(
    row_docs: Sequence[Document],
    methodology_docs: Sequence[Document],
) -> KPICatalog:
    scopes: Dict[str, KPIScopeEntry] = {}
    rows_by_scope: Dict[str, List[Document]] = defaultdict(list)
    methodologies_by_kpi: Dict[str, List[Document]] = defaultdict(list)

    for doc in methodology_docs:
        fields = _extract_labeled_fields(doc.page_content)
        metadata = doc.metadata or {}
        kpi_name = str(metadata.get("kpi_name") or fields.get("KPI") or "").strip()
        kpi_norm = normalize_kpi_text(kpi_name)
        if not kpi_norm:
            continue
        methodologies_by_kpi[kpi_norm].append(doc)

    position_scope_counts: Counter[str] = Counter()
    token_doc_frequency: Counter[str] = Counter()

    for doc in row_docs:
        fields = _extract_labeled_fields(doc.page_content)
        metadata = doc.metadata or {}
        department_path = str(metadata.get("department_path") or fields.get("Путь подразделения") or "").strip()
        center = str(metadata.get("center") or fields.get("Центр ответственности") or "").strip()
        worker_group = str(fields.get("Группа работников") or "").strip()
        position_group = str(fields.get("Группа должностей") or "").strip()
        position = str(metadata.get("position") or fields.get("Должность") or "").strip()
        role_scope = str(metadata.get("role_scope") or fields.get("Контур роли") or "").strip()
        scope_key = _build_scope_key(department_path, center, worker_group, position_group, position)
        if not scope_key:
            continue

        if scope_key not in scopes:
            field_texts = {
                "department_path": department_path,
                "center": center,
                "worker_group": worker_group,
                "position_group": position_group,
                "position": position,
                "role_scope": role_scope,
            }
            field_tokens = {
                name: set(tokenize_kpi_text(value))
                for name, value in field_texts.items()
                if value
            }
            tokens = set().union(*field_tokens.values()) if field_tokens else set()
            scopes[scope_key] = KPIScopeEntry(
                scope_key=scope_key,
                department_path=department_path,
                center=center,
                worker_group=worker_group,
                position_group=position_group,
                position=position,
                role_scope=role_scope,
                search_text=" | ".join(
                    value for value in [department_path, center, worker_group, position_group, position] if value
                ),
                field_texts=field_texts,
                field_tokens=field_tokens,
                tokens=tokens,
            )
            for token in tokens:
                token_doc_frequency[token] += 1
            if position:
                position_scope_counts[normalize_kpi_text(position)] += 1

        rows_by_scope[scope_key].append(doc)

    total_scopes = max(len(scopes), 1)
    token_idf = {
        token: 1.0 + math.log((total_scopes + 1) / (frequency + 1))
        for token, frequency in token_doc_frequency.items()
    }

    aggregated_entries: Dict[Tuple[str, str], KPIAggregatedEntry] = {}
    scope_documents: Dict[str, List[KPIAggregatedEntry]] = defaultdict(list)
    kpi_entries: Dict[str, List[KPIAggregatedEntry]] = defaultdict(list)
    canonical_kpi_names: Dict[str, str] = {}

    for scope_key, docs in rows_by_scope.items():
        grouped_docs: Dict[str, List[Document]] = defaultdict(list)
        for doc in docs:
            fields = _extract_labeled_fields(doc.page_content)
            metadata = doc.metadata or {}
            kpi_name = str(metadata.get("kpi_name") or fields.get("KPI") or "").strip()
            kpi_norm = normalize_kpi_text(kpi_name)
            if not kpi_norm:
                continue
            grouped_docs[kpi_norm].append(doc)
            canonical_kpi_names.setdefault(kpi_norm, kpi_name)

        for kpi_norm, grouped in grouped_docs.items():
            aggregated = _build_aggregated_scope_entry(
                scope=scopes[scope_key],
                row_docs=grouped,
                methodology_docs=methodologies_by_kpi.get(kpi_norm, []),
            )
            entry = KPIAggregatedEntry(
                scope_key=scope_key,
                kpi_name=canonical_kpi_names.get(kpi_norm, ""),
                kpi_name_norm=kpi_norm,
                document=aggregated,
                min_sheet_row=_coerce_sheet_row(aggregated.metadata.get("sheet_row_min")),
            )
            aggregated_entries[(scope_key, kpi_norm)] = entry
            scope_documents[scope_key].append(entry)
            kpi_entries[kpi_norm].append(entry)

    for scope_key, entries in scope_documents.items():
        scope_documents[scope_key] = sorted(
            entries,
            key=lambda item: (
                item.min_sheet_row or 10**9,
                normalize_kpi_text(item.kpi_name),
            ),
        )

    summary_documents = {
        kpi_norm: _build_kpi_summary_document(
            kpi_name=canonical_kpi_names.get(kpi_norm, ""),
            entries=entries,
            methodology_docs=methodologies_by_kpi.get(kpi_norm, []),
            scopes=scopes,
        )
        for kpi_norm, entries in kpi_entries.items()
    }
    slot_values, slot_value_index, slot_entries = _build_slot_indexes(scopes)
    scope_fingerprints = _build_scope_fingerprints(scope_documents)

    return KPICatalog(
        scopes=scopes,
        scope_documents=dict(scope_documents),
        aggregated_entries=aggregated_entries,
        kpi_entries=dict(kpi_entries),
        summary_documents=summary_documents,
        token_idf=token_idf,
        position_scope_counts=dict(position_scope_counts),
        canonical_kpi_names=canonical_kpi_names,
        slot_values=slot_values,
        slot_value_index=slot_value_index,
        slot_entries=slot_entries,
        scope_fingerprints=scope_fingerprints,
    )


def _build_aggregated_scope_entry(
    scope: KPIScopeEntry,
    row_docs: Sequence[Document],
    methodology_docs: Sequence[Document],
) -> Document:
    representative = row_docs[0]
    source = os.path.basename(str(representative.metadata.get("source") or "unknown"))
    sheet_name = str(representative.metadata.get("sheet_name") or "")
    all_fields = [_extract_labeled_fields(doc.page_content) for doc in row_docs]
    kpi_name = str(
        representative.metadata.get("kpi_name")
        or all_fields[0].get("KPI")
        or ""
    ).strip()
    kpi_norm = normalize_kpi_text(kpi_name)
    sheet_rows = sorted(
        row
        for row in (
            _coerce_sheet_row(doc.metadata.get("sheet_row") or _extract_labeled_fields(doc.page_content).get("Строка Excel"))
            for doc in row_docs
        )
        if row
    )

    lines = [
        "Тип записи: KPI агрегированная строка матрицы",
        f"Источник файла: {source}",
    ]
    if sheet_name:
        lines.append(f"Лист: {sheet_name}")
    if sheet_rows:
        lines.append(f"Строки Excel: {', '.join(str(row) for row in sheet_rows)}")
    if scope.department_path:
        lines.append(f"Путь подразделения: {scope.department_path}")
    if scope.role_scope:
        lines.append(f"Контур роли: {scope.role_scope}")
    if scope.center:
        lines.append(f"Центр ответственности: {scope.center}")
    if scope.worker_group:
        lines.append(f"Группа работников: {scope.worker_group}")
    if scope.position_group:
        lines.append(f"Группа должностей: {scope.position_group}")
    if scope.position:
        lines.append(f"Должность: {scope.position}")
    lines.append(f"KPI: {kpi_name}")
    if len(row_docs) > 1:
        lines.append(f"Количество исходных строк: {len(row_docs)}")

    for label in (
        "Детализация расчета",
        "Линия бизнеса",
        "Пул",
        "Прочая аналитика",
        "Периодичность расчета",
        "Специфика расчета",
        "Аналитики",
    ):
        joined = _join_unique(fields.get(label, "") for fields in all_fields)
        if joined:
            lines.append(f"{label}: {joined}")

    methodology_values = []
    for doc in methodology_docs:
        methodology_fields = _extract_labeled_fields(doc.page_content)
        methodology_value = methodology_fields.get("Методика расчета", "").strip()
        if methodology_value:
            methodology_values.append(methodology_value)
    methodology_text = _join_unique(methodology_values, sep="\n")
    if methodology_text:
        lines.append(f"Методика расчета: {methodology_text}")

    return Document(
        page_content="\n".join(lines),
        metadata={
            "source": representative.metadata.get("source", "unknown"),
            "sheet_name": representative.metadata.get("sheet_name", ""),
            "excel": True,
            "excel_doc_type": "kpi_row_aggregated",
            "kpi_name": kpi_name,
            "kpi_name_norm": kpi_norm,
            "position": scope.position,
            "center": scope.center,
            "department_path": scope.department_path,
            "role_scope": scope.role_scope,
            "scope_key": scope.scope_key,
            "sheet_row_min": sheet_rows[0] if sheet_rows else 0,
            "sheet_rows": ",".join(str(row) for row in sheet_rows),
            "aggregated_rows_count": len(row_docs),
        },
    )


def _build_kpi_summary_document(
    kpi_name: str,
    entries: Sequence[KPIAggregatedEntry],
    methodology_docs: Sequence[Document],
    scopes: Dict[str, KPIScopeEntry],
) -> Document:
    lines = [
        "Тип записи: Сводка KPI",
        f"KPI: {kpi_name}",
        f"Количество контуров применения: {len(entries)}",
    ]

    methodology_values = []
    for doc in methodology_docs:
        methodology_fields = _extract_labeled_fields(doc.page_content)
        methodology_value = methodology_fields.get("Методика расчета", "").strip()
        if methodology_value:
            methodology_values.append(methodology_value)
    methodology_text = _join_unique(methodology_values, sep="\n")
    if methodology_text:
        lines.append(f"Методика расчета: {methodology_text}")

    periodicity_values = []
    detail_values = []
    scope_samples = []
    for entry in entries:
        fields = _extract_labeled_fields(entry.document.page_content)
        periodicity_values.append(fields.get("Периодичность расчета", ""))
        detail_values.append(fields.get("Детализация расчета", ""))
        scope = scopes.get(entry.scope_key)
        if scope:
            scope_samples.append(_human_scope_hint(scope))

    periodicity_text = _join_unique(periodicity_values)
    if periodicity_text:
        lines.append(f"Периодичность расчета: {periodicity_text}")
    detail_text = _join_unique(detail_values)
    if detail_text:
        lines.append(f"Детализация расчета: {detail_text}")

    scope_samples = _dedupe_preserve_order(scope_samples)
    if scope_samples:
        lines.append("Примеры контуров применения:")
        for label in scope_samples[:5]:
            lines.append(f"- {label}")

    sample_entry = entries[0]
    return Document(
        page_content="\n".join(lines),
        metadata={
            "source": sample_entry.document.metadata.get("source", "unknown"),
            "excel": True,
            "excel_doc_type": "kpi_summary",
            "kpi_name": kpi_name,
            "kpi_name_norm": normalize_kpi_text(kpi_name),
            "summary_scope_count": len(entries),
        },
    )


class KPIHybridRetriever:
    """Hybrid retrieval specialized for KPI mode."""

    def __init__(
        self,
        catalog: KPICatalog,
        semantic_search: ScoredSearchFn,
        semantic_fetch_k: int = 36,
        scope_confidence_threshold: float = 7.5,
        scope_gap_threshold: float = 1.5,
        rerank_enabled: bool = True,
    ):
        self.catalog = catalog
        self.semantic_search = semantic_search
        self.semantic_fetch_k = max(8, semantic_fetch_k)
        self.scope_confidence_threshold = scope_confidence_threshold
        self.scope_gap_threshold = scope_gap_threshold
        self.rerank_enabled = rerank_enabled

    def retrieve(
        self,
        query: str,
        dialog_messages: Sequence[DialogMessage] = (),
        n_results: int = 6,
    ) -> KPIHybridResult:
        analysis = self._analyze_query(query, dialog_messages)
        if analysis.intent == "list_kpi":
            return self._retrieve_kpi_list(analysis, dialog_messages)
        return self._retrieve_kpi_details(analysis, dialog_messages, n_results)

    def analyze_query(
        self,
        query: str,
        dialog_messages: Sequence[DialogMessage] = (),
    ) -> KPIQueryAnalysis:
        return self._analyze_query(query, dialog_messages)

    def _analyze_query(
        self,
        query: str,
        dialog_messages: Sequence[DialogMessage],
    ) -> KPIQueryAnalysis:
        raw_query = (query or "").strip()
        history = _drop_current_user_message(dialog_messages, raw_query)
        prior_user = _get_last_message(history, "user") or ""
        scope_query = raw_query
        if len(tokenize_kpi_text(raw_query)) <= 4 and prior_user:
            scope_query = f"{prior_user}\n{raw_query}"
        intent_query = scope_query if len(tokenize_kpi_text(raw_query)) <= 6 else raw_query

        explicit_kpi = resolve_kpi_reference(raw_query, dialog_messages)
        if not explicit_kpi:
            explicit_kpi = self._guess_kpi_name(raw_query)

        methodology_follow_up = is_methodology_follow_up(raw_query, dialog_messages)
        needs_methodology = bool(
            _EXPLAIN_RE.search(raw_query)
            or _EXPLAIN_RE.search(intent_query)
            or methodology_follow_up
        )
        looks_like_list = bool(_LIST_RE.search(intent_query) or ("?" not in intent_query and _KPI_RE.search(intent_query)))
        if explicit_kpi and needs_methodology:
            intent = "explain_kpi"
        elif explicit_kpi:
            intent = "explain_kpi"
        elif needs_methodology:
            intent = "explain_kpi"
        elif looks_like_list and not needs_methodology:
            intent = "list_kpi"
        elif _SHORT_DEICTIC_RE.search(raw_query) and dialog_messages:
            intent = "explain_kpi"
        else:
            intent = "semantic_fallback"

        if intent == "semantic_fallback" and not explicit_kpi and not needs_methodology:
            if self._score_scope_candidates(scope_query):
                intent = "list_kpi"

        return KPIQueryAnalysis(
            intent=intent,
            raw_query=raw_query,
            scope_query=scope_query,
            semantic_query=build_kpi_search_query(raw_query, dialog_messages),
            explicit_kpi_name=explicit_kpi,
            needs_methodology=needs_methodology,
        )

    def _guess_kpi_name(self, query: str) -> Optional[str]:
        quoted = _QUOTED_TEXT_RE.search(query or "")
        if quoted:
            return quoted.group(1).strip()

        query_norm = normalize_kpi_text(query)
        query_tokens = set(tokenize_kpi_text(query))
        best_name = ""
        best_score = 0.0
        second_score = 0.0

        for kpi_norm, entries in self.catalog.kpi_entries.items():
            if not entries:
                continue
            score = 0.0
            canonical_name = entries[0].kpi_name
            name_tokens = set(tokenize_kpi_text(canonical_name))
            if kpi_norm and kpi_norm in query_norm:
                score += 12.0
            overlap = query_tokens & name_tokens
            if overlap:
                score += sum(self.catalog.token_idf.get(token, 1.0) for token in overlap) * 2.3
            if score <= 0:
                continue
            if score > best_score:
                second_score = best_score
                best_score = score
                best_name = canonical_name
            elif score > second_score:
                second_score = score

        if best_score >= 8.0 or (best_score >= 5.0 and best_score - second_score >= 1.5):
            return best_name
        return None

    def _score_scope_candidates(self, scope_query: str) -> List[KPIScopeCandidate]:
        query_norm = normalize_kpi_text(scope_query)
        query_tokens = set(tokenize_kpi_text(scope_query))
        candidates: List[KPIScopeCandidate] = []

        for scope in self.catalog.scopes.values():
            score = 0.0
            matched_fields: set[str] = set()
            exact_fields: set[str] = set()
            matched_tokens: set[str] = set()
            for field_name, field_text in scope.field_texts.items():
                normalized_field = normalize_kpi_text(field_text)
                field_tokens = scope.field_tokens.get(field_name, set())
                overlap = query_tokens & field_tokens
                if overlap:
                    matched_fields.add(field_name)
                    matched_tokens.update(overlap)
                    score += sum(self.catalog.token_idf.get(token, 1.0) for token in overlap) * _FIELD_WEIGHTS[field_name]
                if normalized_field and count_kpi_tokens(field_text) >= 2 and normalized_field in query_norm:
                    exact_fields.add(field_name)
                    score += _EXACT_BONUSES[field_name]
                if field_name == "department_path":
                    for segment in [part.strip() for part in _SPLIT_RE.split(field_text) if part.strip()]:
                        normalized_segment = normalize_kpi_text(segment)
                        if normalized_segment and count_kpi_tokens(segment) >= 2 and normalized_segment in query_norm:
                            exact_fields.add(field_name)
                            score += 6.0
                if field_name == "role_scope":
                    for segment in [part.strip() for part in field_text.split("|") if part.strip()]:
                        normalized_segment = normalize_kpi_text(segment)
                        if normalized_segment and count_kpi_tokens(segment) >= 2 and normalized_segment in query_norm:
                            exact_fields.add(field_name)
                            score += 4.0
            if score <= 0:
                continue
            candidates.append(
                KPIScopeCandidate(
                    scope=scope,
                    score=score,
                    matched_fields=matched_fields,
                    exact_fields=exact_fields,
                    matched_tokens=matched_tokens,
                )
            )

        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _choose_scope_candidate(self, candidates: Sequence[KPIScopeCandidate]) -> Tuple[Optional[KPIScopeCandidate], str]:
        if not candidates:
            return None, "no_candidates"

        top = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        gap = top.score - runner_up.score if runner_up else top.score
        position_norm = normalize_kpi_text(top.scope.position)
        position_scope_count = self.catalog.position_scope_counts.get(position_norm, 0)

        if top.score < self.scope_confidence_threshold:
            return None, "low_confidence"
        if not top.exact_fields and len(top.matched_tokens) <= 1:
            return None, "too_few_signals"
        if runner_up and gap < self.scope_gap_threshold:
            return None, "small_gap"
        if top.matched_fields and top.matched_fields <= {"position"} and position_scope_count > 1:
            return None, "position_only"
        return top, "ok"

    def _semantic_scope_fallback(
        self,
        query: str,
    ) -> Tuple[Optional[KPIScopeCandidate], Dict[str, float]]:
        search_results = self.semantic_search(
            query,
            self.semantic_fetch_k,
            {"excel_doc_type": "kpi_row"},
        )
        if not search_results:
            return None, {}

        scope_scores: Dict[str, float] = defaultdict(float)
        for doc, score in search_results:
            scope_key = str(doc.metadata.get("scope_key") or self._scope_key_from_document(doc))
            if not scope_key or scope_key not in self.catalog.scopes:
                continue
            semantic_score = 1.0 / (1.0 + max(float(score), 0.0))
            scope_scores[scope_key] += semantic_score

        if not scope_scores:
            return None, {}

        ordered = sorted(scope_scores.items(), key=lambda item: item[1], reverse=True)
        top_key, top_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        if top_score < 0.75:
            return None, dict(scope_scores)
        if len(ordered) > 1 and top_score - second_score < 0.15:
            return None, dict(scope_scores)

        candidate = KPIScopeCandidate(scope=self.catalog.scopes[top_key], score=top_score)
        return candidate, dict(scope_scores)

    def _retrieve_kpi_list(
        self,
        analysis: KPIQueryAnalysis,
        dialog_messages: Sequence[DialogMessage],
    ) -> KPIHybridResult:
        lexical_candidates = self._score_scope_candidates(analysis.scope_query)
        selected_scope, scope_reason = self._choose_scope_candidate(lexical_candidates)
        diagnostics: Dict[str, object] = {
            "intent": analysis.intent,
            "scope_reason": scope_reason,
            "lexical_top_scope": _human_scope_hint(lexical_candidates[0].scope) if lexical_candidates else "",
        }

        equivalent_docs = self._merge_equivalent_scope_documents(lexical_candidates)
        if equivalent_docs:
            diagnostics["scope_reason"] = "merged_equivalent_scopes"
            diagnostics["selected_scope"] = _human_scope_hint(lexical_candidates[0].scope)
            diagnostics["scope_docs"] = len(equivalent_docs)
            return KPIHybridResult(
                documents=equivalent_docs,
                extra_system_prompt=(
                    "Ниже передан полный агрегированный список KPI для одного смыслового контура роли. "
                    "Если в данных есть несколько технических scope с одинаковым набором KPI, они уже объединены. "
                    "Перечисли все KPI по одному разу и ничего не пропускай."
                ),
                diagnostics=diagnostics,
            )

        if selected_scope is None:
            semantic_scope, semantic_scores = self._semantic_scope_fallback(analysis.semantic_query)
            diagnostics["semantic_scope_scores"] = semantic_scores
            if semantic_scope is not None:
                selected_scope = semantic_scope
                diagnostics["scope_reason"] = "semantic_fallback"

        if selected_scope is None:
            semantic_candidates = self._scope_candidates_from_scores(diagnostics.get("semantic_scope_scores") or {})
            clarification_candidates = semantic_candidates or list(lexical_candidates)
            direct_answer = self._build_scope_clarification(analysis.raw_query, clarification_candidates)
            diagnostics["direct_answer"] = True
            return KPIHybridResult(documents=[], direct_answer=direct_answer, diagnostics=diagnostics)

        scope_docs = [entry.document for entry in self.catalog.scope_documents.get(selected_scope.scope.scope_key, [])]
        diagnostics["selected_scope"] = _human_scope_hint(selected_scope.scope)
        diagnostics["scope_docs"] = len(scope_docs)
        extra_system_prompt = (
            "Ниже передан полный агрегированный список KPI только для одного точного контура роли. "
            "Каждый документ соответствует одному уникальному KPI. "
            "Перечисли все KPI по одному разу и ничего не пропускай."
        )
        return KPIHybridResult(
            documents=scope_docs,
            extra_system_prompt=extra_system_prompt,
            diagnostics=diagnostics,
        )

    def _retrieve_kpi_details(
        self,
        analysis: KPIQueryAnalysis,
        dialog_messages: Sequence[DialogMessage],
        n_results: int,
    ) -> KPIHybridResult:
        diagnostics: Dict[str, object] = {"intent": analysis.intent}
        scope_candidate, scope_reason = self._choose_scope_candidate(self._score_scope_candidates(analysis.scope_query))
        diagnostics["scope_reason"] = scope_reason
        if scope_candidate is not None:
            diagnostics["selected_scope"] = _human_scope_hint(scope_candidate.scope)

        explicit_kpi = analysis.explicit_kpi_name
        if explicit_kpi:
            kpi_norm = normalize_kpi_text(explicit_kpi)
            docs = self._documents_for_explicit_kpi(kpi_norm, scope_candidate)
            if docs:
                diagnostics["explicit_kpi"] = explicit_kpi
                return KPIHybridResult(
                    documents=docs[: max(1, n_results)],
                    extra_system_prompt=(
                        "Сначала объясни ровно тот KPI, который запрошен пользователем. "
                        "Если в документе есть методика расчета, опирайся на нее. "
                        "Если методика не раскрыта, скажи об этом прямо."
                    ),
                    diagnostics=diagnostics,
                )

        semantic_results = self.semantic_search(
            analysis.semantic_query,
            self.semantic_fetch_k,
            {"excel_doc_type": "kpi_row"},
        )
        reranked = self._rerank_semantic_candidates(
            query=analysis.semantic_query,
            semantic_results=semantic_results,
            scope_candidate=scope_candidate,
            limit=max(1, n_results),
        )
        diagnostics["semantic_candidates"] = len(semantic_results)
        diagnostics["reranked_docs"] = len(reranked)
        return KPIHybridResult(
            documents=reranked,
            extra_system_prompt=(
                "Объясни KPI строго по найденным документам. "
                "Если методика расчета присутствует, используй ее как основной источник смысла. "
                "Если нет, не домысливай."
            ),
            diagnostics=diagnostics,
        )

    def _documents_for_explicit_kpi(
        self,
        kpi_norm: str,
        scope_candidate: Optional[KPIScopeCandidate],
    ) -> List[Document]:
        docs: List[Document] = []
        summary_doc = self.catalog.summary_documents.get(kpi_norm)
        if summary_doc is not None:
            docs.append(summary_doc)

        if scope_candidate is not None:
            exact_entry = self.catalog.aggregated_entries.get((scope_candidate.scope.scope_key, kpi_norm))
            if exact_entry is not None:
                docs.append(exact_entry.document)
                return self._unique_documents(docs)

        for entry in self.catalog.kpi_entries.get(kpi_norm, [])[:2]:
            docs.append(entry.document)

        return self._unique_documents(docs)

    def _rerank_semantic_candidates(
        self,
        query: str,
        semantic_results: Sequence[Tuple[Document, float]],
        scope_candidate: Optional[KPIScopeCandidate],
        limit: int,
    ) -> List[Document]:
        if not semantic_results:
            return []

        aggregated_scores: Dict[str, float] = defaultdict(float)
        aggregated_docs: Dict[str, Document] = {}
        query_norm = normalize_kpi_text(query)
        query_tokens = set(tokenize_kpi_text(query))

        for doc, raw_score in semantic_results:
            metadata = doc.metadata or {}
            scope_key = str(metadata.get("scope_key") or self._scope_key_from_document(doc))
            kpi_norm = str(metadata.get("kpi_name_norm") or normalize_kpi_text(metadata.get("kpi_name") or _extract_labeled_fields(doc.page_content).get("KPI") or ""))
            if not scope_key or not kpi_norm:
                continue

            summary_doc = self.catalog.summary_documents.get(kpi_norm)
            if summary_doc is None:
                continue

            semantic_score = 1.0 / (1.0 + max(float(raw_score), 0.0))
            doc_score = semantic_score
            if self.rerank_enabled:
                canonical_kpi_name = self.catalog.canonical_kpi_names.get(kpi_norm, "")
                name_norm = normalize_kpi_text(canonical_kpi_name)
                name_tokens = set(tokenize_kpi_text(canonical_kpi_name))
                overlap = query_tokens & name_tokens
                if overlap:
                    doc_score += sum(self.catalog.token_idf.get(token, 1.0) for token in overlap) * 0.55
                if name_norm and name_norm in query_norm:
                    doc_score += 4.0
                if scope_candidate is not None and scope_key == scope_candidate.scope.scope_key:
                    doc_score += 1.5

            aggregated_scores[kpi_norm] = max(aggregated_scores[kpi_norm], doc_score)
            aggregated_docs[kpi_norm] = summary_doc

        if not aggregated_scores:
            return []

        ordered = sorted(aggregated_scores.items(), key=lambda item: item[1], reverse=True)
        docs = [aggregated_docs[kpi_norm] for kpi_norm, _ in ordered[:limit]]
        return self._unique_documents(docs)

    def _build_scope_clarification(
        self,
        query: str,
        candidates: Sequence[KPIScopeCandidate],
    ) -> str:
        if not candidates:
            return "Чтобы назвать точные KPI, укажите должность и подразделение."

        options = _dedupe_preserve_order(_human_scope_hint(candidate.scope) for candidate in candidates[:3])
        if not options:
            return "Чтобы назвать точные KPI, укажите должность и подразделение."

        if len(options) == 1:
            return f"Для этой должности KPI зависят от подразделения. Уточните, пожалуйста: {options[0]}?"

        joined = ", ".join(options[:-1]) + f" или {options[-1]}"
        return f"Для этой должности KPI зависят от подразделения. Уточните, пожалуйста: {joined}?"

    def _scope_key_from_document(self, doc: Document) -> str:
        metadata = doc.metadata or {}
        if metadata.get("scope_key"):
            return str(metadata.get("scope_key"))
        fields = _extract_labeled_fields(doc.page_content)
        return _build_scope_key(
            str(metadata.get("department_path") or fields.get("Путь подразделения") or ""),
            str(metadata.get("center") or fields.get("Центр ответственности") or ""),
            str(fields.get("Группа работников") or ""),
            str(fields.get("Группа должностей") or ""),
            str(metadata.get("position") or fields.get("Должность") or ""),
        )

    def _scope_candidates_from_scores(self, scope_scores: Dict[str, float]) -> List[KPIScopeCandidate]:
        candidates: List[KPIScopeCandidate] = []
        for scope_key, score in sorted(scope_scores.items(), key=lambda item: item[1], reverse=True):
            scope = self.catalog.scopes.get(scope_key)
            if scope is None:
                continue
            candidates.append(KPIScopeCandidate(scope=scope, score=score))
        return candidates

    def _scope_kpi_fingerprint(self, scope_key: str) -> Tuple[str, ...]:
        entries = self.catalog.scope_documents.get(scope_key, [])
        return tuple(
            sorted(
                str(entry.kpi_name_norm or normalize_kpi_text(entry.kpi_name))
                for entry in entries
                if str(entry.kpi_name_norm or normalize_kpi_text(entry.kpi_name)).strip()
            )
        )

    def _merge_equivalent_scope_documents(
        self,
        candidates: Sequence[KPIScopeCandidate],
    ) -> List[Document]:
        if len(candidates) < 2:
            return []

        top_score = candidates[0].score
        near_top = [
            candidate
            for candidate in candidates
            if top_score - candidate.score < self.scope_gap_threshold
        ]
        if len(near_top) < 2:
            return []

        base_scope = near_top[0].scope
        base_department = normalize_kpi_text(base_scope.department_path)
        base_center = normalize_kpi_text(base_scope.center)
        base_position = normalize_kpi_text(base_scope.position)
        base_fingerprint = self._scope_kpi_fingerprint(base_scope.scope_key)
        if not base_fingerprint:
            return []

        merged_by_kpi: Dict[str, Document] = {}
        for candidate in near_top:
            scope = candidate.scope
            if normalize_kpi_text(scope.department_path) != base_department:
                return []
            if normalize_kpi_text(scope.center) != base_center:
                return []
            if normalize_kpi_text(scope.position) != base_position:
                return []
            if self._scope_kpi_fingerprint(scope.scope_key) != base_fingerprint:
                return []
            for entry in self.catalog.scope_documents.get(scope.scope_key, []):
                merged_by_kpi.setdefault(entry.kpi_name_norm, entry.document)

        ordered_docs = sorted(
            merged_by_kpi.values(),
            key=lambda doc: (
                _coerce_sheet_row(doc.metadata.get("sheet_row_min")),
                normalize_kpi_text(doc.metadata.get("kpi_name") or ""),
            ),
        )
        return self._unique_documents(ordered_docs)

    @staticmethod
    def _unique_documents(documents: Sequence[Document]) -> List[Document]:
        unique_docs: List[Document] = []
        seen_keys: set[Tuple[str, str, str]] = set()
        for doc in documents:
            metadata = doc.metadata or {}
            key = (
                str(metadata.get("excel_doc_type") or ""),
                str(metadata.get("scope_key") or ""),
                str(metadata.get("kpi_name_norm") or ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_docs.append(doc)
        return unique_docs
