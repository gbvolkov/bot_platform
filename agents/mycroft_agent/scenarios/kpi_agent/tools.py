from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from langchain.tools import tool


_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIELD_NAMES = (
    "department_1",
    "department_2",
    "department_3",
    "department_4",
    "department_5",
    "department_6",
    "department_7",
    "department_8",
    "employee_group",
    "position",
)
_FULL_POSITION_FIELD = "full_position_name"
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _normalize_text(value: object) -> str:
    text = str(value or "").replace("\xa0", " ").replace("Ё", "Е").replace("ё", "е")
    text = text.casefold()
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(_normalize_text(value)))


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _candidate_score(query: str, candidate: str) -> float:
    query_norm = _normalize_text(query)
    candidate_norm = _normalize_text(candidate)
    if not query_norm or not candidate_norm:
        return 0.0

    score = _ratio(query_norm, candidate_norm) * 0.45
    if query_norm in candidate_norm or candidate_norm in query_norm:
        score += 0.35

    query_tokens = _tokens(query_norm)
    candidate_tokens = _tokens(candidate_norm)
    if query_tokens and candidate_tokens:
        candidate_token_set = set(candidate_tokens)
        exact_overlap = sum(1 for token in query_tokens if token in candidate_token_set)
        score += (exact_overlap / len(query_tokens)) * 0.25

        fuzzy_scores = []
        for query_token in query_tokens:
            best = max((_ratio(query_token, token) for token in candidate_tokens), default=0.0)
            if best >= 0.72:
                fuzzy_scores.append(best)
        if fuzzy_scores:
            score += (sum(fuzzy_scores) / len(query_tokens)) * 0.25

    return min(score, 1.0)


def _clean_field_value(value: object) -> str:
    return str(value or "").replace("\xa0", " ").strip()


def _compose_full_position_name(fields: dict[str, str]) -> str:
    parts = [fields[field_name] for field_name in _FIELD_NAMES if fields.get(field_name)]
    return " / ".join(parts)


@dataclass(frozen=True)
class KPIStaffStructureCandidate:
    staff_structure_id: int
    fields: dict[str, str]
    full_position_name: str


@dataclass(frozen=True)
class KPIStaffStructureFuzzyIndex:
    candidates: tuple[KPIStaffStructureCandidate, ...]

    @classmethod
    def from_sqlite(cls, database_path: str | Path) -> "KPIStaffStructureFuzzyIndex":
        path = _resolve_repo_path(database_path)
        if not path.is_file():
            raise FileNotFoundError(f"KPI database not found: {path}")

        selected_columns = ("staff_structure_id", *_FIELD_NAMES)
        candidates: list[KPIStaffStructureCandidate] = []
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                f"""
                SELECT {", ".join(selected_columns)}
                FROM kpi_staff_structure
                ORDER BY staff_structure_id
                """
            ).fetchall()

        for row in rows:
            staff_structure_id = int(row[0])
            fields = {
                field_name: value
                for field_name, raw_value in zip(_FIELD_NAMES, row[1:], strict=True)
                if (value := _clean_field_value(raw_value))
            }
            full_position_name = _compose_full_position_name(fields)
            if full_position_name:
                candidates.append(
                    KPIStaffStructureCandidate(
                        staff_structure_id=staff_structure_id,
                        fields=fields,
                        full_position_name=full_position_name,
                    )
                )

        return cls(candidates=tuple(candidates))

    def search(
        self,
        query: str,
        *,
        db_fieldname: str | None = None,
        limit_per_field: int = 8,
        min_score: float = 0.42,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit_per_field or 8), 25))
        threshold = max(0.0, min(float(min_score), 1.0))
        normalized_field = (db_fieldname or "").strip().casefold()

        if normalized_field and normalized_field != _FULL_POSITION_FIELD:
            return {
                "query": query,
                "search_mode": _FULL_POSITION_FIELD,
                "results": [],
                "candidates": [],
                "error": (
                    f"Unknown db_fieldname '{db_fieldname}'. "
                    f"Search is supported only by {_FULL_POSITION_FIELD}."
                ),
            }

        scored = []
        for candidate in self.candidates:
            score = _candidate_score(query, candidate.full_position_name)
            if score >= threshold:
                scored.append((score, candidate))
        scored.sort(key=lambda item: (-item[0], item[1].full_position_name))

        selected = [candidate for _, candidate in scored[:limit]]
        candidate_values = [candidate.full_position_name for candidate in selected]
        staff_structure_ids = [candidate.staff_structure_id for candidate in selected]
        results = []
        if candidate_values:
            results.append(
                {
                    "db_fieldname": _FULL_POSITION_FIELD,
                    "candidate_values": candidate_values,
                    "staff_structure_ids": staff_structure_ids,
                }
            )

        return {
            "query": query,
            "search_mode": _FULL_POSITION_FIELD,
            "results": results,
            "candidates": [
                {
                    "staff_structure_id": candidate.staff_structure_id,
                    "full_position_name": candidate.full_position_name,
                    "fields": candidate.fields,
                }
                for candidate in selected
            ],
        }


def build_kpi_staff_structure_fuzzy_search_tool(
    database_path: str = "data/kpi/kpi.sqlite",
    default_limit_per_field: int = 8,
    default_min_score: float = 0.35,
) -> Any:
    """Build a KPI staff-structure fuzzy search tool with an in-memory index."""

    index = KPIStaffStructureFuzzyIndex.from_sqlite(database_path)

    @tool("kpi_staff_structure_fuzzy_search")
    def kpi_staff_structure_fuzzy_search(
        query: str,
        db_fieldname: str = "",
        limit_per_field: int = default_limit_per_field,
        min_score: float = default_min_score,
    ) -> str:
        """
        Fuzzy-search full position names from kpi_staff_structure.

        Args:
            query: User text fragment to match against full staff-structure position names.
            db_fieldname: Optional field filter. Only full_position_name is supported.
            limit_per_field: Maximum candidate full position names to return.
            min_score: Fuzzy match threshold from 0.0 to 1.0.

        Returns:
            JSON with results shaped as:
            {"query": "...", "results": [{"db_fieldname": "full_position_name", "candidate_values": ["..."], "staff_structure_ids": [1]}]}.
            The candidates field also contains staff_structure_id, full_position_name, and source row fields for BI lookup.
        """
        payload = index.search(
            query=query,
            db_fieldname=db_fieldname,
            limit_per_field=limit_per_field,
            min_score=min_score,
        )
        return json.dumps(payload, ensure_ascii=False)

    return kpi_staff_structure_fuzzy_search
