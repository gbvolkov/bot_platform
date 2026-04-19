from __future__ import annotations

import os
from typing import Any

from langchain.tools import tool

from .documents import GazDocumentsClient


_DEFAULT_GAZ_DOCS_BASE_URL = "http://127.0.0.1:8081"


def _build_client(
    *,
    docs_collection: str,
    docs_base_url: str | None,
    timeout_seconds: float,
) -> GazDocumentsClient:
    return GazDocumentsClient(
        base_url=docs_base_url
        or os.environ.get("GAZ_DOCUMENTS_SERVICE_URL", _DEFAULT_GAZ_DOCS_BASE_URL),
        collection_id=docs_collection,
        timeout_seconds=timeout_seconds,
    )


def _clamp(value: int, *, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _error_payload(tool_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "tool": tool_name,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def build_marketing_document_tools(
    *,
    locale: str = "ru",
    docs_collection: str = "gaz",
    docs_base_url: str | None = None,
    timeout_seconds: float = 20.0,
) -> list[Any]:
    client = _build_client(
        docs_collection=docs_collection,
        docs_base_url=docs_base_url,
        timeout_seconds=timeout_seconds,
    )

    @tool("search_marketing_materials")
    def search_marketing_materials(
        query: str,
        intent: str = "marketing_research",
        families: list[str] | None = None,
        competitor: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Search internal GAZ marketing and sales materials.

        Args:
            query: Natural-language search query.
            intent: Research intent, for example positioning, objections, comparison, or evidence.
            families: Optional product family filters.
            competitor: Optional competitor tag or name.
            top_k: Maximum number of candidates to return.
        """
        try:
            payload = client.search_sales_materials(
                query=query,
                intent=intent,
                families=families or [],
                competitor=competitor,
                top_k=_clamp(top_k, lower=1, upper=20),
            )
        except Exception as exc:  # noqa: BLE001
            return _error_payload("search_marketing_materials", exc)
        return {
            "status": "ok",
            "tool": "search_marketing_materials",
            "locale": locale,
            "collection_id": docs_collection,
            "payload": payload,
        }

    @tool("read_marketing_material")
    def read_marketing_material(
        candidate_id: str,
        focus: str = "",
        max_segments: int = 3,
    ) -> dict[str, Any]:
        """
        Read focused excerpts from one internal GAZ marketing material candidate.

        Args:
            candidate_id: Candidate id returned by search_marketing_materials.
            focus: Specific topic or evidence need for excerpt selection.
            max_segments: Maximum number of document segments to return.
        """
        try:
            payload = client.read_material(
                candidate_id=candidate_id,
                focus=focus,
                max_segments=_clamp(max_segments, lower=1, upper=10),
            )
        except Exception as exc:  # noqa: BLE001
            return _error_payload("read_marketing_material", exc)
        return {
            "status": "ok",
            "tool": "read_marketing_material",
            "locale": locale,
            "collection_id": docs_collection,
            "payload": payload,
        }

    @tool("get_marketing_branch_pack")
    def get_marketing_branch_pack(
        branch: str,
        problem_summary: str = "",
        slots: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Retrieve a branch-oriented pack of internal GAZ materials.

        Args:
            branch: Runtime branch or business segment to retrieve.
            problem_summary: Short customer problem or use case summary.
            slots: Optional structured branch parameters.
            top_k: Maximum number of materials to return.
        """
        try:
            payload = client.get_branch_pack(
                branch=branch,
                slots=slots or {},
                problem_summary=problem_summary,
                top_k=_clamp(top_k, lower=1, upper=20),
            )
        except Exception as exc:  # noqa: BLE001
            return _error_payload("get_marketing_branch_pack", exc)
        return {
            "status": "ok",
            "tool": "get_marketing_branch_pack",
            "locale": locale,
            "collection_id": docs_collection,
            "payload": payload,
        }

    @tool("estimate_marketing_research_cost")
    def estimate_marketing_research_cost(
        query: str,
        intended_depth: str = "standard",
        intent: str = "marketing_research",
        families: list[str] | None = None,
        competitor: str = "",
    ) -> dict[str, Any]:
        """
        Estimate retrieval effort for an internal marketing-materials research request.

        Args:
            query: Natural-language research query.
            intended_depth: Expected depth, for example quick, standard, or deep.
            intent: Research intent.
            families: Optional product family filters.
            competitor: Optional competitor tag or name.
        """
        try:
            payload = client.estimate_research_cost(
                query=query,
                intended_depth=intended_depth,
                intent=intent,
                families=families or [],
                competitor=competitor,
            )
        except Exception as exc:  # noqa: BLE001
            return _error_payload("estimate_marketing_research_cost", exc)
        return {
            "status": "ok",
            "tool": "estimate_marketing_research_cost",
            "locale": locale,
            "collection_id": docs_collection,
            "payload": payload,
        }

    return [
        search_marketing_materials,
        read_marketing_material,
        get_marketing_branch_pack,
        estimate_marketing_research_cost,
    ]
