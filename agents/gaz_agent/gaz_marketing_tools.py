from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

from langchain.tools import tool


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOCS_ROOT = "data/gaz-docs"
_DEFAULT_CACHE_ROOT = "data/gaz_index"


def _resolve_repo_path(raw_path: str | Path | None, *, env_var: str, default: str) -> Path:
    value = raw_path or os.environ.get(env_var) or default
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _clamp(value: int, *, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _error_payload(tool_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "tool": tool_name,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


class _LocalGazMarketingRuntime:
    """Small in-process adapter around the same runtime used by the HTTP service."""

    def __init__(
        self,
        *,
        collection_id: str,
        docs_root: str | Path | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        self.collection_id = collection_id
        self.docs_root = _resolve_repo_path(
            docs_root,
            env_var="GAZ_DOCUMENTS_ROOT",
            default=_DEFAULT_DOCS_ROOT,
        )
        self.cache_root = _resolve_repo_path(
            cache_root,
            env_var="GAZ_INDEX_ROOT",
            default=_DEFAULT_CACHE_ROOT,
        )
        self._lock = RLock()
        self._runtime: Any | None = None

    def _get_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        with self._lock:
            if self._runtime is None:
                Path.cwd().joinpath("logs").mkdir(parents=True, exist_ok=True)
                from services.kb_manager.gaz_runtime import GazRuntimeService

                self._runtime = GazRuntimeService(
                    docs_root=self.docs_root,
                    cache_root=self.cache_root,
                )
        return self._runtime

    def search_sales_materials(
        self,
        *,
        query: str,
        intent: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        return self._get_runtime().search_sales_materials(
            query=query,
            intent=intent,
            families=families or [],
            competitor=competitor,
            top_k=top_k,
            collection_id=self.collection_id,
        )

    def read_material(
        self,
        *,
        candidate_id: str,
        focus: str,
        max_segments: int = 3,
    ) -> dict[str, Any]:
        return self._get_runtime().read_material(
            candidate_id=candidate_id,
            focus=focus,
            max_segments=max_segments,
            collection_id=self.collection_id,
        )

    def get_branch_pack(
        self,
        *,
        branch: str,
        slots: Mapping[str, Any] | None = None,
        problem_summary: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        return self._get_runtime().get_branch_pack(
            branch=branch,
            slots=slots or {},
            problem_summary=problem_summary,
            top_k=top_k,
            collection_id=self.collection_id,
        )

    def estimate_research_cost(
        self,
        *,
        query: str,
        intended_depth: str,
        intent: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
    ) -> dict[str, Any]:
        return self._get_runtime().estimate_research_cost(
            query=query,
            intended_depth=intended_depth,
            intent=intent,
            families=families or [],
            competitor=competitor,
            collection_id=self.collection_id,
        )


def _build_runtime(
    *,
    docs_collection: str,
    docs_root: str | Path | None,
    cache_root: str | Path | None,
) -> _LocalGazMarketingRuntime:
    return _LocalGazMarketingRuntime(
        collection_id=docs_collection,
        docs_root=docs_root,
        cache_root=cache_root,
    )


def build_marketing_document_tools(
    *,
    locale: str = "ru",
    docs_collection: str = "gaz",
    docs_base_url: str | None = None,
    timeout_seconds: float = 20.0,
    docs_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> list[Any]:
    """Build GAZ marketing retrieval tools backed by local runtime assets.

    docs_base_url and timeout_seconds are accepted for compatibility with
    agents.gaz_agent.marketing_tools, but the local runtime does not use HTTP.
    """

    _ = docs_base_url, timeout_seconds
    runtime = _build_runtime(
        docs_collection=docs_collection,
        docs_root=docs_root,
        cache_root=cache_root,
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
            payload = runtime.search_sales_materials(
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
            payload = runtime.read_material(
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
            payload = runtime.get_branch_pack(
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
            payload = runtime.estimate_research_cost(
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


build_gaz_marketing_document_tools = build_marketing_document_tools


__all__ = [
    "build_gaz_marketing_document_tools",
    "build_marketing_document_tools",
]
