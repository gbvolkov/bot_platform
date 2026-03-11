from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


class GazDocumentsClientError(RuntimeError):
    pass


@dataclass
class GazDocumentsClient:
    base_url: str
    collection_id: str = "gaz"
    timeout_seconds: float = 20.0

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = urllib.parse.urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise GazDocumentsClientError(f"{exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise GazDocumentsClientError(str(exc.reason)) from exc
        if not body:
            return {}
        return json.loads(body)

    def get_collection_status(self, collection_id: Optional[str] = None) -> Dict[str, Any]:
        collection = collection_id or self.collection_id
        return self._request("GET", f"/gaz/runtime/collections/{collection}/status")

    def search_sales_materials(
        self,
        query: str,
        intent: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/gaz/runtime/collections/{self.collection_id}/materials/search",
            {
                "query": query,
                "intent": intent,
                "families": list(families or []),
                "competitor": competitor,
                "top_k": top_k,
            },
        )

    def estimate_research_cost(
        self,
        query: str,
        intended_depth: str,
        intent: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/gaz/runtime/collections/{self.collection_id}/materials/estimate",
            {
                "query": query,
                "intended_depth": intended_depth,
                "intent": intent,
                "families": list(families or []),
                "competitor": competitor,
            },
        )

    def get_branch_pack(
        self,
        branch: str,
        slots: Dict[str, Any],
        problem_summary: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/gaz/runtime/collections/{self.collection_id}/packs/{branch}",
            {
                "slots": slots,
                "problem_summary": problem_summary,
                "top_k": top_k,
            },
        )

    def read_material(self, candidate_id: str, focus: str, max_segments: int = 3) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/gaz/runtime/collections/{self.collection_id}/materials/{candidate_id}/read",
            {
                "focus": focus,
                "max_segments": max_segments,
            },
        )
