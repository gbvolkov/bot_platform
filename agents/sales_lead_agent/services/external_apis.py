from __future__ import annotations

from typing import Any

import httpx

from ..schemas import (
    CounterpartyFSSPResponse,
    CounterpartyScoringResponse,
    FSSPGroupedRecord,
    Fincoef,
    ScorePayload,
    TopFactor,
)
from ..settings import SalesLeadAgentSettings


class CounterpartyClients:
    def __init__(self, settings: SalesLeadAgentSettings) -> None:
        self._settings = settings

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._settings.damia_api_key:
            headers["Authorization"] = f"Bearer {self._settings.damia_api_key}"
        return headers

    def _client(self) -> httpx.Client:
        return httpx.Client(headers=self._headers(), timeout=20.0)

    def scoring(self, *, inn: str, model: str | None, include_fincoefs: bool) -> CounterpartyScoringResponse:
        if not self._settings.scoring_base_url:
            return CounterpartyScoringResponse(
                status="failed",
                error="SALES_LEAD_AGENT_SCORING_BASE_URL is not configured.",
                inn=inn,
            )
        try:
            with self._client() as client:
                score_response = client.get(
                    f"{self._settings.scoring_base_url}/scoring/score",
                    params={"inn": inn, "model": model} if model else {"inn": inn},
                )
                score_response.raise_for_status()
                score_payload = score_response.json() if score_response.content else {}
                fincoefs_payload: list[dict[str, Any]] = []
                if include_fincoefs:
                    fincoefs_response = client.get(
                        f"{self._settings.scoring_base_url}/scoring/fincoefs",
                        params={"inn": inn},
                    )
                    fincoefs_response.raise_for_status()
                    fincoefs_payload = fincoefs_response.json() if fincoefs_response.content else []
        except Exception as exc:
            return CounterpartyScoringResponse(status="failed", error=str(exc), inn=inn)

        score = ScorePayload(
            risk_value=_floatish(score_payload.get("risk_value")),
            risk_zone=_stringish(score_payload.get("risk_zone")),
            score_value=_floatish(score_payload.get("score_value")),
            score_zone=_stringish(score_payload.get("score_zone")),
            reliability_value=_floatish(score_payload.get("reliability_value")),
            reliability_zone=_stringish(score_payload.get("reliability_zone")),
            top_factors=[
                TopFactor(
                    name=str(item.get("name") or ""),
                    value=_floatish(item.get("value")),
                    nwoe=_floatish(item.get("nwoe")),
                )
                for item in score_payload.get("top_factors") or []
                if isinstance(item, dict)
            ],
        )
        fincoefs = [
            Fincoef(
                name=str(item.get("name") or ""),
                value=_floatish(item.get("value")),
                norm=_floatish(item.get("norm")),
                comparison=_stringish(item.get("comparison")),
            )
            for item in fincoefs_payload
            if isinstance(item, dict)
        ]
        return CounterpartyScoringResponse(
            status="success",
            error=None,
            inn=inn,
            score=score,
            fincoefs=fincoefs,
        )

    def fssp(
        self,
        *,
        inn: str,
        from_date: str | None,
        to_date: str | None,
        response_format: int,
    ) -> CounterpartyFSSPResponse:
        if not self._settings.fssp_base_url:
            return CounterpartyFSSPResponse(
                status="failed",
                error="SALES_LEAD_AGENT_FSSP_BASE_URL is not configured.",
                inn=inn,
                raw_format=response_format,
            )
        params: dict[str, Any] = {"inn": inn, "format": response_format}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        try:
            with self._client() as client:
                response = client.get(f"{self._settings.fssp_base_url}/fssp/isps", params=params)
                response.raise_for_status()
                payload = response.json() if response.content else []
        except Exception as exc:
            return CounterpartyFSSPResponse(
                status="failed",
                error=str(exc),
                inn=inn,
                raw_format=response_format,
            )

        grouped = [
            FSSPGroupedRecord(
                year=int(item.get("year") or 0),
                status=str(item.get("status") or ""),
                subject=str(item.get("subject") or ""),
                amount=_floatish(item.get("amount")),
                count=int(item.get("count") or 0),
                proceeding_ids=[str(value) for value in item.get("proceeding_ids") or []],
            )
            for item in payload
            if isinstance(item, dict)
        ]
        return CounterpartyFSSPResponse(
            status="success",
            error=None,
            inn=inn,
            grouped=grouped,
            raw_format=response_format,
        )


def _floatish(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _stringish(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
