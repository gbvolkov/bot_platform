from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx

from .config import get_settings
from .schemas import (
    RetrievalConflictResponse,
    RetrievalSnapshot,
    RetrievalUserInputErrorResponse,
)


logger = logging.getLogger(__name__)


class RetrievalServiceConflictError(Exception):
    def __init__(self, payload: RetrievalConflictResponse) -> None:
        super().__init__(payload.message)
        self.payload = payload
        self.snapshot = payload.active_snapshot


class RetrievalServiceUserInputError(Exception):
    def __init__(self, payload: RetrievalUserInputErrorResponse) -> None:
        super().__init__(payload.message)
        self.payload = payload
        self.code = payload.code
        self.suggestion = payload.suggestion
        self.input_field = payload.input_field


class RetrievalServiceClient:
    def __init__(
        self,
        *,
        base_url: str,
        request_timeout: float | None,
        connect_timeout: float,
    ) -> None:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=request_timeout,
            write=request_timeout,
            pool=request_timeout,
        )
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def submit_purchase_search(
        self,
        *,
        conversation_id: str,
        requested_run_id: str | None,
        search_url: str | None,
        query_texts: list[str] | None,
        max_pages: int | None,
        agent_id: str = "sales_lead_agent",
    ) -> RetrievalSnapshot:
        response = await self._client.post(
            "/retrievals/purchase-search",
            json={
                "conversation_id": conversation_id,
                "requested_run_id": requested_run_id,
                "search_url": search_url,
                "query_texts": query_texts,
                "max_pages": max_pages,
                "agent_id": agent_id,
            },
        )
        if response.status_code == httpx.codes.CONFLICT:
            raise RetrievalServiceConflictError(
                RetrievalConflictResponse.model_validate(response.json())
            )
        if response.status_code == httpx.codes.BAD_REQUEST:
            raise RetrievalServiceUserInputError(
                RetrievalUserInputErrorResponse.model_validate(response.json())
            )
        response.raise_for_status()
        return RetrievalSnapshot.model_validate(response.json())

    async def get_latest_for_conversation(
        self,
        *,
        conversation_id: str,
        include_payloads: bool = False,
    ) -> RetrievalSnapshot | None:
        response = await self._client.get(
            f"/retrievals/conversations/{conversation_id}/latest",
            params={"include_payloads": str(include_payloads).lower()},
        )
        response.raise_for_status()
        payload = response.json()
        if payload is None:
            return None
        return RetrievalSnapshot.model_validate(payload)

    async def get_retrieval(
        self,
        *,
        retrieval_id: str,
        include_payloads: bool = False,
    ) -> RetrievalSnapshot | None:
        response = await self._client.get(
            f"/retrievals/{retrieval_id}",
            params={"include_payloads": str(include_payloads).lower()},
        )
        response.raise_for_status()
        payload = response.json()
        if payload is None:
            return None
        return RetrievalSnapshot.model_validate(payload)

    async def mark_announced(self, retrieval_id: str) -> None:
        response = await self._client.post(f"/retrievals/{retrieval_id}/announced")
        response.raise_for_status()


@lru_cache(maxsize=1)
def get_retrieval_service_client() -> RetrievalServiceClient:
    settings = get_settings()
    return RetrievalServiceClient(
        base_url=settings.base_url,
        request_timeout=settings.request_timeout_seconds,
        connect_timeout=settings.connect_timeout_seconds,
    )


async def close_retrieval_service_client() -> None:
    client = get_retrieval_service_client()
    await client.aclose()
    get_retrieval_service_client.cache_clear()
