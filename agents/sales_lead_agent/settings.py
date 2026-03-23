from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import config


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _configured_string(
    env_name: str,
    *,
    default: str | None = None,
    normalize_lower: bool = False,
) -> str:
    raw = os.environ.get(env_name)
    if raw is None:
        value = default
    else:
        value = raw.strip()
    if value is None or value == "":
        raise RuntimeError(f"{env_name} must be configured and must not be blank.")
    return value.lower() if normalize_lower else value


@dataclass(frozen=True)
class SalesLeadAgentSettings:
    work_root: Path
    retention_hours: int
    damia_api_key: str
    scoring_base_url: str
    fssp_base_url: str
    purchase_headless: bool
    open_source_max_concurrency: int
    procurement_search_template: str
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"


@lru_cache(maxsize=1)
def get_settings() -> SalesLeadAgentSettings:
    work_root = Path(
        os.environ.get("SALES_LEAD_AGENT_WORK_ROOT", "./data/sales_lead_agent/runs")
    ).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    embedding_provider = _configured_string(
        "SALES_LEAD_AGENT_EMBEDDING_PROVIDER",
        default="openai",
        normalize_lower=True,
    )
    embedding_model = _configured_string(
        "SALES_LEAD_AGENT_EMBEDDING_MODEL",
        default="text-embedding-3-small",
    )
    if embedding_provider == "openai" and not (config.OPENAI_API_KEY or "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY must be configured when SALES_LEAD_AGENT_EMBEDDING_PROVIDER=openai."
        )
    return SalesLeadAgentSettings(
        work_root=work_root,
        retention_hours=int(os.environ.get("SALES_LEAD_AGENT_RETENTION_HOURS", "72")),
        damia_api_key=os.environ.get("SALES_LEAD_AGENT_DAMIA_API_KEY", ""),
        scoring_base_url=os.environ.get("SALES_LEAD_AGENT_SCORING_BASE_URL", "").rstrip("/"),
        fssp_base_url=os.environ.get("SALES_LEAD_AGENT_FSSP_BASE_URL", "").rstrip("/"),
        purchase_headless=_as_bool(
            os.environ.get("SALES_LEAD_AGENT_PURCHASE_HEADLESS"),
            True,
        ),
        open_source_max_concurrency=int(
            os.environ.get("SALES_LEAD_AGENT_OPEN_SOURCE_MAX_CONCURRENCY", "4")
        ),
        procurement_search_template=os.environ.get(
            "SALES_LEAD_AGENT_PROCUREMENT_TEMPLATE",
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            "?searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2%D0%B0%D0%BD"
            "&morphology=on"
            "&search-filter=%D0%94%D0%B0%D1%82%D0%B5+%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%89%D0%B5%D0%BD%D0%B8%D1%8F"
            "&pageNumber=1"
            "&sortDirection=false"
            "&recordsPerPage=_2"
            "&showLotsInfoHidden=false"
            "&sortBy=UPDATE_DATE"
            "&fz44=on"
            "&fz223=on"
            "&af=on"
            "&currencyIdGeneral=-1"
            "&gws=%D0%92%D1%8B%D0%B1%D0%B5%D1%80%D0%B8%D1%82%D0%B5+%D1%82%D0%B8%D0%BF+%D0%B7%D0%B0%D0%BA%D1%83%D0%BF%D0%BA%D0%B8"
        ),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
