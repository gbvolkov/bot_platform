from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SalesLeadSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SALES_LEAD_",
        validate_assignment=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    export_dir: str = Field(
        default="data/lead_exports",
        description="Directory for persisted sales lead exports.",
    )
    document_store_dir: str = Field(
        default="data/sales_lead_documents",
        description="Directory for downloaded lead-source documents.",
    )
    source_seed_path: str = Field(
        default="data/sales_lead_source_seed.json",
        description="Optional JSON seed file used only as an offline fallback for sales lead adapters.",
    )
    whitelist_path: str = Field(
        default="data/sales_lead_source_whitelist.json",
        description="JSON whitelist of allowed open-source hosts for the sales lead agent.",
    )
    request_rules_path: str = Field(
        default="data/sales_lead_request_rules.json",
        description="Business-editable JSON rules for request understanding and dictionaries.",
    )
    scoring_rules_path: str = Field(
        default="data/sales_lead_scoring_rules.json",
        description="Business-editable JSON rules for lead scoring and prioritization.",
    )
    allow_seed_fallback: bool = Field(
        default=False,
        description="Allow JSON seed fallback when live external integrations are unavailable.",
    )
    user_agent: str = Field(
        default="bot_platform/sales_lead_agent",
        description="User-Agent for live HTTP requests.",
    )
    http_timeout_seconds: float = Field(default=20.0)
    verify_ssl: bool = Field(default=True)

    procurement_base_url: str = Field(default="https://zakupki.gov.ru")
    procurement_results_limit: int = Field(default=10, ge=1, le=50)
    procurement_search_path: str = Field(default="/epz/order/extendedsearch/results.html")
    procurement_search_sort_by: str = Field(default="UPDATE_DATE")
    procurement_page_depth: int = Field(default=1, ge=1, le=5)

    open_search_provider: str = Field(default="duckduckgo")
    open_search_results_limit: int = Field(default=10, ge=1, le=30)
    open_search_region: str = Field(default="ru-ru")
    open_search_max_pages: int = Field(default=10, ge=1, le=30)

    scoring_base_url: str | None = Field(default=None)
    scoring_api_key: str | None = Field(default=None)
    scoring_method: str = Field(default="GET")
    scoring_inn_param: str = Field(default="inn")
    scoring_auth_header: str = Field(default="X-API-Key")
    scoring_timeout_seconds: float = Field(default=20.0)

    fssp_base_url: str | None = Field(default=None)
    fssp_api_key: str | None = Field(default=None)
    fssp_method: str = Field(default="GET")
    fssp_inn_param: str = Field(default="inn")
    fssp_auth_header: str = Field(default="X-API-Key")
    fssp_timeout_seconds: float = Field(default=20.0)

    index_chunk_size: int = Field(default=1200, ge=300, le=4000)
    index_chunk_overlap: int = Field(default=150, ge=0, le=1000)
    index_search_limit: int = Field(default=5, ge=1, le=20)

    request_understanding_llm_enabled: bool = Field(
        default=True,
        description="Enable LLM-assisted refinement for free-form request understanding.",
    )
    request_understanding_llm_provider: str = Field(
        default="openai",
        description="LLM provider used for semantic request understanding fallback.",
    )
    request_understanding_llm_model: str = Field(
        default="nano",
        description="LLM model size used for semantic request understanding fallback.",
    )


@lru_cache
def get_settings() -> SalesLeadSettings:
    return SalesLeadSettings()


settings = get_settings()
