from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SalesLeadRetrievalServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SALES_LEAD_RETRIEVAL_SERVICE_",
        validate_assignment=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_title: str = Field(default="Sales Lead Retrieval Service")
    app_version: str = Field(default="0.1.0")
    api_base_path: str = Field(default="/api")
    base_url: str = Field(
        default="http://localhost:8010/api",
        description="Base URL used by clients that talk to the retrieval service API.",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/sales_lead_retrieval.sqlite",
        description="SQLAlchemy-compatible database URL for the retrieval service.",
    )
    sql_echo: bool = Field(default=False, description="Enable SQL echo for debugging.")
    request_timeout_seconds: float | None = Field(
        default=30.0,
        description="HTTP request timeout for retrieval-service clients.",
    )
    connect_timeout_seconds: float = Field(
        default=5.0,
        description="HTTP connect timeout for retrieval-service clients.",
    )
    poll_interval_seconds: float = Field(
        default=1.0,
        description="Seconds to wait before polling for the next queued retrieval job.",
    )


@lru_cache
def get_settings() -> SalesLeadRetrievalServiceSettings:
    return SalesLeadRetrievalServiceSettings()


settings = get_settings()
