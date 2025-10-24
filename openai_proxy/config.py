from __future__ import annotations

from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_PROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True,
    )

    bot_service_base_url: HttpUrl = Field(
        default="http://localhost:8000/api",
        description="Base URL of the bot_service API.",
    )
    default_user_id: str = Field(
        default="openai-proxy",
        description="Fallback user identifier passed to bot_service.",
    )
    default_user_role: str = Field(
        default="default",
        description="Fallback user role for bot_service conversation config.",
    )
    request_timeout_seconds: float = Field(
        default=180.0,
        description="Total timeout for requests to bot_service.",
    )
    connect_timeout_seconds: float = Field(
        default=10.0,
        description="Connection timeout for requests to bot_service.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

