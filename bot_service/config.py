from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the lightweight bot service."""

    model_config = SettingsConfigDict(
        env_prefix="BOT_SERVICE_",
        validate_assignment=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_title: str = Field(default="Bot Platform API")
    app_version: str = Field(default="0.1.0")
    database_url: str = Field(
        default="mysql+aiomysql://user:password@localhost:3306/bot_platform",
        description="SQLAlchemy compatible database URL.",
    )
    sql_echo: bool = Field(default=False, description="Enable SQL echo for debugging.")
    default_model_provider: Literal["openai", "yandex", "mistral", "gigachat"] = Field(
        default="openai", description="Default provider for agents when not specified explicitly."
    )
    default_user_role: str = Field(
        default="default",
        description="Fallback role passed to agents when user role is not supplied.",
    )
    allow_reset_command: bool = Field(
        default=True,
        description="Enable client-side reset command routing to agents.",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        description="Timeout applied to agent inference calls.",
    )
    api_base_path: str = Field(default="/api")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
