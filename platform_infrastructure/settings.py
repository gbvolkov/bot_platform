from pydantic_settings import BaseSettings, SettingsConfigDict
from platform_contracts.catalog import DEFAULT_CATALOG_PATH
from pydantic import Field


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLATFORM_", env_file=".env", extra="ignore")
    database_path: str = "data/application.sqlite"
    catalog_path: str = str(DEFAULT_CATALOG_PATH)
    artifact_path: str = ".attachments_store/platform"
    worker_agent_ids: str = ""
    service_token: str = ""
    coordinator_url: str = "http://127.0.0.1:8001"
    legacy_url: str = "http://127.0.0.1:8002/api"
    execution_class: str = "interactive"
    lease_seconds: int = Field(default=30, ge=1)
    poll_seconds: float = Field(default=0.25, gt=0)
    redis_url: str | None = None

    def cohorts(self):
        return {name.strip() for name in self.worker_agent_ids.split(",") if name.strip()}
