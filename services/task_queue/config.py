from __future__ import annotations

from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class TaskQueueSettings(BaseSettings):
    """Configuration shared by the task queue dispatcher and workers."""

    model_config = SettingsConfigDict(
        env_prefix="TASK_QUEUE_",
        validate_assignment=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    redis_url: str = Field(default="redis://localhost:6380/0", description="Redis connection URL.")
    queue_key: str = Field(default="agent:jobs", description="Redis list key used for job scheduling.")
    status_prefix: str = Field(
        default="agent:status:",
        description="Prefix for Redis hashes storing job status metadata.",
    )
    channel_prefix: str = Field(
        default="agent:events:",
        description="Prefix for Redis Pub/Sub channels streaming job events.",
    )
    job_ttl_seconds: int = Field(
        default=6 * 60 * 60,
        description="Seconds to retain job status/results after completion.",
    )
    chunk_char_limit: int = Field(
        default=600,
        description="Maximum number of characters per streamed chunk when emitting final text.",
    )
    sse_heartbeat_seconds: int = Field(
        default=10,
        description="Interval in seconds for emitting SSE heartbeat events when idle.",
    )
    worker_heartbeat_seconds: int = Field(
        default=5,
        description="Interval in seconds for workers to send heartbeat updates while processing.",
    )
    heartbeat_stale_after_seconds: int = Field(
        default=60,
        description="If no heartbeat is observed for this many seconds, the job is considered stale.",
    )
    watchdog_interval_seconds: int = Field(
        default=5,
        description="Interval in seconds for watchdog scans that detect stale heartbeats.",
    )
    bot_service_base_url: HttpUrl = Field(
        default="http://localhost:8000/api",
        description="Base URL of the internal bot service API.",
    )
    bot_request_timeout_seconds: float = Field(
        default=180.0,
        description="Worker timeout for requests sent to the bot service (<=0 disables read timeout).",
    )
    bot_connect_timeout_seconds: float = Field(
        default=10.0,
        description="Worker connect timeout for bot service requests.",
    )
    completion_wait_timeout_seconds: float = Field(
        default=210.0,
        description="Maximum seconds the proxy waits for a job to finish when streaming is disabled.",
    )


@lru_cache
def get_settings() -> TaskQueueSettings:
    return TaskQueueSettings()


settings = get_settings()
