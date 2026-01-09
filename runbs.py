from __future__ import annotations

import argparse
import os

import uvicorn


def _env_flag(name: str, default: bool) -> bool:
    """Convert env var to bool using common truthy strings."""
    raw = os.getenv(name)
    return default if raw is None else raw.lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bot service API (FastAPI + uvicorn).")
    parser.add_argument("--host", default=os.getenv("BOT_SERVICE_HOST", "0.0.0.0"), help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=int(os.getenv("BOT_SERVICE_PORT", "8000")), help="Port to listen on.")
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        default=_env_flag("BOT_SERVICE_RELOAD", True),
        help="Enable auto-reload (default: on).",
    )
    parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable auto-reload.")
    parser.add_argument(
        "--log-level",
        default=os.getenv("BOT_SERVICE_LOG_LEVEL", "info"),
        help="Uvicorn log level (debug, info, warning, error).",
    )

    args = parser.parse_args()

    uvicorn.run(
        "bot_service.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
