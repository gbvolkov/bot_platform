from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterator

LOG = logging.getLogger("gaz_index_builder")


def _resolve_path(value: str, *, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ((base_dir or Path.cwd()) / path).resolve()
    return path


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _models_config_context(models_config: str | None) -> Iterator[None]:
    if models_config:
        resolved = _resolve_path(models_config)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Models config does not exist: {resolved}")
        if resolved.name != "models.toml":
            raise ValueError("Models config must be named models.toml because rag_lib resolves that file name directly.")
        with _pushd(resolved.parent):
            yield
        return

    cwd_config = Path.cwd() / "models.toml"
    if cwd_config.exists():
        yield
        return

    package_config = resources.files("gaz_index_builder").joinpath("models.toml")
    with resources.as_file(package_config) as resolved:
        with _pushd(resolved.parent):
            yield


def _load_env_file(env_file: str | None, *, base_dir: Path) -> None:
    candidate: Path | None = None
    if env_file:
        candidate = _resolve_path(env_file, base_dir=base_dir)
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Environment file does not exist: {candidate}")
    else:
        default = base_dir / ".env"
        if default.exists() and default.is_file():
            candidate = default

    if candidate is None:
        return

    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency declared in package metadata
        raise ImportError("python-dotenv is required to load env files.") from exc

    load_dotenv(candidate, override=False)


def _apply_legacy_env_aliases() -> None:
    if "EMBEDDING_MODEL_NAME" not in os.environ and os.environ.get("EMBEDDING_MODEL"):
        os.environ["EMBEDDING_MODEL_NAME"] = os.environ["EMBEDDING_MODEL"]
    if "LLM_MODEL" not in os.environ and os.environ.get("GPT_MODEL"):
        os.environ["LLM_MODEL"] = os.environ["GPT_MODEL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full GAZ document manifest and vector index used by gaz_agent.",
    )
    parser.add_argument(
        "--collection-id",
        default="gaz",
        help="Target collection id. Default: gaz",
    )
    parser.add_argument(
        "--docs-root",
        default="data/gaz-docs",
        help="Source documents directory. Default: data/gaz-docs",
    )
    parser.add_argument(
        "--cache-root",
        default="data/gaz_index",
        help="Output cache/index directory. Default: data/gaz_index",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional env file to load before importing rag_lib settings. Default: .env in current directory if present.",
    )
    parser.add_argument(
        "--models-config",
        default=None,
        help="Optional path to models.toml. Default: current directory models.toml, then package bundled models.toml.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline HF/Transformers loading during index build.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a rebuild even if cached artifacts already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def build_index(args: argparse.Namespace) -> Dict[str, Any]:
    invocation_dir = Path.cwd()
    docs_root = _resolve_path(args.docs_root, base_dir=invocation_dir)
    cache_root = _resolve_path(args.cache_root, base_dir=invocation_dir)

    _load_env_file(args.env_file, base_dir=invocation_dir)
    _apply_legacy_env_aliases()

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not docs_root.exists() or not docs_root.is_dir():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_root}")

    with _models_config_context(args.models_config):
        from .gaz_runtime import GazRuntimeService

        service = GazRuntimeService(
            docs_root=docs_root,
            cache_root=cache_root,
        )
        rebuild_result = service.rebuild_collection(collection_id=args.collection_id, force=args.force)
        status = service.collection_status(args.collection_id)

    return {
        "collection_id": args.collection_id,
        "docs_root": str(docs_root),
        "cache_root": str(cache_root),
        "offline": args.offline,
        "rebuild": rebuild_result,
        "status": status,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        payload = build_index(args)
    except FileNotFoundError as exc:
        LOG.error(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Failed to build GAZ index: %s", exc)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
