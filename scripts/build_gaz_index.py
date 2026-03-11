from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from services.kb_manager.gaz_runtime import GazRuntimeService

LOG = logging.getLogger("build_gaz_index")


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


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
    docs_root = _resolve_path(args.docs_root)
    cache_root = _resolve_path(args.cache_root)

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not docs_root.exists() or not docs_root.is_dir():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_root}")

    from services.kb_manager.gaz_runtime import GazRuntimeService

    service = GazRuntimeService(
        docs_root=docs_root,
        cache_root=cache_root,
        default_collection_id=args.collection_id,
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
