from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("prepare_gaz_agent")


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _default_registry_path() -> str:
    return os.environ.get("BOT_SERVICE_AGENT_CONFIG_PATH", "data/load.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the complete local runtime state required for gaz_agent.",
    )
    parser.add_argument("--agent-id", default="gaz_agent", help="Agent id to verify in registry. Default: gaz_agent")
    parser.add_argument(
        "--registry-path",
        default=_default_registry_path(),
        help="Agent registry path. Default: BOT_SERVICE_AGENT_CONFIG_PATH or data/load.json",
    )
    parser.add_argument("--collection-id", default="gaz", help="Document collection id. Default: gaz")
    parser.add_argument("--docs-root", default="data/gaz-docs", help="Source documents directory. Default: data/gaz-docs")
    parser.add_argument("--cache-root", default="data/gaz_index", help="Output cache directory. Default: data/gaz_index")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline HF/Transformers loading during index build.",
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild of the collection artifacts.")
    parser.add_argument(
        "--service-url",
        default=os.environ.get("GAZ_DOCUMENTS_SERVICE_URL", "http://127.0.0.1:8081"),
        help="KB service base URL used by gaz_agent. Default: GAZ_DOCUMENTS_SERVICE_URL or http://127.0.0.1:8081",
    )
    parser.add_argument(
        "--skip-service-check",
        action="store_true",
        help="Do not ping the KB service after building the local collection state.",
    )
    parser.add_argument(
        "--require-service",
        action="store_true",
        help="Fail unless the KB service is reachable for the built collection.",
    )
    parser.add_argument(
        "--service-timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for KB service ping. Default: 10",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def _load_registry_entry(registry_path: Path, agent_id: str) -> Dict[str, Any]:
    if not registry_path.exists() or not registry_path.is_file():
        return {
            "exists": False,
            "agent_registered": False,
            "registry_path": str(registry_path),
            "agent_entry": None,
        }
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    agents = payload.get("agents") or []
    entry = next((item for item in agents if item.get("id") == agent_id), None)
    return {
        "exists": True,
        "agent_registered": entry is not None,
        "registry_path": str(registry_path),
        "agent_entry": entry,
    }


def _build_collection(args: argparse.Namespace) -> Dict[str, Any]:
    from scripts.build_gaz_index import build_index

    build_args = argparse.Namespace(
        collection_id=args.collection_id,
        docs_root=args.docs_root,
        cache_root=args.cache_root,
        offline=args.offline,
        force=args.force,
        log_level=args.log_level,
    )
    return build_index(build_args)


def _check_service(service_url: str, collection_id: str, timeout_seconds: float) -> Dict[str, Any]:
    from agents.gaz_agent.documents import GazDocumentsClient, GazDocumentsClientError

    client = GazDocumentsClient(base_url=service_url, collection_id=collection_id, timeout_seconds=timeout_seconds)
    try:
        status = client.get_collection_status(collection_id)
    except GazDocumentsClientError as exc:
        return {
            "checked": True,
            "available": False,
            "base_url": service_url,
            "error": str(exc),
        }
    return {
        "checked": True,
        "available": True,
        "base_url": service_url,
        "status": status,
    }


def _next_steps(service_check: Dict[str, Any]) -> list[str]:
    steps: list[str] = []
    if not service_check.get("available"):
        steps.append("Start the KB service with `python -m services.kb_manager.app`.")
    steps.append("Ensure `GAZ_DOCUMENTS_SERVICE_URL` points to the KB service instance used by gaz_agent.")
    return steps


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    registry_path = _resolve_path(args.registry_path)
    registry = _load_registry_entry(registry_path, args.agent_id)
    collection = _build_collection(args)

    service_check: Dict[str, Any]
    if args.skip_service_check:
        service_check = {
            "checked": False,
            "available": False,
            "base_url": args.service_url,
        }
    else:
        service_check = _check_service(args.service_url, args.collection_id, args.service_timeout)

    status = collection.get("status") or {}
    artifacts_ready = bool(status.get("available")) and int(status.get("doc_count") or 0) > 0
    index_ready = bool(status.get("rag_index_built"))
    service_reachable = bool(service_check.get("available"))
    service_requirement_satisfied = (not args.require_service) or service_reachable
    agent_ready = (
        artifacts_ready
        and index_ready
        and registry.get("agent_registered", False)
        and service_requirement_satisfied
    )

    return {
        "agent_id": args.agent_id,
        "ready": agent_ready,
        "checks": {
            "registry_registered": registry.get("agent_registered", False),
            "artifacts_ready": artifacts_ready,
            "index_ready": index_ready,
            "service_checked": service_check.get("checked", False),
            "service_reachable": service_reachable,
            "service_requirement_satisfied": service_requirement_satisfied,
        },
        "registry": registry,
        "collection": collection,
        "service": service_check,
        "next_steps": _next_steps(service_check),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        payload = prepare(args)
    except FileNotFoundError as exc:
        LOG.error(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Failed to prepare gaz_agent runtime: %s", exc)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if not payload["checks"]["registry_registered"]:
        LOG.error("gaz_agent is not registered in %s", payload["registry"]["registry_path"])
        return 3
    if not payload["checks"]["artifacts_ready"]:
        LOG.error("GAZ collection manifest was not built successfully")
        return 4
    if not payload["checks"]["index_ready"]:
        LOG.error("GAZ vector index was not built successfully")
        return 5
    if args.require_service and not payload["checks"]["service_requirement_satisfied"]:
        LOG.error("KB service is required but unavailable at %s", args.service_url)
        return 6
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
