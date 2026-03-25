from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path

from agents.sales_lead_agent.tools import (
    DocumentPreparationService,
    RunWorkspaceManager,
    SalesLeadAgentSettings,
)

DEFAULT_PURCHASE_ID = "32615827409"
DEFAULT_PURCHASE_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "sales_lead_agent"
    / "permanent_index"
    / "purchase_downloads"
    / DEFAULT_PURCHASE_ID
)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _build_settings(temp_root: Path) -> SalesLeadAgentSettings:
    shared_index_id = os.environ.get("SALES_LEAD_AGENT_SHARED_INDEX_ID", "sales_lead_permanent").strip()
    if not shared_index_id:
        shared_index_id = "sales_lead_permanent"

    embedding_provider = os.environ.get("SALES_LEAD_AGENT_EMBEDDING_PROVIDER", "openai").strip().lower()
    if not embedding_provider:
        embedding_provider = "openai"

    embedding_model = os.environ.get("SALES_LEAD_AGENT_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    if not embedding_model:
        embedding_model = "text-embedding-3-small"

    return SalesLeadAgentSettings(
        work_root=temp_root / "runs",
        permanent_index_root=temp_root / "permanent_index",
        shared_index_id=shared_index_id,
        procurement_search_template=os.environ.get(
            "SALES_LEAD_AGENT_PROCUREMENT_TEMPLATE",
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test",
        ),
        purchase_headless=_bool_env("SALES_LEAD_AGENT_PURCHASE_HEADLESS", True),
        open_source_max_concurrency=int(
            os.environ.get("SALES_LEAD_AGENT_OPEN_SOURCE_MAX_CONCURRENCY", "4")
        ),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        damia_api_key=os.environ.get("SALES_LEAD_AGENT_DAMIA_API_KEY", "").strip(),
        scoring_base_url=os.environ.get("SALES_LEAD_AGENT_SCORING_BASE_URL", "").strip().rstrip("/"),
        fssp_base_url=os.environ.get("SALES_LEAD_AGENT_FSSP_BASE_URL", "").strip().rstrip("/"),
    )


def _resolve_document_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()

    candidates = sorted(
        path.resolve()
        for path in DEFAULT_PURCHASE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".doc"
    )
    if len(candidates) != 1:
        found = [path.name for path in candidates]
        raise RuntimeError(
            f"Expected exactly one .doc file in {DEFAULT_PURCHASE_DIR}, found {len(candidates)}: {found}"
        )
    return candidates[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same document-preparation flow used by purchase_search_tool on one local document.",
    )
    parser.add_argument(
        "--file",
        dest="file_path",
        default = "C:\\Projects\\bot_platform\\data\\sales_lead_agent\\permanent_index\\purchase_downloads\\32615827409\\Документация.doc",
        help="Optional override for the document path. Defaults to the only .doc file in purchase 32615827409.",
    )
    parser.add_argument(
        "--source-url",
        dest="source_url",
        default=None,
        help="Optional provenance URL stored on the prepared document metadata.",
    )
    parser.add_argument(
        "--with-index",
        action="store_true",
        help="Also run the embedding/indexing step instead of skipping it.",
    )
    return parser.parse_args()


def _emit_payload(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> int:
    args = _parse_args()
    document_path = _resolve_document_path(args.file_path)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    temp_root = Path(tempfile.mkdtemp(prefix="sales-lead-doc-load-"))
    settings = _build_settings(temp_root)
    workspace = RunWorkspaceManager(settings).create_run()
    service = DocumentPreparationService(settings)

    if not args.with_index:
        service._index_documents = lambda **kwargs: None  # type: ignore[method-assign]

    payload: dict[str, object] = {
        "document_path": str(document_path),
        "purchase_id": document_path.parent.name,
        "workspace_root": str(workspace.root_dir),
        "permanent_index_root": str(settings.permanent_index_root),
        "indexing_enabled": args.with_index,
    }

    try:
        prepared = service.prepare_files(
            workspace=workspace,
            origin="purchase",
            bundle_id=document_path.parent.name,
            registry_number=document_path.parent.name,
            source_url=args.source_url,
            file_paths=[str(document_path)],
        )
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _emit_payload(payload)
        return 1

    payload.update(
        {
            "status": "success",
            "prepared_documents": [item.model_dump() for item in prepared],
        }
    )
    _emit_payload(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
