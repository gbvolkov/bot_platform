from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import (
    generate_batch_run,
    generate_run,
    mark_approved,
    mark_preview_passed,
    publish_run,
    validate_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ismart-content-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--request", required=True)
    generate_parser.add_argument("--out", required=True)

    batch_parser = subparsers.add_parser("generate-batch")
    batch_parser.add_argument("--request", required=True)
    batch_parser.add_argument("--out", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--run", required=True)

    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("--run", required=True)
    approve_parser.add_argument("--content-id", required=True)

    preview_parser = subparsers.add_parser("preview-pass")
    preview_parser.add_argument("--run", required=True)
    preview_parser.add_argument("--content-id", required=True)
    preview_parser.add_argument("--artifact", required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--run", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            run_dir = generate_run(Path(args.request), Path(args.out))
            print(run_dir)
        elif args.command == "generate-batch":
            run_dir = generate_batch_run(Path(args.request), Path(args.out))
            print(run_dir)
        elif args.command == "validate":
            validate_run(Path(args.run))
        elif args.command == "approve":
            mark_approved(Path(args.run), args.content_id)
        elif args.command == "preview-pass":
            mark_preview_passed(Path(args.run), args.content_id, args.artifact)
        elif args.command == "publish":
            publish_run(Path(args.run))
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
