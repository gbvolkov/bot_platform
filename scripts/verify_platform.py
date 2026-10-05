"""Run the deterministic release checks without downloading model assets."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("access", "worker"), default="worker")
    parser.add_argument("--with-local-ismart-assets", action="store_true")
    args = parser.parse_args()
    selected = ["tests/unit"] if args.scope == "worker" else [
        "tests/unit/test_platform_runs.py", "tests/unit/test_platform_boundaries.py",
        "tests/unit/test_task_queue_delivery.py"]
    if args.scope == "worker":
        # Inactive/removed implementation is explicitly outside this release.
        selected += ["--ignore=tests/unit/test_ismart_content_agent_mvp.py"]
        # This test intentionally constructs real neural privacy models. It is
        # an environment smoke check, separate from deterministic conformance.
        selected += ["--ignore=tests/unit/test_simulate_privacy_flow.py"]
        if not args.with_local_ismart_assets:
            selected += ["--ignore=tests/unit/test_ismart_materials_agent.py"]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OPENAI_API_KEY": "deterministic-test-placeholder",
        "OPENAI_API_KEY_PERSONAL": "deterministic-test-placeholder",
        "YA_API_KEY": "deterministic-test-placeholder", "YA_FOLDER_ID": "deterministic-test-folder"}
    (ROOT / ".tmp").mkdir(parents=True, exist_ok=True)
    return subprocess.call([sys.executable, "-m", "pytest", *selected, "-q", "--tb=short",
        "--basetemp=" + str(ROOT / ".tmp" / ("verify-" + uuid.uuid4().hex))], cwd=ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
