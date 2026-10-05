"""Run with python -I, after installing wheels, from outside the checkout."""
import argparse
import importlib.abc
import importlib.util
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("component", choices=("access", "capabilities"))
parser.add_argument("--target", type=Path, help="Optional wheel installation target directory")
args = parser.parse_args()
if args.target:
    sys.path.insert(0, str(args.target.resolve()))

forbidden = {"agents", "bot_service"}
if args.component == "access":
    forbidden |= {"agent_runtime", "langgraph", "langchain", "langchain_core", "torch", "transformers"}
for package in forbidden:
    assert importlib.util.find_spec(package) is None, f"Unexpected installed dependency: {package}"


class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden:
            raise AssertionError("Forbidden installed-package import: " + fullname)


sys.meta_path.insert(0, Boundary())
if args.component == "access":
    import platform_access.main
    import platform_access.coordinator
    import openai_proxy.main
    import web_chat.main
    from platform_contracts.catalog import AgentCatalog, DEFAULT_CATALOG_PATH
    assert len([a for a in AgentCatalog.load(DEFAULT_CATALOG_PATH).agents.values() if a.active]) == 17
    assert "/api/runs/{run_id}" in platform_access.main.app.openapi()["paths"]
else:
    import platform_capabilities.models
    import platform_capabilities.procurement
    import platform_capabilities.ingestion
    import services.sales_lead_retrieval.main
    import services.kb_manager.app
    from platform_guardrails.config import load_guardrail_policy_config
    assert load_guardrail_policy_config()["policies"]
print(f"Installed {args.component} package boundaries passed.")
