import ast
import json
import subprocess
import sys
from pathlib import Path

from platform_contracts.catalog import AgentCatalog

ROOT = Path(__file__).resolve().parents[2]


def imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_contracts_are_standard_library_only():
    for path in (ROOT / "platform_contracts").glob("*.py"):
        assert set(imported_roots(path)) <= sys.stdlib_module_names


def test_application_does_not_import_adapters():
    for path in (ROOT / "platform_application").glob("*.py"):
        assert set(imported_roots(path)) <= sys.stdlib_module_names | {"platform_contracts"}


def test_public_process_and_capabilities_do_not_import_business_agents():
    for package in ("platform_access", "platform_client", "platform_capabilities", "platform_tools", "platform_guardrails", "services/kb_manager", "services/sales_lead_retrieval"):
        for path in (ROOT / package).rglob("*.py"):
            if "build" in path.parts:
                continue
            assert not (set(imported_roots(path)) & {"agents", "bot_service"}), str(path)
    config = json.loads((ROOT / "platform_tools/tools.json").read_text(encoding="utf-8"))
    for tool in config["internal_tools"]:
        assert not (tool.get("import") or tool.get("factory")).startswith("agents.")


def test_all_active_ids_are_in_metadata_without_importing_plugins():
    catalog = AgentCatalog.load(ROOT / "config_defaults/agents.json")
    active = {a.id for a in catalog.agents.values() if a.active}
    assert active == {"kpi_agent", "marketing_analyst", "mycroft_gaz_agent", "new_theodor_agent",
        "theodor_agent", "ideator", "new_ideator", "ismart_tutor_agent", "artifact_creator_agent",
        "simple_agent", "simple_agent_en", "ismart_task_variator_agent", "product_Car", "gaz_agent",
        "sales_lead_agent", "ismart_generator_agent", "sysadmin_agent"}
    assert catalog.get("ismart_generator_agent").execution_class == "batch"


def test_public_import_blocks_agent_and_model_dependencies():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'agents','agent_runtime','langchain','langchain_core','langgraph','torch','transformers','bot_service'}:
            raise AssertionError('Forbidden public dependency: '+fullname)
sys.meta_path.insert(0, Block())
import platform_access.main, platform_access.coordinator, openai_proxy.main
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
