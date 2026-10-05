"""Cold-start compatibility imports must work while plugins load in parallel."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("facade", ["agents/utils.py", "agents/tools/yandex_search.py"])
def test_waiting_from_import_receives_complete_compatibility_module(facade):
    source = (ROOT / facade).read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = next(node.args[2].value for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "alias_module")
    code = '''
import importlib, importlib.abc, importlib.util, sys, threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
source, target = sys.argv[1:]
entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
marker = object()
class Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        if module.__name__ == 'compat_fixture':
            exec(source, module.__dict__)
        else:
            entered.set()
            assert release.wait(5)
            module.Marker = marker
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'compat_fixture', implementation}:
            return importlib.util.spec_from_loader(fullname, Loader())
implementation = target
parts = target.split('.')
for index in range(1, len(parts)):
    name = '.'.join(parts[:index])
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
sys.meta_path.insert(0, Finder())
lock = importlib._bootstrap._get_module_lock('compat_fixture')
acquire = lock.acquire
def observed_acquire():
    if threading.current_thread().name.endswith('_1'):
        waiting.set()
    return acquire()
lock.acquire = observed_acquire
def use_facade():
    from compat_fixture import Marker
    return Marker
with ThreadPoolExecutor(2) as executor:
    first = executor.submit(use_facade)
    assert entered.wait(5)
    second = executor.submit(use_facade)
    assert waiting.wait(5)
    release.set()
    assert first.result(5) is marker and second.result(5) is marker
assert importlib.import_module('compat_fixture') is sys.modules[target]
'''
    result = subprocess.run([sys.executable, "-c", code, source, target], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("agent_id", ["gaz_pricing_bi_int", "kpi_bi_int"])
def test_delegated_bi_settings_match_actual_constructor(agent_id):
    catalog = json.loads((ROOT / "config_defaults/agents.json").read_text(encoding="utf-8"))
    entry = next(item for item in catalog["agents"] if item["id"] == agent_id)
    tree = ast.parse((ROOT / "agents/bi_agent/bi_agent.py").read_text(encoding="utf-8"))
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "initialize_agent")
    supported = {arg.arg for arg in [*factory.args.args, *factory.args.kwonlyargs]}
    assert set(entry["params"]) <= supported
    assert entry["init_context"]["database_url"].startswith("sqlite:///data/")
    removed = json.loads((ROOT / "config_defaults/removed_ignored_settings.json").read_text(encoding="utf-8"))
    assert removed["agents"][agent_id]["locale"] == "ru"
