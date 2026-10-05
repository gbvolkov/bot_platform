"""Generate component build metadata and lock subsets from the frozen uv lock.

No dependency resolution or upgrades. Re-run after intentionally updating the
root lock; review generated metadata and lockfiles together.
"""
import json
from pathlib import Path
import tomllib
import subprocess

ROOT = Path(__file__).resolve().parents[1]

COMPONENTS = {
    "platform_contracts": ("contracts", [], []),
    "platform_sdk": ("sdk", ["contracts"], ["pydantic"]),
    "platform_application": ("application", ["contracts"], []),
    "platform_infrastructure": ("infrastructure", ["application"], ["pydantic-settings", "redis"]),
    "platform_client": ("client", ["contracts", "sdk"], ["httpx"]),
    "platform_access": ("access", ["application", "infrastructure", "client", "sdk"], ["fastapi", "uvicorn"]),
    "agent_runtime": ("runtime", ["contracts", "sdk", "client", "infrastructure", "capabilities"], ["langgraph", "langgraph-checkpoint-sqlite", "langchain-core", "pydantic-settings"]),
    "platform_capabilities": ("capabilities", ["contracts", "sdk", "client"], ["langchain", "langchain-openai", "langchain-mistralai", "boto3", "xhtml2pdf", "rag-lib", "torch", "sentence-transformers", "sqlalchemy", "aiosqlite", "palimpsest", "llm-guard", "langchain-mcp-adapters", "telegramify-markdown",
        "fastapi", "uvicorn", "httpx", "pydantic-settings", "langchain-core", "langchain-community",
        "langchain-huggingface", "langchain-classic", "langchain-text-splitters", "langgraph", "numpy", "pandas",
        "requests", "chardet", "filelock", "mistletoe", "scikit-learn", "umap-learn", "trafilatura",
        "duckduckgo-search", "assemblyai", "pydub", "rank-llm", "zakupki-crawler", "langfuse"]),
    "agents": ("agents", ["runtime", "capabilities"], ["deepagents"]),
    "generators": ("generators", ["runtime"], ["httpx"]),
    "openai_proxy": ("openai-proxy", ["client", "sdk"], ["fastapi", "httpx", "uvicorn", "pydantic-settings"]),
    "web_chat": ("web-chat", ["client", "sdk"], ["fastapi", "httpx", "uvicorn", "pydantic-settings", "jinja2"]),
}


def main():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {p["name"]: p for p in lock["package"]}
    for folder, (name, local, external) in COMPONENTS.items():
        deps = [f"bot-platform-{dep}==0.1.0" for dep in local]
        deps += [f"{dep}=={packages[dep]['version']}" for dep in external]
        included = [f"{folder}*"]
        if folder == "platform_contracts":
            included += ["config_defaults*"]
        if folder == "agents":
            included += ["data.config.mycroft*"]
        if folder == "platform_capabilities":
            included += ["platform_guardrails*", "platform_tools*", "platform_utils*", "user_manager*",
                         "services.kb_manager*", "services.sales_lead_retrieval*", "data.config.guardrails*"]
        content = f'''[build-system]
requires = ["setuptools==82.0.1"]
build-backend = "setuptools.build_meta"

[project]
name = "bot-platform-{name}"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
dependencies = {json.dumps(deps)}

[tool.setuptools.packages.find]
where = [".."]
include = {json.dumps(included)}
namespaces = true
exclude = ["*.build", "*.build.*", "*.tests", "*.tests.*"]

[tool.setuptools.package-data]
"*" = ["*.json", "*.toml", "*.md", "*.txt", "*.yaml", "*.html", "static/*", "templates/*"]
'''
        if folder == "platform_capabilities":
            content += '\n[tool.setuptools]\npy-modules = ["config"]\npackage-dir = {"" = ".."}\n'
        (ROOT / folder / "pyproject.toml").write_text(content, encoding="utf-8")
    # Export an independently installable access environment from the SAME lock.
    roots = {"fastapi", "httpx", "uvicorn", "pydantic-settings", "redis", "pytest", "setuptools", "jinja2"}
    seen = set()
    def visit(name):
        if name in seen:
            return
        seen.add(name)
        for dep in packages[name].get("dependencies", []):
            visit(dep["name"])
    for name in roots:
        visit(name)
    deploy = ROOT / "deploy"
    deploy.mkdir(exist_ok=True)
    lines = ["# Access/test dependency closure from uv.lock; no model frameworks."]
    for name in sorted(seen):
        package = packages[name]
        marker = '; sys_platform == "win32"' if name == "colorama" else ''
        lines.append(f"{name}=={package['version']}{marker}")
    (deploy / "requirements.access.lock").write_text("\n".join(lines)+"\n", encoding="utf-8")
    baseline = subprocess.run(["uv", "export", "--frozen", "--no-dev", "--format", "requirements-txt", "--no-hashes"],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    (deploy / "requirements.worker.lock").write_text(baseline, encoding="utf-8")
    retrieval_roots = set(COMPONENTS["platform_capabilities"][2]) | {"rag-lib", "fastapi", "uvicorn", "httpx", "pydantic-settings", "langchain",
        "langchain-core", "langchain-community", "langchain-huggingface", "langchain-openai", "langchain-mistralai",
        "langgraph", "pandas", "python-dotenv", "openpyxl", "chardet", "numpy", "zakupki-crawler",
        "boto3", "xhtml2pdf", "palimpsest", "llm-guard", "sentence-transformers", "unstructured",
        "sqlalchemy", "aiosqlite", "langchain-mcp-adapters", "telegramify-markdown", "langgraph-checkpoint-sqlite"}
    seen.clear()
    for name in retrieval_roots:
        visit(name)
    import re
    selected = ["# Retrieval dependency closure; excludes business-agent packages."]
    for line in baseline.splitlines():
        if line.startswith("./vendor/") and "llm-guard" in seen:
            selected.append(line)
        match = re.match(r"^([A-Za-z0-9_-]+)(?:==|\[| @ )", line)
        if match and match[1].lower().replace("_", "-") in seen:
            selected.append(line)
    (deploy / "requirements.retrieval.lock").write_text("\n".join(selected)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
