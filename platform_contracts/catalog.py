"""Parse metadata without importing plugins, providers, or frameworks."""
import hashlib
import json
from pathlib import Path

from . import AgentDescriptor, ContentType, ModelType

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "config_defaults" / "agents.json"


class AgentCatalog:
    def __init__(self, config: dict):
        entries = config.get("agents", config.get("modules"))
        if not isinstance(entries, list):
            raise ValueError("Manifest must contain an agents list")
        self.agents = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Agent entry must be an object")
            for key in ("id", "name", "description", "module"):
                if not isinstance(entry.get(key), str) or not entry[key].strip():
                    raise ValueError(f"Missing agent {key}")
            if entry["id"] in self.agents:
                raise ValueError(f"Duplicate agent ID: {entry['id']}")
            settings = entry.get("params", {})
            if not isinstance(settings, dict):
                raise ValueError("Agent params must be an object")
            provider = settings.get("provider", "openai")
            if provider not in {m.value for m in ModelType}:
                raise ValueError(f"Unknown provider: {provider}")
            content_types = entry.get("supported_content_types", [])
            for value in content_types:
                ContentType(value)
            execution = entry.get("execution_class", "interactive")
            if execution not in {"interactive", "batch"}:
                raise ValueError(f"Invalid execution class: {execution}")
            for key in ("capabilities", "required_services"):
                value = entry.get(key, [])
                if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
                    raise ValueError(f"{key} must be a list of names")
            for key in ("is_active", "privacy_affinity"):
                if key in entry and not isinstance(entry[key], bool):
                    raise ValueError(f"{key} must be boolean")
            revision = entry.get("revision") or hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError("Agent revision must be a nonempty string")
            exporter = entry.get("artifact_exporter")
            if exporter is not None and (not isinstance(exporter, str) or exporter.count(":") != 1):
                raise ValueError("Artifact exporter must name a module:callable")
            self.agents[entry["id"]] = AgentDescriptor(
                id=entry["id"], name=entry["name"], description=entry["description"],
                implementation=entry["module"], revision=revision, settings=settings,
                capabilities=tuple(entry.get("capabilities", [])),
                required_services=tuple(entry.get("required_services", [])),
                execution_class=execution, active=entry.get("is_active", True),
                privacy_affinity=entry.get("privacy_affinity", False),
                supported_content_types=tuple(content_types), configuration=entry,
            )

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, agent_id: str) -> AgentDescriptor:
        descriptor = self.agents[agent_id]
        if not descriptor.active:
            raise KeyError(f"Inactive agent: {agent_id}")
        return descriptor
