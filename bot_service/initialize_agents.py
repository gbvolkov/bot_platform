from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ModelType(str, Enum):
    GPT = "openai"
    MISTRAL = "mistral"


@dataclass(frozen=True)
class CheckpointSaver:
    backend: str


DEFAULTS: dict[str, Any] = {
    "provider": ModelType.GPT,
    "use_platform_store": False,
    "locale": "ru",
    "checkpoint_saver": None,
    "notify_on_reload": True,
    "role": "default",
}

_CHECKPOINT_SAVER_CACHE: dict[str, CheckpointSaver] = {}


def init_postgresql_saver() -> CheckpointSaver:
    return CheckpointSaver(backend="PostgreSQL")


def init_sqlite_saver() -> CheckpointSaver:
    return CheckpointSaver(backend="SQLite")


def init_redis_saver() -> CheckpointSaver:
    return CheckpointSaver(backend="Redis")


def normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def resolve_checkpoint_saver(value: Any) -> CheckpointSaver | None:
    if value is None:
        return None
    if isinstance(value, CheckpointSaver):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "~"}:
        return None

    key = normalize_key(text)
    if key in {"postgresql", "postgesql", "postgres", "pg"}:
        return _get_cached_saver("postgresql", init_postgresql_saver)
    if key in {"sqlite", "sqllite"}:
        return _get_cached_saver("sqlite", init_sqlite_saver)
    if key in {"redis"}:
        return _get_cached_saver("redis", init_redis_saver)
    raise ValueError(f"Unknown checkpoint_saver value: {value!r}")


def _get_cached_saver(key: str, init_fn) -> CheckpointSaver:
    if key not in _CHECKPOINT_SAVER_CACHE:
        _CHECKPOINT_SAVER_CACHE[key] = init_fn()
    return _CHECKPOINT_SAVER_CACHE[key]


def coerce_provider(value: Any) -> ModelType | str:
    if value is None:
        return ModelType.GPT
    if isinstance(value, ModelType):
        return value
    text = str(value).strip()
    for candidate in ModelType:
        if text.lower() == candidate.value or text.upper() == candidate.name:
            return candidate
    return text


def load_config() -> dict[str, Any]:
    config_path = find_config_path()
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    return parse_yaml(config_path.read_text(encoding="utf-8"))


def find_config_path() -> Path:
    candidates = [Path("load.yaml"), Path("load.json")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Expected load.yaml or load.json in the working directory.")


def parse_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
        raise ValueError("YAML config must be a mapping at the top level.")
    except ModuleNotFoundError:
        return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    entries = preprocess_yaml_lines(text)
    if not entries:
        return {}
    data, index = parse_block(entries, 0, entries[0][0])
    if index != len(entries):
        raise ValueError("Unexpected trailing YAML content.")
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at the top level.")
    return data


def preprocess_yaml_lines(text: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = strip_inline_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.lstrip(" ")
        entries.append((indent, content))
    return entries


def strip_inline_comment(line: str) -> str:
    result: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            result.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            result.append(ch)
        elif ch == "#" and not in_single and not in_double:
            break
        else:
            result.append(ch)
        i += 1
    return "".join(result).rstrip()


def parse_block(entries: list[tuple[int, str]], index: int, base_indent: int) -> tuple[Any, int]:
    if entries[index][1].startswith("- "):
        return parse_list(entries, index, base_indent)
    return parse_dict(entries, index, base_indent)


def parse_list(entries: list[tuple[int, str]], index: int, base_indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    i = index
    while i < len(entries):
        indent, content = entries[i]
        if indent < base_indent:
            break
        if indent != base_indent:
            raise ValueError("Invalid list indentation in YAML.")
        if not content.startswith("- "):
            raise ValueError("Expected list item in YAML.")
        item_text = content[2:].strip()
        if not item_text:
            i += 1
            if i < len(entries) and entries[i][0] > indent:
                value, i = parse_block(entries, i, entries[i][0])
            else:
                value = None
            items.append(value)
            continue

        key, value_str = split_key_value(item_text)
        if key is not None:
            item: dict[str, Any]
            if value_str == "":
                i += 1
                if i < len(entries) and entries[i][0] > indent:
                    value, i = parse_block(entries, i, entries[i][0])
                else:
                    value = None
                item = {key: value}
            else:
                item = {key: parse_scalar(value_str)}
                i += 1
            if i < len(entries) and entries[i][0] > indent:
                extra, i = parse_dict(entries, i, entries[i][0])
                item.update(extra)
            items.append(item)
            continue

        items.append(parse_scalar(item_text))
        i += 1
    return items, i


def parse_dict(entries: list[tuple[int, str]], index: int, base_indent: int) -> tuple[dict[str, Any], int]:
    data: dict[str, Any] = {}
    i = index
    while i < len(entries):
        indent, content = entries[i]
        if indent < base_indent:
            break
        if indent != base_indent:
            raise ValueError("Invalid mapping indentation in YAML.")
        key, value_str = split_key_value(content)
        if key is None:
            raise ValueError("Expected mapping entry in YAML.")
        if value_str == "":
            i += 1
            if i < len(entries) and entries[i][0] > indent:
                value, i = parse_block(entries, i, entries[i][0])
            else:
                value = None
        else:
            value = parse_scalar(value_str)
            i += 1
        data[key] = value
    return data, i


def split_key_value(text: str) -> tuple[str | None, str | None]:
    in_single = False
    in_double = False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            key = text[:idx].strip()
            value = text[idx + 1 :].strip()
            if not key:
                return None, None
            return key, value
    return None, None


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return unescape_string(value[1:-1], quote=value[0])
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null", "~"}:
        return None
    if is_int(value):
        return int(value)
    if is_float(value):
        return float(value)
    return value


def unescape_string(value: str, quote: str) -> str:
    if quote not in {'"', "'"}:
        return value
    value = value.replace("\\\\", "\\")
    if quote == '"':
        return value.replace('\\"', '"')
    return value.replace("\\'", "'")


def is_int(value: str) -> bool:
    if value.startswith(("-", "+")):
        value = value[1:]
    return value.isdigit()


def is_float(value: str) -> bool:
    try:
        float(value)
        return "." in value or "e" in value.lower()
    except ValueError:
        return False


def extract_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("settings") or config.get("config")
    if isinstance(settings, dict):
        return settings
    return {key: config[key] for key in DEFAULTS if key in config}


def extract_modules(config: dict[str, Any]) -> list[Any]:
    modules = config.get("modules") or config.get("agents")
    if modules is None:
        raise ValueError("Config must contain a 'modules' list.")
    if not isinstance(modules, list):
        raise ValueError("'modules' must be a list.")
    return modules


def parse_module_entry(entry: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        module_name = entry.get("module") or entry.get("name") or entry.get("path")
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError("Module entry must include a module name.")
        params = entry.get("params") or entry.get("settings")
        if isinstance(params, dict):
            return module_name, params
        overrides = {k: v for k, v in entry.items() if k not in {"module", "name", "path"}}
        return module_name, overrides
    raise ValueError(f"Unsupported module entry: {entry!r}")


def initialize_module(module_name: str, params: dict[str, Any]) -> Any:
    module = importlib.import_module(module_name)
    init_fn = getattr(module, "initialize_agent", None)
    if not callable(init_fn):
        raise AttributeError(f"{module_name} does not expose initialize_agent().")
    return init_fn(**params)


def main() -> None:
    config = load_config()
    settings = extract_settings(config)
    defaults = DEFAULTS.copy()
    defaults.update(settings)

    modules = extract_modules(config)
    agents = []
    for entry in modules:
        module_name, overrides = parse_module_entry(entry)
        params = defaults.copy()
        params.update(overrides)
        params["provider"] = coerce_provider(params.get("provider"))
        params["checkpoint_saver"] = resolve_checkpoint_saver(params.get("checkpoint_saver"))
        agents.append(initialize_module(module_name, params))

    print(f"Loaded {len(agents)} agent(s).")
    for agent in agents:
        print(f" - {agent!r}")


if __name__ == "__main__":
    main()
