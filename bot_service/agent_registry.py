from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from collections.abc import AsyncIterator

#from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.utils import ModelType

from .config import settings
from .initialize_agents import parse_yaml
from .schemas import AgentInfo, ContentType


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    name: str
    description: str
    factory: Callable[..., Any]
    default_provider: ModelType
    supported_content_types: Tuple[ContentType, ...]
    allow_raw_attachments: bool = False
    stream_modes: Tuple[str, ...] = ("messages", "values")
    stream_subgraphs: bool = False
    is_active: bool = True
    init_params: Dict[str, Any] = field(default_factory=dict)
    checkpoint_saver: Any = None
    param_names: frozenset[str] = field(default_factory=frozenset)
    accepts_kwargs: bool = False


_SKIP_CHECKPOINTER = object()


def _resolve_config_path(raw_path: str | None) -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    if not raw_path:
        raw_path = "bot_service/load.json"
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_agent_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = parse_yaml(text)
    if not isinstance(data, dict):
        raise ValueError("Agent config must be a mapping at the top level.")
    return data


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _normalize_checkpoint_saver(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "~"}:
            return None
        return _normalize_key(text)
    return _normalize_key(str(value))


def _coerce_provider(value: Any, default: ModelType) -> ModelType:
    if value is None:
        return default
    if isinstance(value, ModelType):
        return value
    text = str(value).strip()
    if not text:
        return default
    for candidate in ModelType:
        if text.lower() == candidate.value.lower() or text.upper() == candidate.name:
            return candidate
    raise ValueError(f"Unknown provider value: {value!r}")


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Agent config field '{field_name}' must be a non-empty string.")
    return value


def _coerce_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"Agent config field '{field_name}' must be a boolean.")


def _coerce_content_type(value: Any) -> ContentType:
    if isinstance(value, ContentType):
        return value
    if isinstance(value, str):
        return ContentType(value)
    raise ValueError(f"Unsupported content type value: {value!r}")


def _parse_supported_content_types(
    value: Any,
    default: Tuple[ContentType, ...],
) -> Tuple[ContentType, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(_coerce_content_type(item) for item in value)
    raise ValueError("Agent config field 'supported_content_types' must be a list.")


def _parse_streaming_config(
    value: Any,
    default_modes: Tuple[str, ...],
    default_subgraphs: bool,
) -> tuple[Tuple[str, ...], bool]:
    if value is None:
        return default_modes, default_subgraphs
    if not isinstance(value, dict):
        raise ValueError("Agent config field 'streaming' must be a mapping.")
    modes_value = value.get("modes")
    modes = default_modes
    if modes_value is not None:
        if not isinstance(modes_value, (list, tuple)) or not modes_value:
            raise ValueError("Agent config field 'streaming.modes' must be a non-empty list.")
        modes = tuple(_require_str(item, "streaming.modes[]") for item in modes_value)
    subgraphs = _coerce_bool(value.get("subgraphs"), "streaming.subgraphs", default_subgraphs)
    return modes, subgraphs


def _import_initialize_agent(module_path: str) -> Callable[..., Any]:
    module = importlib.import_module(module_path)
    init_fn = getattr(module, "initialize_agent", None)
    if not callable(init_fn):
        raise AttributeError(f"{module_path} does not expose initialize_agent().")
    return init_fn


def _get_signature_info(init_fn: Callable[..., Any]) -> tuple[frozenset[str], bool]:
    signature = inspect.signature(init_fn)
    params = signature.parameters.values()
    param_names = frozenset(signature.parameters.keys())
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    return param_names, accepts_kwargs


def _build_definitions_from_config(
    config: dict[str, Any],
    default_provider: ModelType,
    default_content_types: Tuple[ContentType, ...],
) -> Dict[str, AgentDefinition]:
    entries = config.get("agents") or config.get("modules")
    if entries is None:
        raise ValueError("Agent config must contain an 'agents' list.")
    if not isinstance(entries, list):
        raise ValueError("Agent config field 'agents' must be a list.")

    definitions: Dict[str, AgentDefinition] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each agent entry must be a mapping.")
        agent_id = _require_str(entry.get("id"), "id")
        if agent_id in definitions:
            raise ValueError(f"Duplicate agent id '{agent_id}' in config.")
        name = _require_str(entry.get("name"), "name")
        description = _require_str(entry.get("description"), "description")
        module_path = _require_str(entry.get("module") or entry.get("path"), "module")
        supported_content_types = _parse_supported_content_types(
            entry.get("supported_content_types"),
            default_content_types,
        )
        allow_raw_attachments = _coerce_bool(
            entry.get("allow_raw_attachments"),
            "allow_raw_attachments",
            False,
        )
        stream_modes, stream_subgraphs = _parse_streaming_config(
            entry.get("streaming"),
            ("messages", "values"),
            False,
        )
        is_active = _coerce_bool(entry.get("is_active"), "is_active", True)
        params = entry.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"Agent '{agent_id}' params must be a mapping.")
        params = dict(params)
        provider = _coerce_provider(params.pop("provider", None), default_provider)
        checkpoint_saver = params.pop("checkpoint_saver", None)
        init_fn = _import_initialize_agent(module_path)
        param_names, accepts_kwargs = _get_signature_info(init_fn)
        definitions[agent_id] = AgentDefinition(
            id=agent_id,
            name=name,
            description=description,
            factory=init_fn,
            default_provider=provider,
            supported_content_types=supported_content_types,
            allow_raw_attachments=allow_raw_attachments,
            stream_modes=stream_modes,
            stream_subgraphs=stream_subgraphs,
            is_active=is_active,
            init_params=params,
            checkpoint_saver=checkpoint_saver,
            param_names=param_names,
            accepts_kwargs=accepts_kwargs,
        )
    return definitions




class AgentRegistry:
    def __init__(self) -> None:
        default_provider = _coerce_provider(settings.default_model_provider, ModelType.GPT)
        default_content_types: Tuple[ContentType, ...] = (
            #ContentType.TEXT_FILES,
            #ContentType.MARKDOWN,
            #ContentType.DOCX_DOCUMENTS,
            #ContentType.PDFS,
            #ContentType.CSVS,
            #ContentType.EXCELS,
        )
        
        #persistent checkpointers
        self._checkpointer_cm: AsyncIterator[AsyncSqliteSaver] | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._checkpointer_lock = asyncio.Lock()

        config_path = _resolve_config_path(settings.agent_config_path)
        config = _load_agent_config(config_path)
        self._definitions = _build_definitions_from_config(
            config,
            default_provider,
            default_content_types,
        )
        LOG.info("Loaded %d agent definitions from %s", len(self._definitions), config_path)
        self._instances: Dict[str, Any] = {}
        self._init_tasks: Dict[str, Future] = {}
        self._init_errors: Dict[str, BaseException] = {}

    def list_agents(self) -> List[AgentInfo]:
        return [
            AgentInfo(
                id=definition.id,
                name=definition.name,
                description=definition.description,
                provider=definition.default_provider.value,
                supported_content_types=list(definition.supported_content_types),
            )
            for definition in self._definitions.values()
            if definition.is_active
        ]

    def list_ready_agents(self) -> List[AgentInfo]:
        ready_ids = set(self._instances)
        return [
            AgentInfo(
                id=definition.id,
                name=definition.name,
                description=definition.description,
                provider=definition.default_provider.value,
                supported_content_types=list(definition.supported_content_types),
            )
            for definition in self._definitions.values()
            if definition.id in ready_ids and definition.is_active
        ]

    async def _ensure_checkpointer(self) -> AsyncSqliteSaver:
        """Create a single shared SQLite checkpointer (lazy, once per process)."""
        if self._checkpointer is not None:
            return self._checkpointer

        async with self._checkpointer_lock:
            if self._checkpointer is not None:
                return self._checkpointer

            # settings.checkpoint_sqlite_path example: "data/checkpoints.sqlite"
            conn_string = settings.checkpointer_db_url
            cm = AsyncSqliteSaver.from_conn_string(conn_string)

            # Enter once and keep it open for the whole app lifetime
            saver = await cm.__aenter__()
            await saver.setup()  # idempotent

            self._checkpointer_cm = cm
            self._checkpointer = saver
            return saver

    async def aclose(self) -> None:
        """Call on app shutdown to close the shared checkpointer."""
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(None, None, None)
        self._checkpointer_cm = None
        self._checkpointer = None

    def _start_initialization(self, agent_id: str) -> None:
        definition = self._definitions[agent_id]
        provider = definition.default_provider

        async def build_async() -> Any:
            params = dict(definition.init_params)
            params["provider"] = provider

            checkpoint_key = _normalize_checkpoint_saver(definition.checkpoint_saver)
            checkpoint_saver = _SKIP_CHECKPOINTER
            if checkpoint_key in {"sqlite", "sqllite"}:
                checkpoint_saver = await self._ensure_checkpointer()
            elif checkpoint_key is not None:
                LOG.info(
                    "Agent '%s' checkpoint_saver '%s' not supported yet; skipping.",
                    agent_id,
                    definition.checkpoint_saver,
                )

            if checkpoint_saver is not _SKIP_CHECKPOINTER:
                params["checkpoint_saver"] = checkpoint_saver

            if not definition.accepts_kwargs:
                params = {key: value for key, value in params.items() if key in definition.param_names}

            loop = asyncio.get_running_loop()

            # Keep your current pattern: build agent in executor thread
            return await loop.run_in_executor(None, lambda: definition.factory(**params))

        loop = asyncio.get_running_loop()
        task = loop.create_task(build_async())

        def on_done(fut: Future) -> None:
            try:
                instance = fut.result()
            except BaseException as exc:  # noqa: BLE001
                LOG.exception("Agent '%s' initialization failed", agent_id)
                self._init_errors[agent_id] = exc
            else:
                self._instances[agent_id] = instance
                self._init_errors.pop(agent_id, None)
                LOG.info("Agent '%s' initialization complete.", agent_id)
            finally:
                self._init_tasks.pop(agent_id, None)

        task.add_done_callback(on_done)
        self._init_tasks[agent_id] = task

    async def ensure_agent_ready(self, agent_id: str) -> bool:
        if agent_id not in self._definitions:
            raise KeyError(f"Unknown agent '{agent_id}'")
        if agent_id in self._instances:
            return True
        if agent_id in self._init_errors:
            exc = self._init_errors.pop(agent_id)
            raise RuntimeError(f"Failed to initialize agent '{agent_id}'") from exc
        task = self._init_tasks.get(agent_id)
        if task is None:
            self._start_initialization(agent_id)
            return False
        if task.done():
            try:
                instance = task.result()
            except BaseException as exc:  # noqa: BLE001
                self._init_errors[agent_id] = exc
                self._init_tasks.pop(agent_id, None)
                raise RuntimeError(f"Failed to initialize agent '{agent_id}'") from exc
            else:
                self._instances[agent_id] = instance
                self._init_tasks.pop(agent_id, None)
                self._init_errors.pop(agent_id, None)
                return True
        return False

    def get_agent(self, agent_id: str) -> Any:
        if agent_id not in self._definitions:
            raise KeyError(f"Unknown agent '{agent_id}'")
        if agent_id in self._instances:
            return self._instances[agent_id]
        if agent_id in self._init_errors:
            raise RuntimeError(f"Agent '{agent_id}' failed to initialize") from self._init_errors[agent_id]
        raise RuntimeError(f"Agent '{agent_id}' is still initializing")

    def is_ready(self, agent_id: str) -> bool:
        return agent_id in self._instances

    def initialization_status(self, agent_id: str) -> str:
        if agent_id in self._instances:
            return "ready"
        if agent_id in self._init_errors:
            return "error"
        if agent_id in self._init_tasks:
            return "initializing"
        if agent_id in self._definitions:
            return "pending"
        return "unknown"

    def supported_content_types(self, agent_id: str) -> Tuple[ContentType, ...]:
        if agent_id not in self._definitions:
            raise KeyError(f"Unknown agent '{agent_id}'")
        return self._definitions[agent_id].supported_content_types

    def allows_raw_attachments(self, agent_id: str) -> bool:
        if agent_id not in self._definitions:
            raise KeyError(f"Unknown agent '{agent_id}'")
        return self._definitions[agent_id].allow_raw_attachments

    def stream_config(self, agent_id: str | None) -> tuple[List[str], bool]:
        if not agent_id:
            return ["messages", "values"], False
        definition = self._definitions.get(agent_id)
        if definition is None:
            return ["messages", "values"], False
        return list(definition.stream_modes), definition.stream_subgraphs

    def preload_all(self) -> None:
        for agent_id in self._definitions:
            if agent_id in self._instances:
                continue
            if agent_id in self._init_tasks:
                continue
            if agent_id in self._init_errors:
                self._init_errors.pop(agent_id, None)
            self._start_initialization(agent_id)


agent_registry = AgentRegistry()
