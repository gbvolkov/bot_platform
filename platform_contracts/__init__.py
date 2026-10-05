"""Wire and plugin contracts. Standard library only."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Mapping, Protocol


class ContentType(str, Enum):
    IMAGES = "images"
    PDFS = "pdfs"
    TEXT_FILES = "text_files"
    MARKDOWN = "mds"
    DOCX_DOCUMENTS = "docx_documents"
    CSVS = "csvs"
    EXCELS = "excels"
    SOUNDS = "sounds"
    VIDEOS = "videos"
    JSONS = "jsons"


class ModelType(Enum):
    GPT = ("openai", "GPT")
    YA = ("yandex", "YandexGPT")
    SBER = ("gigachat", "Sber")
    MISTRAL = ("mistral", "MistralAI")
    GPT4 = ("openai_4", "GPT4")
    GPT_PERS = ("openai_pers", "GPT_PERS")
    GPT_THINK = ("openai_think", "GPT_THINK")

    def __init__(self, value, display_name):
        self._value_ = value
        self.display_name = display_name


TERMINAL = frozenset({"completed", "failed", "interrupted", "cancelled", "recovery_required"})


class ArtifactError(RuntimeError):
    """An input/output artifact could not be processed or exported."""


@dataclass(frozen=True)
class AgentDescriptor:
    id: str
    name: str
    description: str
    implementation: str
    revision: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    required_services: tuple[str, ...] = ()
    execution_class: str = "interactive"
    active: bool = True
    privacy_affinity: bool = False
    supported_content_types: tuple[str, ...] = ()
    configuration: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    owner_id: str
    conversation_id: str
    media_type: str
    storage_key: str
    filename: str


@dataclass(frozen=True)
class AgentContext:
    user_id: str
    user_role: str
    conversation_id: str
    run_id: str
    parent_run_id: str | None = None
    trace: Mapping[str, str] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRequest:
    operation: str
    input: Mapping[str, Any]
    agent_revision: str
    conversation_sequence: int
    artifacts: tuple[ArtifactRef, ...] = ()
    idempotency_key: str | None = None


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    attempt_id: str | None
    sequence: int
    type: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class RunResult:
    output: Mapping[str, Any]
    artifacts: tuple[ArtifactRef, ...] = ()
    interrupt: Mapping[str, Any] | None = None
    error: str | None = None
    checkpoint: str | None = None


class AgentHandle(Protocol):
    async def invoke(self, request: RunRequest, context: AgentContext) -> RunResult: ...
    def stream(self, request: RunRequest, context: AgentContext) -> AsyncIterator[RunEvent]: ...
    async def resume(self, interrupt_id: str, response: Any, context: AgentContext) -> RunResult: ...
    async def reset(self, context: AgentContext) -> RunResult: ...
    async def close(self) -> None: ...


class AgentResolver(Protocol):
    async def resolve(self, agent_id: str, *, state_scope: str) -> Any: ...
    def describe(self, agent_id: str) -> AgentDescriptor: ...
