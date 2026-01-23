from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections.abc import AsyncIterator
from urllib.parse import unquote, urlparse

#from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.find_job_agent import initialize_agent as init_job_agent
from agents.sd_ass_agent.agent import initialize_agent as init_sd_agent
from agents.ingos_product_agent import initialize_agent as init_product_agent
from agents.bi_agent import initialize_agent as init_aibi_agent
from agents.ideator_agent import initialize_agent as init_ideator_agent
from agents.ideator_old_agent import initialize_agent as init_ideator_old_agent
from agents.ismart_tutor_agent import initialize_agent as init_ismart_tutor_agent
from agents.simple_agent.agent import initialize_agent as init_simple_agent
from agents.theodor_agent.agent import initialize_agent as init_theodor_agent
from agents.utils import ModelType

from .config import settings
from .schemas import AgentInfo, ContentType


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    name: str
    description: str
    factory: Callable[[ModelType], Any]
    default_provider: ModelType
    supported_content_types: Tuple[ContentType, ...]
    allow_raw_attachments: bool = False
    is_active: bool = True


PROVIDER_MAPPING: Dict[str, ModelType] = {
    "openai": ModelType.GPT,
    "yandex": ModelType.YA,
    "mistral": ModelType.MISTRAL,
    "gigachat": ModelType.SBER,
}


def _resolve_provider(provider_name: str) -> ModelType:
    return PROVIDER_MAPPING.get(provider_name.lower(), ModelType.GPT)


PRODUCT_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"




class AgentRegistry:
    def __init__(self) -> None:
        default_provider = _resolve_provider(settings.default_model_provider)
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

        self._definitions: Dict[str, AgentDefinition] = {
            "find_job": AgentDefinition(
                id="find_job",
                name="Job Finder",
                description="Подыскивает вакансии на основе резюме пользователя.",
                factory=lambda provider, checkpoint_saver: init_job_agent(provider=provider, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=(
                    ContentType.TEXT_FILES,
                    ContentType.MARKDOWN,
                    ContentType.DOCX_DOCUMENTS,
                    ContentType.PDFS,
                ),
                is_active = False,
            ),
            "service_desk": AgentDefinition(
                id="service_desk",
                name="Service Desk Assistant",
                description="Отвечает на вопросы сотрудников и консультирует по внутренним процессам.",
                factory=lambda provider, checkpoint_saver: init_sd_agent(provider=provider, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=default_content_types,
                is_active = False,
            ),
            "ai_bi": AgentDefinition(
                id="ai_bi",
                name="BI analyst",
                description="Отвечает на вопросы по датасету и строит графики.",
                factory=lambda provider, checkpoint_saver: init_aibi_agent(provider=provider, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=default_content_types,
                is_active = False,
            ),
            "theodor_agent": AgentDefinition(
                id="theodor_agent",
                name="Theodor AI (Product Mentor)",
                description="Продуктовый наставник. Ведет по методологии Фёдора (13 артефактов).",
                factory=lambda provider, checkpoint_saver: init_theodor_agent(provider=provider, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=default_content_types,
                is_active = False,
            ),
            "ideator": AgentDefinition(
                id="ideator",
                name="Ideator (Новости → идеи)",
                description="Генерирует смысловые линии и идеи, опираясь на новости из отчёта.",
                factory=lambda provider, checkpoint_saver: init_ideator_agent(provider=provider, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=default_content_types + (ContentType.JSONS,),
                allow_raw_attachments=True,
                is_active = True,
            ),
            "ideator_old": AgentDefinition(
                id="ideator_old",
                name="Ideator (Новости → идеи). Old version",
                description="Генерирует смысловые линии и идеи, опираясь на новости из отчёта.",
                factory=lambda provider, checkpoint_saver: init_ideator_old_agent(provider=provider, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=default_content_types + (ContentType.JSONS,),
                allow_raw_attachments=True,
                is_active = False,
            ),
            "ismart_tutor_agent": AgentDefinition(
                id="ismart_tutor_agent",
                name="iSmart Tutor",
                description="Educational tutor that provides concise hints and clarifying questions (without giving full solutions).",
                factory=lambda provider, checkpoint_saver: init_ismart_tutor_agent(provider=provider, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=default_content_types + (ContentType.IMAGES,),
                allow_raw_attachments=True,
            ),
            "simple_agent": AgentDefinition(
                id="simple_agent",
                name="Simple Agent",
                description="General-purpose assistant with optional system prompt setup.",
                factory=lambda provider, checkpoint_saver: init_simple_agent(provider=provider, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=default_content_types,
            ),
        }
        self._definitions.update(self._build_product_definitions(default_provider, default_content_types))
        self._instances: Dict[str, Any] = {}
        self._init_tasks: Dict[str, Future] = {}
        self._init_errors: Dict[str, BaseException] = {}

    def _build_product_definitions(
        self,
        default_provider: ModelType,
        default_content_types: Tuple[ContentType, ...],
    ) -> Dict[str, AgentDefinition]:
        product_definitions: Dict[str, AgentDefinition] = {}
        docs_dir = PRODUCT_DOCS_DIR
        if not docs_dir.exists():
            return product_definitions

        for product_dir in sorted(docs_dir.iterdir()):
            if not product_dir.is_dir():
                continue
            product_name = product_dir.name
            agent_id = f"product_{product_name}"
            product_definitions[agent_id] = AgentDefinition(
                id=agent_id,
                name=product_name,
                description=f"Assistant of Ingosstrakh Product '{product_name}'.",
                factory=lambda provider, checkpoint_saver, product=product_name: init_product_agent(provider=provider, product=product, checkpoint_saver=checkpoint_saver),
                default_provider=default_provider,
                supported_content_types=default_content_types,
                is_active = False
            )
        return product_definitions

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
            checkpoint_saver = await self._ensure_checkpointer()
            loop = asyncio.get_running_loop()

            # Keep your current pattern: build agent in executor thread
            return await loop.run_in_executor(
                None,
                lambda: definition.factory(provider, checkpoint_saver=checkpoint_saver),
            )

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
