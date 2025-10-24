from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from agents.find_job_agent import initialize_agent as init_job_agent
from agents.sd_ass_agent.agent import initialize_agent as init_sd_agent
from agents.utils import ModelType

from .config import settings
from .schemas import AgentInfo


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    name: str
    description: str
    factory: Callable[[ModelType], Any]
    default_provider: ModelType


def _resolve_provider(provider_name: str) -> ModelType:
    try:
        return ModelType(provider_name)
    except ValueError:
        return ModelType(settings.default_model_provider)


class AgentRegistry:
    def __init__(self) -> None:
        default_provider = _resolve_provider(settings.default_model_provider)
        self._definitions: Dict[str, AgentDefinition] = {
            "find_job": AgentDefinition(
                id="find_job",
                name="Job Finder",
                description="Подыскивает вакансии на основе резюме пользователя.",
                factory=lambda provider: init_job_agent(provider=provider),
                default_provider=default_provider,
            ),
            "service_desk": AgentDefinition(
                id="service_desk",
                name="Service Desk Assistant",
                description="Отвечает на вопросы сотрудников и консультирует по внутренним процессам.",
                factory=lambda provider: init_sd_agent(provider=provider),
                default_provider=default_provider,
            ),
        }
        self._instances: Dict[str, Any] = {}

    def list_agents(self) -> List[AgentInfo]:
        return [
            AgentInfo(
                id=definition.id,
                name=definition.name,
                description=definition.description,
                provider=definition.default_provider.value,
            )
            for definition in self._definitions.values()
        ]

    def get_agent(self, agent_id: str) -> Any:
        if agent_id not in self._definitions:
            raise KeyError(f"Unknown agent '{agent_id}'")
        if agent_id not in self._instances:
            definition = self._definitions[agent_id]
            provider = definition.default_provider
            self._instances[agent_id] = definition.factory(provider)
        return self._instances[agent_id]


agent_registry = AgentRegistry()
