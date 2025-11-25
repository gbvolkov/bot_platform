from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union, Callable
from langchain.agents import AgentState


class ArtifactDefinition(TypedDict):
    id: int
    name: str
    goal: str
    methodology: str
    criteria: List[str]

class CriteriaEstimation(TypedDict):
    """Estimation of artifact criteria. 
    Высокоуровневая оценка выполнения критерия.
    """
    criteria_estimation: Annotated[str, ..., "Estimation of artifact criteria. Высокоуровневая оценка выполнения критерия."]

class ArtifactOption(TypedDict):
    """Option for an artifact. 
    Вариант для артефакта.
    """
    artifact_option: Annotated[str, ..., "Option for an artifact. Вариант для артефакта."]
    criteris_estimations: Annotated[List[CriteriaEstimation], ..., "List of estimations for criterias applicable for the aftifact. Перечень оценок выполнения критериев для артефакта."]

class ArtifactOptions(TypedDict):
    """List of options for an artifact.
        Список вариантов для артефакта.
    """
    general_considerations: Annotated[str, ..., "General considerations for an artifact. Общие рассуждения для артефакта."]
    artifact_options: Annotated[List[ArtifactOption], ..., "Unnumbered list of options for an artifact. Ненумерованный список вариантов для артефакта."]

class AftifactFinalText(TypedDict):
    """Final text for an artifact.
    Финальный текст для артефакта.
    """
    artifact_final_text: Annotated[str, ..., "Final text for an artifact. Финальный текст для артефакта."]

class ArtifactDetails(TypedDict):
    artifact_definition: ArtifactDefinition
    artifact_options: ArtifactOptions
    selected_option: int
    artifact_final_text: str

from enum import IntEnum

class ArtifactState(IntEnum):
    INIT = 0
    OPTIONS_GENERATED = 4
    OPTION_SELECTED = 6
    ARTIFACT_GENERATED = 8
    ARTIFACT_CONFIRMED = 12

class ArtifactAgentState(AgentState[ArtifactOptions]):
    # Merge helpers to allow concurrent writes without InvalidUpdateError
    def _merge_latest(a, b):
        return b if b is not None else a

    def _merge_dict(a: Dict[str, Any] | None, b: Dict[str, Any] | None):
        return {**(a or {}), **(b or {})}

    def _merge_artifacts(a: Dict[int, ArtifactDetails] | None, b: Dict[int, ArtifactDetails] | None):
        return {**(a or {}), **(b or {})}

    user_info: NotRequired[Annotated[Dict[str, Any], _merge_dict]]
    user_prompt: Annotated[str, _merge_latest]
    artifacts: NotRequired[Annotated[Dict[int, ArtifactDetails], _merge_artifacts]]
    current_artifact_id: NotRequired[Annotated[int, _merge_latest]]
    current_artifact_state: NotRequired[Annotated[ArtifactState, _merge_latest]]
    current_artifact_text: Annotated[str, _merge_latest]
    #generated_artifacts: List[ArtifactDetails]
    #selected_option: NotRequired[int]

class ArtifactAgentContext(TypedDict):
    user_prompt: str
    generated_artifacts: List[ArtifactDetails]

def get_artifacts_list()-> str:
    artifact_list = ""
    for artifact in ARTIFACTS:
        artifact_list += f"{artifact['id']+1}: {artifact['name']}\n"
    return artifact_list

ARTIFACTS: List[ArtifactDefinition] = [
    {
        "id": 0,
        "name": "Продуктовая троица",
        "goal": "Определить сегмент, проблему и решение.",
        "methodology": "Сегмент должен быть растущим, боль реальной, решение масштабируемым.",
        "criteria": ["Сегмент растущий", "Реальная боль на языке клиента", "Потенциал 2x-30x", "Тезисы проверяемы"]
    },
    {
        "id": 1,
        "name": "Карточка инициативы",
        "goal": "Сформировать паспорт проекта.",
        "methodology": "Заполнить все разделы: проблема, решение, рынок, команда, метрики.",
        "criteria": ["Все разделы заполнены", "Сегменты конкретны", "Проблема на языке клиента", "Относительные метрики"]
    },
    {
        "id": 2,
        "name": "Карта стейкхолдеров",
        "goal": "Определить всех заинтересованных лиц и их влияние.",
        "methodology": "Матрица влияния/интереса.",
        "criteria": ["Роли/интересы определены", "Влияние оценено", "Риски учтены", "Матрица взаимодействия"]
    },
    {
        "id": 3,
        "name": "Бэклог гипотез",
        "goal": "Собрать список гипотез для проверки.",
        "methodology": "HADI циклы, приоритезация ICE/RICE.",
        "criteria": ["Формула гипотезы", "Метрика успеха", "Приоритет (ICE/RICE)", "Связь с болью"]
    },
    {
        "id": 4,
        "name": "Глубинное интервью",
        "goal": "Подтвердить проблему через интервью.",
        "methodology": "CustDev, открытые вопросы.",
        "criteria": ["Целевая выборка", "Сценарий", "Инсайты с цитатами", "Ссылки на сырьё"]
    },
    {
        "id": 5,
        "name": "Ценностное предложение",
        "goal": "Сформулировать выгоду для клиента.",
        "methodology": "Value Proposition Canvas.",
        "criteria": ["Связка боль->выгода", "Top-3 ценности", "Проверяемые обещания"]
    },
    {
        "id": 6,
        "name": "Карта путешествия клиента (CJM)",
        "goal": "Визуализировать путь клиента.",
        "methodology": "Путь от возникновения потребности до решения.",
        "criteria": ["Стадии", "Боли/эмоции", "Точки контакта", "Возможности улучшения"]
    },
    {
        "id": 7,
        "name": "Карта бизнес-процессов",
        "goal": "Описать процессы AS-IS и TO-BE.",
        "methodology": "BPMN или простая схема.",
        "criteria": ["AS-IS/TO-BE", "Входы/выходы", "Владельцы", "Узкие места"]
    },
    {
        "id": 8,
        "name": "Конкурентный анализ",
        "goal": "Сравнить с конкурентами.",
        "methodology": "Сравнение по фичам, ценам, опыту.",
        "criteria": [">=5 альтернатив", "Сравнительная таблица", "Дифференциация"]
    },
    {
        "id": 9,
        "name": "УТП",
        "goal": "Сформулировать Уникальное Торговое Предложение.",
        "methodology": "Почему купят именно у нас.",
        "criteria": ["Одна чёткая формула", "Доказуемые преимущества", "Релевантно сегменту"]
    },
    {
        "id": 10,
        "name": "Финансовая модель",
        "goal": "Оценить экономику проекта.",
        "methodology": "Unit-экономика, P&L.",
        "criteria": ["Ключевые допущения", "LTV/CAC/маржа", "Чувствительность", "Сценарии"]
    },
    {
        "id": 11,
        "name": "Дорожная карта",
        "goal": "План реализации.",
        "methodology": "Gantt или список релизов.",
        "criteria": ["Релизы", "Цели/метрики", "Ресурсы/риски", "Вехи"]
    },
    {
        "id": 12,
        "name": "Карточка проекта",
        "goal": "Финальная сводка.",
        "methodology": "Summary всего проекта.",
        "criteria": ["Сводка по 1-12", "Роли/ответственность", "Критерии готовности", "Go/No-Go"]
    }
]
