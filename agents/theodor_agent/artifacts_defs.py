import copy

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
    criteria_estimation: Annotated[str, ..., "Estimation of artifact criteria. Высокоуровневая оценка выполнения критерия.Format as MarkdownV2."]

class ArtifactOption(TypedDict):
    """Option for an artifact. 
    Вариант для артефакта.
    """
    artifact_option: Annotated[str, ..., "Option for an artifact. Вариант для артефакта.Format as MarkdownV2."]
    criteris_estimations: Annotated[List[CriteriaEstimation], ..., "List of estimations for criterias applicable for the aftifact. Перечень оценок выполнения критериев для артефакта."]

class ArtifactOptions(TypedDict):
    """List of options for an artifact.
        Список вариантов для артефакта.
    """
    general_considerations: Annotated[str, ..., "Critics for an artifact and for user's comments. Критические замечания для артефакта и для комментариев пользователя. Format as MarkdownV2."]
    artifact_options: Annotated[List[ArtifactOption], ..., "Unnumbered list of options for an artifact. Ненумерованный список вариантов для артефакта."]

class AftifactFinalText(TypedDict):
    """Final text for an artifact.
    Финальный текст для артефакта.
    """
    artifact_estimation: Annotated[str, ..., "Estimation for an artifact. Here you can criticize. Оценка применимости и полноты артифакта. Здесь можно критиковать."]
    artifact_final_text: Annotated[str, ..., "Final text for an artifact - no discussion with user, just artifact text. Финальный текст для артефакта - без рассуждений и ответа пользователю - только текст артефакта! Format as MarkdownV2."]


class CriteriaEstimationEn(TypedDict):
    """Estimation of artifact criteria."""
    criteria_estimation: Annotated[str, ..., "Estimation of artifact criteria. High-level assessment. Format as MarkdownV2."]


class ArtifactOptionEn(TypedDict):
    """Option for an artifact."""
    artifact_option: Annotated[str, ..., "Option for an artifact. Format as MarkdownV2."]
    criteris_estimations: Annotated[List[CriteriaEstimationEn], ..., "List of estimations for criteria applicable to the artifact."]


class ArtifactOptionsEn(TypedDict):
    """List of options for an artifact."""
    general_considerations: Annotated[str, ..., "Critics for an artifact and for user's comments. Format as MarkdownV2."]
    artifact_options: Annotated[List[ArtifactOptionEn], ..., "Unnumbered list of options for an artifact."]


class AftifactFinalTextEn(TypedDict):
    """Final text for an artifact."""
    artifact_estimation: Annotated[str, ..., "Estimation for an artifact. Here you can criticize."]
    artifact_final_text: Annotated[str, ..., "Final text for an artifact - no discussion with user, just artifact text. Format as MarkdownV2."]

class ArtifactDetails(TypedDict):
    artifact_definition: ArtifactDefinition
    # Stored in state as a flat list of options for this artifact.
    artifact_options: List[ArtifactOption]
    # 0-based index into artifact_options, -1 when not chosen yet.
    selected_option: int
    # Stored as a rendered string (final text + estimation).
    artifact_final_text: str
    # Compact memory note about the discussion (not the artifact text itself).
    artifact_summary: NotRequired[str]

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


ARTIFACTS_RU: List[ArtifactDefinition] = [
    {
        "id": 0,
        "stage": "Генерация идей",
        "stage_goal": "Сформулировать и «упаковать» первоначальную бизнес-идею в понятный формат для её первичной оценки.",
        "name": "Продуктовая троица",
        "goal": "Этот инструмент используется для стратегического анализа и поиска возможностей для кратного, экспоненциального роста («в иксы»). Он помогает убедиться, что ваша идея нацелена на растущий рынок с реальной проблемой",
        "components": [
            "Клиентский сегмент: Поиск агрессивно растущих сегментов.",
            "Проблема (Потребность): Анализ рынка проблем, конкурирующих за внимание сегмента",
            "Ценность: Анализ рынка ценностей, которые решают проблему",
            "Решение: Анализ рынка решений, которые доставляют ценность"
        ],
        "methodology": 
"""Продуктовую троицу.
1.	Начните с любого из трех элементов. Например, найдите быстрорастущий клиентский сегмент (люди, работающие удаленно8; молодые люди, совершающие онлайн-транзакции).
2.	Определите их самую острую, критическую потребность (нехватка «живого» общения).
3.	Сформулируйте ценность, которая решает эту проблему.
4.	Проанализируйте, какие решения могут доставить эту ценность, и ищите экспоненциально растущие драйверы (продукты из данных, дашборды, UBER-изация).
5.	Задавайте себе вопрос: «Где здесь иксы (рост 2x-30x)?». Если кратного роста нет ни в сегменте, ни в проблеме, ни в ценности — ищите дальше
""",
        "criteria": ["Сегмент растущий", "Реальная боль на языке клиента", "Потенциал 2x-30x", "Тезисы проверяемы"]
    },
    {
        "id": 1,
        "stage": "Генерация идей",
        "stage_goal": "Сформулировать и «упаковать» первоначальную бизнес-идею в понятный формат для её первичной оценки.",
        "name": "Карточка инициативы (продуктовый канвас)",
        "goal": "Структурировать и «упаковать» вашу бизнес-идею в единый, понятный формат.",
        "components": [
            "Сегменты клиентов: Конкретные группы пользователей, чью проблему вы решаете.",
            "Проблема: Формулировка боли клиента на его языке",
            "Альтернативные решения: Как клиенты решают проблему сейчас?",
            "Источники прибыли: Модель монетизации. Как вы будете зарабатывать?",
            "Решение: Как именно ваш продукт будет решать проблему?",
            "Каналы коммуникации: Как клиенты узнают о вашем решении?",
            "Ключевые метрики: Как вы будете измерять успех?",
            "Структура затрат: На что потребуются ресурсы?",
            "Бизнес-процессы: Какие внутренние процессы компании будут затронуты?",
        ],
        "methodology": 
"""Заполнить все разделы: проблема, решение, рынок, команда, метрики.
1.	Название: Сделайте его кратким, ёмким и понятным. Оно должно отражать суть.
2.	Сегменты: Будьте конкретны. «Ритейл-клиенты» — плохой сегмент. «Текущие клиенты нашего банка, имеющие дебетовые и кредитные карты» — хороший. Избегайте слишком широких или слишком узких сегментов.
3.	Проблема: Не путайте причину и следствие. «Отсутствие методики» — это не проблема клиента. «Невозможность экспортировать продукцию из-за отсутствия сертификации» — вот его боль. Говорите на языке клиента.
4.	Источники прибыли и Затраты: Сделайте предварительную оценку, покажите порядок цифр. Это ваша гипотеза о P&L.
5.	Метрики: Выбирайте относительные метрики (например, процент успешных авторизаций), а не абсолютные (количество авторизаций). Метрики должны быть связаны с источниками прибыли.
6.	Кросс-аналитика: Постоянно проверяйте связь между блоками. Решает ли ваше «Решение» указанную «Проблему» для данного «Сегмента»? Соответствуют ли «Метрики» вашим «Источникам прибыли»?
""",
        "criteria": ["Все разделы заполнены", "Сегменты конкретны", "Проблема на языке клиента", "Относительные метрики"]
    },
    {
        "id": 2,
        "stage": "Генерация идей",
        "stage_goal": "Сформулировать и «упаковать» первоначальную бизнес-идею в понятный формат для её первичной оценки.",
        "name": "Карта стейкхолдеров",
        "goal": "Идентифицировать всех людей и группы, на которых влияет ваша инициатива или которые могут повлиять на неё. Это помогает выстроить правильную коммуникацию и управлять ожиданиями",
        "components": [
            "Список стейкхолдеров: Все, кто имеет отношение к проекту (топ-менеджмент, команда, юристы, финансы, пользователи и т.д.).",
            "Матрица 'Власть/Интерес': Инструмент для классификации стейкхолдеров\n"
            "-- Высокая власть / Высокий интерес (Ключевые игроки): Управлять плотно.\n"
            "-- Высокая власть / Низкий интерес (Создатели контекста): Держать удовлетворенными.\n"
            "-- Низкая власть / Высокий интерес (Субъекты): Держать в курсе.\n"
            "-- Низкая власть / Низкий интерес (Толпа): Мониторить с минимальными усилиями.",
        ],
        "methodology": 
"""Составляем матрицу влияния/интереса.
1.	Составьте полный список всех, кто может быть заинтересован в вашем проекте или затронут им.
2.	Проанализируйте каждого стейкхолдера, оценив его уровень власти (влияния на проект) и уровень интереса (насколько проект важен для него).
3.	Разместите каждого в соответствующем квадранте матрицы.
4.	Разработайте стратегию коммуникации для каждой группы. Например, с ключевыми игроками нужно взаимодействовать регулярно и вовлекать в принятие решений, а группу "Толпа" достаточно периодически информировать через общие рассылки.
""",
        "criteria": ["Роли/интересы определены", "Влияние оценено", "Риски учтены", "Матрица взаимодействия"]
    },
    {
        "id": 3,
        "stage": "Исследования",
        "stage_goal": "Проверить все гипотезы из «Карточки инициативы» с помощью данных и общения с клиентами. Снизить риски, «убив» нежизнеспособные идеи до начала дорогостоящей разработки.",
        "name": "Бэклог гипотез",
        "goal": "Собрать и приоритизировать все предположения о проблемах, сегментах и решениях для их систематической проверки.\n"
                "Сформулируй не менее 7 гипотез. Используй инструмент web_search_summary для сбора данных.\n"
                "Для каждой гипотезы сделай краткое резюме (3–5 пунктов), спроси «Учесть эти инсайты?», при «Да» интегрируй и отметь источник.",
        "data_source": "Всегда спрашивай: «Хотите загрузить реальные данные (интервью, таблицы, отчёты) или создаём вручную?\n"
                "Если файл загружен — сделай краткое резюме (3–5 пунктов), спроси «Учесть эти инсайты?», при «Да» интегрируй и отметь источник.",
        "components": [
            "Гипотеза: Сформулированная по принципу «Если..., то...».",
            "Сегмент: Для какой группы пользователей проверяется гипотеза.",
            "Приоритет: Насколько важна эта гипотеза.",
            "Способ проверки: Например, глубинное интервью, A/B тест, прототип.",
        ],
        "methodology": 
"""HADI циклы, приоритезация ICE/RICE.
1.	Перенесите все предположения из «Карточки инициативы» в бэклог.
2.	Сформулируйте их в виде гипотез. Например: "Если мы упростим процедуру регистрации, то процент успешных авторизаций увеличится с 40% до 80%".
3.	Проведите приоритизацию гипотез, чтобы начать с самых рискованных и важных.
""",
        "criteria": ["Формула гипотезы", "Метрика успеха", "Приоритет (ICE/RICE)", "Связь с болью"]
    },
    {
        "id": 4,
        "stage": "Исследования",
        "stage_goal": "Проверить все гипотезы из «Карточки инициативы» с помощью данных и общения с клиентами. Снизить риски, «убив» нежизнеспособные идеи до начала дорогостоящей разработки.",
        "name": "Глубинное интервью (CustDev)",
        "goal": "Получить качественные данные о проблемах, целях и текущем опыте пользователей для проверки гипотез.",
        "data_source": "Всегда спрашивай: «Хотите загрузить реальные данные (интервью, таблицы, отчёты) или создаём вручную?\n"
                "Если файл загружен — сделай краткое резюме (3–5 пунктов), спроси «Учесть эти инсайты?», при «Да» интегрируй и отметь источник.",
        "components": [
            "Гипотезы: Какие предположения вы проверяете.",
            "Цели интервью: Что конкретно вы хотите узнать.",
            "Структура и вопросы: Подробный скрипт с открытыми вопросами.",
            "Тайминг: Сколько времени отводится на каждый блок.",
        ],
        "methodology": 
"""CustDev, открытые вопросы.
1.	Подготовка — 90% успеха!. Тщательно пропишите план.
2.	Задавайте открытые вопросы о прошлом опыте. Не спрашивайте «Хотели бы вы...?» (форсайтный вопрос). Спрашивайте «Расскажите, как вы в последний раз решали проблему...?» (ретроспективный вопрос).
3.	Используйте правило «5 почему», чтобы докопаться до первопричины.
4.	Внимательно слушайте, не перебивайте. Большую часть времени должен говорить собеседник.
5.	Фиксируйте результаты в виде таблицы инсайтов по каждому респонденту.
""",
        "criteria": ["Целевая выборка", "Сценарий", "Инсайты с цитатами", "Ссылки на сырьё"]
    },
    {
        "id": 5,
        "stage": "Исследования",
        "stage_goal": "Проверить все гипотезы из «Карточки инициативы» с помощью данных и общения с клиентами. Снизить риски, «убив» нежизнеспособные идеи до начала дорогостоящей разработки.",
        "name": "Ценностное предложение",
        "goal": "Систематически сопоставить потребности клиента и характеристики вашего продукта, чтобы убедиться в их соответствии и четко сформулировать главную выгоду. Этот артефакт помогает «продать» концепцию продукта клиентам и стейкхолдерам.",
        "data_source": "Всегда спрашивай: «Хотите загрузить реальные данные (интервью, таблицы, отчёты) или создаём вручную?\n"
                "Если файл загружен — сделай краткое резюме (3–5 пунктов), спроси «Учесть эти инсайты?», при «Да» интегрируй и отметь источник.",
        "components": [
            "Профиль потребителя (Круг):\n"
            "-- Задачи потребителя: Что клиент пытается сделать (функциональные, социальные, эмоциональные)\n"
            "-- Проблемы потребителя: Что его беспокоит и мешает\n"
            "-- Выгоды потребителя: Какие результаты и преимущества он хочет получить",
            "Карта ценности (Квадрат):\n"
            "-- Товары и услуги: Перечень того, что вы предлагаете.\n"
            "-- Факторы помощи (Pain Relievers): Как ваш продукт решает проблемы клиента.\n"
            "-- Факторы выгоды (Gain Creators): Как ваш продукт создает выгоды для клиента.\n",
        ],
        "methodology": 
"""Value Proposition Canvas.
1.	Начните с Профиля потребителя. На основе данных из глубинных интервью заполните задачи, проблемы и выгоды клиента.
2.	Затем заполните Карту ценности. Опишите, как ваш продукт и его функции помогают клиенту, снимая его боли и создавая выгоды.
3.	Найдите соответствие (fit). Убедитесь, что ваши факторы помощи и выгоды нацелены на самые важные проблемы и желаемые выгоды клиента.
4.	Сформулируйте ценностное предложение. Используйте простую формулу: «Наш [продукт] помогает [сегменту клиентов], которые хотят [выполнить задачу], тем, что [устраняет проблему] и [создает выгоду]».
""",
        "criteria": ["Связка боль->выгода", "Top-3 ценности", "Проверяемые обещания"]
    },
    {
        "id": 6,
        "stage": "Исследования",
        "stage_goal": "Проверить все гипотезы из «Карточки инициативы» с помощью данных и общения с клиентами. Снизить риски, «убив» нежизнеспособные идеи до начала дорогостоящей разработки.",
        "name": "Карта путешествия клиента (CJM)",
        "goal": "Визуализировать весь опыт клиента при взаимодействии с компанией или продуктом, чтобы выявить барьеры, болевые точки и негативные эмоции.",
        "data_source": "Всегда спрашивай: «Хотите загрузить реальные данные (интервью, таблицы, отчёты) или создаём вручную?\n"
                "Если файл загружен — сделай краткое резюме (3–5 пунктов), спроси «Учесть эти инсайты?», при «Да» интегрируй и отметь источник.",
        "components": [
            "Этапы: Ключевые фазы взаимодействия (поиск, покупка, использование).",
            "Действия: Что конкретно делает клиент на каждом этапе.",
            "Точки касания: Где происходит взаимодействие (сайт, приложение, колл-центр).",
            "Проблемы и барьеры: Что мешает клиенту.",
            "Эмоции: Что чувствует клиент на каждом шаге (отображается графиком).",
            "Решения: Идеи по устранению барьеров.",
        ],
        "methodology": 
"""Путь от возникновения потребности до решения.
1.	Соберите информацию: Используйте данные из глубинных интервью.
2.	Опишите персонажа: Создайте собирательный образ вашего клиента.
3.	Нанесите на карту все этапы, действия и точки касания.
4.	Определите барьеры и отметьте, где клиент испытывает негативные эмоции.
5.	Сгенерируйте гипотезы о решениях для устранения этих барьеров.
""",
        "criteria": ["Стадии", "Боли/эмоции", "Точки контакта", "Возможности улучшения"]
    },
    {
        "id": 7,
        "stage": "Исследования",
        "stage_goal": "Проверить все гипотезы из «Карточки инициативы» с помощью данных и общения с клиентами. Снизить риски, «убив» нежизнеспособные идеи до начала дорогостоящей разработки.",
        "name": "Карта бизнес-процессов",
        "goal": "Отразить текущую ситуацию и систематизировать внутренние процессы организации, на которые влияет ваша инициатива. В отличие от CJM, этот артефакт показывает взаимодействие всех внутренних ролей, а не только клиента.",
        "data_source": "Всегда спрашивай: «Хотите загрузить реальные данные (интервью, таблицы, отчёты) или создаём вручную?\n"
                "Если файл загружен — сделай краткое резюме (3–5 пунктов), спроси «Учесть эти инсайты?», при «Да» интегрируй и отметь источник.",
        "components": [
            "Роли: Все участники процесса (не только клиент).",
            "Действия: Последовательность операций.",
            "Длительность: Время выполнения каждого действия.",
            "Инструменты/Системы: Какие программы или документы используются.",
            "Гипотезы о проблемах и решениях: Где есть «узкие места» и как их можно улучшить.",
        ],
        "methodology": 
"""BPMN или простая схема.
1.	Соберите информацию: Поговорите с бизнес-аналитиками и сотрудниками, вовлеченными в процесс.
2.	Определите точки входа и выхода процесса, чтобы задать четкие границы.
3.	Нанесите на карту все действия, роли и используемые инструменты.
4.	Проанализируйте процесс и зафиксируйте гипотезы о проблемах (например, «оформление печатных форм занимает слишком много времени») и решениях.
""",
        "criteria": ["AS-IS/TO-BE", "Входы/выходы", "Владельцы", "Узкие места"]
    },
    {
        "id": 8,
        "stage": "Исследования",
        "stage_goal": "Проверить все гипотезы из «Карточки инициативы» с помощью данных и общения с клиентами. Снизить риски, «убив» нежизнеспособные идеи до начала дорогостоящей разработки.",
        "name": "Конкурентный анализ",
        "goal": "Понять, как другие компании решают схожие проблемы, чтобы правильно позиционировать свой продукт, избежать чужих ошибок и найти конкурентные преимущества.",
        "data_source": "Всегда спрашивай: «Хотите загрузить реальные данные (интервью, таблицы, отчёты) или создаём вручную?\n"
                "Если файл загружен — сделай краткое резюме (3–5 пунктов), спроси «Учесть эти инсайты?», при «Да» интегрируй и отметь источник.\n"
                "**ВАЖНО**: Обязательно используй инструмент web_search_summary для поиска конкурентов",
        "components": [
            "Конкуренты: Список прямых и косвенных конкурентов.",
            "Сегменты клиентов: На кого нацелены конкуренты.",
            "Ценностное предложение (УТП): Чем они привлекают клиентов.",
            "Модель монетизации: Как они зарабатывают.",
            "Характеристики продукта: Ключевые фичи.",
            "Цена/стоимость.",
            "Отзывы клиентов.",
        ],
        "methodology": 
"""Сравнение по фичам, ценам, опыту.
1.	Найдите конкурентов: Используйте поиск, данные маркетинга, отраслевые отчеты. Анализируйте не только прямых конкурентов, но и тех, кто борется за ту же клиентскую ценность (например, для микрокредитов конкурентами являются и ломбарды).
2.	Соберите информацию по всем ключевым пунктам и занесите в сравнительную таблицу.
3.	Станьте клиентом конкурента, чтобы изучить его продукт изнутри.
4.	Проанализируйте сильные и слабые стороны и определите, в чем вы можете быть лучше. Не конкурируйте ценой в последнюю очередь!
""",
        "criteria": [">=5 альтернатив", "Сравнительная таблица", "Дифференциация"]
    },
    {
        "id": 9,
        "stage": "Исследования",
        "stage_goal": "Проверить все гипотезы из «Карточки инициативы» с помощью данных и общения с клиентами. Снизить риски, «убив» нежизнеспособные идеи до начала дорогостоящей разработки.",
        "name": "Уникальное торговое предложение (УТП)",
        "goal": "На основе понимания клиента и рынка сформулировать одно ясное, короткое и убедительное предложение, которое объясняет, почему ваш продукт — лучший выбор.",
        "components": [
            "Целевая аудитория: Для кого ваш продукт.",
            "Проблема: Какую боль вы решаете.",
            "Решение/Продукт: Что вы предлагаете.",
            "Уникальное отличие: Что делает вас лучше конкурентов.",
        ],
        "methodology": 
"""Почему купят именно у нас.
1.	Синтезируйте знания: Вернитесь к артефактам «Ценностное предложение» (что нужно клиенту) и «Конкурентный анализ» (что предлагают другие).
2.	Найдите свою уникальность: В чем ваше ключевое преимущество, которое важно для клиента и которое сложно скопировать? Это может быть новая технология, особый сервис, более удобный дизайн и т.д
3.	Сформулируйте УТП: Используйте простую и понятную структуру. Например: "Для [целевой сегмент], которые сталкиваются с [проблема], наш [продукт] предоставляет [ключевое преимущество], в отличие от [конкуренты]".
4.	Проверьте его: Ваше УТП должно быть конкретным, легко запоминающимся и вызывать желание узнать больше.
""",
        "criteria": ["Одна чёткая формула", "Доказуемые преимущества", "Релевантно сегменту"]
    },
    {
        "id": 10,
        "stage": "Проектирование",
        "stage_goal": "На основе проверенных гипотез спроектировать детальное решение, рассчитать его экономику и составить план реализации.",
        "name": "Финансовая модель",
        "goal": "Детально оценить экономический эффект продукта, рассчитав все доходы и расходы, и определить точку безубыточности.",
        "data_source": "Всегда спрашивай: «Хотите загрузить реальные данные (интервью, таблицы, отчёты) или создаём вручную?\n"
                "Если файл загружен — сделай краткое резюме (3–5 пунктов), спроси «Учесть эти инсайты?», при «Да» интегрируй и отметь источник.",
        "components": [
            "Затраты (расходы)\n"
            "-- Переменные: Команда проекта, внешние подрядчики.\n"
            "-- Постоянные: Поддержка, лицензии.",
            "Доходы (экономический эффект): Рассчитываются на основе ключевых метрик.",
            "Точка безубыточности (TCO - Total Cost of Ownership): Момент, когда доходы начинают превышать расходы.",
        ],
        "methodology": 
"""Unit-экономика, P&L.
1.	Соберите все типы расходов: Команда, подрядчики, оборудование, маркетинг, поддержка и т.д..
2.	Распределите расходы по месяцам в соответствии со стадиями фреймворка.
3.	Рассчитайте доходы: Возьмите ключевые метрики и покажите, как их изменение влияет на деньги. Например, «увеличение конверсии на 10% принесет Х денег».
4.	Постройте три сценария: негативный, стандартный и позитивный.
5.	Определите точку окупаемости. В продуктовом подходе она должна быть в горизонте 3-6 месяцев.
""",
        "criteria": ["Ключевые допущения", "LTV/CAC/маржа", "Чувствительность", "Сценарии"]
    },
    {
        "id": 11,
        "stage": "Проектирование",
        "stage_goal": "На основе проверенных гипотез спроектировать детальное решение, рассчитать его экономику и составить план реализации.",
        "name": "Дорожная карта",
        "goal": "Создать комплексный план работ, визуализирующий основные этапы, задачи и зависимости для координации команды и информирования стейкхолдеров.",
        "data_source": "**ВАЖНО**: Обязательно используй инструмент web_search_summary для получения рыночных цен.",
        "components": [
            "Задачи/Пакеты работ: Декомпозиция крупных целей.",
            "Сроки: Длительность выполнения каждой задачи.",
            "Ответственные: Кто выполняет задачу.",
            "Вехи (Milestones): Ключевые события, отмечающие завершение этапа.",
            "Критический путь: Самая длинная цепочка задач, определяющая общую длительность проекта.",
        ],
        "methodology": 
"""План работ.
1.	Определите цели и этапы разработки.
2.	Декомпозируйте их на задачи.
3.	Определите последовательность, длительность и связи между задачами.
4.	Назначьте ответственных.
5.	Установите вехи в конце каждого значимого этапа (например, «Тестирование завершено»).
6.	Управляйте рисками, добавляя временные буферы (резерв времени).
""",
        "criteria": ["Релизы", "Цели/метрики", "Ресурсы/риски", "Вехи"]
    },
    {
        "id": 12,
        "stage": "Проектирование",
        "stage_goal": "На основе проверенных гипотез спроектировать детальное решение, рассчитать его экономику и составить план реализации.",
        "name": "Карточка проекта",
        "goal": "Финальный документ для защиты проекта и получения бюджета на стадию Development. Он объединяет и суммирует всю проделанную работу на предыдущих этапах.",
        "components": [
            "Резюме проекта: Краткое описание решения, подтвержденная проблематика и целевые сегменты.",
            "Описание MVP: Задачи, пилотная зона, метрики успешности.",
            "Экономика: Данные из финансовой модели (доходы, затраты, затраты на команду).",
            "Команда проекта: Роли, компетенции, вовлеченность (FTE).",
            "Риски: Что может пойти не так и как вы планируете это решать.",
        ],
        "methodology": 
"""Summary всего проекта.
1.	Перенесите подтвержденные данные: В отличие от «Карточки инициативы», здесь должны быть только валидированные гипотезы о проблемах и решениях3.
2.	Детально опишите MVP: Что именно вы будете реализовывать, на каком сегменте и по каким метрикам будете оценивать успех.
3.	Интегрируйте финансовую модель: Укажите точные цифры по доходам и расходам.
4.	Представьте команду, необходимую для реализации.
5.	Продумайте риски и предложите способы их митигации.
""",
        "criteria": ["Сводка по 1-12", "Роли/ответственность", "Критерии готовности", "Go/No-Go"]
    }
]


ARTIFACTS_EN: List[ArtifactDefinition] = [
    {
        "id": 0,
        "stage": "Ideation",
        "stage_goal": "Formulate and \"package\" the initial business idea into a clear format for its initial evaluation.",
        "name": "Product Trinity",
        "goal": "This tool is used for strategic analysis and finding opportunities for multiplicative, exponential growth (\"in multiples\"). It helps ensure that your idea targets a growing market with a real problem.",
        "components": [
            "Customer segment: Search for aggressively growing segments.",
            "Problem (Need): Analyze the market of problems competing for the segment's attention.",
            "Value: Analyze the market of values that solve the problem.",
            "Solution: Analyze the market of solutions that deliver the value",
        ],
        "methodology":
"""Product Trinity.
1.\tStart with any of the three elements. For example, find a fast-growing customer segment (people working remotely; young people making online transactions).
2.\tIdentify their most acute, critical need (lack of \"live\" communication).
3.\tFormulate the value that solves this problem.
4.\tAnalyze which solutions can deliver this value, and look for exponentially growing drivers (data products, dashboards, Uber-ization).
5.\tAsk yourself: \"Where are the multiples (2x-30x growth)?\" If there is no multiplicative growth in the segment, the problem, or the value - keep searching.
""",
        "criteria": ["Growing segment", "Real pain in the customer's language", "2x-30x potential", "Theses are testable"]
    },
    {
        "id": 1,
        "stage": "Ideation",
        "stage_goal": "Formulate and \"package\" the initial business idea into a clear format for its initial evaluation.",
        "name": "Initiative Card (Product Canvas)",
        "goal": "Structure and \"package\" your business idea into a single, clear format.",
        "components": [
            "Customer segments: Specific user groups whose problem you solve.",
            "Problem: The customer's pain in their language.",
            "Alternative solutions: How do customers solve the problem now?",
            "Revenue sources: Monetization model. How will you earn money?",
            "Solution: How exactly will your product solve the problem?",
            "Communication channels: How will customers learn about your solution?",
            "Key metrics: How will you measure success?",
            "Cost structure: What will require resources?",
            "Business processes: Which internal company processes will be affected?",
        ],
        "methodology":
"""Fill all sections: problem, solution, market, team, metrics.
1.\tName: Make it short, clear, and meaningful. It should capture the essence.
2.\tSegments: Be specific. "Retail clients" is a bad segment. "Current customers of our bank with debit and credit cards" is a good one. Avoid too broad or too narrow segments.
3.\tProblem: Do not confuse cause and effect. "Lack of a methodology" is not the client's problem. "Inability to export products due to missing certification" is the pain. Speak in the customer's language.
4.\tRevenue sources and Costs: Make a preliminary estimate, show the order of magnitude. This is your P&L hypothesis.
5.\tMetrics: Choose relative metrics (e.g., percentage of successful authorizations), not absolute (number of authorizations). Metrics should be linked to revenue sources.
6.\tCross-analysis: Constantly check consistency between blocks. Does your "Solution" solve the stated "Problem" for this "Segment"? Do "Metrics" align with your "Revenue sources"?
""",
        "criteria": ["All sections filled", "Segments are specific", "Problem in the customer's language", "Relative metrics"]
    },
    {
        "id": 2,
        "stage": "Ideation",
        "stage_goal": "Formulate and \"package\" the initial business idea into a clear format for its initial evaluation.",
        "name": "Stakeholder Map",
        "goal": "Identify all people and groups affected by your initiative or who can influence it. This helps build proper communication and manage expectations",
        "components": [
            "Stakeholder list: Everyone related to the project (top management, team, legal, finance, users, etc.).",
            "Power/Interest matrix: A tool for classifying stakeholders",
            "-- High power / High interest (Key players): Manage closely.",
            "-- High power / Low interest (Context setters): Keep satisfied.",
            "-- Low power / High interest (Subjects): Keep informed.",
            "-- Low power / Low interest (Crowd): Monitor with minimal effort.",
        ],
        "methodology":
"""Build an influence/interest matrix.
1.\tMake a complete list of everyone who may be interested in your project or affected by it.
2.\tAnalyze each stakeholder by assessing their level of power (influence on the project) and level of interest (how important the project is to them).
3.\tPlace each in the appropriate quadrant of the matrix.
4.\tDevelop a communication strategy for each group. For example, key players should be engaged regularly and involved in decisions, while the "Crowd" can be informed periodically via general updates.
""",
        "criteria": ["Roles/interests defined", "Influence assessed", "Risks considered", "Interaction matrix"]
    },
    {
        "id": 3,
        "stage": "Discovery",
        "stage_goal": "Validate all hypotheses from the Initiative Card using data and customer conversations. Reduce risk by killing non-viable ideas before expensive development.",
        "name": "Hypothesis Backlog",
        "goal": "Collect and prioritize all assumptions about problems, segments, and solutions for systematic validation.\nFormulate at least 7 hypotheses. Use the web_search_summary tool to gather data.\nFor each hypothesis, make a brief summary (3-5 bullets), ask \"Use these insights?\", on \"Yes\" integrate and mark the source.",
        "data_source": "Always ask: \"Do you want to upload real data (interviews, tables, reports) or create manually?\n"
                "If a file is uploaded — make a brief summary (3-5 bullets), ask \"Use these insights?\", on \"Yes\" integrate and mark the source.",
        "components": [
            "Hypothesis: Formulated as \"If..., then...\".",
            "Segment: Which user group the hypothesis is for.",
            "Priority: How important this hypothesis is.",
            "Validation method: For example, in-depth interview, A/B test, prototype.",
        ],
        "methodology":
"""HADI cycles, ICE/RICE prioritization.
1.\tMove all assumptions from the Initiative Card to the backlog.
2.\tFormulate them as hypotheses. For example: "If we simplify the registration process, the percentage of successful authorizations will increase from 40% to 80%".
3.\tPrioritize hypotheses to start with the riskiest and most important.
""",
        "criteria": ["Hypothesis formula", "Success metric", "Priority (ICE/RICE)", "Link to pain"]
    },
    {
        "id": 4,
        "stage": "Discovery",
        "stage_goal": "Validate all hypotheses from the Initiative Card using data and customer conversations. Reduce risk by killing non-viable ideas before expensive development.",
        "name": "In-depth Interview (CustDev)",
        "goal": "Obtain qualitative data about problems, goals, and current user experience to validate hypotheses.",
        "data_source": "Always ask: \"Do you want to upload real data (interviews, tables, reports) or create manually?\n"
                "If a file is uploaded — make a brief summary (3-5 bullets), ask \"Use these insights?\", on \"Yes\" integrate and mark the source.",
        "components": [
            "Hypotheses: Which assumptions you are testing.",
            "Interview goals: What exactly you want to learn.",
            "Structure and questions: Detailed script with open-ended questions.",
            "Timing: How much time is allocated to each block.",
        ],
        "methodology":
"""CustDev, open-ended questions.
1.\tPreparation is 90% of success! Write the plan carefully.
2.\tAsk open questions about past experience. Do not ask "Would you...?" (foresight question). Ask "Tell me how you last solved the problem..." (retrospective question).
3.\tUse the "5 whys" rule to get to the root cause.
4.\tListen carefully, do not interrupt. Most of the time the respondent should speak.
5.\tRecord results in a table of insights for each respondent.
""",
        "criteria": ["Target sample", "Script", "Insights with quotes", "Links to raw data"]
    },
    {
        "id": 5,
        "stage": "Discovery",
        "stage_goal": "Validate all hypotheses from the Initiative Card using data and customer conversations. Reduce risk by killing non-viable ideas before expensive development.",
        "name": "Value Proposition",
        "goal": "Systematically match customer needs and product characteristics to ensure fit and clearly articulate the main benefit. This artifact helps \"sell\" the product concept to customers and stakeholders.",
        "data_source": "Always ask: \"Do you want to upload real data (interviews, tables, reports) or create manually?\n"
                "If a file is uploaded — make a brief summary (3-5 bullets), ask \"Use these insights?\", on \"Yes\" integrate and mark the source.",
        "components": [
            "Customer Profile (Circle):",
            "-- Customer jobs: What the customer is trying to do (functional, social, emotional).",
            "-- Customer pains: What bothers them and gets in the way.",
            "-- Customer gains: What outcomes and benefits they want.",
            "Value Map (Square):",
            "-- Products and services: What you offer.",
            "-- Pain relievers: How your product solves customer pains.",
            "-- Gain creators: How your product creates gains for the customer.",
        ],
        "methodology":
"""Value Proposition Canvas.
1.\tStart with the Customer Profile. Based on data from in-depth interviews, fill in customer jobs, pains, and gains.
2.\tThen fill in the Value Map. Describe how your product and its features help the customer by relieving pains and creating gains.
3.\tFind fit. Ensure your pain relievers and gain creators target the most important pains and gains.
4.\tFormulate the value proposition. Use a simple formula: "Our [product] helps [customer segment] who want to [do a job] by [relieving pain] and [creating gain]".
""",
        "criteria": ["Pain-to-benefit link", "Top-3 values", "Testable promises"]
    },
    {
        "id": 6,
        "stage": "Discovery",
        "stage_goal": "Validate all hypotheses from the Initiative Card using data and customer conversations. Reduce risk by killing non-viable ideas before expensive development.",
        "name": "Customer Journey Map (CJM)",
        "goal": "Visualize the full customer experience when interacting with the company or product to identify barriers, pain points, and negative emotions.",
        "data_source": "Always ask: \"Do you want to upload real data (interviews, tables, reports) or create manually?\n"
                "If a file is uploaded — make a brief summary (3-5 bullets), ask \"Use these insights?\", on \"Yes\" integrate and mark the source.",
        "components": [
            "Stages: Key phases of interaction (search, purchase, usage).",
            "Actions: What the customer does at each stage.",
            "Touchpoints: Where interaction happens (site, app, call center).",
            "Problems and barriers: What gets in the customer's way.",
            "Emotions: What the customer feels at each step (shown on a graph).",
            "Solutions: Ideas to remove barriers.",
        ],
        "methodology":
"""Path from need to decision.
1.\tCollect information: Use data from in-depth interviews.
2.\tDescribe the persona: Create a composite image of your customer.
3.\tMap all stages, actions, and touchpoints.
4.\tIdentify barriers and note where the customer experiences negative emotions.
5.\tGenerate hypotheses for solutions to remove these barriers.
""",
        "criteria": ["Stages", "Pains/emotions", "Touchpoints", "Improvement opportunities"]
    },
    {
        "id": 7,
        "stage": "Discovery",
        "stage_goal": "Validate all hypotheses from the Initiative Card using data and customer conversations. Reduce risk by killing non-viable ideas before expensive development.",
        "name": "Business Process Map",
        "goal": "Capture the current situation and systematize internal processes affected by your initiative. Unlike a CJM, this artifact shows interaction of internal roles, not just the customer.",
        "data_source": "Always ask: \"Do you want to upload real data (interviews, tables, reports) or create manually?\n"
                "If a file is uploaded — make a brief summary (3-5 bullets), ask \"Use these insights?\", on \"Yes\" integrate and mark the source.",
        "components": [
            "Roles: All participants in the process (not just the customer).",
            "Actions: Sequence of operations.",
            "Duration: Time for each action.",
            "Tools/Systems: Which programs or documents are used.",
            "Hypotheses about problems and solutions: Where bottlenecks are and how to improve them.",
        ],
        "methodology":
"""BPMN or a simple diagram.
1.\tCollect information: Talk to business analysts and employees involved in the process.
2.\tDefine the process entry and exit points to set clear boundaries.
3.\tMap all actions, roles, and tools used.
4.\tAnalyze the process and capture hypotheses about problems (e.g., "printing forms takes too much time") and solutions.
""",
        "criteria": ["AS-IS/TO-BE", "Inputs/outputs", "Owners", "Bottlenecks"]
    },
    {
        "id": 8,
        "stage": "Discovery",
        "stage_goal": "Validate all hypotheses from the Initiative Card using data and customer conversations. Reduce risk by killing non-viable ideas before expensive development.",
        "name": "Competitive Analysis",
        "goal": "Understand how other companies solve similar problems to position your product correctly, avoid others' mistakes, and find competitive advantages.",
        "data_source": "Always ask: \"Do you want to upload real data (interviews, tables, reports) or create manually?\n"
                "If a file is uploaded — make a brief summary (3-5 bullets), ask \"Use these insights?\", on \"Yes\" integrate and mark the source.\n"
                "**IMPORTANT**: Always use the web_search_summary tool to find competitors.",
        "components": [
            "Competitors: List of direct and indirect competitors.",
            "Customer segments: Who competitors target.",
            "Value proposition (USP): How they attract customers.",
            "Monetization model: How they earn money.",
            "Product characteristics: Key features.",
            "Price/cost.",
            "Customer reviews.",
        ],
        "methodology":
"""Compare by features, prices, experience.
1.\tFind competitors: Use search, marketing data, and industry reports. Analyze not only direct competitors, but also those who fight for the same customer value (e.g., for microloans, competitors include pawn shops).
2.\tCollect information on all key points and put it into a comparison table.
3.\tBecome a competitor's customer to study the product from the inside.
4.\tAnalyze strengths and weaknesses and determine where you can be better. Do not compete on price as the first option.
""",
        "criteria": [">=5 alternatives", "Comparison table", "Differentiation"]
    },
    {
        "id": 9,
        "stage": "Discovery",
        "stage_goal": "Validate all hypotheses from the Initiative Card using data and customer conversations. Reduce risk by killing non-viable ideas before expensive development.",
        "name": "Unique Selling Proposition (USP)",
        "goal": "Based on understanding of the customer and market, formulate one clear, short, and compelling statement that explains why your product is the best choice.",
        "components": [
            "Target audience: Who the product is for.",
            "Problem: What pain you solve.",
            "Solution/Product: What you offer.",
            "Unique differentiator: What makes you better than competitors.",
        ],
        "methodology":
"""Why customers will buy from us.
1.\tSynthesize knowledge: Return to "Value Proposition" (what the customer needs) and "Competitive Analysis" (what others offer).
2.\tFind your uniqueness: What is your key advantage that matters to the customer and is hard to copy? It could be a new technology, special service, more convenient design, etc.
3.\tFormulate the USP: Use a simple structure. For example: "For [target segment] who face [problem], our [product] provides [key advantage], unlike [competitors]".
4.\tValidate it: Your USP should be specific, memorable, and make people want to learn more.
""",
        "criteria": ["One clear formula", "Provable advantages", "Relevant to the segment"]
    },
    {
        "id": 10,
        "stage": "Design",
        "stage_goal": "Based on validated hypotheses, design a detailed solution, calculate its economics, and create an implementation plan.",
        "name": "Financial Model",
        "goal": "Assess the product's economic effect in detail by calculating all revenues and costs and determining the breakeven point.",
        "data_source": "Always ask: \"Do you want to upload real data (interviews, tables, reports) or create manually?\n"
                "If a file is uploaded — make a brief summary (3-5 bullets), ask \"Use these insights?\", on \"Yes\" integrate and mark the source.",
        "components": [
            "Costs (expenses)",
            "-- Variable: Project team, external contractors.",
            "-- Fixed: Support, licenses.",
            "Revenues (economic effect): Calculated based on key metrics.",
            "Breakeven point (TCO - Total Cost of Ownership): The moment when revenues start to exceed costs.",
        ],
        "methodology":
"""Unit economics, P&L.
1.\tCollect all types of expenses: Team, contractors, equipment, marketing, support, etc.
2.\tDistribute expenses by month according to the framework stages.
3.\tCalculate revenues: Take key metrics and show how changes impact money. For example, "a 10% conversion increase brings X revenue".
4.\tBuild three scenarios: negative, baseline, and positive.
5.\tDefine the breakeven point. In the product approach, it should be within 3-6 months.
""",
        "criteria": ["Key assumptions", "LTV/CAC/margin", "Sensitivity", "Scenarios"]
    },
    {
        "id": 11,
        "stage": "Design",
        "stage_goal": "Based on validated hypotheses, design a detailed solution, calculate its economics, and create an implementation plan.",
        "name": "Roadmap",
        "goal": "Create a comprehensive work plan visualizing key phases, tasks, and dependencies for team coordination and stakeholder communication.",
        "data_source": "**IMPORTANT**: Always use the web_search_summary tool to obtain market prices.",
        "components": [
            "Tasks/Work packages: Decomposition of large goals.",
            "Timelines: Duration of each task.",
            "Owners: Who performs the task.",
            "Milestones: Key events marking phase completion.",
            "Critical path: The longest chain of tasks that defines the overall project duration.",
        ],
        "methodology":
"""Work plan.
1.\tDefine goals and development phases.
2.\tDecompose them into tasks.
3.\tDefine sequence, duration, and dependencies between tasks.
4.\tAssign owners.
5.\tSet milestones at the end of each significant phase (e.g., "Testing complete").
6.\tManage risks by adding time buffers (time reserve).
""",
        "criteria": ["Releases", "Goals/metrics", "Resources/risks", "Milestones"]
    },
    {
        "id": 12,
        "stage": "Design",
        "stage_goal": "Based on validated hypotheses, design a detailed solution, calculate its economics, and create an implementation plan.",
        "name": "Project Card",
        "goal": "Final document for project defense and budget approval for the Development stage. It combines and summarizes all work done in previous stages.",
        "components": [
            "Project summary: Brief description of the solution, validated problems, and target segments.",
            "MVP description: Tasks, pilot area, success metrics.",
            "Economics: Data from the financial model (revenues, costs, team costs).",
            "Project team: Roles, competencies, involvement (FTE).",
            "Risks: What can go wrong and how you plan to address it.",
        ],
        "methodology":
"""Project summary.
1.\tTransfer validated data: Unlike the Initiative Card, this should include only validated hypotheses about problems and solutions.
2.\tDescribe the MVP in detail: What exactly you will build, on which segment, and how success will be measured.
3.\tIntegrate the financial model: Provide exact revenue and cost numbers.
4.\tPresent the team needed for implementation.
5.\tThink through risks and propose mitigation approaches.
""",
        "criteria": ["Summary of 1-12", "Roles/responsibility", "Readiness criteria", "Go/No-Go"]
    },
]


ARTIFACTS: List[ArtifactDefinition] = []


def set_artifacts_locale(locale: str = "ru") -> None:
    target = ARTIFACTS_EN if locale == "en" else ARTIFACTS_RU
    ARTIFACTS.clear()
    ARTIFACTS.extend(copy.deepcopy(target))


def get_artifact_schemas(locale: str = "ru") -> Dict[str, Any]:
    if locale == "en":
        return {"options": ArtifactOptionsEn, "final": AftifactFinalTextEn}
    return {"options": ArtifactOptions, "final": AftifactFinalText}


set_artifacts_locale("ru")


ARTIFACTS_OLD: List[ArtifactDefinition] = [
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
