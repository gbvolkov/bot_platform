from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from platform_guardrails.middleware import PrivacyModelRequestMiddleware
from platform_guardrails.privacy import DEFAULT_PALIMPSEST_ENTITIES, PrivacyRail, map_strings


# Edit this block when you want to run the simulation by variable assignment.
SYSTEM_PROMPT = """Ты помогаешь писать красивые короткие тексты.
Если для создания текста требуется получить дополнительную информацию, используй инструмент web_search.



После того, как пользователь окончательно подтвердил финальные текст артефакта, вызови commit_artifact_final_text(final_text) и передай ТОЛЬКО текст артефакта.

"""

USER_REQUEST = """Напиши красивый текст про улицу Плющиха. Используй факты о людях, кто там проживал и напиши про окружающую местность" \
"""
TOOL_CALL_PARAMETERS = {"query": """улица Плющиха кто жил на Плющихе известные жители факты окружающая местность Девичье поле Новодевичий монастырь"""}
TOOL_CALLS_AFTER_ANONYMIZATION = {"query": """улица Плющиха кто жил на Плющихе известные жители факты окружающая местность Девичье поле Новодевичий монастырь
"""}
RAW_LLM_RESPONSE = """Плющиха — улица тихого московского достоинства, спрятанная в Хамовниках между Садовым кольцом и Москвой-рекой. Когда-то это была дорога на Смоленск: по ней ехали из города к переправам и дальше — в западные земли. Название закрепилось к концу XVII–XVIII века; старые московские предания связывают его с кабаком, стоявшим здесь, а сама улица долго помнила слободы, подворья и дворцовые хозяйства неподалёку — конюшенные дворы, ремесленные владения, служилую Москву.

Плющиха удивительно уцелела в 1812-м: большой пожар прошёл мимо многих домов, и потому здесь особенно ясно читается «домашняя» Москва XIX века — не парадная, а жилая. В переулках рядом и на самой улице селились люди, которые делали город культурным и умным: здесь жили представители семьи Толстых (с 1837 года), бывали профессора и врачи, художники и литераторы. Плющиха вообще умеет хранить частные биографии — не громко, а как хранят семейные письма: за окнами, за воротами, в глубине дворов.

И местность вокруг под стать этому настроению. Отсюда рукой подать до Девичьего поля и Новодевичьего монастыря — пространства, где Москва вдруг становится просторнее и светлее. Близость реки добавляет воздуха, а старые переулки держат тишину даже тогда, когда центр шумит. Поэтому Плющиха и сегодня воспринимается не как «улица-магистраль», а как длинная прогулка по памяти: шаг за шагом — от истории дороги на Смоленск к камерной Москве, где у каждого дома есть свой голос."""

USER_LLM_RESPONSE = """Плющиха — улица тихого московского достоинства, спрятанная в Хамовниках между Садовым кольцом и Москвой-рекой. Когда-то это была дорога на Смоленск: по ней ехали из города к переправам и дальше — в западные земли. Название закрепилось к концу XVII–XVIII века; старые московские предания связывают его с кабаком, стоявшим здесь, а сама улица долго помнила слободы, подворья и дворцовые хозяйства неподалёку — конюшенные дворы, ремесленные владения, служилую Москву.

Плющиха удивительно уцелела в 1812-м: большой пожар прошёл мимо многих домов, и потому здесь особенно ясно читается «домашняя» Москва XIX века — не парадная, а жилая. В переулках рядом и на самой улице селились люди, которые делали город культурным и умным: здесь жили представители семьи Толстых (с 1837 года), бывали профессора и врачи, художники и литераторы. Плющиха вообще умеет хранить частные биографии — не громко, а как хранят семейные письма: за окнами, за воротами, в глубине дворов.

И местность вокруг под стать этому настроению. Отсюда рукой подать до Девичьего поля и Новодевичьего монастыря — пространства, где Москва вдруг становится просторнее и светлее. Близость реки добавляет воздуха, а старые переулки держат тишину даже тогда, когда центр шумит. Поэтому В. Ф. Снегирёв и сегодня воспринимается не как «улица-магистраль», а как длинная прогулка по памяти: шаг за шагом — от истории дороги на Смоленск к камерной Москве, где у каждого дома есть свой голос.
"""

# Optional fields for the same simulation.
TOOL_NAME = "web_search"
TOOL_RESULT = """Краткий обзор по теме: улица Плющиха, известные жители, факты и ближайшая окрестность

- Общая информация:
  - Улица Плющиха находится в районе Хамовники Центрального округа Москвы. Протягивается от Смоленской улицы до 2-го переулка Тружеников.
  - Название возникло в XVIII веке по кабаку Плющева. Старые названия: Саввинская улица (по Саввинскому монастырю), Смоленская улица (по церкви Смоленской иконы Божьей Матери).
  - История улицы восходит к XV веку. По дороге на Смоленск вдоль Плющихи располагалось подворье ростовского архиерея; позже формировались слободы вокруг подворья.
  - В XV–XVII веках вдоль Плющихи формировались различные владения: западная сторона — Благовещенская/Бережковская, восточная — дворцовая Ружейная, рядом Новый Конюшенный двор. В 1649 году владения частично передали городу, но многие прежние собственники продолжали торговлю и строительство.
  - Московский пожар 1812 года затронул улицу мало: западная часть пострадала минимально.
  - К 1850 году число дворов на восточной стороне составляло 26, на западной — 27; каменные дома встречались, в основном у мелких чиновников и купцов. Угол с Смоленской улицей — церковь Смоленской иконы Богородицы и харчевня возле рынка.

- Знаменитые жители и памятные дома:
  - Дом 11 (Щербачёв) — с 1837 года там жили Толстые: семья Льва Толстого.
  - Дом 6 (сейчас — часть ГАИ) — в 1840-х годах там жил профессор А. О. Армфельд; гостили Аксаковы, М. П. Погодин, Н. В. Гоголь.
  - Дом 32 — Ф. А. Страхов, сторонник Толстого.
  - Дом 36 — поэт А. А. Фет.
  - Дом 20 — жил знаменитый художник В. И. Суриков (дом Ахматова).
  - Дом 23 — писатель И. И. Лажечников (1867).
  - Дом 28 — художник С. В. Иванов.
  - Дом 43 — художник С. В. Малютин.
  - Дом 62 — профессор В. Ф. Снегирёв, основоположник советской гинекологии.
  - Рядом — Долгий переулок (ныне улица Бурденко) — нейрохирург Н. Н. Бурденко.
  - Дорогомилово-Тишинский переулок — художник А. К. Саврасов до 1897 года (автор картины «Грачи прилетели»).
  - Угол Плющихи и Ружейного переулка — русский философ Иван Ильин.
  - Дом 31 — последние годы провела А. В. Книпер; в этом же доме жили Владимир Тимирёв и Елена Васильевна Сафонова.
  - Примечание: многие перечисленные дома после модернизаций не сохранились физически.

- Поэзия и кино:
  - Поэзия: 1907 год — стихотворение Ивана Алексеевича Бунина «На Плющихе».
  - Искусство и кино: на Плющихе снимались эпизоды фильмов и сцен клубного театра:
    - «Три тополя на Плющихе» упоминается как часть кинематографических проектов.
    - Клуб «Каучук» (Плющиха, 64) снимал сцены «Гамлета» для фильма «Берегись автомобиля» (1966 год).
    - В эпизодах фильма «Испытательный срок» (О. Ефремов, О. Табаков, В. Невинный) использована финальная сцена на Плющихе («Покровские ворота» и т. д.).

- Окрестности и датады:
  - Рядом исторически располагались слободы вокруг архиерейского подворья, Конюшенный двор и память о строениях, связанных с Оружейной палатой.
  - Важная часть—окрестности Софийского и Смоленского направлений, а также упоминания о соседних переулках и улицах, связанных с историей Москвы периода XVIII–XIX веков.

- О резюмирующая картина:
  - Улица Плющиха известна как старейшая часть центра Москвы с богатой историей владений и резиденций, где жили известные деятели культуры, искусства и медицины, а также связана с литературой, кино и Московской архитектурой прошлых столетий.
  - Окрестности улицы — важная часть Девичьего поля и Новодевичьего монастыря в контексте близлежащего ландшафта и исторических переулков, хотя прямой текст о Девичьем поле здесь не акцентирован.

Если нужно, могу дополнительно выделить конкретные годы жизни жильцов, связи между мастерами Оружейной палаты и архитектурой, или подготовить карту с адресами упомянутых домов.

** Ссылка на статью: https://vk.com/wall-51696045_52481 **
========= END OF DOCUMENT ============



Ключевые факты о улице Плющиха и окружающей местности:

- История названия и характер улицы:
  - Плющиха стала названием к концу XVII века, произошло от кабака купца Плющева; ранее по улице проходила дорога на Смоленск (Смоленская).
  - Кабак Плющева был популярен и хорошо обслуживался.

- Пожар 1812 года:
  - Западная часть улицы пострадала значительно менее; жилищные здания купцов и мелких чиновников сохранились по большей части.
  - В 80-е годы XIX века на месте деревянных зданий возводились современные.

- География и вид на реку:
  - Плющиха лежит между Садовым кольцом и Москвой-рекой; жители, выходящие окнами на реку, радуются, что перед ними не планируется застройка.

- Художественная и литературная история:
  - На Плющихе жили и творили многие известные деятели культуры: Василий Суриков (дом №20), Сергей Иванов (дом №28), Афанасий Фет (дом №36); Илья Репин писал здесь «Царевна Софья» после поездки в Новодевичий монастырь.
  - Подвал доходного дома А.Л. Баумгартена (№53/25) обитал писатель Андрей Белый с женой.
  - В 1990-е годы на Плющихе жил Леонид Маленков (полиция), известный за арест Чикатило.
  - Для фильма «Три тополя на Плющихе» использовался не сам дом на Плющихе, а дом №5 на Ростовской набережной; на Плющихе для съемок построили стеклянное кафе.

- Легендарный «дом Толстого»:
  - В начале улицы, за отелем «Золотое кольцо», стоит дом №?, где в XIX веке жили близ Толстого. Этот дом известен как «дом Толстого» и описан Левом Толстым в повести Детство, в том числе эпизод с падением с крыши и последовавшим выздоровлением.

- Современная недвижимость и цены (примерные):
  - Довоенный кирпичный дом (№13): квартира 34 м² на 2-м этаже — 9 млн ₽; студийная планировка, новые стеклопакеты, паркет.
  - Модерн-стиль (№22): квартира 165 м² на 2-м этаже — 69 млн ₽; стиль «мягкий модерн», кухня-гостиная, мастер-спальня и гардеробная.
  - Элитный Bunin (№37/21): квартира 68 м² — 68 млн ₽; двухкомнатная, есть встроенная техника; четырехкомнатная 183 м² — 218,12 млн ₽.
  - Дом Совмина (№42): однокомнатная 46 м² — 17,5 млн ₽.
  - Дом №44/2 (1910, с надстройкой 1950-х): 59 м² — 18 млн ₽; варианты мансардного использования.
  - Кирпичная сталинка (№58): однокомнатная 18 м² на 1-м этаже — 5,9 млн ₽; теплый дом.

Примечание по стилевым моментам в тексте:
- В обзоре встречаются упоминания различной эпохи застройки и конкретных квартир, а также краткие сведения об истории и знаменитых жителях.
- Адреса и характеристики домов даны с акцентом на архитектуру и стоимость квартир.

** Ссылка на статью: https://dzen.ru/a/X3JEUU-t4woqVG3E **
========= END OF DOCUMENT ============

"""
TENANT_ID = "demo_tenant"
USER_ID = "demo_user"
THREAD_ID = "demo_thread"
PALIMPSEST_LOCALE = "ru-RU"
PALIMPSEST_RUN_ENTITIES = DEFAULT_PALIMPSEST_ENTITIES


@dataclass(frozen=True)
class PrivacyFlowParams:
    system_prompt: str
    user_request: str
    tool_call_parameters: dict[str, Any]
    tool_calls_after_anonymization: dict[str, Any]
    raw_llm_response: str
    user_llm_response: str
    tool_name: str = "lookup_customer"
    tool_result: str = "CRM record: Ivan Petrov, phone +7 999 111-22-33, status VIP."
    tenant_id: str = "demo_tenant"
    user_id: str = "demo_user"
    thread_id: str = "demo_thread"
    palimpsest_locale: str = "ru-RU"
    palimpsest_run_entities: list[str] = field(default_factory=lambda: list(DEFAULT_PALIMPSEST_ENTITIES))


def _runtime(params: PrivacyFlowParams) -> SimpleNamespace:
    return SimpleNamespace(
        execution_info=None,
        config={
            "configurable": {
                "tenant_id": params.tenant_id,
                "user_id": params.user_id,
                "thread_id": params.thread_id,
            }
        },
    )


def _build_privacy_rail(params: PrivacyFlowParams) -> PrivacyRail:
    return PrivacyRail.from_palimpsest(
        locale=params.palimpsest_locale,
        run_entities=params.palimpsest_run_entities,
        verbose=False,
    )


def _build_middleware(rail: PrivacyRail) -> PrivacyModelRequestMiddleware:
    return PrivacyModelRequestMiddleware(
        rail,
        agent_name="simulate_privacy_flow",
        guard_tool_calls=True,
    )


def _scope_key(params: PrivacyFlowParams) -> str:
    return f"{params.tenant_id}|{params.user_id}|{params.thread_id}"


def _palimpsest_table(session: Any) -> list[dict[str, Any]]:
    ctx = getattr(session, "_ctx", None)
    rows: list[dict[str, Any]] = []
    faked = getattr(ctx, "_faked", {}) if ctx is not None else {}
    for item in faked.values():
        rows.append(
            {
                "entity_type": item.get("type") or item.get("entity_type"),
                "real": item.get("true"),
                "anonymous": item.get("fake"),
            }
        )
    if rows:
        return rows

    for anonymous, entry in getattr(session, "_anon_entries_by_text", {}).items():
        rows.append(
            {
                "entity_type": getattr(entry, "entity_type", None),
                "real": None,
                "anonymous": anonymous,
            }
        )
    return rows


def simulate_flow(params: PrivacyFlowParams) -> dict[str, Any]:
    rail = _build_privacy_rail(params)
    middleware = _build_middleware(rail)
    runtime = _runtime(params)

    first_model_capture: dict[str, Any] = {}
    first_request = ModelRequest(
        model=object(),
        system_prompt=params.system_prompt,
        messages=[HumanMessage(content=params.user_request)],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=runtime,
    )

    def first_llm_handler(updated_request: ModelRequest) -> ModelResponse:
        first_model_capture["system_prompt"] = updated_request.system_prompt
        first_model_capture["user_request"] = updated_request.messages[0].content
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": params.tool_name,
                            "args": params.tool_calls_after_anonymization,
                            "id": "call-1",
                        }
                    ],
                )
            ]
        )

    first_model_result = middleware.wrap_model_call(first_request, first_llm_handler)
    model_visible_tool_call = first_model_result.result[0].tool_calls[0]
    session = rail.sessions.get_session(_scope_key(params))
    palimpsest_tool_call_parameters_anonymized = map_strings(
        params.tool_call_parameters,
        session.anonymize,
    )

    tool_capture: dict[str, Any] = {}
    tool_request = ToolCallRequest(
        tool_call={
            "name": params.tool_name,
            "args": params.tool_calls_after_anonymization,
            "id": "call-1",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=runtime,
    )

    def tool_handler(updated_request: ToolCallRequest) -> ToolMessage:
        tool_capture["parameters"] = updated_request.tool_call["args"]
        return ToolMessage(content=params.tool_result, tool_call_id="call-1")

    tool_result = middleware.wrap_tool_call(tool_request, tool_handler)

    second_model_capture: dict[str, Any] = {}
    second_request = ModelRequest(
        model=object(),
        system_prompt=params.system_prompt,
        messages=[
            HumanMessage(content=params.user_request),
            tool_result,
        ],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=runtime,
    )

    def final_llm_handler(updated_request: ModelRequest) -> ModelResponse:
        second_model_capture["tool_result"] = updated_request.messages[1].content
        return ModelResponse(result=[AIMessage(content=params.raw_llm_response)])

    final_result = middleware.wrap_model_call(second_request, final_llm_handler)
    final_user_response = final_result.result[0].content
    palimpsest_raw_llm_response_deanonymized = session.deanonymize(params.raw_llm_response)

    checks = {
        "system_prompt_changed_by_palimpsest": first_model_capture["system_prompt"] != params.system_prompt,
        "user_request_changed_by_palimpsest": first_model_capture["user_request"] != params.user_request,
        "provided_tool_calls_match_palimpsest_anonymization": (
            params.tool_calls_after_anonymization == palimpsest_tool_call_parameters_anonymized
        ),
        "model_tool_call_deanonymized": model_visible_tool_call["args"] == params.tool_call_parameters,
        "tool_execution_parameters_deanonymized": tool_capture["parameters"] == params.tool_call_parameters,
        "tool_result_changed_by_palimpsest_for_llm": second_model_capture["tool_result"] != params.tool_result,
        "provided_user_response_matches_palimpsest_deanonymization": (
            params.user_llm_response == palimpsest_raw_llm_response_deanonymized
        ),
        "final_response_deanonymized": final_user_response == params.user_llm_response,
    }

    return {
        "checks": checks,
        "palimpsest_table": _palimpsest_table(session),
        "first_model_request": first_model_capture,
        "palimpsest_tool_call_parameters_anonymized": palimpsest_tool_call_parameters_anonymized,
        "provided_tool_calls_after_anonymization": params.tool_calls_after_anonymization,
        "model_visible_tool_call": model_visible_tool_call,
        "tool_execution_parameters": tool_capture["parameters"],
        "tool_result_visible_to_next_llm": second_model_capture["tool_result"],
        "raw_llm_response": params.raw_llm_response,
        "palimpsest_raw_llm_response_deanonymized": palimpsest_raw_llm_response_deanonymized,
        "user_llm_response": final_user_response,
    }


def demo_params() -> PrivacyFlowParams:
    return PrivacyFlowParams(
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
        tool_call_parameters=TOOL_CALL_PARAMETERS,
        tool_calls_after_anonymization=TOOL_CALLS_AFTER_ANONYMIZATION,
        raw_llm_response=RAW_LLM_RESPONSE,
        user_llm_response=USER_LLM_RESPONSE,
        tool_name=TOOL_NAME,
        tool_result=TOOL_RESULT,
        tenant_id=TENANT_ID,
        user_id=USER_ID,
        thread_id=THREAD_ID,
        palimpsest_locale=PALIMPSEST_LOCALE,
        palimpsest_run_entities=PALIMPSEST_RUN_ENTITIES,
    )


def _load_params(path: Path) -> PrivacyFlowParams:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("replacements", None)
    return PrivacyFlowParams(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate privacy anonymization/deanonymization flow.")
    parser.add_argument("--input", type=Path, help="JSON file with PrivacyFlowParams fields.")
    args = parser.parse_args()
    params = _load_params(args.input) if args.input else demo_params()
    print(json.dumps(simulate_flow(params), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
