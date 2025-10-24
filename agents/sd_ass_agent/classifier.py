import requests
import json
import time

import logging
import config

#from langchain_openai import ChatOpenAI
#from langchain_community.chat_models import ChatYandexGPT
#from langchain_core.prompts import ChatPromptTemplate

from ..llm_utils import get_llm


model_uri=f'cls://{config.YA_FOLDER_ID}/yandexgpt-lite/latest'
header = {
    "Content-Type": "application/json",
    "Authorization": f"Api-Key {config.YA_API_KEY}",
}
api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification"

classification_task = "Определи тип запроса пользователя к базе знаний компании."
samples = [
    {
        "text": "Как оформить коммандировку?\n\nUser role: Сотрудник компании Интерлизинг",
        "label": "управление персоналом"
    },
    {
        "text": "Как оформить коммандировку?\n\nUser role: Сотрудник отдела продаж",
        "label": "управление персоналом"
    },
    {
        "text": "Как оформить коммандировку?\n\nUser role: Сотрудник техподдержки",
        "label": "управление персоналом"
    },
    {
        "text": "Какие параметры нужно указывать в лизинговой заявке?\n\nUser role: Сотрудник отдела продаж",
        "label": "продажи и продукты"
    },
    {
        "text": "Какие параметры нужно указывать в лизинговой заявке?\n\nUser role: Сотрудник техподдержки",
        "label": "техническая поддержка"
    },
    {
        "text": "Какие параметры нужно указывать в лизинговой заявке?\n\nUser role: Сотрудник компании Интерлизинг",
        "label": "продажи и продукты"
    },
    {
        "text": "не работает предрешение, что делать?\n\nUser role: Сотрудник техподдержки",
        "label": "техническая поддержка"
    },
    {
        "text": "не работает предрешение, что делать?\n\nUser role: Сотрудник компании Интерлизинг",
        "label": "управление персоналом"
    },
    {
        "text": "Какие есть субсидии?\n\nUser role: Сотрудник отдела продаж",
        "label": "продажи и продукты"
    },
    {
        "text": "Какие есть субсидии?\n\nUser role: Сотрудник техподдержки",
        "label": "техническая поддержка"
    },
    ]
lables = {
    "продажи и продукты": "sm_agent",
    "управление персоналом": "default_agent",
    "техническая поддержка": "sd_agent",
    "другое": "default_agent"
    }

#summariser_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

summariser_llm = get_llm("nano", provider="yandex", temperature=0)
#model_name=f'gpt://{config.YA_FOLDER_ID}/yandexgpt-lite/latest'
#summariser_llm = ChatYandexGPT(
#    #iam_token = None,
#    api_key = config.YA_API_KEY, 
#    folder_id=config.YA_FOLDER_ID, 
#    model_uri=model_name,
#    temperature=0
#    )


def summarise_request(request: str, maxlen: int = 256) -> str:
    if len(request) <= maxlen:
        return request
    try:    
        prompt = ("You have as an input series of user's requests to knowledgebase.\n" 
            "Please prepare ONE final request, which will request all information user needs to retrieve.\n" 
            "Always answer in Russian.\n" 
            f"UserRequest: {request}.")
        result = summariser_llm.invoke(prompt)
    except Exception as e:
        logging.error("Error occured at summarise_request.\nException: {e}")
        raise e
    return result.content

def classify_request(request: str) -> str:
    prompt = {
        "modelUri": model_uri,
        "text": request,
        "task_description": classification_task,
        "labels": list(lables.keys()),
        "samples": samples
    }
    try:
        response = requests.post(api_url, headers=header, json=prompt)
        response.raise_for_status()
        result = json.loads(response.text)
        predictions = result["predictions"]
        predictions.sort(key = lambda x: x["confidence"], reverse=True)
        defined_class = predictions[0]["label"]
        return lables[defined_class]
    except Exception as e:
        logging.error("Error occured at classify_request. Return 'default_agent'\nException: {e}")
        return "default_agent"

if __name__ == "__main__":
    user_queries =  [
        #"Какой порядок выхода в отпуск?",
        #"Какие виды субсидий МПТ существуют?",
        #"Как настроить калькулятор для коммерческого менеджера?",
        #"Какая погода в Бомбее?",
        "Подскажите, как оформить субсидию?",
        "А какие виды бывают?",
        "А какая процедура оформления?",
        "",
        "А какие льготы есть?",
        "Не, при оформлении сделки?",
        "А субсидии от правительства?",
        "Расскажи подробнее",
        "А от партнёров?",
        "Для клиентов",
        "КАкие партнёрс кие программы для клиентов вообще есть?",
        "Не надо мне присылать льготы для сотрудников!!!!!!!!!!!",
    ]

    query_str = ""
    for query in user_queries:
        if query == "":
            query_str = ""
            continue
        query_str = query_str + ("" if query_str=="" else ";") + query
        final_query = summarise_request(query_str, maxlen=8)
        query_class = classify_request(final_query)
        print(f"{query}:{query_str}:{final_query}:{query_class}")
        time.sleep(1)





