import requests
from typing import Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import base64
from xml.etree import ElementTree as ET
import trafilatura
from duckduckgo_search.utils import _normalize, _normalize_url

from agents.llm_utils import get_llm
import logging



SEARCH_TOOL_POLICY_PROMPT_RU = """

==================================================================================================================================================
### Web Search
1. **Запрет самовольного поиска.**  
   Веб-поиск запрещён без явного запроса пользователя.  
2. **Вызов `web_search`.**  
   Если пользователь явно попросил интернет/внешние источники, ты **ДОЛЖЕН** вызвать `web_search`.  
   Во всех остальных случаях **НЕ** вызывай `web_search`.  
3. **Язык запроса.**  
   Сначала пробуй на русском, затем на английском при необходимости.  
4. **Упорный поиск.**  
   Если результатов недостаточно, расширяй запрос (синонимы, альтернативные термины) и повторяй, пока не получишь данные или не исчерпаешь разумные варианты.  
   *ВАЖНО*: не более 3 поисков за ход.  
5. **Разделение источников.**  
   Внешние данные явно отделяй от отчёта «Разведчика» и не выдавай гипотезы за факты.  
6. **Формат ссылок.**  
   Внешние ссылки выводи полностью в Markdown и в угловых скобках: Название/домен — <https://...> (не сокращать).  
7. **Тайминг ответа.**  
   Не отправляй свободный текст пользователю, пока не обработал результаты `web_search` (если вызван).
==================================================================================================================================================
"""

SEARCH_TOOL_POLICY_PROMPT_EN = """

==================================================================================================================================================
### Web Search
1. **No autonomous search.**  
   Web search is forbidden without an explicit user request.  
2. **Calling `web_search`.**  
   If the user explicitly asks for internet/external sources, you **MUST** call `web_search`.  
   Otherwise you **MUST NOT** call `web_search`.  
3. **Query language.**  
   Use English whenever it is possble.  
4. **Persistent search.**  
   If results are insufficient, broaden the query (synonyms, alternatives) and retry until you have enough data or exhaust reasonable options.  
   *IMPORTANT*: Max 3 searches per turn.  
5. **Source separation.**  
   Clearly separate external data from the «Разведчик» report and do not present hypotheses as facts.  
6. **Link format.**  
   Output external links in full Markdown with angle brackets: Title/domain — <https://...> (no shortening).  
7. **Answer timing.**  
   Do **not** send any free-text response to the user until you have processed `web_search` results (if invoked).
==================================================================================================================================================
"""


summariser_llm = get_llm("nano", provider="openai", temperature=0, streaming=False)


def extract_text_from_content(content: Any) -> str:
    parts: list[str] = []

    def _append(text: str) -> None:
        if text:
            parts.append(text)

    def _walk(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            _append(obj)
            return
        if isinstance(obj, bytes):
            try:
                _append(obj.decode("utf-8"))
            except Exception:
                _append(obj.decode(errors="ignore"))
            return
        if isinstance(obj, dict):
            for key in ("text", "output_text", "input_text"):
                value = obj.get(key)
                if isinstance(value, str):
                    _append(value)
            if "content" in obj:
                _walk(obj.get("content"))
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)
            return

        text_attr = getattr(obj, "text", None)
        if isinstance(text_attr, str):
            _append(text_attr)
            return
        content_attr = getattr(obj, "content", None)
        if content_attr is not None:
            _walk(content_attr)
            return

    _walk(content)
    if parts:
        return "\n".join(p for p in parts if p)
    return str(content)
def summarise_content(content: str, thematic: str, maxlen: int = 4096) -> str:
    if len(content) <= maxlen:
        return content
    try:    
        prompt = ("You have as an input a content of webpages specific for a given thematic.\n" 
            "Please prepare summary of the content related to the information for a given thematic.\n" 
            "Always answer in Russian.\n" 
            f"Thematic: {thematic}."
            f"Content: {content}.\n"
            f"The length of the response shall not exceed {maxlen} characters.")
        result = summariser_llm.invoke(prompt)
    except Exception as e:
        logging.error("Error occured at summarise_content.\nException: %s", e)
        raise
    return extract_text_from_content(getattr(result, "content", result))


def _build_payload(api_key: str, query: str, max_results: int, folder_id: str):
    _headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }
    _endpoint = "https://searchapi.api.cloud.yandex.net/v2/web/search"
    _payload = {
        "query": {
            "searchType": "SEARCH_TYPE_RU",
            "queryText": query
        },
        "groupSpec": {
            "groupMode": "GROUP_MODE_FLAT",
            "groupsOnPage": max_results,
        },
        "folderId": folder_id,
        "responseFormat": "FORMAT_XML"
    }
    return _endpoint, _headers, _payload


# Input schema for the tool
class YandexSearchInput(BaseModel):
    query: str = Field(..., description="Search query for web search. Can be in Russian (prefferred) on in English.")

class YandexSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "A wrapper around Yandex Search. "
        "Useful for when you need an information from internet."
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = YandexSearchInput

    api_key: str
    folder_id: str
    max_results: int = 5
    max_size: int = 16384
    summarize: bool = False
    
    query: str = ""

    def _run(self, query: str) -> str:
        self.query = query

        endpoint, headers, payload = _build_payload(self.api_key, query, self.max_results, self.folder_id)

        try:
            return self._get_data(endpoint, headers, payload)
        except Exception as e:
            logging.error(f"Error occured at summarise_request.\nException: {e}")
            return f"Yandex Search failed: {e}"

    def _extract_url_content(self, url: str) -> str:
        html = trafilatura.fetch_url(url)
        return trafilatura.extract(html)

    def _get_data(self, endpoint, headers, payload):
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        data = base64.b64decode(result["rawData"]).decode("utf-8")
        root = ET.fromstring(data)
        results = []
        for doc in root.findall(".//doc"):
            title = doc.findtext("title")
            url = doc.findtext("saved-copy-url")
            href = doc.findtext("url")
            try:
                body = self._extract_url_content(url)
            except Exception as e:
                continue
            href = _normalize_url(href)
            if self.summarize:
                body = summarise_content(_normalize(body)[:self.max_size], self.query, 2048)
            results.append(
                {
                    "title": _normalize(title),
                    "href": href,
                    "body": f"{_normalize(body)[:self.max_size]}\n\n** Ссылка на статью: {href} **\n========= END OF DOCUMENT ============\n\n",
                }
            )

        return "\n\n".join(r["body"] for r in results)



class YandexRetrieveSummaryTool(YandexSearchTool):
    name: str = "web_search_summary"
    description: str = (
        "A wrapper and summariser around Yandex Search. "
        "You shall use the tool to get summary of related information from the internet. "
        "Input should be a search query."
    )

    def _run(self, query: str) -> str:
        self.query = query

        endpoint, headers, payload = _build_payload(self.api_key, query, self.max_results, self.folder_id)

        try:
            docs = self._get_data(endpoint, headers, payload)
            return summarise_content(docs, self.query, maxlen=8096)
        except Exception as e:
            logging.error(f"Error occured on data scrapping and summarisation.\nException: {e}")
            return f"Yandex Search failed: {e}"
