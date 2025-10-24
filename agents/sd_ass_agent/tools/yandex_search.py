import requests
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import base64
from xml.etree import ElementTree as ET
import trafilatura
from duckduckgo_search.utils import _normalize, _normalize_url

import logging

# Input schema for the tool
class YandexSearchInput(BaseModel):
    query: str = Field(..., description="Search query for web search")

class YandexSearchTool(BaseTool):
    name: str = "yandex_web_search"
    description: str = (
        "A wrapper around Yandex Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = YandexSearchInput

    api_key: str
    folder_id: str
    max_results: int = 5
    max_size: int = 16384

    def _run(self, query: str) -> str:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = "https://searchapi.api.cloud.yandex.net/v2/web/search"
        payload = {
            "query": {
                "searchType": "SEARCH_TYPE_RU",
                "queryText": query
            },
            "groupSpec": {
                "groupMode": "GROUP_MODE_FLAT",
                "groupsOnPage": self.max_results,
            },
            "folderId": self.folder_id,
            "responseFormat": "FORMAT_XML"
        }

        try:
            return self._get_data(endpoint, headers, payload)
        except Exception as e:
            logging.error("Error occured at summarise_request.\nException: {e}")
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
            results.append(
                {
                    "title": _normalize(title),
                    "href": href,
                    "body": f"{_normalize(body)[:self.max_size]}\n\n** Ссылка на статью: {href} **\n========= END OF DOCUMENT ============\n\n",
                }
            )

        return "\n\n".join(r["body"] for r in results)
