from __future__ import annotations

import functools
import json
import requests
from typing import List, Optional, Dict
from pydantic import BaseModel

import pandas as pd

import config

from langchain.schema import Document

from .utils.load_table import (
    get_data_from_json
    , get_glossary_data
    , get_st_data
)
from .utils.build_documents import (
    get_documents_for_sd_qa
    , get_documents_for_sd_tickets
    , get_documents_for_glossary
)

from abc import ABC, abstractmethod
import logging

# ---------------------------------------------------------------------------
# Helper decorator – unchanged except for typing tweaks
# ---------------------------------------------------------------------------

def _authorization_wrapper(func):
    """Refreshes / exchanges the token if an API call returns 401."""
    @functools.wraps(func)
    def wrapper(self: "TeamlyAPIWrapper", *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except requests.HTTPError as http_err:
            #TODO!!! After Teamly fix, remove 500 error from here!
            if http_err.response.status_code not in [401, 500]:
                raise
            # token may have expired → refresh & retry once
            self.auth_code = self._get_token()
            return func(self, *args, **kwargs)
    return wrapper

def _get_article_text(base_url: str, article_info: Dict) -> str:
    """
    Traverse a ProseMirror-like editor JSON (embedded as a string inside the
    editorContentObject) and build a single text string where:

    • every node whose type is "text" contributes its text value  
    • every node (or mark) whose type is "url" *or* "link" contributes the URL  
    • every node whose type is "paragraph" contributes a newline after its children  

    The traversal is depth-first and keeps the document’s original order.
    """
    # 1. Parse the JSON that lives in editorContentObject["content"]
    raw_doc = article_info["editorContentObject"]["content"]
    doc = json.loads(raw_doc)

    pieces: List[str] = [article_info["title"], "\n"]

    def walk(nodes: Dict | list) -> None:
        if not isinstance(nodes, list):
            nodes = [nodes]
        for node in nodes:
            ntype = node.get("type")

            # ---  text nodes ----------------------------------------------------
            if ntype == "text":
                pieces.append(node.get("text", ""))

                # links may be attached as marks on a text node  -----------------
                for mark in node.get("marks", []):
                    if mark.get("type") in {"url", "link"}:
                        url_obj = mark.get("attrs", {}).get("link") or mark.get("attrs", {})
                        url_placement = url_obj.get("type", "external")
                        url = url_obj.get("url")
                        if url and isinstance(url, str):
                            #if url_placement == "internal":
                            #    url = f" {self.base_url}{url}"
                            if not url.startswith("https:") and not url.startswith("mailto:"):
                                url = base_url + url
                            pieces.append(f" {url} ")
                    elif mark.get("type") in {"media"}:
                        url = mark.get("attrs", {}).get("src")
                        if url and isinstance(url, str) and not url.startswith("data:"):
                            if not url.startswith("https:"):
                                url = base_url + url
                            pieces.append(f" {url} ")

            # ---  dedicated url / link nodes ------------------------------------
            elif ntype in {"url", "link"}:
                url_obj = node.get("attrs", {}).get("link") or node.get("attrs", {})
                url_placement = url_obj.get("type", "external")
                url = url_obj.get("url")
                if url and isinstance(url, str):
                    #if url_placement == "internal":
                    #    url = f" {self.base_url}{url}"
                    if not url.startswith("https:") and not url.startswith("mailto:"):
                        url = base_url + url
                    pieces.append(f" {url} ")

            # ---  dedicated url / link nodes ------------------------------------
            elif ntype in {"media"}:
                url = node.get("attrs", {}).get("src") or node.get("attrs", {})
                if url and isinstance(url, str) and not url.startswith("data:"):
                    if not url.startswith("https:") and not url.startswith("mailto:"):
                        url = base_url + url
                    pieces.append(f" {url} ")

            # ---  newline after a paragraph -------------------------------------
            elif ntype == "paragraph":
                pieces.append("\n")

            # ---  recurse into children -----------------------------------------
            for child in node.get("content", []):
                walk(child)


    # The root is usually a {"type": "doc", ...}
    walk(doc)
    return "".join(pieces)[:128000]

# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

GLOSSARY_ARTICLE_IDS=[
    "91c22083-a2cc-4928-bfad-5925b2da021f"
    , "edf2f482-f2a6-46ac-91df-1da136a5d3ef"
    , "fc5c69d0-0f39-4f6f-a8f8-5ba2ba592866"
    , "6e2d33ed-094d-4261-9aa9-058d56994ead"
    , "685fda38-24f0-4560-8afc-a623535b5a8a"
    , "ff5a5245-8f6d-4e29-af09-35acd6245e2e"
]

class TeamlyAPIWrapper(BaseModel, ABC):
    """
    Teamly API Wrapper.

    Required auth/connection data are read from *auth_data_store* – the same
    JSON structure you used before:

    ```json
    {
      "base_url":      "...",
      "client_id":     "...",
      "client_secret": "...",
      "auth_code":     "...",   # either refresh-token or one-shot code
      "redirect_uri":  "..."
    }
    ```
    """
    base_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    auth_code: str = ""
    redirect_uri: str = ""
    sd_documents: List[Document] = []
    articles_json_path: str = ""
    articles_data_path: str = ""
    rename_map: dict = {}
    k: int = config.MAX_TEAMLY_DOCS
    
    # ---------------------------------------------------------------------
    # Construction / auth helpers
    # ---------------------------------------------------------------------

    def __init__(self, auth_data_store: str) -> None:
        """
        Parameters
        ----------
        auth_data_store:
            Path to the JSON file that stores credentials and the last
            (refresh) token. The file will be updated in-place whenever a new
            token is issued.
        k:
            How many documents to return for every query.
        """
        super().__init__()

        self._auth_path = auth_data_store
        with open(auth_data_store, "r", encoding="utf-8") as f:
            self._auth_data = json.load(f)

        self.base_url: str = self._auth_data["base_url"].rstrip("/")
        self.client_id: str = self._auth_data["client_id"]
        self.client_secret: str = self._auth_data["client_secret"]
        self.auth_code: str = self._auth_data["auth_code"]  # may be refresh-token
        self.redirect_uri: str = self._auth_data["redirect_uri"]
        # dynamic header (gets rebuilt after refresh)
        self._headers = lambda: {
            "X-Account-Slug": "default",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_code}",
        }

        self._load_sd_documents()

    @abstractmethod
    def get_documents(self, df: pd.DataFrame) -> list[Document]:
        pass

    def parse_json(self, data: str, space_id: str, article_id: str, article_title: str):
        return get_data_from_json(data, space_id, article_id, article_title, self.rename_map)

    def _get_article_data(self, article_id: str)-> pd.DataFrame:
        article_info = self.get_article_info(article_id)
        space_id = article_info["space_id"]
        article_title = article_info["title"]
        raw_doc = article_info["editorContentObject"]["content"]
        doc = json.loads(raw_doc)
        return self.parse_json(doc, space_id, article_id, article_title)

    def _load_sd_documents(self):
        if config.RETRIEVER_TYPE == "teamly":
            with open (self.articles_json_path, "r") as f:
                articles = json.load(f)
            df = pd.DataFrame()
            for article_id in articles["articles"]:
                article_df = self._get_article_data(article_id)
                df = pd.concat([df, article_df], ignore_index=True)
            self.sd_documents = self.get_documents(df) 
        else:
            with open(self.articles_data_path, "r", encoding="utf-8") as f:
                docs_json = json.load(f)
            self.sd_documents = [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in docs_json]

    def get_documents_from_teamly_search(self, query: str) -> List[Document]:
        """Synchronous retrieval used by most LC components."""
        try:
            raw_hits = self._semantic_search(query)[: self.k]
            
            # Group hits by space_id and article_id
            grouped_hits = {}
            for hit in raw_hits:
                hit_type = hit["type"]

                if hit_type == "article" and hit["source_id"] in GLOSSARY_ARTICLE_IDS:
                    continue
                key = (hit["container_id"], hit["source_id"], hit_type)
                if key not in grouped_hits:
                    grouped_hits[key] = []
                grouped_hits[key].append(hit)
            
            # Merge hits with same space_id and article_id
            documents = []
            for hits in grouped_hits.values():
                if len(hits) == 1:
                    documents.append(self._to_document(hits[0]))
                else:
                    # Merge multiple hits into one document
                    hits = sorted(hits, key=lambda x: x["offset"])

                    merged_text = "\n\n".join(hit["text"] for hit in hits)
                    max_score = max(hit["score"] for hit in hits)
                    total_token_length = sum(hit["chunk_token_length"] for hit in hits)
                    total_length = sum(hit["length"] for hit in hits)
                    offsets = [hit["offset"] for hit in hits]
                    
                    # Use the first hit as base for metadata
                    base_hit = hits[0]
                    hit_type = base_hit["type"]
                    pretty_link = ""
                    if hit_type == "article":
                        pretty_link = f"Ссылка на статью:https://kb.ileasing.ru{base_hit['link']}"
                    elif hit_type == "doc":
                        pretty_link = f"Ссылка на документ:https://kb.ileasing.ru/api/v1/disk{base_hit['link']}/view_pdf"
                    document = Document(
                        page_content=f"{merged_text}\n\n{pretty_link}",
                        metadata={
                            "docid": f"{base_hit['container_id']}_{base_hit['source_id']}_{hit_type}_merged",
                            "space_id": base_hit["container_id"],
                            "article_id": base_hit["source_id"],
                            "article_title": base_hit["title"],
                            "score": max_score,
                            "chunk_token_length": total_token_length,
                            "offset": offsets,
                            "length": total_length,
                            "merged": True,  # Flag indicating this is a merged document
                            "original_hits_count": len(hits),  # How many hits were merged
                            "type": hit_type,
                            "source": f"semantic"
                        },
                    )
                    documents.append(document)
        except Exception as e:
            logging.error(f"Error occured at 'TeamlyAPIWrapper::get_documents_from_teamly_search'.\nException: {e}")
            raise e
        return documents

    # ---------------------------------------------------------------------
    # Internal request helpers
    # ---------------------------------------------------------------------

    @_authorization_wrapper
    def _post(self, url: str, payload: dict) -> dict:
        try:
            resp = requests.post(f"{self.base_url}{url}", json=payload, headers=self._headers())
            resp.raise_for_status()
        except Exception as e:
            logging.error(f"Non-critical exception occured at 'TeamlyAPIWrapper::_post'.\nException: {e}\nResponse: {resp}\nURL:{url}\nPayload: {payload}")
            raise e
        return resp.json()

    def _auth_post(self, url: str, payload: dict) -> str:
        # No bearer header during auth flow
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json", "Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        refresh_token = data.get("refresh_token")
        access_token  = data.get("access_token")

        # Persist fresh refresh-token
        self.auth_code = refresh_token or access_token
        self._auth_data["auth_code"] = self.auth_code
        with open(self._auth_path, "w", encoding="utf-8") as fp:
            json.dump(self._auth_data, fp, ensure_ascii=False, indent=4)

        return access_token

    # token endpoints -----------------------------------------------------

    def _refresh_token(self) -> str:
        url = f"{self.base_url}/api/v1/auth/integration/refresh"
        return self._auth_post(
            url,
            {
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.auth_code,
            },
        )

    def _authorise(self) -> str:
        url = f"{self.base_url}/api/v1/auth/integration/authorize"
        return self._auth_post(
            url,
            {
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri":  self.redirect_uri,
                "code":          self.auth_code,
            },
        )

    def _get_token(self) -> str:
        try:
            return self._refresh_token()
        except requests.HTTPError:
            return self._authorise()

    # ---------------------------------------------------------------------
    # Semantic search & conversion
    # ---------------------------------------------------------------------

    def _semantic_search(self, query: str) -> list[dict]:
        """Raw semantic search – returns the provider’s JSON hits."""
        url = "/api/v1/semantic/external/search"
        payload = {
            "query": query,
            "limit_count": self.k
        }
        return self._post(url, payload)

    def get_article(self, article_id: str, max_length: int = 0) -> str:
        article_info = self.get_article_info(article_id, max_length)
        return _get_article_text(self.base_url, article_info)

    def get_article_info(self, article_id: str, max_length: int = 0) -> str:
        url = "/api/v1/wiki/ql/article"
        payload = {
            "query": {
                "__filter": {
                            "id": article_id,
                            "editorContentAfterVersionAt": 1711433043
                        },
                "title": True,
                "space_id": True,
                "editorContentObject": {
                    "content": True
                }                
            }
        }
        return self._post(url, payload)

    def get_smart_table(self, st_id: str) -> str:
        url = "/api/v1/ql/content-database/content"

        payload = {
            "query": {
                "__filter": {
                            "contentDatabaseId": st_id
                        },
                "id": True,
                "title": True,
                "container": {
                    "id": True,
                },
                "schemaProperties": {
                    "id": True,
                    "spaceId": True,
                    "propertyId": True,
                    "name": True,
                    "type": True,
                    "code": True
                },
                "content": {
                    "article": {
                        "id": True,
                        "properties": {
                            "properties": True
                        }
                    },
                    "hasNested": True
                },            
            }
        }
        return self._post(url, payload)  

    def get_smart_table_content(self, st_id: str):
        row_content = self.get_smart_table(st_id)
        return get_st_data(row_content)

    def _to_document(self, hit: dict) -> Document:
        """
        Convert one semantic-search hit (see sample payload below) into a
        LangChain Document.  We keep the full chunk text as page_content
        and put *everything else* into metadata so you can reference it
        later for filtering, citation, etc.

        Sample hit:
        {
            "space_id": "211ca458-...",
            "article_id": "f3d0cc75-...",
            "article_title": "Искусственный интеллект в медицине",
            "score": 0.80609,
            "text": "...",
            "chunk_token_length": 341,
            "offset": 0,
            "length": 1429
        }
        """
        hit_type = hit['type']
        pretty_link = ""
        link = ""
        if hit_type == "article":
            link = f"https://kb.ileasing.ru{hit['link']}"
            pretty_link = f"Ссылка на статью:{link}"
        elif hit_type == "doc":
            link = f"https://kb.ileasing.ru/api/v1/disk{hit['link']}/view_pdf"
            pretty_link = f"Ссылка на документ:{link}"
        return Document(
            page_content=f"{hit["text"]}\n\n{pretty_link}",
            metadata={
                "docid": f"{hit['container_id']}_{hit['source_id']}_{hit['type']}_merged",
                "space_id": hit["container_id"],
                "article_id": hit["source_id"],
                "article_title": hit["title"],
                "link": link,
                "score": hit["score"],
                "chunk_token_length": hit["chunk_token_length"],
                "offset": hit["offset"],
                "length": hit["length"],
                "merged": False,  # Flag indicating this is a merged document
                "original_hits_count": 1,  # How many hits were merged
                "type": hit_type,
                "source": "semantic"
            },
        )

class TeamlyAPIWrapper_SD_QA(TeamlyAPIWrapper):
    rename_map: dict = {
        "ИС": "it_system",
        "Описание проблемы": "problem_description",
        "Решение": "problem_solution",
        # Add others if you need them:
        "Пример Тикетов": "ticket_example"
    }
    articles_json_path: str = "./data/sd_articles.json"
    articles_data_path: str = "./data/QA_data.json"

    def get_documents(self, df: pd.DataFrame) -> list[Document]:
        return get_documents_for_sd_qa(df)


class TeamlyAPIWrapper_SD_Tickets(TeamlyAPIWrapper):
    rename_map: dict = {
        "Номер": "ticket_no",
        "Тема": "topic",
        "Дата регистрации": "ticket_dt",
        "Критерий ошибки": "ticket_type",
        "Описание текст": "problem",
        "Описание решения": "solution"
    }
    articles_json_path: str = "./data/sd_tickets.json"
    articles_data_path: str = "./data/tickets_data.json"


    def _get_article_data(self, article_id: str)-> pd.DataFrame:
        row_content = self.get_smart_table(article_id)
        return get_st_data(row_content)
        

    def get_documents(self, df: pd.DataFrame) -> list[Document]:
        return get_documents_for_sd_tickets(df)


class TeamlyAPIWrapper_Glossary(TeamlyAPIWrapper):
    rename_map: dict = {
    }
    articles_json_path: str = "./data/glossary_articles.json"
    articles_data_path: str = "./data/glossary_data.json"
    def parse_json(self, data: str, space_id: str, article_id: str, article_title: str):
        return get_glossary_data(data, space_id, article_id, article_title, self.rename_map)
    def get_documents(self, df: pd.DataFrame) -> list[Document]:
        return get_documents_for_glossary(df)



if __name__ == "__main__":
    from typing import Any
    from pprint import pprint
    query = "Как параметры АД зависят от EL?"

    teamply_wrapper = TeamlyAPIWrapper_Glossary("./auth.json")
    docs = teamply_wrapper.sd_documents
    docs_json = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(teamply_wrapper.articles_data_path, "w", encoding="utf-8") as f:
        json.dump(docs_json, f)

    result = teamply_wrapper.get_documents_from_teamly_search(query)
    pprint(result)

    pprint(teamply_wrapper.sd_documents)

    query = "Как удалить карточку 51 счёта?"
    teamply_wrapper = TeamlyAPIWrapper_SD_Tickets("./auth.json")
    docs = teamply_wrapper.sd_documents
    docs_json = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(teamply_wrapper.articles_data_path, "w", encoding="utf-8") as f:
        json.dump(docs_json, f)


    result = teamply_wrapper.get_documents_from_teamly_search(query)
    pprint(result)

    pprint(teamply_wrapper.sd_documents)


    query = "Как удалить карточку 51 счёта?"
    teamply_wrapper = TeamlyAPIWrapper_SD_QA("./auth.json")
    docs = teamply_wrapper.sd_documents
    docs_json = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(teamply_wrapper.articles_data_path, "w", encoding="utf-8") as f:
        json.dump(docs_json, f)


    result = teamply_wrapper.get_documents_from_teamly_search(query)
    pprint(result)

    pprint(teamply_wrapper.sd_documents)
