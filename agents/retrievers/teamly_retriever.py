from __future__ import annotations

import logging

from typing import Any
from asyncio import get_running_loop
from abc import ABC, abstractmethod

import re

import config

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain_classic.schema import Document
from langchain_classic.schema.retriever import BaseRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from .utils.build_index import get_retrievers
from .teamly_api_wrapper import (
    TeamlyAPIWrapper_SD_QA
    , TeamlyAPIWrapper_SD_Tickets
    , TeamlyAPIWrapper_Glossary
    , TeamlyAPIWrapper
)



# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class HybridTeamlyRetriever(BaseRetriever, ABC):
    """Shared logic for hybrid FAISS + BM25 + Teamly remote search."""
    wrapper: TeamlyAPIWrapper = None
    k: int = 5
    idx_vectors: FAISS = None
    idx_bm25: BM25Retriever = None

    def __init__(
        self,
        wrapper: TeamlyAPIWrapper,
        *,
        k: int = 10,
    ):
        # cooperative init: wrapper already ran, BaseRetriever next
        super().__init__()
        self.wrapper = wrapper
        self.k = k
        self.load_index()


    def _index_search(self, query: str) -> list[Document]:
        try:
            docs = []
            if self.idx_vectors:
                docs.extend(self.idx_vectors.similarity_search(query, k=self.k))
            if self.idx_bm25:
                docs.extend(self.idx_bm25.invoke(query)[: self.k])

            return [
                d
                for seen in [set()]
                for d in docs
                if not (d.page_content in seen or seen.add(d.page_content))
            ]
        except Exception as e:
            logging.error(f"Error occured at 'HybridTeamlyRetriever::_index_search'.\nException: {e}")
            raise e

    @abstractmethod
    def _get_relevant_documents(self, query: str, *, run_manager, **kw):
        pass

    async def _aget_relevant_documents(self, query: str, **kw):
        loop = get_running_loop()
        return await loop.run_in_executor(None, self._get_relevant_documents, query)

    def load_index(self):
        (self.idx_vectors, self.idx_bm25) = get_retrievers(self.wrapper.sd_documents)            

    def refresh(self):
        self.wrapper._load_sd_documents()
        self.load_index()

class TeamlyRetriever_Tickets(HybridTeamlyRetriever):
    def __init__(self, auth_data, **kw):
        wrapper = TeamlyAPIWrapper_SD_Tickets(auth_data_store=auth_data)
        super().__init__(wrapper, **kw)
    def _get_relevant_documents(self, query: str, *, run_manager, **kw):
        return (
            self._index_search(query)
        )

class TeamlyRetriever(HybridTeamlyRetriever):
    def __init__(self, auth_data_store, **kw):
        wrapper = TeamlyAPIWrapper_SD_QA(auth_data_store=auth_data_store)
        super().__init__(wrapper, **kw)
    def _get_relevant_documents(self, query: str, *, run_manager, **kw):
        return (
            self.wrapper.get_documents_from_teamly_search(query)
            + self._index_search(query)
        )

class TeamlyRetriever_Glossary(HybridTeamlyRetriever):
    def __init__(self, auth_data_store, **kw):
        wrapper = TeamlyAPIWrapper_Glossary(auth_data_store=auth_data_store)
        super().__init__(wrapper, **kw)
    def load_index(self):
        #docs = [
        #    d for d in self.wrapper.sd_documents
        #    if d.metadata.get("source") == "term"
        #]
        (self.idx_vectors, self.idx_bm25) = (None, None) #get_retrievers(docs)
    def get_abbreviations(self, query: str):
        q = query.upper()
        return [
            d for d in self.wrapper.sd_documents
            if d.metadata.get("source") == "abbr"
            and re.search(rf"\b{re.escape(d.metadata['term'])}\b", q)
        ]
    def _get_relevant_documents(self, query: str, *, run_manager, **kw):
        return (
            self.get_abbreviations(query)
            #+ self._index_search(query)
        )


class TeamlyContextualCompressionRetriever(ContextualCompressionRetriever):
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            documents = super()._get_relevant_documents(query, run_manager=run_manager, **kwargs)
            if isinstance(self.base_retriever, TeamlyRetriever):
                for doc in documents:
                    article_id = doc.metadata.get("article_id")
                    space_id = doc.metadata.get("space_id", "")
                    link = doc.metadata.get("link", "")
                    source = doc.metadata.get("source", "")
                    doc_type = doc.metadata.get("type", "")
                    try:
                        if article_id and source == "semantic" and doc_type == "article":
                            doc.page_content = f"{self.base_retriever.wrapper.get_article(article_id)}\n\nСсылка на статью: {link}"
                    except Exception as e:
                        logging.warning(f"Cannot get article content. Continue with incomplete document'.\nException: {e}\nArticle: {article_id}")
            
        except Exception as e:
            logging.error(f"Error occured at 'TeamlyContextualCompressionRetriever::_get_relevant_documents'.\nException: {e}")
            raise e
        return documents

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
    from pprint import pprint

    class KBDocumentPromptTemplate(StringPromptTemplate):
        max_length : int = 0
        def __init__(self, max_length: int, **kwargs: Any):
            super().__init__(**kwargs)
            self.max_length = max_length

        def format(self, **kwargs: Any) -> str:
            page_content = kwargs.pop("page_content")
            #problem_number = kwargs.pop("problem_number")
            #chunk_size = kwargs.pop("actual_chunk_size")
            #here additional data could be retrieved based on problem_number
            result = page_content
            if self.max_length > 0:
                result = result[:self.max_length]
            return result

        @property
        def _prompt_type(self) -> str:
            return "kb_document"
        
    retriever = TeamlyRetriever_Glossary(auth_data_store="./auth.json", k=5)

    llm = ChatOpenAI(model="gpt-5-mini")
    with open("./prompt.txt", encoding="utf-8") as f:
        prompt_txt = f.read()
    system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_txt),
                ("human", "User request: \n{input}\n\nContext: \n{context}"),
            ]
        )

    my_prompt = KBDocumentPromptTemplate(0, input_variables=["page_content", "article_id", "article_title"])

    docs_chain = create_stuff_documents_chain(llm, system_prompt, document_prompt=my_prompt, document_separator='\n#EOD\n\n')
    rag_chain = create_retrieval_chain(retriever, docs_chain)

    result = rag_chain.invoke({"input": "Что делать если в АД EL выше допустимого? Выносить на КК?"})
    pprint(result)
