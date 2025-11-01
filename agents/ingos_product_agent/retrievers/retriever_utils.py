
from __future__ import annotations

import logging

from typing import TYPE_CHECKING, List, Any, Optional, Dict, Tuple, TypedDict, Annotated
import os, torch, pickle
from functools import lru_cache

import config

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryByteStore

from langchain_classic.docstore.document import Document
from langchain_core.tools import tool

from ...retrievers.cross_encoder_reranker_with_score import CrossEncoderRerankerWithScores, TournamentCrossEncoderReranker
from ...retrievers.utils.models_builder import (
    getEmbeddingModel,
    getRerankerModel,
)
from ...retrievers.utils.load_common_retrievers import (
    load_vectorstore
)

from .vector_store import VectorStore

from palimpsest import Palimpsest

import config
#_faiss_reranker_retriever: Optional[TournamentCrossEncoderReranker] = None

_MAX_RETRIEVALS=5

if TYPE_CHECKING:
    from services.kb_manager.notifications import KBReloadContext

logger = logging.getLogger(__name__)

@lru_cache(maxsize=64)
def buildFAISSRetriever(product: str = "default")-> ContextualCompressionRetriever:
    #global _faiss_reranker_retriever
    _faiss_reranker_retriever: Optional[TournamentCrossEncoderReranker] = None
    if _faiss_reranker_retriever is None:
        logging.info("loading FAISSRetriever reranked")
        vector_store_path = config.PRODUCT_INDEX_FOLDER
        vectorstore = load_vectorstore(vector_store_path)

        #with open(f'{vector_store_path}/docstore.pkl', 'rb') as file:
        #    documents = pickle.load(file)
        #doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]

        store = InMemoryByteStore()
        #id_key = "problem_number"
        #multi_retriever = MultiVectorRetriever(
        #    vectorstore=vectorstore,
        #    byte_store=store,
        #    id_key=id_key,
        #    search_kwargs={"k": _MAX_RETRIEVALS},
        #)
        #multi_retriever.docstore.mset(list(zip(doc_ids, documents)))
        reranker_model = getRerankerModel()
        #reranker = CrossEncoderRerankerWithScores(
        #    model=reranker_model, 
        #   top_n=_MAX_RETRIEVALS, 
        #    min_ratio=float(config.MIN_RERANKER_RATIO)
        #)
        reranker = TournamentCrossEncoderReranker(
            model=reranker_model, 
            top_n=_MAX_RETRIEVALS, 
            tournament_size=10,
            min_ratio=float(config.MIN_RERANKER_RATIO)
        )
        search_kwargs = {}
        if product != "default":
            search_kwargs={"filter": {"product": product}}

        _faiss_reranker_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=vectorstore.as_retriever(search_kwargs=search_kwargs)
        )

    return _faiss_reranker_retriever



def get_retriever_faiss(product: str = "default"):
    MAX_RETRIEVALS = 3
    retriever = buildFAISSRetriever(product)
    def search(query: str) -> List[Document]:
        try:
            result = retriever.invoke(
                query, 
                search_kwargs={
                    "k": MAX_RETRIEVALS,
                },
            )

        except Exception as e:
            logging.error(f"Error occured during faiss search tool calling.\nException: {e}")
            raise e
        return result
    return search

def get_retriever(product: str = "default"):
    MAX_RETRIEVALS = 3
    retriever  = VectorStore(docs_path="./data/docs", vector_store_path="./data/vector_store")
    product = product
    def search(query: str) -> List[Document]:
        try:
            result = retriever.search(query=query, n_results=MAX_RETRIEVALS, product=product)
        except Exception as e:
            logging.error(f"Error occured during faiss search tool calling.\nException: {e}")
            raise e
        return result
    return search



#@lru_cache(maxsize=64)
def get_search_tool(product: str = "default", anonymizer: Palimpsest = None):
    product = product
    if config.INGOS_RETRIEVER == "faiss":
        search = get_retriever_faiss(product)
    else:
        search = get_retriever(product)
    @tool
    def search_kb(query: str) -> str:
        """Retrieves from knowledgebase context suitable for the query. Shall be always used when user asks question.
        Args:
            query: a query to knowledgebase which helps answer user's question
        Returns:
            Context from knowledgebase suitable for the query.
        """
        print(f"search for product:{product}")
        found_docs = search(query)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            if anonymizer:
                result = anonymizer.anonimize(result)
            return result
        else:
            return "No matching information found."
    return search_kb


def reload_retrievers(context: "KBReloadContext | None" = None) -> None:
    """Clear cached retriever instances so fresh indexes are loaded on demand."""
    reason = context.reason if context else "unspecified"
    logger.info("Reloading product retrievers (reason=%s)", reason)
    buildFAISSRetriever.cache_clear()
