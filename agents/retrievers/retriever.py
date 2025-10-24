import logging

from typing import List, Any, Optional, Dict, Tuple, TypedDict, Annotated
import os
import pickle
import torch

from langchain_community.document_loaders import NotionDBLoader
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.docstore.document import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition
from palimpsest import Palimpsest
from .teamly_retriever import (
    TeamlyRetriever,
    TeamlyRetriever_Tickets,
    TeamlyRetriever_Glossary,
    TeamlyContextualCompressionRetriever
)
import config

from .utils.load_common_retrievers import (
    buildMultiRetriever,
    buildTeamlyRetriever,
    buildFAISSRetriever,
    getTeamlyTicketsRetriever,
    getTeamlyGlossaryRetriever,
)

from .cross_encoder_reranker_with_score import CrossEncoderRerankerWithScores


def get_retriever_multi():
    k = 5

    retriever = buildMultiRetriever(
        [config.NOTION_INDEX_FOLDER, config.CHATS_INDEX_FOLDER],
        search_kwargs={"k": k},
        weights=[0.5, 0.5]
    )
    def search(query: str) -> List[Document]:
        result = retriever.invoke(query, search_kwargs={"k": k})
        # docs = retriever.similarity_search_with_score(query, k=5)
        # result = [doc for doc, score in docs if score >= 0.20]
        return result
    return search

def get_retriever_teamly():
    max_retrievals = 3
    retriever = buildTeamlyRetriever()

    def search(query: str) -> List[Document]:
        try:
            result = retriever.invoke(query, search_kwargs={"k": max_retrievals})
        except Exception as e:
            logging.error(f"Error occured during teamly search tool calling.\nException: {e}")
            raise e
        # torch.cuda.empty_cache()
        return result
    return search

def get_retriever_faiss():
    MAX_RETRIEVALS = 3
    retriever = buildFAISSRetriever()
    def search(query: str) -> List[Document]:
        try:
            result = retriever.invoke(query, search_kwargs={"k": MAX_RETRIEVALS})
        except Exception as e:
            logging.error(f"Error occured during faiss search tool calling.\nException: {e}")
            raise e
        return result
    return search

def get_retriever():
    retriever_type = config.RETRIEVER_TYPE
    if retriever_type == "teamly":
        return get_retriever_teamly()
    return get_retriever_faiss()

# Initialize the search function with the selected retrieverx
search = get_retriever()


def get_search_tool(anonymizer: Palimpsest = None):
    @tool
    def search_kb(query: str) -> str:
        """Retrieves from knowledgebase context suitable for the query. Shall be always used when user asks question.
        Args:
            query: a query to knowledgebase which helps answer user's question
        Returns:
            Context from knowledgebase suitable for the query.
        """
        found_docs = search(query)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            if anonymizer:
                result = anonymizer.anonimize(result)
            return result
        else:
            return "No matching information found."
    return search_kb

def get_tickets_search_tool(anonymizer: Palimpsest = None):
    MAX_RETRIEVALS = 3

    tickets_retriever = getTeamlyTicketsRetriever()
    @tool
    def search_tickets(query: str) -> str:
        """Retrieves from tickets knowledgebase context suitable for the query. Shall be always used when user asks question.
        Args:
            query: a query to knowledgebase which helps answer user's question
        Returns:
            Context from knowledgebase suitable for the query.
        """
        found_docs = tickets_retriever.invoke(query)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            if anonymizer:
                result = anonymizer.anonimize(result)
            return result
        else:
            return "No matching information found."
    return search_tickets

def get_term_and_defition_tools(anonymizer: Palimpsest = None):
    MAX_RETRIEVALS = 3
    glossary_retriever = getTeamlyGlossaryRetriever()
    
    @tool
    def lookup_term(term: str) -> str:
        """
        Look up the definition of a term or abbreviation in the reference dictionary.

        This tool is designed to retrieve the meaning of either a full term 
        or an abbreviation from a predefined reference source. 
        All abbreviations in the reference are stored in uppercase. 
        All terms in the reference are stored in singular nominative case. 

        The input must strictly follow these conventions:
        - Abbreviations: uppercase only (e.g., "HTTP", "NASA", "АД").
        - Terms: singular nominative case (e.g., "server", "network", "лизинговая заявка").

        Args:
            name (str): The term or abbreviation to look up.
                Must match the format and casing conventions of the reference.

        Returns:
            str: The definition or description of the provided term or abbreviation.
                Currently returns a constant placeholder string.
        """
        found_docs = glossary_retriever.invoke(term)
        if found_docs:
            result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
            if anonymizer:
                result = anonymizer.anonimize(result)
            return result
        else:
            return "No matching information found."
    return lookup_term

if __name__ == '__main__':
    search_kb = get_search_tool()
    answer = search_kb("Кто такие кей юзеры?")
    print(answer)
