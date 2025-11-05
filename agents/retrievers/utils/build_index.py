import pandas as pd
import os

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

import config

from .models_builder import getEmbeddingModel

embedding = getEmbeddingModel() #HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})

#def get_retrievers(df):
#    embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
#    documents = get_documents(df)
#    # Initialize the embedding model (E5 large). `normalize_embeddings=True` to use cosine similarity.#
#
#    # Create a FAISS vector store from the documents
#    vector_store = FAISS.from_documents(documents, embedding)
#    #vector_store = None
#
#    # Create a BM25 retriever from the same documents
#    bm25_retriever = BM25Retriever.from_documents(documents)
#    #bm25_retriever = None
#    return (vector_store, bm25_retriever)


def get_retrievers(documents):
    vector_store = None
    bm25_retriever = None
    logging.info("Loading retrievers...")
    try:
        # Initialize the embedding model (E5 large). `normalize_embeddings=True` to use cosine similarity.

        # Create a FAISS vector store from the documents
        vector_store = FAISS.from_documents(documents, embedding)

        # Create a BM25 retriever from the same documents
        bm25_retriever = BM25Retriever.from_documents(documents)
    except Exception as e:
        logging.error(f"Error while building retrievers: {e}")
    #bm25_retriever = None
    logging.info("...complete loading retrievers")
    return (vector_store, bm25_retriever)
