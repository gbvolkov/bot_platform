import logging

import torch

from typing import List, Any, Optional, Dict, Tuple, TypedDict, Annotated

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import config


_device = "cuda" if (config.FORCE_CPU != "1") and torch.cuda.is_available() else "cpu"

_embedding_model: Optional[HuggingFaceEmbeddings] = None
_reranker_model: Optional[HuggingFaceCrossEncoder] = None

def getEmbeddingModel()-> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logging.info(f"loading model for embedding:  {config.EMBEDDING_MODEL}")
        model_kwargs = {"device": _device}
        if _device == "cpu":
            # Force transformer loader to materialize weights on CPU instead of meta tensors
            model_kwargs["model_kwargs"] = {
                "low_cpu_mem_usage": False,
                "device_map": None,
            }
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_model

def getRerankerModel()-> HuggingFaceCrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        logging.info(f"loading model for reranker: {config.RERANKING_MODEL}")
        _reranker_model = HuggingFaceCrossEncoder(
            model_name=config.RERANKING_MODEL, 
            model_kwargs={'trust_remote_code': True, "device": _device}
        )
    return _reranker_model
