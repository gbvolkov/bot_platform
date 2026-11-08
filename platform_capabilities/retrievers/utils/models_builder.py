import logging

import torch

from typing import List, Any, Optional, Dict, Tuple, TypedDict, Annotated

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import config
from platform_utils.model_loading import MODEL_LOADING_LOCK


_device = "cuda" if (config.FORCE_CPU != "1") and torch.cuda.is_available() else "cpu"

_embedding_model: Optional[HuggingFaceEmbeddings] = None
_embedding_revision_models: Dict[str, HuggingFaceEmbeddings] = {}
_reranker_model: Optional[HuggingFaceCrossEncoder] = None

def getEmbeddingModel(*, revision: Optional[str] = None)-> HuggingFaceEmbeddings:
    global _embedding_model
    with MODEL_LOADING_LOCK:
        if revision is not None:
            if revision in _embedding_revision_models:
                return _embedding_revision_models[revision]
            if _embedding_model is not None and _embedding_model.model_kwargs.get("revision") == revision:
                return _embedding_model
            model_kwargs = {"device": _device, "revision": revision}
            if _device == "cpu":
                model_kwargs["model_kwargs"] = {"low_cpu_mem_usage": False, "device_map": None}
            selected = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs=model_kwargs,
                                            encode_kwargs={"normalize_embeddings": True})
            _embedding_revision_models[revision] = selected
            if _embedding_model is None:
                _embedding_model = selected
            return selected
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
    with MODEL_LOADING_LOCK:
        if _reranker_model is None:
            logging.info(f"loading model for reranker: {config.RERANKING_MODEL}")
            _reranker_model = HuggingFaceCrossEncoder(
                model_name=config.RERANKING_MODEL,
                model_kwargs={'trust_remote_code': True, "device": _device}
            )
    return _reranker_model
