from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except ImportError:  # pragma: no cover - optional dependency guard
    HuggingFaceCrossEncoder = None  # type: ignore[assignment]

from ...retrievers.utils.models_builder import (
    getEmbeddingModel,
    getRerankerModel,
)

logger = logging.getLogger(__name__)


@dataclass
class SemanticScoreWeights:
    """Weights applied to individual semantic similarity components."""

    title_position: float = 0.35
    skills_resume: float = 0.2
    description_resume: float = 0.45
    experience_resume: float = 0

    def as_mapping(self) -> Dict[str, float]:
        return asdict(self)


class SemanticJobScorer:
    """Scores vacancies against resume-derived features using semantic similarity."""

    def __init__(
        self,
        *,
        embedding_model: Optional[Any] = None,
        cross_encoder: Optional[Any] = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._cross_encoder = cross_encoder
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._embedding_unavailable = False
        self._attempted_cross_encoder_load = cross_encoder is not None

    def score(
        self,
        job: Dict[str, Any],
        resume_text: str,
        features: Optional[Dict[str, Any]] = None,
        *,
        weights: Optional[SemanticScoreWeights] = None,
        return_components: bool = False,
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        features = features or {}
        weights = weights or SemanticScoreWeights()

        components: Dict[str, float] = {}

        title = (job.get("title") or "").strip()
        positions = [
            item.strip()
            for item in features.get("positions", [])
            if isinstance(item, str) and item.strip()
        ]
        if title and positions:
            title_scores = [self._pair_similarity(title, pos) for pos in positions]
            components["title_position"] = max(title_scores, default=0.0)
        elif title and resume_text:
            components["title_position"] = self._pair_similarity(title, resume_text)
        else:
            components["title_position"] = 0.0

        job_skills = job.get("skills") or []
        skill_terms: List[str] = [str(item) for item in job_skills if isinstance(item, str)]
        if not skill_terms:
            extra_skills = features.get("skills")
            if isinstance(extra_skills, Iterable):
                skill_terms = [str(item) for item in extra_skills if item]
        skills_text = ", ".join(skill_terms).strip()
        if skills_text and resume_text:
            components["skills_resume"] = self._pair_similarity(resume_text, skills_text)
        else:
            components["skills_resume"] = 0.0

        description = (job.get("description") or "").strip()
        if description and resume_text:
            components["description_resume"] = self._pair_similarity(resume_text, description)
        else:
            components["description_resume"] = 0.0

        experience = (job.get("experience") or "").strip()
        if experience and resume_text:
            components["experience_resume"] = self._pair_similarity(resume_text, experience)
        else:
            components["experience_resume"] = 0.0

        weighted_score, normalisation = 0.0, 0.0
        for key, weight in weights.as_mapping().items():
            component_value = components.get(key)
            if component_value is None:
                continue
            weighted_score += component_value * weight
            normalisation += weight

        final_score = weighted_score / normalisation if normalisation else 0.0

        if return_components:
            return final_score, components
        return final_score

    def _pair_similarity(self, left: str, right: str) -> float:
        left_text = (left or "").strip()
        right_text = (right or "").strip()
        if not left_text or not right_text:
            return 0.0
        
        ce_score = None
        #ce_score = self._cross_encoder_similarity(left_text, right_text)
        embedding_score = self._embedding_similarity(left_text, right_text)

        if ce_score is not None and embedding_score is not None:
            return (ce_score + embedding_score) / 2.0
        if ce_score is not None:
            return ce_score
        if embedding_score is not None:
            return embedding_score
        return 0.0

    def _embedding_similarity(self, left: str, right: str) -> Optional[float]:
        left_vector = self._embedding_vector(left)
        right_vector = self._embedding_vector(right)
        if left_vector is None or right_vector is None:
            return None
        similarity = float(np.dot(left_vector, right_vector))
        similarity = max(-1.0, min(1.0, similarity))
        return 0.5 * (similarity + 1.0)

    def _embedding_vector(self, text: str) -> Optional[np.ndarray]:
        text_key = text.strip()
        if not text_key:
            return None
        if text_key in self._embedding_cache:
            return self._embedding_cache[text_key]
        model = self._ensure_embedding_model()
        if model is None:
            return None
        try:
            vector = model.embed_query(text_key)
        except Exception as exc:  # pragma: no cover - runtime model issues
            logger.warning("Embedding generation failed: %s", exc)
            return None
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        if norm:
            array = array / norm
        self._embedding_cache[text_key] = array
        return array

    def _cross_encoder_similarity(self, left: str, right: str) -> Optional[float]:
        cross_encoder = self._ensure_cross_encoder()
        if cross_encoder is None:
            return None
        try:
            scores = cross_encoder.score([(left, right)])
        except Exception as exc:  # pragma: no cover - runtime model issues
            logger.warning("Cross encoder scoring failed: %s", exc)
            return None
        if not scores:
            return None
        return self._sigmoid(float(scores[0]))

    def _ensure_embedding_model(self) -> Optional[Any]:
        if self._embedding_model is None and not self._embedding_unavailable:
            try:
                self._embedding_model = getEmbeddingModel()
            except Exception as exc:  # pragma: no cover - model load failure
                logger.warning("Failed to load embedding model: %s", exc)
                self._embedding_unavailable = True
        return self._embedding_model

    def _ensure_cross_encoder(self) -> Optional[Any]:
        if self._cross_encoder is not None:
            return self._cross_encoder
        if self._attempted_cross_encoder_load:
            return None
        self._attempted_cross_encoder_load = True
        if HuggingFaceCrossEncoder is None:
            logger.debug("Cross encoder dependency not available; skipping load.")
            return None
        try:
            self._cross_encoder = getRerankerModel()
        except Exception as exc:  # pragma: no cover - model load failure
            logger.warning("Failed to load cross encoder model: %s", exc)
            self._cross_encoder = None
        return self._cross_encoder

    @staticmethod
    def _sigmoid(value: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-value))
        except OverflowError:  # pragma: no cover - defensive clamp
            return 0.0 if value < 0 else 1.0


_default_scorer: Optional[SemanticJobScorer] = None


def get_default_scorer() -> SemanticJobScorer:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = SemanticJobScorer()
    return _default_scorer


def semantic_score_job(
    job: Dict[str, Any],
    resume_text: str,
    features: Optional[Dict[str, Any]] = None,
    *,
    weights: Optional[SemanticScoreWeights] = None,
    scorer: Optional[SemanticJobScorer] = None,
    return_components: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    active_scorer = scorer or get_default_scorer()
    return active_scorer.score(
        job,
        resume_text,
        features,
        weights=weights,
        return_components=return_components,
    )


__all__ = [
    "SemanticJobScorer",
    "SemanticScoreWeights",
    "get_default_scorer",
    "semantic_score_job",
]
