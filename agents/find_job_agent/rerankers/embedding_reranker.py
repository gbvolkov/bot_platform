from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from langchain_classic.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from ...retrievers.utils.models_builder import (
    getEmbeddingModel,
    getRerankerModel,
)

try:  # pragma: no cover - optional dependency handling
    import torch  # type: ignore
except Exception:  # noqa: S110 - best effort optional import
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRerankOptions:
    """Optional knobs for the embedding-based vacancy reranker."""

    top_k: Optional[int] = None
    use_cross_encoder: bool = False
    cross_encoder_pool_size: int = 25
    cross_encoder_weight: Optional[float] = None


def _normalize_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmax = float(values.max())
    vmin = float(values.min())
    if np.isclose(vmax, vmin):
        return np.ones_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


class VacancyEmbeddingReranker:
    """Ranks vacancies using multilingual-e5 embeddings with optional BGE reranking."""

    def __init__(
        self,
        embedding_model: Optional[HuggingFaceEmbeddings] = None,
        reranker_model: Optional[HuggingFaceCrossEncoder] = None,
    ) -> None:
        self._embedding_model: HuggingFaceEmbeddings = embedding_model or getEmbeddingModel()
        self._reranker_model: Optional[HuggingFaceCrossEncoder] = reranker_model
        self._attempted_reranker_load = reranker_model is not None

    def rerank(
        self,
        vacancies: Sequence[Dict[str, Any]],
        resume_text: str,
        features: Optional[Dict[str, Any]] = None,
        *,
        options: Optional[EmbeddingRerankOptions] = None,
        top_k: Optional[int] = None,
        use_cross_encoder: Optional[bool] = None,
        cross_encoder_pool_size: Optional[int] = None,
        cross_encoder_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not vacancies:
            return []
        try:
            opts = options or EmbeddingRerankOptions()
            limit = top_k if top_k is not None else opts.top_k
            apply_ce = use_cross_encoder if use_cross_encoder is not None else opts.use_cross_encoder
            pool_size = cross_encoder_pool_size if cross_encoder_pool_size is not None else opts.cross_encoder_pool_size
            blend_weight = cross_encoder_weight if cross_encoder_weight is not None else opts.cross_encoder_weight

            if blend_weight is not None and not 0.0 <= blend_weight <= 1.0:
                raise ValueError("cross_encoder_weight must be between 0 and 1")

            documents = [self._vacancy_to_document(vacancy) for vacancy in vacancies]
            query_text = self._compose_query(resume_text, features or {})

            query_vector = np.array(
                self._embedding_model.embed_query(self._format_query_text(query_text)),
                dtype=np.float32,
            )
            document_embeddings = self._embedding_model.embed_documents(
                [self._format_passage_text(doc.page_content) for doc in documents]
            )
            embedding_scores = [
                float(np.dot(query_vector, np.array(doc_vec, dtype=np.float32)))
                for doc_vec in document_embeddings
            ]

            embedding_scores_array = np.array(embedding_scores, dtype=np.float32)
            embedding_scores_norm = _normalize_array(embedding_scores_array)

            reranker_scores: Dict[int, float] = {}
            reranker_scores_norm: Dict[int, float] = {}

            if apply_ce:
                reranker = self._get_reranker_model()
                if reranker is None:
                    logger.warning("Cross encoder reranker requested but could not be loaded.")
                else:
                    candidate_count = len(documents) if pool_size is None else min(pool_size, len(documents))
                    ranked_indices = sorted(
                        range(len(embedding_scores)),
                        key=lambda idx: embedding_scores[idx],
                        reverse=True,
                    )
                    candidate_indices = ranked_indices[:candidate_count]
                    pairs = [
                        (query_text, documents[idx].page_content)
                        for idx in candidate_indices
                    ]

                    if not pairs:
                        logger.debug("No candidate pairs for cross encoder reranking.")
                    else:
                        try:
                            raw_scores = reranker.score(pairs)
                        except Exception as exc:  # pragma: no cover - runtime model issues
                            logger.warning("Cross encoder scoring failed: %s", exc)
                            raw_scores = []

                        scores_list = self._to_float_list(raw_scores)
                        reranker_scores = {
                            idx: score for idx, score in zip(candidate_indices, scores_list)
                        }
                        if reranker_scores:
                            norm_values = _normalize_array(np.array(list(reranker_scores.values()), dtype=np.float32))
                            reranker_scores_norm = {
                                idx: float(norm)
                                for idx, norm in zip(reranker_scores.keys(), norm_values)
                            }

            final_scores = embedding_scores_norm.copy()
            if reranker_scores:
                if blend_weight is None:
                    for idx, score in reranker_scores.items():
                        final_scores[idx] = float(score)
                else:
                    for idx, norm_score in reranker_scores_norm.items():
                        final_scores[idx] = (
                            blend_weight * norm_score
                            + (1.0 - blend_weight) * float(embedding_scores_norm[idx])
                        )

            ranked: List[Dict[str, Any]] = []
            for idx, vacancy in enumerate(vacancies):
                item = dict(vacancy)
                item["embedding_score"] = round(float(embedding_scores[idx]), 4)
                item["embedding_score_norm"] = round(float(embedding_scores_norm[idx]), 4)
                if idx in reranker_scores:
                    item["reranker_score"] = round(float(reranker_scores[idx]), 4)
                    item["reranker_score_norm"] = round(
                        float(reranker_scores_norm.get(idx, 0.0)), 4
                    )
                item["final_rank_score"] = round(float(final_scores[idx]), 4)
                ranked.append(item)

            ranked.sort(key=lambda entry: entry["final_rank_score"], reverse=True)
            #ranked.sort(key=lambda entry: entry["reranker_score"], reverse=True)

            if limit is not None:
                ranked = ranked[:limit]
        except Exception as exc:
            logger.warning("Error reranking with embeddings: %s", exc)
            ranked = []

        return ranked

    def _get_reranker_model(self) -> Optional[HuggingFaceCrossEncoder]:
        if self._reranker_model is not None or self._attempted_reranker_load:
            return self._reranker_model
        try:
            self._reranker_model = getRerankerModel()
        except Exception as exc:  # pragma: no cover - model load failure
            logger.warning("Failed to load reranker model: %s", exc)
            self._attempted_reranker_load = True
            return None
        self._attempted_reranker_load = True
        return self._reranker_model

    @staticmethod
    def _compose_query(resume_text: str, features: Dict[str, Any]) -> str:
        sections: List[str] = []
        feature_str = VacancyEmbeddingReranker._features_to_string(features)
        if feature_str:
            sections.append(feature_str)
        if resume_text and resume_text.strip():
            sections.append(resume_text.strip())
        if not sections:
            return "job search preferences"
        return "\n".join(sections)

    @staticmethod
    def _features_to_string(features: Dict[str, Any]) -> str:
        lines: List[str] = []
        for key, value in features.items():
            if value in (None, "", [], {}, ()):  # skip empty entries
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, dict):
                parts = [
                    f"{child_key}: {child_value}"
                    for child_key, child_value in value.items()
                    if child_value not in (None, "", [])
                ]
                if parts:
                    lines.append(f"{label}: {'; '.join(parts)}")
            elif isinstance(value, (list, tuple, set)):
                items = [str(item) for item in value if item]
                if items:
                    lines.append(f"{label}: {', '.join(items)}")
            else:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _vacancy_to_document(vacancy: Dict[str, Any]) -> Document:
        parts: List[str] = []
        title = vacancy.get("title")
        company = vacancy.get("company")
        location = vacancy.get("location")
        description = vacancy.get("description")
        experience = vacancy.get("experience")
        skills = vacancy.get("skills")
        salary = vacancy.get("salary")

        if title:
            parts.append(f"Title: {title}")
        if company:
            parts.append(f"Company: {company}")
        if location:
            parts.append(f"Location: {location}")
        if isinstance(salary, dict) and (salary.get("min") or salary.get("max")):
            parts.append(
                "Salary: "
                + "-".join(
                    str(salary.get(bound))
                    for bound in ("min", "max")
                    if salary.get(bound)
                )
                + f" {salary.get('currency', 'RUB')}"
            )
        if description:
            parts.append(f"Description: {description}")
        if experience:
            parts.append(f"Experience: {experience}")
        if isinstance(skills, list) and skills:
            skill_str = ", ".join(str(skill) for skill in skills if skill)
            if skill_str:
                parts.append(f"Skills: {skill_str}")

        metadata = {
            "id": vacancy.get("id"),
            "source": vacancy.get("source"),
            "url": vacancy.get("url"),
        }
        return Document(
            page_content="\n".join(parts),
            metadata={k: v for k, v in metadata.items() if v},
        )

    @staticmethod
    def _format_query_text(text: str) -> str:
        text = text.strip()
        if not text.lower().startswith("query:"):
            return f"query: {text}"
        return text

    @staticmethod
    def _format_passage_text(text: str) -> str:
        text = text.strip()
        if not text.lower().startswith("passage:"):
            return f"passage: {text}"
        return text

    @staticmethod
    def _to_float_list(raw_scores: Any) -> List[float]:
        if raw_scores is None:
            return []
        if torch is not None and isinstance(raw_scores, torch.Tensor):  # type: ignore[attr-defined]
            return raw_scores.detach().cpu().numpy().astype(float).tolist()
        if isinstance(raw_scores, np.ndarray):
            return raw_scores.astype(float).tolist()
        if isinstance(raw_scores, (list, tuple)):
            return [float(item) for item in raw_scores]
        try:
            return [float(raw_scores)]
        except (TypeError, ValueError):
            logger.warning("Unexpected reranker score type: %s", type(raw_scores))
            return []


if __name__ == "__main__":  # pragma: no cover - manual smoke test helper
    sample_vacancies = [
        {
            "id": "1",
            "title": "Senior Python Developer",
            "company": "Tech Corp",
            "location": "Remote",
            "description": "Develop backend services with FastAPI and PostgreSQL.",
            "skills": ["Python", "FastAPI", "PostgreSQL"],
            "experience": "3-5 years",
            "salary": {"min": 250000, "max": 300000, "currency": "RUB"},
            "url": "https://example.com/jobs/1",
        },
        {
            "id": "2",
            "title": "Data Scientist",
            "company": "AI Labs",
            "location": "Moscow",
            "description": "Build ML models for customer analytics and reporting.",
            "skills": ["Python", "Pandas", "Machine Learning"],
            "experience": "2+ years",
            "salary": {"min": 220000, "max": 280000, "currency": "RUB"},
            "url": "https://example.com/jobs/2",
        },
    ]

    sample_features = {
        "positions": ["Python Developer", "Machine Learning Engineer"],
        "skills": ["Python", "FastAPI", "ML"],
        "locations": ["Remote", "Moscow"],
    }

    reranker = VacancyEmbeddingReranker()
    results = reranker.rerank(
        sample_vacancies,
        resume_text="Experienced Python developer with ML background and FastAPI expertise.",
        features=sample_features,
        use_cross_encoder=False,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
