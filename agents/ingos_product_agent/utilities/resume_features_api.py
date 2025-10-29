from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ...llm_utils import get_llm
from .feature_extractor import extract_features_from_resume

logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Feature Extraction API")


class ResumeRequest(BaseModel):
    """Incoming payload containing resume text to analyse."""

    resume_text: str = Field(..., min_length=1, description="Raw resume text that should be processed.")


class SalaryRangeResponse(BaseModel):
    """Salary range structure returned to clients."""

    min: Optional[float] = Field(default=None, description="Minimum desired salary.")
    max: Optional[float] = Field(default=None, description="Maximum desired salary.")
    currency: Optional[str] = Field(default="RUB", description="Currency code for the salary values.")


class ResumeFeaturesResponse(BaseModel):
    """Structured resume features returned by the service."""

    positions: List[str]
    locations: List[str]
    skills: List[str]
    competencies: List[str]
    salary_range: SalaryRangeResponse


@lru_cache(maxsize=1)
def _get_extractor_llm():
    """Initialise and cache the LLM instance used for feature extraction."""

    llm = get_llm(model="nano", temperature=0)
    if not hasattr(llm, "invoke"):
        raise AttributeError("Configured LLM instance does not expose an 'invoke' method.")
    return llm


def _normalise_features(raw_features: Dict[str, Any]) -> ResumeFeaturesResponse:
    """Ensure the extracted feature payload conforms to the API schema."""

    positions = [p for p in raw_features.get("positions", []) if isinstance(p, str)]
    locations = [l for l in raw_features.get("locations", []) if isinstance(l, str)]
    skills = [l for l in raw_features.get("skills", []) if isinstance(l, str)]
    competencies = [l for l in raw_features.get("competencies", []) if isinstance(l, str)]

    salary_payload = raw_features.get("salary_range") or {}
    if not isinstance(salary_payload, dict):
        salary_payload = {}

    salary = SalaryRangeResponse(
        min=salary_payload.get("min"),
        max=salary_payload.get("max"),
        currency=salary_payload.get("currency") or "RUB",
    )

    return ResumeFeaturesResponse(positions=positions, locations=locations, skills=skills, salary_range=salary, competencies=competencies)


@app.post("/resume/features", response_model=ResumeFeaturesResponse)
def extract_resume_features(payload: ResumeRequest) -> ResumeFeaturesResponse:
    """Extract structured search features from a plain-text resume."""

    resume_text = payload.resume_text.strip()
    if not resume_text:
        raise HTTPException(status_code=400, detail="resume_text must not be empty")

    try:
        extractor_llm = _get_extractor_llm()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Failed to initialise extractor LLM: %s", exc)
        raise HTTPException(status_code=500, detail="LLM initialisation failed") from exc

    try:
        raw_features = extract_features_from_resume(resume_text, extractor_llm)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Feature extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Feature extraction failed") from exc

    return _normalise_features(raw_features)


@app.get("/health", summary="Service health check")
def health_check() -> Dict[str, str]:
    """Simple readiness endpoint for monitoring."""

    return {"status": "ok"}

