from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


GazStage = Literal["OPENING", "SELL", "RESEARCH", "RECOMMEND", "FOLLOWUP"]
GazBranch = Literal[
    "tco",
    "configuration",
    "comparison",
    "service_risk",
    "internal_approval",
    "passenger_route",
    "special_body",
    "special_conditions",
    "unknown_selection",
]
ClientIntent = Literal["overview", "compare", "specs", "financing", "objection", "recommendation", "materials", "next_step"]
AnswerDepth = Literal["broad", "bounded", "justified", "deep_research"]
CustomerTemperature = Literal["neutral", "impatient", "irritated", "competitor_risk"]


class GazSlots(BaseModel):
    customer_goal: Optional[str] = None
    transport_type: Optional[str] = None
    route_type: Optional[str] = None
    route_mode: Optional[str] = None
    body_type: Optional[str] = None
    capacity_or_payload: Optional[str] = None
    competitor: Optional[str] = None
    decision_criterion: Optional[str] = None
    decision_role: Optional[str] = None
    special_conditions: List[str] = Field(default_factory=list)


class IntentFlags(BaseModel):
    requested_price: bool = False
    requested_materials: bool = False
    asks_for_recommendation: bool = False
    requested_portfolio_overview: bool = False
    requested_financing: bool = False
    challenged_questions: bool = False
    expressed_friction: bool = False
    requested_comparison_table: bool = False
    requested_specs: bool = False
    requested_versions: bool = False
    requested_competitor_comparison: bool = False
    expressed_impatience: bool = False
    requested_concrete_numbers: bool = False
    threatened_competitor_switch: bool = False


class BranchClassificationResult(BaseModel):
    active_branch: Optional[GazBranch] = None
    branch_conflict: List[GazBranch] = Field(default_factory=list)
    reasoning: str = ""


class MaterialCandidate(BaseModel):
    candidate_id: str
    title: str
    doc_kind: str
    rationale: str
    preview_snippet: Optional[str] = None
    branch_relevance: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReadExcerpt(BaseModel):
    excerpt: str
    relevance_reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReadMaterialResult(BaseModel):
    candidate_id: str
    title: str
    focus: str
    excerpts: List[ReadExcerpt] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceCandidateRef(BaseModel):
    candidate_id: str
    title: str
    doc_kind: str


class SalesDirectionDigest(BaseModel):
    group_id: str
    title: str
    families: List[str] = Field(default_factory=list)
    main_characteristics: List[str] = Field(default_factory=list)
    typical_use_cases: List[str] = Field(default_factory=list)
    financial_angle: str = ""
    key_tradeoffs: List[str] = Field(default_factory=list)
    evidence_highlights: List[str] = Field(default_factory=list)
    source_candidates: List[SourceCandidateRef] = Field(default_factory=list)


class SalesLandscapeResult(BaseModel):
    topic: str = ""
    audience: str = ""
    use_case: str = ""
    focus: str = ""
    directions: List[SalesDirectionDigest] = Field(default_factory=list)
    finance_options: List[str] = Field(default_factory=list)
    recommended_next_narrowing: str = ""
    source_candidates: List[SourceCandidateRef] = Field(default_factory=list)


class ComparisonProductDigest(BaseModel):
    family_id: str
    label: str
    main_use_cases: List[str] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)
    financial_angle: str = ""
    source_candidates: List[SourceCandidateRef] = Field(default_factory=list)


class ComparisonDigestResult(BaseModel):
    query: str = ""
    products_compared: List[ComparisonProductDigest] = Field(default_factory=list)
    comparison_axes: List[str] = Field(default_factory=list)
    high_level_differences: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    source_candidates: List[SourceCandidateRef] = Field(default_factory=list)


class ProductSnapshotEntry(BaseModel):
    family_id: str
    label: str
    facts: List[str] = Field(default_factory=list)
    source_candidates: List[SourceCandidateRef] = Field(default_factory=list)


class ProductDimensionBaseline(BaseModel):
    family_id: str
    label: str
    dimension: str
    evidence: List[str] = Field(default_factory=list)


class ProductSnapshotResult(BaseModel):
    query: str = ""
    dimensions_requested: List[str] = Field(default_factory=list)
    products: List[ProductSnapshotEntry] = Field(default_factory=list)
    value_ranges_or_baselines: List[ProductDimensionBaseline] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    source_candidates: List[SourceCandidateRef] = Field(default_factory=list)


class TurnIntentExtractionResult(BaseModel):
    slot_updates: GazSlots = Field(default_factory=GazSlots)
    intent_flags: IntentFlags = Field(default_factory=IntentFlags)
    problem_summary_candidate: str = ""
    current_client_intent: ClientIntent = "overview"
    customer_temperature: CustomerTemperature = "neutral"


class AnswerPlanResult(BaseModel):
    current_client_intent: ClientIntent = "overview"
    answer_depth: AnswerDepth = "broad"
    customer_temperature: CustomerTemperature = "neutral"
    work_mode: GazStage = "SELL"
    clarification_allowed: bool = False
    should_offer_provisional_recommendations: bool = True
    search_query: str = ""
    branch_hint: Optional[str] = None
    provisional_recommendations: List[str] = Field(default_factory=list)


class ShortlistEntry(BaseModel):
    family_id: str
    fit_reason: str
    risk_note: Optional[str] = None


class FollowupDocument(BaseModel):
    candidate_id: str
    title: str
    why_it_matters: str


class FollowupPack(BaseModel):
    decision_role: Optional[str] = None
    recommended_action: str
    documents: List[FollowupDocument] = Field(default_factory=list)


class PolicyValidationResult(BaseModel):
    is_valid: bool = True
    violations: List[str] = Field(default_factory=list)
    suggested_fix: Optional[str] = None
