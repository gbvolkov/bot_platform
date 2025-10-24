from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

from langchain_core.prompts import ChatPromptTemplate

import logging
logger = logging.getLogger(__name__)


class SalaryRange(BaseModel):
    """
    Target compensation as integers. Either bound may be null.
    If only a single expected value is known, set it to `min` and leave `max` null.
    Currency defaults to RUB when unclear.
    """
    min: Optional[int] = Field(
        default=None,
        description=(
            "Lower bound of expected salary as an integer WITHOUT separators or symbols "
            "(e.g., 150000). Leave null if unknown."
        ),
        ge=0,
        json_schema_extra={"examples": [120000, 250000, None]},
    )
    max: Optional[int] = Field(
        default=None,
        description=(
            "Upper bound of expected salary as an integer WITHOUT separators or symbols. "
            "Leave null if unknown."
        ),
        ge=0,
        json_schema_extra={"examples": [180000, None]},
    )
    currency: Literal["RUB", "USD", "EUR"] = Field(
        default="RUB",
        description="Three-letter currency code. Default is RUB when not specified.",
        json_schema_extra={"examples": ["RUB", "USD", "EUR"]},
    )

    @field_validator("currency", mode="before")
    @classmethod
    def _coerce_currency(cls, v):
        if not v:
            return "RUB"
        s = str(v).upper()
        if "RUB" in s or "РУБ" in s or "₽" in s:
            return "RUB"
        if "USD" in s or "$" in s:
            return "USD"
        if "EUR" in s or "€" in s:
            return "EUR"
        return "RUB"

    @model_validator(mode="after")
    def _order_range(self):
        if self.min and self.max and self.min > self.max:
            self.min, self.max = self.max, self.min
        return self

class ResumeFeatures(BaseModel):
    """
    Extracted search criteria from a resume.
    Keep items atomic, de-duplicated, and high-signal.
    If information is missing, return empty lists or nulls.
    """
    positions: List[str] = Field(
        default_factory=list,
        description=(
            "Job titles or roles the candidate fits or seeks. "
            "Use concise, standard titles (e.g., 'Senior Backend Engineer', 'Data Scientist'). "
            "Avoid company names or departments. Max ~5 items."
        ),
        json_schema_extra={"examples": [["Senior Backend Engineer", "Tech Lead", "Python Developer"]]},
    )
    locations: List[str] = Field(
        default_factory=list,
        description=(
            "Geographic preferences or bases (city/region/country). "
            "One place per item (e.g., 'Berlin', 'Moscow', 'Remote (EU time)') "
            "— avoid addresses. Max ~5 items."
        ),
        json_schema_extra={"examples": [["Berlin", "Moscow", "Remote"]]},
    )
    skills: List[str] = Field(
        default_factory=list,
        description=(
            "Hard skills/technologies as atomic tokens, not sentences. "
            "Examples: 'Python', 'PostgreSQL', 'Kubernetes', 'ETL'. "
            "Do not include proficiency adjectives (e.g., 'strong'). Max ~10 items."
        ),
        json_schema_extra={"examples": [["Python", "Django", "PostgreSQL", "Docker", "Kubernetes"]]},
    )
    competencies: List[str] = Field(
        default_factory=list,
        description=(
            "Functional/behavioral competencies or domains (often broader than skills). "
            "Examples: 'People Management', 'Stakeholder Management', 'Data Warehousing', 'SRE'. "
            "Max ~10 items."
        ),
        json_schema_extra={"examples": [["People Management", "System Design", "Observability"]]},
    )
    salary_range: SalaryRange = Field(
        default_factory=SalaryRange,
        description=(
            "Expected compensation range. Use integers for min/max; currency is RUB by default if unclear. "
            "If only a single number is implied, put it in 'min' and leave 'max' null."
        ),
    )

    @model_validator(mode="after")
    def _normalize(self):
        def norm(xs, cap):
            # trim, drop empties, dedupe (casefold), cap
            seen, out = set(), []
            for x in xs:
                if not isinstance(x, str):
                    continue
                y = " ".join(x.split()).strip()
                k = y.casefold()
                if y and k not in seen:
                    seen.add(k)
                    out.append(y)
                if len(out) >= cap:
                    break
            return out
        self.positions = norm(self.positions, 5)
        self.locations = norm(self.locations, 5)
        self.skills = norm(self.skills, 10)
        self.competencies = norm(self.competencies, 10)
        return self
    


import re
from typing import List, Optional, Any

# ---------------- helpers ----------------

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _split_list_line(line: str) -> List[str]:
    # split on commas, semicolons, slashes, bullets, and "and/и"
    parts = re.split(r"[;,/|•·••]|(?:\s+and\s+)|(?:\s+и\s+)", line)
    return [p.strip(" -—·•\t") for p in parts if p and p.strip()]

def _dedupe_keep_order(items: List[str], cap: int) -> List[str]:
    seen, out = set(), []
    for it in items:
        k = it.casefold()
        if k and k not in seen:
            seen.add(k)
            out.append(it)
        if len(out) >= cap:
            break
    return out

def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""

# Try to instantiate either SalaryRange(min, max, currency) or SalaryRange(minimum, maximum, currency)
def _make_salary_range(min_val: Optional[int], max_val: Optional[int], currency: str):
    try:
        return SalaryRange(min=min_val, max=max_val, currency=currency)  # type: ignore
    except TypeError:
        return SalaryRange(minimum=min_val, maximum=max_val, currency=currency)  # type: ignore

# ---------------- compiled regexes ----------------

_POS_PAT = re.compile(r"(?:position|role|title|должн\w*)\s*[:.\-–—]\s*([^\n]+)", re.IGNORECASE)
_SKILL_PAT = re.compile(r"(?:skill|ability|experienced in|навык\w*|умени\w*|опыт\s+в)\s*[:.\-–—]\s*([^\n]+)", re.IGNORECASE)
_COMP_PAT = re.compile(r"(?:competenc(?:e|y|ies)|компетен\w*)\s*[:.\-–—]\s*([^\n]+)", re.IGNORECASE)

_LOC_PAT = re.compile(
    r"(?:location|based in|lives in|located in|город|локац\w*|прожива\w*)\s*[:.\-–—]?\s*([^\n,;]+)",
    re.IGNORECASE,
)
_IN_CITY_PAT = re.compile(
    r"(?:\b(?:in|at|near|within|в|из|по|г\.)\b)\s+([A-ZА-Я][\w.\- ]{2,})",
    re.IGNORECASE | re.UNICODE,
)

# Salary-related
_SALARY_LINE_PAT = re.compile(
    r"(?:salary|compensation|pay|оклад|зарплат\w*|вознагражд\w*|доход)\s*[:.\-–—]?\s*([^\n]+)",
    re.IGNORECASE,
)
_CURRENCY_PAT = re.compile(r"(RUB|USD|EUR|руб\.?|₽|€|\$)", re.IGNORECASE)
_RANGE_PAT = re.compile(r"(?P<a>\d[\d\s.,]{2,})\s*[-–—]\s*(?P<b>\d[\d\s.,]{2,})", re.IGNORECASE)
_SINGLE_AMT_PAT = re.compile(r"(?P<a>\d[\d\s.,]{2,})", re.IGNORECASE)
_MULTIPLIER_PAT = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(k|m|тыс|млн)\b", re.IGNORECASE)

# ---------------- fallbacks ----------------

def _fallback_positions(text: str) -> List[str]:
    hits = _POS_PAT.findall(text)
    if hits:
        items: List[str] = []
        for h in hits:
            items.extend(_split_list_line(h))
        return _dedupe_keep_order([_norm_text(x) for x in items], cap=3)

    # Headline often contains a role/title
    headline = _first_nonempty_line(text)
    return [_norm_text(headline[:80])] if headline else []

def _fallback_locations(text: str) -> List[str]:
    # Strong signals: explicit labels
    hits = [h.strip() for h in _LOC_PAT.findall(text) if h and h.strip()]
    if hits:
        return _dedupe_keep_order([_norm_text(h) for h in hits], cap=3)

    # Next-best: preposition + place
    in_hits: List[str] = []
    for m in _IN_CITY_PAT.finditer(text):
        cand = m.group(1).strip(" ,.;:()[]")
        if cand:
            in_hits.append(cand)
    if in_hits:
        return _dedupe_keep_order([_norm_text(h) for h in in_hits], cap=3)

    # Avoid noisy “capitalized word” heuristic
    return []

def _fallback_skills(text: str) -> List[str]:
    hits = _SKILL_PAT.findall(text)
    if hits:
        items: List[str] = []
        for h in hits:
            items.extend(_split_list_line(h))
        return _dedupe_keep_order([_norm_text(x) for x in items], cap=3)
    return []

def _fallback_competencies(text: str) -> List[str]:
    hits = _COMP_PAT.findall(text)
    if hits:
        items: List[str] = []
        for h in hits:
            items.extend(_split_list_line(h))
        return _dedupe_keep_order([_norm_text(x) for x in items], cap=3)
    return []

# ---- salary utils ----

def _parse_multiplier(token: str) -> Optional[int]:
    t = token.casefold()
    if t in {"k", "тыс"}:
        return 1_000
    if t in {"m", "млн"}:
        return 1_000_000
    return None

def _to_int_amount(s: str) -> Optional[int]:
    s = s.strip()
    # "120k", "1.2m", "120 тыс", "1,2 млн"
    m = _MULTIPLIER_PAT.fullmatch(s)
    if m:
        num = float(m.group(1).replace(",", "."))
        mult = _parse_multiplier(m.group(2)) or 1
        return int(round(num * mult))
    # Plain number with spaces/commas/dots
    cleaned = re.sub(r"[^\d]", "", s)
    return int(cleaned) if cleaned.isdigit() else None

def _detect_currency(s: str) -> str:
    m = _CURRENCY_PAT.search(s)
    if not m:
        return "RUB"
    t = m.group(0).upper()
    if "RUB" in t or "РУБ" in t or "₽" in t:
        return "RUB"
    if "USD" in t or "$" in t:
        return "USD"
    if "EUR" in t or "€" in t:
        return "EUR"
    return "RUB"

def _parse_salary_from_snippet(snippet: str) -> Optional[Any]:
    # Prefer range if present
    rng = _RANGE_PAT.search(snippet)
    if rng:
        a = _to_int_amount(rng.group("a"))
        b = _to_int_amount(rng.group("b"))
        if a and b:
            lo, hi = sorted([a, b])
            return lo, hi, _detect_currency(snippet)

    # Else single amount
    # try multiplier forms first, then generic numbers
    # Scan all numbers, pick the largest (likely salary, not a year)
    nums = []
    for m in _MULTIPLIER_PAT.finditer(snippet):
        val = _to_int_amount(m.group(0))
        if val:
            nums.append(val)
    if not nums:
        for m in _SINGLE_AMT_PAT.finditer(snippet):
            val = _to_int_amount(m.group("a"))
            if val:
                nums.append(val)
    if nums:
        return max(nums), None, _detect_currency(snippet)
    return None

def _fallback_salary(text: str) -> "SalaryRange":
    # 1) Look in explicit salary lines
    for m in _SALARY_LINE_PAT.finditer(text):
        parsed = _parse_salary_from_snippet(m.group(1))
        if parsed:
            lo, hi, cur = parsed
            return _make_salary_range(lo, hi, cur)

    # 2) Otherwise search the whole text (less precise)
    parsed = _parse_salary_from_snippet(text)
    if parsed:
        lo, hi, cur = parsed
        return _make_salary_range(lo, hi, cur)

    # 3) Unknown
    return _make_salary_range(None, None, "RUB")


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze resumes and extract search criteria.\n"
     "Return ONLY the requested fields; if missing, use empty lists/nulls.\n"
     "Use RUB by default if currency is unclear."
    ),
    ("human",
     "Resume:\n{resume_text}")
])

def _build_structured_chain(extractor_llm):
    # key line: force structured (typed) output via Pydantic schema
    structured_llm = extractor_llm.with_structured_output(ResumeFeatures)
    return _PROMPT | structured_llm

# ---------------- Adapter for your existing fallbacks ----------------

def _coerce_salary_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts variants like {'min':..,'max':..} or {'minimum':..,'maximum':..},
    returns normalized dict with 'min','max','currency'.
    """
    if not isinstance(d, dict):
        return {"min": None, "max": None, "currency": "RUB"}
    min_ = d.get("min", d.get("minimum"))
    max_ = d.get("max", d.get("maximum"))
    cur = d.get("currency", "RUB")
    return SalaryRange(min=min_, max=max_, currency=cur).model_dump()

# ---------------- Public function (drop-in) ----------------

def extract_features_from_resume(resume_text: str, extractor_llm) -> Dict[str, Any]:
    """
    Refactored: uses LangChain with_structured_output to avoid manual JSON parsing.
    Falls back to your regex-based helpers on any exception.
    """
    chain = _build_structured_chain(extractor_llm)

    try:
        features: ResumeFeatures = chain.invoke({"resume_text": resume_text})
        return features.model_dump()
    except Exception as exc:  # defensive fallback, mirrors your original behavior
        logger.warning("Structured extraction failed, using fallbacks: %s", exc)
        return {
            "positions": _fallback_positions(resume_text),
            "locations": _fallback_locations(resume_text),
            "skills": _fallback_skills(resume_text),
            "competencies": _fallback_competencies(resume_text),
            "salary_range": _coerce_salary_dict(_fallback_salary(resume_text).to_dict()
                                                if hasattr(_fallback_salary(resume_text), "to_dict")
                                                else _fallback_salary(resume_text)),
        }