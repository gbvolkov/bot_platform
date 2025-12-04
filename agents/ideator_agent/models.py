from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

IMPORTANCE_SCORE = {"high": 3, "medium": 2, "low": 1}


@dataclass
class ArticleRecord:
    id: int
    title: str
    summary: str
    url: str
    importance: str
    date: str
    processed_at: str
    search_country: str
    search_language: str
    source_file: str
    word_count: int
    importance_reasoning: str
    search_keywords: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)

    def region_label(self) -> str:
        country = (self.search_country or "").lower()
        if country in ("ru", "rus", "russia", "россия"):
            return "РФ — релевантно"
        if country:
            return "Зарубежный рынок — требует адаптации"
        return "Регион не указан"

    def norm_importance(self) -> str:
        val = (self.importance or "").lower()
        return val if val in IMPORTANCE_SCORE else "medium"

    def importance_weight(self) -> int:
        return IMPORTANCE_SCORE.get(self.norm_importance(), 1)

    def display_title(self, fallback_words: int = 8) -> str:
        if self.title:
            return self.title
        snippet = self.summary.split()
        return " ".join(snippet[:fallback_words]) if snippet else "<без заголовка>"

    def display_date(self) -> str:
        if self.date:
            return self.date[:10]
        if self.processed_at:
            return self.processed_at[:10]
        return "n/a"

    def fact_ref(self) -> str:
        return (
            f"[{self.search_country or 'n/a'} | "
            f"{self.norm_importance()} | "
            f"\"{self.display_title()}\" | "
            f"{self.display_date()} | "
            f"{self.url}]"
        )


@dataclass
class IdeatorReport:
    generated_at: str
    report_date: str
    total_articles: int
    high_importance: int
    medium_importance: int
    low_importance: int
    search_goal: str
    search_prompt: str
    articles: List[ArticleRecord]

    def sorted_articles(self) -> List[ArticleRecord]:
        return sorted(
            self.articles,
            key=lambda a: (-a.importance_weight(), -a.word_count, a.id),
        )

    def filter_by_ids(self, ids: List[int]) -> List[ArticleRecord]:
        id_set = set(ids)
        return [a for a in self.articles if a.id in id_set]

    def countries(self) -> List[str]:
        return sorted({a.search_country for a in self.articles if a.search_country})
