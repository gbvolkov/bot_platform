from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ..schemas import SearchFilters
from ..settings import SalesLeadAgentSettings

#_WORD_RE = re.compile(r"[0-9A-Za-zЀ-ӿ]{2,}")
_WORD_RE = re.compile(r"[0-9A-Za-z\u0400-\u04FF]{2,}")
_STOPWORDS = {
    "для",
    "под",
    "при",
    "или",
    "оказание",
    "or",
    "the",
    "and",
    "with",
}


class ProcurementQueryBuilder:
    def __init__(self, settings: SalesLeadAgentSettings) -> None:
        self._template = settings.procurement_search_template

    def build_search_string(self, filters: SearchFilters) -> str:
        raw_parts = [
            filters.query_text or "",
            filters.customer_name or "",
            filters.supplier_hint or "",
        ]
        tokens: list[str] = []
        seen: set[str] = set()
        for part in raw_parts:
            for token in _WORD_RE.findall(part):
                normalized = token.strip().lower()
                if normalized in _STOPWORDS or normalized in seen:
                    continue
                seen.add(normalized)
                tokens.append(normalized)
        if not tokens and filters.query_text:
            tokens = [filters.query_text.strip().lower()]
        return "+".join(tokens[:5])

    def build_url(self, filters: SearchFilters) -> str:
        parsed = urlparse(self._template)
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        params["searchString"] = self.build_search_string(filters)
        new_query = urlencode(params, doseq=True)
        return urlunparse(parsed._replace(query=new_query))
