from typing import List, Optional, Dict, Tuple, Any
import re

import config

from ..retrievers.teamly_api_wrapper import TeamlyAPIWrapper_Glossary
from difflib import SequenceMatcher

from langchain.schema import Document

import pymorphy2
try:
    morph = pymorphy2.MorphAnalyzer()
except Exception:
    morph = None

ARTICLE_URL_TMPL = "https://kb.ileasing.ru/space/{space_id}/article/{article_id}"

teamly_glossary_wrapper = TeamlyAPIWrapper_Glossary(auth_data_store="./auth_glossary.json")
tnd_docs = teamly_glossary_wrapper.sd_documents

#glossary_retriever = TeamlyRetriever_Glossary(auth_data_store="./auth_glossary.json", k=3)
def get_terms_and_definitions(query: str) -> str:
    q = query.upper()
    return "\n\n".join(
        d.page_content.strip()
        for d in tnd_docs
        if getattr(d, "page_content", None) in q
    )

def get_abbreviation_meaning(
    tnd_docs: List[Document],
    query: str
) -> List[Document]:
    q = query.upper()
    return [
        d for d in tnd_docs
        if d.metadata.get("source") == "abbr"
        and d.metadata['term'].upper() == q
    ]

def get_term_meanings(
    tnd_docs: List[Document],
    query: str,
    top_k: int = 5,
    min_score: float = 0.60,
    token_sim_threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """
    Fuzzy lookup of term meanings (non-abbreviation terms) in the T&T list.

    The matcher:
        - Ignores case and word order.
        - Tolerates minor typos via per-token similarity.
        - Optionally lemmatizes (if pymorphy2 is installed) to better align with
        your catalog where terms are stored in the singular nominative case.
        - Excludes abbreviation entries (source == "abbr").

    Args:
        query (str): The user-provided phrase containing a term.
            The input can contain words in any order and in any case.
        top_k (int, optional): Maximum number of best-scoring matches to return.
            Defaults to 5.
        min_score (float, optional): Minimum combined similarity score [0..1]
            required for a result to be included. Defaults to 0.60.
        token_sim_threshold (float, optional): Per-token similarity threshold [0..1]
            for mapping term tokens to query tokens (minor typos allowed).
            Defaults to 0.80.

    Returns:
        List[Dict[str, Any]]: A list of match objects (best first). Each item contains:
            - "term" (str): Canonical stored term (singular nominative).
            - "definition" (str): The definition text.
            - "score" (float): Combined fuzzy score in [0..1].
            - "matched_tokens" (List[Tuple[str, str, float]]): Pairs of
                (term_token, query_token, token_similarity).
            - "url" (str): Link to the article.
            - "space_id" (Any), "article_id" (Any), "article_title" (str), "source" (str)

    Notes:
        - Abbreviations are searched via your existing get_abbreviations(); this
            function intentionally focuses on terms (i.e., non-"abbr" sources).
        - If pymorphy2 is installed, tokens are lemmatized to help match the
            stored singular nominative forms. If not installed, a robust
            lowercased, alphanumeric token match is used.
    """
    # --- helpers ---------------------------------------------------------

    def _normalize(text: str) -> str:
        # Strip punctuation-like chars, collapse whitespace, lowercase
        # Keeps Cyrillic and Latin letters and digits
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text, re.U)
        return " ".join(tokens).strip()

    def _tokenize(text: str) -> List[str]:
        norm = _normalize(text).lower()
        return norm.split() if norm else []

    def _lemmatize_ru(tokens: List[str]) -> List[str]:
        # Optional: better alignment with nominative singular storage.
        if morph is None:
            return tokens  # fallback: no lemmatization
        lemmas = []
        for t in tokens:
            try:
                p = morph.parse(t)[0]
                lemmas.append(p.normal_form)
            except Exception:
                lemmas.append(t)
        return lemmas

    def _prep_tokens(text: str) -> List[str]:
        return _lemmatize_ru(_tokenize(text))

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() if a and b else 0.0

    def _best_token_alignment(
        term_tokens: List[str],
        query_tokens: List[str],
        threshold: float,
    ) -> Tuple[float, float, List[Tuple[str, str, float]]]:
        """
        Greedy one-to-one token alignment using similarity; returns:
            coverage = matched_term_tokens / len(term_tokens)
            overlap  = matched_term_tokens / |union(term_tokens, query_tokens)|
            matches  = list of (term_tok, query_tok, sim)
        """
        matches: List[Tuple[str, str, float]] = []
        q_remaining = list(query_tokens)
        matched = 0

        for tt in term_tokens:
            best_sim = 0.0
            best_qi = -1
            for qi, qt in enumerate(q_remaining):
                sim = _ratio(tt, qt)
                if sim > best_sim:
                    best_sim, best_qi = sim, qi
            if best_sim >= threshold and best_qi >= 0:
                matched += 1
                matches.append((tt, q_remaining[best_qi], best_sim))
                q_remaining.pop(best_qi)

        union_size = len(set(term_tokens) | set(query_tokens)) or 1
        coverage = matched / (len(term_tokens) or 1)
        overlap = matched / union_size
        return coverage, overlap, matches

    # --- prep query ------------------------------------------------------
    q_tokens = _prep_tokens(query)

    # --- iterate over term docs (exclude abbreviations) ------------------
    candidates = []

    for d in tnd_docs:
        if d.metadata.get("source") == "abbr":
            continue  # handled elsewhere by get_abbreviations

        term = (d.metadata.get("term") or "").strip()
        if not term:
            continue

        # Cache normalized tokens per doc to avoid recompute
        cache_key = "_norm_term_tokens"
        if cache_key not in d.metadata:
            d.metadata[cache_key] = _prep_tokens(term)

        t_tokens: List[str] = d.metadata[cache_key]

        # Skip if either side is empty after normalization
        if not t_tokens or not q_tokens:
            continue

        # Order-insensitive and order-sensitive similarities
        cov, ovl, token_matches = _best_token_alignment(
            t_tokens, q_tokens, token_sim_threshold
        )
        order_insensitive = _ratio(" ".join(sorted(t_tokens)), " ".join(sorted(q_tokens)))
        order_sensitive = _ratio(" ".join(t_tokens), " ".join(q_tokens))

        # Combine scores (weights tuned for good default feel)
        score = (
            0.50 * cov +           # ensure most term words are present
            0.20 * ovl +           # reward compact overlap
            0.20 * order_insensitive +  # handle word reordering
            0.10 * order_sensitive      # reward close phrasing when it happens
        )

        if score >= min_score:
            candidates.append({
                "page_content": d.page_content,
                "term": term,
                "definition": d.metadata.get("definition", ""),
                "score": round(score, 4),
                "matched_tokens": token_matches,
                "url": ARTICLE_URL_TMPL.format(
                    space_id=d.metadata.get("space_id"),
                    article_id=d.metadata.get("article_id"),
                ),
                "space_id": d.metadata.get("space_id"),
                "article_id": d.metadata.get("article_id"),
                "article_title": d.metadata.get("article_title"),
                "source": d.metadata.get("source"),
                "_doc": d,  # keep the original doc if you need it
            })

    # Sort best-first and trim
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]
