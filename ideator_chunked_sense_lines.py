from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from hashlib import blake2b
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional
    tiktoken = None


DEFAULT_MAX_TOKENS = 32000
DEFAULT_OVERHEAD_TOKENS = 2000
DEFAULT_ENCODING_NAME = "cl100k_base"


@dataclass
class Prompt:
    system: str
    user: str


@dataclass
class ArticleCard:
    id: int
    title: str
    summary: str
    region: str = ""
    importance: str = ""


@dataclass
class LineCard:
    id: str
    short_title: str
    description: str
    region_note: str = ""
    exemplar_article_ids: List[int] = field(default_factory=list)
    exemplar_titles: List[str] = field(default_factory=list)


@dataclass
class Assignment:
    article_id: int
    line_id: str
    confidence: float
    rationale: str = ""


@dataclass
class DiscoverResult:
    assignments: List[Assignment]
    new_lines: List[LineCard]
    unassigned: List[ArticleCard]


@dataclass
class MergeResult:
    merged_lines: List[LineCard]
    id_map: Dict[str, str]


@dataclass
class AssignConfig:
    emb_threshold: float = 0.74
    score_threshold: float = 0.76
    margin_threshold: float = 0.07
    weight_emb: float = 0.62
    weight_kw: float = 0.18
    weight_ent: float = 0.12
    weight_region: float = 0.08
    region_match: float = 1.0
    region_unknown: float = 0.7
    region_mismatch: float = 0.4


@dataclass
class MergeConfig:
    merge_threshold: float = 0.78
    merge_soft_threshold: float = 0.72
    weight_emb: float = 0.55
    weight_kw: float = 0.20
    weight_ent: float = 0.15
    weight_exemplar: float = 0.10
    region_penalty: float = 0.10
    kw_threshold: float = 0.40
    exemplar_overlap_threshold: float = 0.33


EmbedFn = Callable[[str], Sequence[float]]
KeywordFn = Callable[[str, int], List[str]]
EntityFn = Callable[[str, int], List[str]]
LLMCall = Callable[[Prompt], Any]


ASSIGN_SYSTEM_PROMPT = (
    "You are a strict classifier. Assign each article to an existing sense line "
    "ONLY if the fit is strong and supported by the article summary. If no strong "
    "match, mark NEW. Do not invent facts. Return JSON only."
)

ASSIGN_USER_TEMPLATE = """\
{{
  "sense_lines": {sense_lines},
  "articles": {articles},
  "rules": {{
    "assign_only_if_strong": true,
    "max_lines_per_article": 1
  }}
}}
"""

NEW_LINES_SYSTEM_PROMPT = (
    "Group the unassigned articles into 1-4 NEW sense lines. "
    "Each line must have: short_title (6-12 words), description (1-2 factual sentences), "
    "region_note, exemplar_article_ids (2-3). Return JSON only."
)

NEW_LINES_USER_TEMPLATE = """\
{{
  "unassigned_articles": {articles},
  "max_new_lines": {max_new_lines}
}}
"""

MERGE_TIEBREAK_SYSTEM_PROMPT = (
    "Decide if two sense lines are duplicates or near-duplicates. "
    "Merge ONLY if they describe the same underlying topic with the same mechanism. "
    "Return JSON only."
)

MERGE_TIEBREAK_USER_TEMPLATE = """\
{{
  "pair": {{
    "A": {line_a},
    "B": {line_b}
  }}
}}
"""

MERGE_SYNTH_SYSTEM_PROMPT = (
    "Combine these line variants into one canonical sense line. "
    "Keep it factual, 1-2 sentences, no new facts. Return JSON only."
)

MERGE_SYNTH_USER_TEMPLATE = """\
{{
  "canonical_id": "{canonical_id}",
  "members": {members}
}}
"""


def chunk_articles_by_tokens(
    articles: Sequence[ArticleCard],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overhead_tokens: int = DEFAULT_OVERHEAD_TOKENS,
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> List[List[ArticleCard]]:
    encoder = _get_encoder(encoding_name)
    limit = max_tokens - max(0, overhead_tokens)
    chunks: List[List[ArticleCard]] = []
    current: List[ArticleCard] = []
    current_tokens = 0
    for art in articles:
        art_tokens = _count_tokens(_serialize_article(art), encoder)
        if art_tokens > limit:
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            chunks.append([art])
            continue
        if current and current_tokens + art_tokens > limit:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(art)
        current_tokens += art_tokens
    if current:
        chunks.append(current)
    return chunks


def discover_lines(
    chunk_articles: Sequence[ArticleCard],
    existing_lines: Sequence[LineCard],
    *,
    embed_fn: Optional[EmbedFn] = None,
    keyword_fn: Optional[KeywordFn] = None,
    entity_fn: Optional[EntityFn] = None,
    llm_call: Optional[LLMCall] = None,
    assign_config: AssignConfig = AssignConfig(),
    cluster_threshold: float = 0.62,
    max_new_lines: int = 4,
    min_support: int = 2,
    use_llm_assignment: bool = False,
) -> DiscoverResult:
    article_lookup = {a.id: a for a in chunk_articles}
    embedder = _ensure_embedder(embed_fn)
    keyword_fn = keyword_fn or _extract_keywords
    entity_fn = entity_fn or _extract_entities

    assignments: List[Assignment] = []
    unassigned: List[ArticleCard] = []

    if existing_lines and use_llm_assignment and llm_call:
        assignments, unassigned = _llm_assign(
            chunk_articles,
            existing_lines,
            llm_call,
        )
    else:
        line_sigs = _build_line_signatures(
            existing_lines,
            article_lookup,
            embedder,
            keyword_fn,
            entity_fn,
        )
        for art in chunk_articles:
            best = _score_article_against_lines(
                art,
                line_sigs,
                embedder,
                keyword_fn,
                entity_fn,
                assign_config,
            )
            if best is None:
                unassigned.append(art)
            else:
                assignments.append(best)

    new_lines: List[LineCard] = []
    remaining = list(unassigned)
    if remaining:
        new_line_assignments: List[Assignment] = []
        if llm_call:
            new_lines, mapping = _llm_new_lines(
                remaining,
                llm_call,
                max_new_lines=max_new_lines,
            )
            for art_id, line_id in mapping.items():
                new_line_assignments.append(
                    Assignment(article_id=art_id, line_id=line_id, confidence=1.0)
                )
            assigned_ids = set(mapping.keys())
            remaining = [a for a in remaining if a.id not in assigned_ids]
        if not new_lines and remaining:
            clusters = _cluster_unassigned_by_embeddings(
                remaining,
                embedder,
                cluster_threshold=cluster_threshold,
            )
            for idx, cluster in enumerate(clusters, start=1):
                if len(cluster) < min_support:
                    continue
                line = _summarize_cluster(
                    cluster,
                    temp_id=f"N{idx}",
                    keyword_fn=keyword_fn,
                )
                new_lines.append(line)
                for art in cluster:
                    new_line_assignments.append(
                        Assignment(article_id=art.id, line_id=line.id, confidence=1.0)
                    )
        if new_line_assignments:
            assignments.extend(new_line_assignments)
            assigned_ids = {a.article_id for a in new_line_assignments}
            remaining = [a for a in remaining if a.id not in assigned_ids]

    return DiscoverResult(assignments=assignments, new_lines=new_lines, unassigned=remaining)


def merge_lines(
    existing_lines: Sequence[LineCard],
    new_lines: Sequence[LineCard],
    *,
    embed_fn: Optional[EmbedFn] = None,
    keyword_fn: Optional[KeywordFn] = None,
    entity_fn: Optional[EntityFn] = None,
    llm_call: Optional[LLMCall] = None,
    merge_config: MergeConfig = MergeConfig(),
    min_support: int = 2,
    consider_existing_pairs: bool = False,
) -> MergeResult:
    embedder = _ensure_embedder(embed_fn)
    keyword_fn = keyword_fn or _extract_keywords
    entity_fn = entity_fn or _extract_entities
    existing_ids = {line.id for line in existing_lines}
    all_lines = list(existing_lines) + list(new_lines)
    if not all_lines:
        return MergeResult(merged_lines=[], id_map={})

    line_sigs = _build_line_signatures(
        all_lines,
        {},
        embedder,
        keyword_fn,
        entity_fn,
    )

    candidates: List[Tuple[str, str, float, float, float, float, float]] = []
    for a, b in combinations(all_lines, 2):
        if not consider_existing_pairs and a.id in existing_ids and b.id in existing_ids:
            continue
        score, emb_cos, kw_j, ent_j, overlap = _merge_metrics(a, b, line_sigs, merge_config)
        if score >= merge_config.merge_soft_threshold:
            candidates.append((a.id, b.id, score, emb_cos, kw_j, ent_j, overlap))

    parents = {line.id: line.id for line in all_lines}

    for a_id, b_id, score, emb_cos, kw_j, _ent_j, overlap in sorted(
        candidates, key=lambda x: x[2], reverse=True
    ):
        passes_gate = (
            emb_cos >= 0.74
            or kw_j >= merge_config.kw_threshold
            or overlap >= merge_config.exemplar_overlap_threshold
        )
        if score >= merge_config.merge_threshold and passes_gate:
            _union(parents, a_id, b_id)
        else:
            if llm_call and _llm_merge_tiebreak(a_id, b_id, line_sigs, llm_call):
                _union(parents, a_id, b_id)

    groups: Dict[str, List[LineCard]] = {}
    for line in all_lines:
        root = _find(parents, line.id)
        groups.setdefault(root, []).append(line)

    merged_lines: List[LineCard] = []
    id_map: Dict[str, str] = {}

    for group_lines in groups.values():
        canonical_id = _choose_canonical_id(group_lines, existing_ids)
        for line in group_lines:
            id_map[line.id] = canonical_id
        merged = _synthesize_line(
            canonical_id,
            group_lines,
            llm_call=llm_call,
        )
        if len(merged.exemplar_article_ids) >= min_support or not merged.exemplar_article_ids:
            merged_lines.append(merged)

    return MergeResult(merged_lines=merged_lines, id_map=id_map)


def build_assign_prompt(
    articles: Sequence[ArticleCard],
    sense_lines: Sequence[LineCard],
) -> Prompt:
    payload = ASSIGN_USER_TEMPLATE.format(
        sense_lines=json.dumps([_line_to_json(line) for line in sense_lines]),
        articles=json.dumps([_article_to_json(a) for a in articles]),
    )
    return Prompt(system=ASSIGN_SYSTEM_PROMPT, user=payload)


def build_new_lines_prompt(
    articles: Sequence[ArticleCard],
    max_new_lines: int,
) -> Prompt:
    payload = NEW_LINES_USER_TEMPLATE.format(
        articles=json.dumps([_article_to_json(a) for a in articles]),
        max_new_lines=max_new_lines,
    )
    return Prompt(system=NEW_LINES_SYSTEM_PROMPT, user=payload)


def build_merge_tiebreak_prompt(line_a: LineCard, line_b: LineCard) -> Prompt:
    payload = MERGE_TIEBREAK_USER_TEMPLATE.format(
        line_a=json.dumps(_line_to_json(line_a)),
        line_b=json.dumps(_line_to_json(line_b)),
    )
    return Prompt(system=MERGE_TIEBREAK_SYSTEM_PROMPT, user=payload)


def build_merge_synthesis_prompt(canonical_id: str, members: Sequence[LineCard]) -> Prompt:
    payload = MERGE_SYNTH_USER_TEMPLATE.format(
        canonical_id=canonical_id,
        members=json.dumps([_line_to_json(m) for m in members]),
    )
    return Prompt(system=MERGE_SYNTH_SYSTEM_PROMPT, user=payload)


def _serialize_article(article: ArticleCard) -> str:
    return (
        f"[{article.id}] {article.title}\n"
        f"{article.summary}\n"
        f"region: {article.region}\n"
        f"importance: {article.importance}"
    )


def _get_encoder(encoding_name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def _count_tokens(text: str, encoder) -> int:
    if encoder is None:
        return max(1, len(text) // 4)
    return len(encoder.encode(text))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\-]{3,}", _normalize_text(text), flags=re.UNICODE)


def _extract_keywords(text: str, max_terms: int = 12) -> List[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    counts: Dict[str, int] = {}
    for t in tokens:
        if t in _STOPWORDS:
            continue
        counts[t] = counts.get(t, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in ordered[:max_terms]]


def _extract_entities(text: str, max_terms: int = 8) -> List[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", text or "")
    if not candidates:
        return []
    counts: Dict[str, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in ordered[:max_terms]]


def _ensure_embedder(embed_fn: Optional[EmbedFn]) -> EmbedFn:
    return embed_fn or _hashing_embed


def _hashing_embed(text: str, dims: int = 256) -> List[float]:
    vec = [0.0] * dims
    tokens = _tokenize(text)
    for tok in tokens:
        h = blake2b(tok.encode("utf-8"), digest_size=4).digest()
        idx = int.from_bytes(h, "little") % dims
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
    return sum(x * y for x, y in zip(a, b))


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _region_score(line_region: str, article_region: str, cfg: AssignConfig) -> float:
    line_region_norm = _normalize_text(line_region)
    article_region_norm = _normalize_text(article_region)
    if not line_region_norm or not article_region_norm:
        return cfg.region_unknown
    if article_region_norm in line_region_norm:
        return cfg.region_match
    return cfg.region_mismatch


def _build_line_signatures(
    lines: Sequence[LineCard],
    article_lookup: Dict[int, ArticleCard],
    embed_fn: EmbedFn,
    keyword_fn: KeywordFn,
    entity_fn: EntityFn,
) -> Dict[str, Dict[str, Any]]:
    sigs: Dict[str, Dict[str, Any]] = {}
    for line in lines:
        exemplar_titles: List[str] = list(line.exemplar_titles)
        if not exemplar_titles:
            for art_id in line.exemplar_article_ids:
                art = article_lookup.get(art_id)
                if art and art.title:
                    exemplar_titles.append(art.title)
        text_sig = f"{line.short_title}. {line.description}. {' '.join(exemplar_titles)}"
        sigs[line.id] = {
            "id": line.id,
            "emb": list(embed_fn(text_sig)),
            "kw": keyword_fn(text_sig, 12),
            "ent": entity_fn(text_sig, 8),
            "region": line.region_note,
            "exemplar_ids": list(line.exemplar_article_ids),
            "line": line,
        }
    return sigs


def _score_article_against_lines(
    article: ArticleCard,
    line_sigs: Dict[str, Dict[str, Any]],
    embed_fn: EmbedFn,
    keyword_fn: KeywordFn,
    entity_fn: EntityFn,
    cfg: AssignConfig,
) -> Optional[Assignment]:
    if not line_sigs:
        return None
    art_text = f"{article.title}. {article.summary}"
    art_emb = embed_fn(art_text)
    art_kw = keyword_fn(art_text, 12)
    art_ent = entity_fn(art_text, 8)

    scored: List[Tuple[str, float, float]] = []
    for line_id, sig in line_sigs.items():
        emb_cos = _cosine(art_emb, sig["emb"])
        kw_j = _jaccard(art_kw, sig["kw"])
        ent_j = _jaccard(art_ent, sig["ent"])
        reg = _region_score(sig["region"], article.region, cfg)
        score = (
            cfg.weight_emb * emb_cos
            + cfg.weight_kw * kw_j
            + cfg.weight_ent * ent_j
            + cfg.weight_region * reg
        )
        scored.append((line_id, score, emb_cos))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_id, best_score, best_emb = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    margin = best_score - second_score

    if best_emb >= cfg.emb_threshold and best_score >= cfg.score_threshold and margin >= cfg.margin_threshold:
        return Assignment(article_id=article.id, line_id=best_id, confidence=best_score)
    if best_emb >= (cfg.emb_threshold + 0.08):
        return Assignment(article_id=article.id, line_id=best_id, confidence=best_score)
    if _jaccard(art_kw, line_sigs[best_id]["kw"]) >= 0.50:
        return Assignment(article_id=article.id, line_id=best_id, confidence=best_score)
    return None


def _cluster_unassigned_by_embeddings(
    articles: Sequence[ArticleCard],
    embed_fn: EmbedFn,
    cluster_threshold: float,
) -> List[List[ArticleCard]]:
    if not articles:
        return []
    embs = {a.id: embed_fn(f"{a.title}. {a.summary}") for a in articles}
    parents = {a.id: a.id for a in articles}
    ids = [a.id for a in articles]
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            if _cosine(embs[a_id], embs[b_id]) >= cluster_threshold:
                _union(parents, a_id, b_id)
    groups: Dict[int, List[ArticleCard]] = {}
    for art in articles:
        root = _find(parents, art.id)
        groups.setdefault(root, []).append(art)
    return list(groups.values())


def _summarize_cluster(
    cluster: Sequence[ArticleCard],
    *,
    temp_id: str,
    keyword_fn: KeywordFn,
) -> LineCard:
    text = " ".join(f"{a.title} {a.summary}" for a in cluster)
    keywords = keyword_fn(text, 6)
    title = " / ".join(keywords[:3]) if keywords else "New sense line"
    description = " ".join(
        [a.summary.strip() for a in cluster[:2] if a.summary.strip()]
    ).strip()
    if not description:
        description = "Clustered articles share a common topic."
    region_note = _infer_region_note(cluster)
    exemplar_ids = [a.id for a in cluster[:3]]
    return LineCard(
        id=temp_id,
        short_title=title,
        description=description,
        region_note=region_note,
        exemplar_article_ids=exemplar_ids,
        exemplar_titles=[a.title for a in cluster[:3] if a.title],
    )


def _infer_region_note(cluster: Sequence[ArticleCard]) -> str:
    regions = [a.region for a in cluster if a.region]
    if not regions:
        return ""
    first = regions[0]
    if all(r == first for r in regions):
        return first
    return "requires adaptation"


def _merge_metrics(
    a: LineCard,
    b: LineCard,
    sigs: Dict[str, Dict[str, Any]],
    cfg: MergeConfig,
) -> Tuple[float, float, float, float, float]:
    sa = sigs[a.id]
    sb = sigs[b.id]
    emb_cos = _cosine(sa["emb"], sb["emb"])
    kw_j = _jaccard(sa["kw"], sb["kw"])
    ent_j = _jaccard(sa["ent"], sb["ent"])
    overlap = _exemplar_overlap(sa["exemplar_ids"], sb["exemplar_ids"])
    penalty = cfg.region_penalty if _regions_conflict(a.region_note, b.region_note) else 0.0
    score = (
        cfg.weight_emb * emb_cos
        + cfg.weight_kw * kw_j
        + cfg.weight_ent * ent_j
        + cfg.weight_exemplar * overlap
        - penalty
    )
    return score, emb_cos, kw_j, ent_j, overlap


def _exemplar_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def _regions_conflict(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return _normalize_text(a) != _normalize_text(b)


def _choose_canonical_id(lines: Sequence[LineCard], existing_ids: set[str]) -> str:
    existing = [l for l in lines if l.id in existing_ids]
    if existing:
        return _min_line_id(existing)
    return _min_line_id(lines)


def _min_line_id(lines: Sequence[LineCard]) -> str:
    def key(line: LineCard) -> Tuple[int, str]:
        m = re.match(r"[A-Za-z]+(\d+)", line.id or "")
        if m:
            return (int(m.group(1)), line.id)
        return (10**9, line.id)

    return min(lines, key=key).id


def _synthesize_line(
    canonical_id: str,
    members: Sequence[LineCard],
    *,
    llm_call: Optional[LLMCall],
) -> LineCard:
    if llm_call:
        prompt = build_merge_synthesis_prompt(canonical_id, members)
        data = _call_llm_json(llm_call, prompt)
        if isinstance(data, dict):
            try:
                return LineCard(
                    id=str(data.get("id") or canonical_id),
                    short_title=str(data.get("short_title") or ""),
                    description=str(data.get("description") or ""),
                    region_note=str(data.get("region_note") or ""),
                    exemplar_article_ids=[int(x) for x in data.get("exemplar_article_ids", [])],
                    exemplar_titles=[str(x) for x in data.get("exemplar_titles", [])],
                )
            except Exception:
                pass
    exemplar_ids: List[int] = []
    exemplar_titles: List[str] = []
    for line in members:
        exemplar_ids.extend(line.exemplar_article_ids)
        exemplar_titles.extend(line.exemplar_titles)
    exemplar_ids = list(dict.fromkeys(exemplar_ids))[:5]
    exemplar_titles = list(dict.fromkeys(exemplar_titles))[:5]
    region_note = _merge_region_notes([l.region_note for l in members])
    best = _best_line_by_support(members)
    return LineCard(
        id=canonical_id,
        short_title=best.short_title,
        description=best.description,
        region_note=region_note,
        exemplar_article_ids=exemplar_ids,
        exemplar_titles=exemplar_titles,
    )


def _merge_region_notes(notes: Sequence[str]) -> str:
    notes_clean = [n for n in notes if n]
    if not notes_clean:
        return ""
    norm = [_normalize_text(n) for n in notes_clean]
    if all(n == norm[0] for n in norm):
        return notes_clean[0]
    if any("requires adaptation" in n for n in norm):
        return "requires adaptation"
    return notes_clean[0]


def _best_line_by_support(lines: Sequence[LineCard]) -> LineCard:
    return max(lines, key=lambda l: len(l.exemplar_article_ids))


def _llm_assign(
    articles: Sequence[ArticleCard],
    lines: Sequence[LineCard],
    llm_call: LLMCall,
) -> Tuple[List[Assignment], List[ArticleCard]]:
    prompt = build_assign_prompt(articles, lines)
    data = _call_llm_json(llm_call, prompt)
    assignments: List[Assignment] = []
    assigned_ids: set[int] = set()
    if isinstance(data, dict):
        for item in data.get("assignments", []) or []:
            try:
                art_id = int(item.get("article_id"))
                line_id = str(item.get("line_id") or "").strip()
                if line_id.lower() in {"new", "none", "null"}:
                    continue
                confidence = float(item.get("confidence") or 0.0)
                rationale = str(item.get("rationale") or "")
                assignments.append(
                    Assignment(
                        article_id=art_id,
                        line_id=line_id,
                        confidence=confidence,
                        rationale=rationale,
                    )
                )
                assigned_ids.add(art_id)
            except Exception:
                continue
    unassigned = [a for a in articles if a.id not in assigned_ids]
    return assignments, unassigned


def _llm_new_lines(
    unassigned: Sequence[ArticleCard],
    llm_call: LLMCall,
    *,
    max_new_lines: int,
) -> Tuple[List[LineCard], Dict[int, str]]:
    prompt = build_new_lines_prompt(unassigned, max_new_lines)
    data = _call_llm_json(llm_call, prompt)
    new_lines: List[LineCard] = []
    mapping: Dict[int, str] = {}
    if isinstance(data, dict):
        for idx, item in enumerate(data.get("new_lines", []) or [], start=1):
            try:
                temp_id = str(item.get("temp_id") or item.get("id") or f"N{idx}")
                new_lines.append(
                    LineCard(
                        id=temp_id,
                        short_title=str(item.get("short_title") or ""),
                        description=str(item.get("description") or ""),
                        region_note=str(item.get("region_note") or ""),
                        exemplar_article_ids=[int(x) for x in item.get("exemplar_article_ids", [])],
                        exemplar_titles=[str(x) for x in item.get("exemplar_titles", [])],
                    )
                )
            except Exception:
                continue
        for item in data.get("article_to_new_line", []) or []:
            try:
                art_id = int(item.get("article_id"))
                temp_id = str(item.get("temp_id") or item.get("line_id") or "")
                if temp_id:
                    mapping[art_id] = temp_id
            except Exception:
                continue
    return new_lines, mapping


def _llm_merge_tiebreak(
    a_id: str,
    b_id: str,
    sigs: Dict[str, Dict[str, Any]],
    llm_call: LLMCall,
) -> bool:
    line_a = sigs[a_id]["line"]
    line_b = sigs[b_id]["line"]
    prompt = build_merge_tiebreak_prompt(line_a, line_b)
    data = _call_llm_json(llm_call, prompt)
    if isinstance(data, dict):
        return bool(data.get("merge"))
    return False


def _call_llm_json(llm_call: LLMCall, prompt: Prompt) -> Any:
    response = llm_call(prompt)
    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        return None
    return _safe_json_loads(response)


def _safe_json_loads(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _line_to_json(line: LineCard) -> Dict[str, Any]:
    return {
        "id": line.id,
        "short_title": line.short_title,
        "description": line.description,
        "region_note": line.region_note,
        "exemplar_article_ids": line.exemplar_article_ids,
        "exemplar_titles": line.exemplar_titles,
    }


def _article_to_json(article: ArticleCard) -> Dict[str, Any]:
    return {
        "id": article.id,
        "title": article.title,
        "summary": article.summary,
        "region": article.region,
        "importance": article.importance,
    }


def _find(parents: Dict[Any, Any], x: Any) -> Any:
    if parents[x] != x:
        parents[x] = _find(parents, parents[x])
    return parents[x]


def _union(parents: Dict[Any, Any], a: Any, b: Any) -> None:
    ra = _find(parents, a)
    rb = _find(parents, b)
    if ra != rb:
        parents[rb] = ra


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over",
    "under", "between", "about", "their", "there", "which", "while", "where",
    "were", "been", "being", "also", "such", "will", "shall", "could", "would",
    "should", "may", "might", "than", "then", "them", "they", "your", "yours",
    "our", "ours", "their", "theirs", "his", "her", "hers", "its", "a", "an",
    "to", "of", "in", "on", "at", "by", "as", "is", "are", "was", "be", "it",
}


def load_articles_from_path(path: str, encoding: str = "utf-8") -> List[ArticleCard]:
    text = _read_text(path, encoding=encoding)
    data = None
    if path.lower().endswith(".json") or text.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(text)
        except Exception:
            data = None
    if data is not None:
        return _articles_from_json(data)
    return _articles_from_text(text)


def assign_articles_to_lines(
    articles: Sequence[ArticleCard],
    lines: Sequence[LineCard],
    *,
    embed_fn: Optional[EmbedFn] = None,
    keyword_fn: Optional[KeywordFn] = None,
    entity_fn: Optional[EntityFn] = None,
    assign_config: AssignConfig = AssignConfig(),
    force_assign: bool = False,
) -> Tuple[List[Assignment], List[ArticleCard]]:
    if not lines:
        return [], list(articles)
    article_lookup = {a.id: a for a in articles}
    embedder = _ensure_embedder(embed_fn)
    keyword_fn = keyword_fn or _extract_keywords
    entity_fn = entity_fn or _extract_entities
    line_sigs = _build_line_signatures(
        lines,
        article_lookup,
        embedder,
        keyword_fn,
        entity_fn,
    )
    assignments: List[Assignment] = []
    unassigned: List[ArticleCard] = []
    for art in articles:
        if not line_sigs:
            unassigned.append(art)
            continue
        art_text = f"{art.title}. {art.summary}"
        art_emb = embedder(art_text)
        art_kw = keyword_fn(art_text, 12)
        art_ent = entity_fn(art_text, 8)
        scored: List[Tuple[str, float, float]] = []
        for line_id, sig in line_sigs.items():
            emb_cos = _cosine(art_emb, sig["emb"])
            kw_j = _jaccard(art_kw, sig["kw"])
            ent_j = _jaccard(art_ent, sig["ent"])
            reg = _region_score(sig["region"], art.region, assign_config)
            score = (
                assign_config.weight_emb * emb_cos
                + assign_config.weight_kw * kw_j
                + assign_config.weight_ent * ent_j
                + assign_config.weight_region * reg
            )
            scored.append((line_id, score, emb_cos))
        if not scored:
            unassigned.append(art)
            continue
        scored.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score, best_emb = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        passes = (
            best_emb >= assign_config.emb_threshold
            and best_score >= assign_config.score_threshold
            and margin >= assign_config.margin_threshold
        )
        if not passes and best_emb >= (assign_config.emb_threshold + 0.08):
            passes = True
        if not passes and _jaccard(art_kw, line_sigs[best_id]["kw"]) >= 0.50:
            passes = True
        if passes or force_assign:
            assignments.append(
                Assignment(article_id=art.id, line_id=best_id, confidence=best_score)
            )
        else:
            unassigned.append(art)
    return assignments, unassigned


def run_chunked_pipeline(
    articles: Sequence[ArticleCard],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overhead_tokens: int = DEFAULT_OVERHEAD_TOKENS,
    cluster_threshold: float = 0.62,
    min_support: int = 2,
    final_assign: bool = True,
    force_assign: bool = False,
) -> Dict[str, Any]:
    chunks = chunk_articles_by_tokens(
        articles,
        max_tokens=max_tokens,
        overhead_tokens=overhead_tokens,
    )
    lines: List[LineCard] = []
    assignments_map: Dict[int, Assignment] = {}

    for chunk in chunks:
        res = discover_lines(
            chunk,
            lines,
            cluster_threshold=cluster_threshold,
            min_support=min_support,
        )
        if res.new_lines:
            if lines:
                merge = merge_lines(lines, res.new_lines, min_support=min_support)
                lines = merge.merged_lines
                id_map = merge.id_map
                for art_id, assignment in list(assignments_map.items()):
                    mapped = id_map.get(assignment.line_id, assignment.line_id)
                    if mapped != assignment.line_id:
                        assignments_map[art_id] = Assignment(
                            article_id=assignment.article_id,
                            line_id=mapped,
                            confidence=assignment.confidence,
                            rationale=assignment.rationale,
                        )
            else:
                lines = list(res.new_lines)
                id_map = {line.id: line.id for line in lines}
        else:
            id_map = {line.id: line.id for line in lines}

        for assignment in res.assignments:
            mapped = id_map.get(assignment.line_id, assignment.line_id)
            assignments_map[assignment.article_id] = Assignment(
                article_id=assignment.article_id,
                line_id=mapped,
                confidence=assignment.confidence,
                rationale=assignment.rationale,
            )

    if final_assign:
        final_assignments, final_unassigned = assign_articles_to_lines(
            articles,
            lines,
            force_assign=force_assign,
        )
        assignments = final_assignments
        unassigned_ids = [a.id for a in final_unassigned]
    else:
        assignments = list(assignments_map.values())
        assigned_ids = {a.article_id for a in assignments}
        unassigned_ids = [a.id for a in articles if a.id not in assigned_ids]

    return {
        "sense_lines": [_line_to_json(l) for l in lines],
        "assignments": [
            {
                "article_id": a.article_id,
                "line_id": a.line_id,
                "confidence": a.confidence,
            }
            for a in assignments
        ],
        "unassigned": unassigned_ids,
        "stats": {
            "articles": len(articles),
            "lines": len(lines),
            "chunks": len(chunks),
        },
    }


def _read_text(path: str, encoding: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding=encoding, errors="replace")


def _articles_from_json(data: Any) -> List[ArticleCard]:
    items: List[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("articles"), list):
            items = data["articles"]
        elif isinstance(data.get("items"), list):
            items = data["items"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    articles: List[ArticleCard] = []
    for idx, raw in enumerate(items):
        if isinstance(raw, str):
            summary = raw.strip()
            title = _derive_title(summary)
            articles.append(ArticleCard(id=idx, title=title, summary=summary))
            continue
        if not isinstance(raw, dict):
            continue
        raw_id = raw.get("id", idx)
        title = str(raw.get("title") or "").strip()
        summary = str(
            raw.get("summary")
            or raw.get("text")
            or raw.get("content")
            or raw.get("body")
            or ""
        ).strip()
        if not summary and title:
            summary = title
        if not title:
            title = _derive_title(summary)
        region = str(raw.get("region") or raw.get("search_country") or raw.get("country") or "").strip()
        importance = str(raw.get("importance") or "").strip()
        articles.append(
            ArticleCard(
                id=_coerce_int(raw_id, idx),
                title=title,
                summary=summary,
                region=region,
                importance=importance,
            )
        )
    return articles


def _articles_from_text(text: str) -> List[ArticleCard]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(raw_paragraphs) <= 1:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 1:
            raw_paragraphs = lines
    articles: List[ArticleCard] = []
    for idx, para in enumerate(raw_paragraphs):
        lines = [l.strip() for l in para.splitlines() if l.strip()]
        if not lines:
            continue
        if len(lines) >= 2 and len(lines[0]) <= 120:
            title = lines[0]
            summary = " ".join(lines[1:]).strip()
            if not summary:
                summary = title
        else:
            summary = " ".join(lines).strip()
            title = _derive_title(summary)
        articles.append(ArticleCard(id=idx, title=title, summary=summary))
    return articles


def _derive_title(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Untitled"
    first_line = text.splitlines()[0].strip()
    if len(first_line) <= 120:
        return first_line
    words = first_line.split()
    return " ".join(words[:10]).strip() or first_line[:120].strip()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _write_output(path: str, payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if path:
        Path(path).write_text(text, encoding="utf-8")
        return
    print(text)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked sense line discovery from text or JSON input."
    )
    parser.add_argument(
        "-i",
        "--input",
        default = "./docs/ideator/report.json",
        help="Path to a text file, JSON report, or '-' for stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./docs/ideator/lines.json",
        help="Write JSON output to this path (default: stdout).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens per chunk.",
    )
    parser.add_argument(
        "--overhead-tokens",
        type=int,
        default=DEFAULT_OVERHEAD_TOKENS,
        help="Reserved tokens for prompt overhead.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Input file encoding.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum articles per new line when clustering.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.62,
        help="Embedding similarity threshold for clustering.",
    )
    parser.add_argument(
        "--no-final-assign",
        action="store_true",
        help="Skip the final pass that assigns all articles to final lines.",
    )
    parser.add_argument(
        "--force-assign",
        action="store_true",
        help="Assign every article to a line in the final pass even if thresholds fail.",
    )
    args = parser.parse_args()

    articles = load_articles_from_path(args.input, encoding=args.encoding)
    payload = run_chunked_pipeline(
        articles,
        max_tokens=args.max_tokens,
        overhead_tokens=args.overhead_tokens,
        cluster_threshold=args.cluster_threshold,
        min_support=args.min_support,
        final_assign=not args.no_final_assign,
        force_assign=args.force_assign,
    )
    _write_output(args.output, payload)


if __name__ == "__main__":
    _main()
