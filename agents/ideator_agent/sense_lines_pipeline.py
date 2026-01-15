#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sense-lines generator:
Local: Embeddings (Sentence-Transformers) + KMeans + c-TF-IDF tags + extractive MMR description + region heuristic
Default: LLM mode (LangChain) generates per-cluster short_title, description, region_note
          using only representative items (title + summary + importance_reasoning + url), while
          preserving stable ids and line ordering from the local clustering step.

Key behavior:
- Clustering and tags are produced from article content:
  (title + summary + importance_reasoning)
- sense_lines.articles includes ALL cluster articles (stable order: centroid distance asc, then original idx)
- Descriptions can be up to N sentences (default 4): --max_description_sentences
- Stopwords handling: CountVectorizer always receives stop_words as LIST or None (validation-safe)

Install (local-only):
  pip install sentence-transformers scikit-learn numpy

Optional better RU stopwords:  pip install stopwordsiso

LLM mode (LangChain + OpenAI example):
  pip install langchain langchain-openai
  export OPENAI_API_KEY="..."

Run:
  python sense_lines.py --input report.json --output out.json
  python sense_lines.py --input report.json --output out.json --k 4 --max_lines 4
  python sense_lines.py --input report.json --output out.json --k 4 --max_lines 4 --llm_model gpt-4o-mini
  python sense_lines.py --input report.json --output out.json --no-llm --max_description_sentences 4
"""


from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer

from .models import IdeatorReport
from .prompts import get_locale


# -----------------------------
# Defaults / heuristics
# -----------------------------

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

FALLBACK_STOPWORDS_RU = {
    "и", "в", "во", "на", "но", "а", "что", "это", "как", "к", "ко", "из", "у", "по",
    "за", "от", "для", "при", "про", "до", "над", "под", "же", "ли", "бы", "не", "ни",
    "мы", "вы", "они", "он", "она", "оно", "его", "ее", "их", "наш", "ваш", "этот",
    "эта", "эти", "того", "тому", "там", "тут", "здесь", "тогда", "сейчас", "уже",
    "еще", "очень", "может", "могут", "будет", "быть", "были", "есть", "нет",
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as", "by",
}

# Lightweight region dictionary (extend as needed)
REGION_TERMS = [
    "Россия", "РФ", "России", "Украина", "Украины", "Казахстан", "Казахстана",
    "Беларусь", "Белоруссия", "Армения", "Грузия", "Азербайджан", "Узбекистан",
    "Кыргызстан", "Киргизия", "Таджикистан", "Туркменистан", "Молдова",
    "Финляндия", "Финляндии", "Германия", "Германии", "Франция", "Франции",
    "Италия", "Испания", "Польша", "Чехия", "США", "Америка", "Китай", "Индия",
    "ЕС", "Евросоюз", "Европа",
    "Москва", "Москве", "Санкт-Петербург", "Петербург", "СПб",
    "Казань", "Новосибирск", "Екатеринбург", "Краснодар",
    "Мариуполь", "Донецк", "Луганск", "Крым",
]

REGION_REGEX = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(t) for t in sorted(REGION_TERMS, key=len, reverse=True)) + r")(?!\w)",
    flags=re.IGNORECASE,
)

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?…])\s+")


# -----------------------------
# Data models
# -----------------------------

@dataclass
class Article:
    idx: int
    title: str
    summary: str
    importance_reasoning: str
    url: str
    date: str = ""
    importance: str = ""
    source_id: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None

    def full_text(self) -> str:
        parts = []
        for x in (self.title, self.summary, self.importance_reasoning):
            if x := (x or "").strip():
                parts.append(x)
        return "\n".join(parts).strip()

    def label(self) -> str:
        return (self.title or "").strip() or self.url


@dataclass
class SenseLine:
    id: str
    short_title: str
    description: str
    articles: List[Dict[str, str]]  # [{title,url}, ...]  <-- ALL cluster articles
    region_note: str


class ArticleRef(TypedDict):
    id: int
    title: str
    summary: str


class SenseLineItem(TypedDict):
    id: str
    short_title: str
    description: str
    articles: List[ArticleRef]
    region_note: str


# -----------------------------
# IO
# -----------------------------

def load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_source_id(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_articles(report: Dict[str, Any]) -> List[Article]:
    items = report.get("articles", [])
    if not isinstance(items, list):
        raise ValueError("report['articles'] must be a list")

    out: List[Article] = []
    for i, a in enumerate(items):
        if not isinstance(a, dict):
            continue
        if url := str(a.get("url", "") or "").strip():
            source_id = _coerce_source_id(a.get("id"))
            out.append(
                Article(
                    idx=i,
                    title=str(a.get("title", "") or "").strip(),
                    summary=str(a.get("summary", "") or "").strip(),
                    importance_reasoning=str(a.get("importance_reasoning", "") or "").strip(),
                    url=url,
                    date=str(a.get("date", "") or "").strip(),
                    importance=str(a.get("importance", "") or "").strip(),
                    source_id=source_id,
                    raw=a,
                )
            )
    return out


def save_output(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Stopwords (validation-safe for scikit-learn)
# -----------------------------

def get_stopwords_ru_list() -> Optional[List[str]]:
    """
    Return stopwords as a LIST (or None), so CountVectorizer(stop_words=...) always receives list/None.
    Deterministic ordering via sorting.
    """
    sw: Optional[set] = None
    try:
        import stopwordsiso  # type: ignore
        s = set(stopwordsiso.stopwords("ru"))
        sw = s or None
    except Exception:
        sw = None

    if sw is None:
        sw = set(FALLBACK_STOPWORDS_RU)

    out = sorted(sw)
    return out or None


# -----------------------------
# Embeddings + clustering
# -----------------------------

def build_embedder(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def embed_texts(model: SentenceTransformer, texts: Sequence[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32, copy=False)


def choose_k_by_silhouette(
    emb: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int,
) -> int:  # sourcery skip: use-assigned-variable
    n = emb.shape[0]
    min_k = max(2, min_k)
    max_k = max(min_k, max_k)
    max_k = min(max_k, n - 1) if n > 2 else 2

    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(emb)
        counts = Counter(labels)
        if min(counts.values()) < 2:
            continue
        score = silhouette_score(emb, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_kmeans(emb: np.ndarray, k: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(emb)
    centers = normalize(km.cluster_centers_)
    return labels, centers


def cosine_distance_to_centroid(A: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    # A: (n, d) normalized; centroid: (d,) normalized
    return 1.0 - (A @ centroid.reshape(-1, 1)).reshape(-1)


# -----------------------------
# c-TF-IDF tags
# -----------------------------

def compute_ctfidf_tags(
    cluster_texts: List[str],
    stopwords_list: Optional[List[str]],
    top_n: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
) -> List[List[str]]:
    token_pattern = r"(?u)\b[0-9A-Za-zА-Яа-яЁё\-]{2,}\b"
    vectorizer = CountVectorizer(
        stop_words=stopwords_list,  # LIST or None (validation-safe)
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        token_pattern=token_pattern,
    )
    X = vectorizer.fit_transform(cluster_texts)

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
    X_tfidf = tfidf.fit_transform(X)

    vocab = np.array(vectorizer.get_feature_names_out())
    tags_per_cluster: List[List[str]] = []
    for i in range(X_tfidf.shape[0]):
        row = X_tfidf.getrow(i)
        if row.nnz == 0:
            tags_per_cluster.append([])
            continue
        idx = row.indices
        data = row.data
        top_idx = idx[np.argsort(-data)[:top_n]]
        tags = vocab[top_idx].tolist()

        clean = []
        for t in tags:
            if re.fullmatch(r"\d+", t):
                continue
            if len(t) > 56:
                continue
            clean.append(t)
        tags_per_cluster.append(clean[:top_n])
    return tags_per_cluster


def make_short_title_from_tags(tags: List[str], max_len: int = 56) -> str:
    if not tags:
        return "Miscellaneous"
    title = " · ".join(tags[:3]).strip()
    return title if len(title) <= max_len else tags[0][:max_len].rstrip()


# -----------------------------
# Extractive description (MMR)
# -----------------------------

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = SENTENCE_SPLIT_REGEX.split(text)
    out = []
    for p in parts:
        s = re.sub(r"\s+", " ", p.strip())
        if not s:
            continue
        if len(s) < 35:
            continue
        if not re.search(r"[.!?…]$", s):
            s += "."
        out.append(s)
    return out


def mmr_select(
    sent_emb: np.ndarray,
    centroid: np.ndarray,
    top_n: int,
    lambda_param: float = 0.75,
) -> List[int]:
    if sent_emb.shape[0] == 0:
        return []
    sim_to_centroid = (sent_emb @ centroid.reshape(-1, 1)).reshape(-1)
    selected: List[int] = []

    for _ in range(min(top_n, sent_emb.shape[0])):
        if not selected:
            selected.append(int(np.argmax(sim_to_centroid)))
            continue
        sel_emb = sent_emb[selected]
        sim_to_selected = sent_emb @ sel_emb.T
        max_sim_sel = sim_to_selected.max(axis=1)

        mmr = lambda_param * sim_to_centroid - (1.0 - lambda_param) * max_sim_sel
        for sidx in selected:
            mmr[sidx] = -1e9
        selected.append(int(np.argmax(mmr)))
    return selected


def build_cluster_description_extractive(
    embedder: SentenceTransformer,
    cluster_articles: List[Article],
    centroid: np.ndarray,
    batch_size: int,
    max_sentences: int,
) -> str:
    candidates: List[str] = []
    # Prefer "importance_reasoning" sentences first (more ideation-oriented)
    for a in cluster_articles:
        candidates.extend(split_sentences(a.importance_reasoning))

    # Add summary sentences if not enough variety
    if len(candidates) < 10:
        for a in cluster_articles:
            candidates.extend(split_sentences(a.summary))

    # Last resort: titles
    if len(candidates) < 2:
        for a in cluster_articles:
            t = (a.title or "").strip()
            if t and len(t) >= 35:
                candidates.append(t if t.endswith(".") else f"{t}.")

    # Deduplicate
    seen = set()
    sentences = []
    for s in candidates:
        key = re.sub(r"\s+", " ", s.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        sentences.append(s)

    if not sentences:
        return "Articles in this cluster share a common theme; more source content is required."

    sent_emb = embed_texts(embedder, sentences, batch_size=batch_size)
    pick = mmr_select(sent_emb, centroid=centroid, top_n=max_sentences, lambda_param=0.75)
    chosen = [sentences[i] for i in pick[:max_sentences]]
    return " ".join(chosen).strip()


# -----------------------------
# Region heuristic
# -----------------------------

def region_note_for_cluster(
    cluster_articles: List[Article],
    dominance_threshold: float = 0.60,
    min_mentions: int = 2,
) -> str:
    counts = Counter()
    for a in cluster_articles:
        text = f"{a.title} {a.summary} {a.importance_reasoning}"
        hits = [m.group(1) for m in REGION_REGEX.finditer(text)]
        for h in set(hits):
            counts[h] += 1

    if not counts:
        return ""

    top_region, top_count = counts.most_common(1)[0]
    n = len(cluster_articles)
    if top_count >= min_mentions and (top_count / n) >= dominance_threshold:
        return f"Primarily relevant to {top_region} context; applicability may vary by region."
    return ""


# -----------------------------
# Stable cluster ordering + ALL articles list
# -----------------------------

def ordered_indices_by_centroid(
    doc_emb: np.ndarray,
    indices: List[int],
    centroid: np.ndarray,
) -> List[int]:
    emb_cluster = doc_emb[indices]
    d = cosine_distance_to_centroid(emb_cluster, centroid)
    # stable tie-break by original index
    order = sorted(range(len(indices)), key=lambda i: (float(d[i]), indices[i]))
    return [indices[i] for i in order]


def build_cluster_manifest(
    articles: List[Article],
    doc_emb: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    max_lines: int,
    llm_rep_docs: int,
    top_tags: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
) -> List[Dict[str, Any]]:
    """
    Returns cluster manifests with stable ordering independent of LLM:
      - cluster_key (size desc, nearest_doc_idx asc)
      - ordered_indices_all (ALL cluster docs, ordered by centroid distance)
      - rep_indices_llm (top N for LLM prompt)
      - tags, local_short_title
    """
    stopwords_list = get_stopwords_ru_list()

    cluster_to_idxs: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        cluster_to_idxs[int(lab)].append(i)

    # Keep largest clusters if too many
    clusters = sorted(cluster_to_idxs.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    clusters = clusters[:max_lines]

    # c-TF-IDF class-docs
    cluster_texts: List[str] = []
    cluster_ids: List[int] = []
    for cid, idxs in clusters:
        cluster_ids.append(cid)
        joined = "\n".join(articles[i].full_text() for i in idxs if articles[i].full_text())
        cluster_texts.append(joined[:2_000_000])

    tags_per_cluster = compute_ctfidf_tags(
        cluster_texts=cluster_texts,
        stopwords_list=stopwords_list,
        top_n=top_tags,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
    )

    manifests: List[Dict[str, Any]] = []
    for pos, cid in enumerate(cluster_ids):
        idxs = cluster_to_idxs[cid]
        centroid = centers[cid]

        ordered_all = ordered_indices_by_centroid(doc_emb, idxs, centroid)
        nearest_doc_idx = ordered_all[0] if ordered_all else min(idxs)
        cluster_key = (-len(idxs), nearest_doc_idx)  # stable ordering key

        tags = tags_per_cluster[pos]
        local_short_title = make_short_title_from_tags(tags)

        rep_llm = ordered_all[: max(1, llm_rep_docs)]

        manifests.append(
            {
                "cluster_key": cluster_key,
                "cluster_id": cid,
                "size": len(idxs),
                "ordered_indices_all": ordered_all,
                "rep_indices_llm": rep_llm,
                "tags": tags,
                "local_short_title": local_short_title,
            }
        )

    # Stable ordering by key only
    manifests.sort(key=lambda m: (m["cluster_key"][0], m["cluster_key"][1]))
    return manifests


# -----------------------------
# LangChain LLM per cluster (optional)
# -----------------------------

def safe_json_extract(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    with contextlib.suppress(Exception):
        return json.loads(s)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def llm_refine_cluster_fields_langchain(
    *,
    llm,  # LangChain ChatModel
    cluster_tags: List[str],
    rep_items: List[Dict[str, str]],
    region_hint: str,
    local_fallback_title: str,
    local_fallback_desc: str,
    max_description_sentences: int,
    locale: str = "en",
) -> Tuple[str, str, str]:
    locale_prompts = get_locale(locale)["prompts"]
    system = locale_prompts["sense_line_llm_system"].format(
        max_description_sentences=max_description_sentences
    )

    items_text = []
    items_text.extend(
        f"[{i}] title: {it.get('title', '')}\nsummary: {it.get('summary', '')}\nimportance_reasoning: {it.get('importance_reasoning', '')}\nurl: {it.get('url', '')}\n"
        for i, it in enumerate(rep_items, start=1)
    )
    user = locale_prompts["sense_line_llm_user"].format(
        cluster_tags=", ".join(cluster_tags) if cluster_tags else "(none)",
        region_hint=region_hint or "(none)",
        items_text="\n".join(items_text),
    )

    from langchain_core.messages import SystemMessage, HumanMessage

    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(resp, "content", "") or ""
    data = safe_json_extract(content)

    short_title = str(data.get("short_title", "") or "").strip() or local_fallback_title
    description = str(data.get("description", "") or "").strip() or local_fallback_desc
    region_note = str(data.get("region_note", "") or "").strip()

    # Enforce up to max_description_sentences sentences
    parts = SENTENCE_SPLIT_REGEX.split(description.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > max_description_sentences:
        description = " ".join(parts[:max_description_sentences])
        if not re.search(r"[.!????]$", description):
            description += "."

    if region_note is None:
        region_note = ""
    return short_title, description, region_note


def build_llm_model_langchain(
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise SystemExit(
            "Missing langchain-openai. Install: pip install langchain langchain-openai"
        ) from e
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        #api_key=OPENAI_API_KEY,
    )


# -----------------------------
# Sense lines assembly
# -----------------------------

def build_sense_lines(
    *,
    embedder: SentenceTransformer,
    articles: List[Article],
    centers: np.ndarray,
    cluster_manifests: List[Dict[str, Any]],
    batch_size: int,
    use_llm: bool,
    llm_model_name: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_timeout: int,
    max_description_sentences: int,
    locale: str = "en",
) -> List[SenseLine]:
    llm = None
    if use_llm:
        llm = build_llm_model_langchain(
            model_name=llm_model_name,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout,
        )

    sense_lines: List[SenseLine] = []
    for i, m in enumerate(cluster_manifests, start=1):
        cid = int(m["cluster_id"])
        ordered_all: List[int] = m["ordered_indices_all"]
        rep_llm: List[int] = m["rep_indices_llm"]
        tags: List[str] = m["tags"]
        local_title: str = m["local_short_title"]

        cluster_articles = [articles[j] for j in ordered_all]  # stable order
        centroid = centers[cid]

        local_region = region_note_for_cluster(cluster_articles)
        local_desc = build_cluster_description_extractive(
            embedder=embedder,
            cluster_articles=cluster_articles,
            centroid=centroid,
            batch_size=batch_size,
            max_sentences=max_description_sentences,
        )

        short_title = local_title
        description = local_desc
        region_note = local_region

        # LLM sees only representative subset (for cost/latency) but output applies to whole cluster
        rep_items = []
        for j in rep_llm[: min(6, len(rep_llm))]:
            a = articles[j]
            rep_items.append(
                {
                    "title": a.title,
                    "summary": a.summary,
                    "importance_reasoning": a.importance_reasoning,
                    "url": a.url,
                }
            )

        if use_llm and llm is not None:
            try:
                short_title, description, region_note = llm_refine_cluster_fields_langchain(
                    llm=llm,
                    cluster_tags=tags[:8],
                    rep_items=rep_items,
                    region_hint=local_region,
                    local_fallback_title=local_title,
                    local_fallback_desc=local_desc,
                    max_description_sentences=max_description_sentences,
                    locale=locale,
                )
            except Exception:
                short_title, description, region_note = local_title, local_desc, local_region

        # Include ALL cluster articles
        links_all = [{"title": articles[j].label(), "url": articles[j].url} for j in ordered_all]

        sense_lines.append(
            SenseLine(
                id=f"{i:02d}",
                short_title=short_title,
                description=description,
                articles=links_all,
                region_note=region_note or "",
            )
        )

    return sense_lines


def build_output_object(sense_lines: List[SenseLine], total_articles: int, k: int, llm_used: bool) -> Dict[str, Any]:
    assistant_message = (
        f"Generated {len(sense_lines)} sense lines from {total_articles} articles (k={k}). "
        f"{'LLM refinement enabled for title/description/region_note.' if llm_used else 'Local-only (no LLM) mode.'} "
        "Each sense line includes all articles assigned to its cluster."
    )
    return {
        "assistant_message": assistant_message,
        "sense_lines": [
            {
                "id": sl.id,
                "short_title": sl.short_title,
                "description": sl.description,
                "articles": sl.articles,
                "region_note": sl.region_note,
            }
            for sl in sense_lines
        ],
        "decision": {
            "selected_line_index": None,
            "custom_line_text": None,
            "consent_generate": False,
            "regen_lines": False,
            "finish": False,
        },
    }


# -----------------------------
# Pipeline helper (Ideator agent)
# -----------------------------

def _report_to_articles(report: IdeatorReport) -> List[Article]:
    payload = {
        "articles": [
            {
                "id": art.id,
                "title": art.title,
                "summary": art.summary,
                "importance_reasoning": art.importance_reasoning,
                "url": art.url,
                "date": art.date,
                "importance": art.importance,
            }
            for art in report.articles
        ]
    }
    return parse_articles(payload)


def _article_ref(article: Article) -> ArticleRef:
    source_id = article.source_id if article.source_id is not None else article.idx
    title = (article.title or "").strip() or article.url
    summary = (article.summary or "").strip() or (article.importance_reasoning or "").strip()
    return {
        "id": int(source_id),
        "title": title,
        "summary": summary,
    }


def _build_sense_line_items(
    *,
    sense_lines: List[SenseLine],
    articles: List[Article],
    cluster_manifests: List[Dict[str, Any]],
) -> List[SenseLineItem]:
    items: List[SenseLineItem] = []
    for idx, (line, manifest) in enumerate(zip(sense_lines, cluster_manifests), start=1):
        ordered_all: List[int] = manifest["ordered_indices_all"]
        article_refs = [_article_ref(articles[j]) for j in ordered_all]
        items.append(
            {
                "id": f"L{idx}",
                "short_title": line.short_title,
                "description": line.description,
                "articles": article_refs,
                "region_note": line.region_note or "",
            }
        )
    return items


def build_sense_lines_from_report(
    report: IdeatorReport,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    batch_size: int = 64,
    k: int = 0,
    min_k: int = 3,
    max_k: int = 6,
    max_lines: int = 6,
    random_state: int = 42,
    top_tags: int = 8,
    ngram_max: int = 2,
    min_df: int = 2,
    max_df: float = 0.9,
    max_description_sentences: int = 4,
    llm_rep_docs: int = 6,
    llm: bool = True,
    llm_model: str = "gpt-4.1-mini",
    llm_temperature: float = 0.0,
    llm_max_tokens: int = 1024,
    llm_timeout: int = 60,
    locale: str = "en",
) -> List[SenseLineItem]:
    articles = _report_to_articles(report)
    if len(articles) < 3:
        raise SystemExit("Need at least 3 articles with URLs to build sense lines.")

    embedder = build_embedder(embedding_model, device)

    docs = [a.full_text() for a in articles]
    docs = [d if d.strip() else (articles[i].title or articles[i].url) for i, d in enumerate(docs)]
    doc_emb = embed_texts(embedder, docs, batch_size=batch_size)

    if k and k >= 2:
        chosen_k = k
    else:
        max_k = min(max_k, max_lines)
        min_k = min(min_k, max_k)
        chosen_k = choose_k_by_silhouette(doc_emb, min_k=min_k, max_k=max_k, random_state=random_state)

    chosen_k = max(2, min(chosen_k, len(articles) - 1))
    labels, centers = cluster_kmeans(doc_emb, k=chosen_k, random_state=random_state)

    manifests = build_cluster_manifest(
        articles=articles,
        doc_emb=doc_emb,
        labels=labels,
        centers=centers,
        max_lines=max_lines,
        llm_rep_docs=max(1, llm_rep_docs),
        top_tags=top_tags,
        ngram_max=ngram_max,
        min_df=min_df,
        max_df=max_df,
    )

    sense_lines = build_sense_lines(
        embedder=embedder,
        articles=articles,
        centers=centers,
        cluster_manifests=manifests,
        batch_size=batch_size,
        use_llm=llm,
        llm_model_name=llm_model,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_timeout=llm_timeout,
        max_description_sentences=max(1, max_description_sentences),
        locale=locale,
    )
    return _build_sense_line_items(
        sense_lines=sense_lines,
        articles=articles,
        cluster_manifests=manifests,
    )

    #out_obj = build_output_object(
    #    sense_lines,
    #    total_articles=len(articles),
    #    k=chosen_k,
    #    llm_used=bool(llm),
    #)
    #return out_obj["sense_lines"]


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="report.json", help="Path to report JSON")
    p.add_argument("--output", default="out.json", help="Path to output JSON")

    # Local models
    p.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--batch_size", type=int, default=64)

    # Clustering
    p.add_argument("--k", type=int, default=0, help="Fixed k (0 => auto via silhouette)")
    p.add_argument("--min_k", type=int, default=3)
    p.add_argument("--max_k", type=int, default=6)
    p.add_argument("--max_lines", type=int, default=6)
    p.add_argument("--random_state", type=int, default=42)

    # Tags
    p.add_argument("--top_tags", type=int, default=8)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.9)

    # Description length
    p.add_argument(
        "--max_description_sentences",
        type=int,
        default=4,
        help="Max sentences in description (local and LLM-trim).",
    )

    # LLM prompt representatives (NOT output size)
    p.add_argument("--llm_rep_docs", type=int, default=6, help="Representative docs per cluster for LLM prompt")

    # LLM (LangChain) optional
    p.add_argument(
        "--llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM refinement per cluster (LangChain). Use --no-llm to disable.",
    )
    p.add_argument("--llm_model", default="gpt-4.1-mini", help="Chat model name (LangChain provider)")
    p.add_argument("--llm_temperature", type=float, default=0.0)
    p.add_argument("--llm_max_tokens", type=int, default=1024)
    p.add_argument("--llm_timeout", type=int, default=60)

    args = p.parse_args()

    report = load_report(args.input)
    articles = parse_articles(report)
    if len(articles) < 3:
        raise SystemExit("Need at least 3 articles with URLs to build sense lines.")

    embedder = build_embedder(args.embedding_model, args.device)

    # IMPORTANT: clustering input uses title + summary + importance_reasoning
    docs = [a.full_text() for a in articles]
    docs = [d if d.strip() else (articles[i].title or articles[i].url) for i, d in enumerate(docs)]
    doc_emb = embed_texts(embedder, docs, batch_size=args.batch_size)

    # Choose k (cap by max_lines)
    if args.k and args.k >= 2:
        k = int(args.k)
    else:
        max_k = min(args.max_k, args.max_lines)
        min_k = min(args.min_k, max_k)
        k = choose_k_by_silhouette(doc_emb, min_k=min_k, max_k=max_k, random_state=args.random_state)

    k = max(2, min(k, len(articles) - 1))
    labels, centers = cluster_kmeans(doc_emb, k=k, random_state=args.random_state)

    # Stable manifests (ordering, tags, ALL article ordering)
    manifests = build_cluster_manifest(
        articles=articles,
        doc_emb=doc_emb,
        labels=labels,
        centers=centers,
        max_lines=args.max_lines,
        llm_rep_docs=max(1, int(args.llm_rep_docs)),
        top_tags=args.top_tags,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    sense_lines = build_sense_lines(
        embedder=embedder,
        articles=articles,
        centers=centers,
        cluster_manifests=manifests,
        batch_size=args.batch_size,
        use_llm=bool(args.llm),
        llm_model_name=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout=args.llm_timeout,
        max_description_sentences=max(1, int(args.max_description_sentences)),
    )

    out_obj = build_output_object(sense_lines, total_articles=len(articles), k=k, llm_used=bool(args.llm))
    save_output(args.output, out_obj)

    print(
        f"OK: wrote {args.output} "
        f"(sense_lines={len(sense_lines)}, k={k}, articles={len(articles)}, llm={bool(args.llm)})"
    )

if __name__ == "__main__":
    main()
