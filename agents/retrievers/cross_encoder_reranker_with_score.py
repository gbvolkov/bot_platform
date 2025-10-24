
from copy import deepcopy
from math import ceil
from typing import List, Optional

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document

class CrossEncoderRerankerWithScores(CrossEncoderReranker):
    """Same as CrossEncoderReranker but stores the rerank score in doc.metadata['rerank_score']."""
    min_ratio: float = 0.0  # ratio of max score used as a threshold

    def __init__(self, *args, min_ratio: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_ratio = float(min_ratio)

    def _score_and_tag(self, documents: List[Document], query: str) -> List[Document]:
        """Score a list of docs and return deep-copied docs with rerank scores in metadata."""
        if not documents:
            return []

        scores = self.model.score([(query, d.page_content) for d in documents])
        out: List[Document] = []
        for d, s in zip(documents, scores):
            d2 = deepcopy(d)
            md = dict(d2.metadata or {})
            md["rerank_score"] = float(s)
            d2.metadata = md
            out.append(d2)
        return out

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[object] = None,
    ) -> List[Document]:
        if not documents:
            return []

        # Score all at once (original behavior)
        scored = self._score_and_tag(documents, query)
        max_s = max(d.metadata["rerank_score"] for d in scored)
        threshold = self.min_ratio * max_s
        passed_docs = [d for d in scored if d.metadata["rerank_score"] >= threshold]
        passed_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
        return passed_docs[: self.top_n]
    
class TournamentCrossEncoderReranker(CrossEncoderRerankerWithScores):
    """
    Reranks using tournament-style batching:

    - Split docs into chunks of size N (tournament_size).
    - For each chunk, score within the chunk and keep top ceil(chunk_size/2).
    - Merge the survivors and repeat while len(docs) >= N.
    - Final pass: score the remaining docs together, apply min_ratio, sort desc, return top_n.
    """
    tournament_size: int = 20

    def __init__(
        self,
        *args,
        tournament_size: int = 20,
        min_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, min_ratio=min_ratio, **kwargs)
        if tournament_size <= 0:
            raise ValueError("tournament_size must be > 0")
        self.tournament_size = int(tournament_size)

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[object] = None,
    ) -> List[Document]:
        if not documents:
            return []

        #N = self.tournament_size
        current = list(documents)

        # Run tournaments until pool is smaller than one full tournament
        while len(current) > self.tournament_size:
            next_round: List[Document] = []

            # Process in fixed-size chunks (last chunk may be smaller)
            for i in range(0, len(current), self.tournament_size):
                chunk = current[i : i + self.tournament_size]
                # Score within the chunk and keep top half (ceil)
                scored_chunk = self._score_and_tag(chunk, query)
                scored_chunk.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
                keep_k = max(1, ceil(len(chunk) / 2))
                next_round.extend(scored_chunk[:keep_k])

            current = next_round

        # Final normalization pass across the remaining docs
        final_scored = self._score_and_tag(current, query)
        if not final_scored:
            return []

        max_s = max(d.metadata["rerank_score"] for d in final_scored)
        threshold = self.min_ratio * max_s
        passed = [d for d in final_scored if d.metadata["rerank_score"] >= threshold]
        passed.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)

        return passed[: self.top_n]