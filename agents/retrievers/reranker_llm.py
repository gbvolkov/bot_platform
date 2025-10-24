from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence
from langchain_core.callbacks.manager import Callbacks
from rank_llm.data import Candidate, Query, Request
from copy import deepcopy

from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.schema import Document


class RankLLMRerank_GV(RankLLMRerank):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        request = Request(
            query=Query(text=query, qid=1),
            candidates=[
                Candidate(doc={"text": doc.page_content}, docid=index, score=1)
                for index, doc in enumerate(documents)
            ],
        )

        rerank_results = self.client.rerank(
            request,
            rank_end=len(documents),
            window_size=min(20, len(documents)),
            step=10,
        )
        final_results = []
        if isinstance(rerank_results, list) and hasattr(rerank_results[0], "candidates"):
            rerank_results = rerank_results[0]
        if hasattr(rerank_results, "candidates"):
            # Old API format
            for res in rerank_results.candidates:
                doc = documents[int(res.docid)]
                doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
                final_results.append(doc_copy)
        else:
            for res in rerank_results:
                doc = documents[int(res.docid)]
                doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
                final_results.append(doc_copy)

        return final_results[: self.top_n]