from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import config
from rag_lib.core.domain import Segment
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.vectors.factory import create_vector_store

from .utils.loader import load_single_document

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".png",
    ".jpg",
    ".jpeg",
    ".msg",
    ".eml",
}

_PASSENGER_KEYWORDS = (
    "\u043f\u0430\u0441\u0441\u0430\u0436\u0438\u0440",
    "\u0430\u0432\u0442\u043e\u0431\u0443\u0441",
    "\u043c\u0430\u0440\u0448\u0440\u0443\u0442",
    "vector",
    "citymax",
    "\u043f\u0430\u0437",
)
_SPECIAL_BODY_KEYWORDS = {
    "refrigerator": ("\u0445\u043e\u043b\u043e\u0434", "\u0440\u0435\u0444\u0440\u0438\u0436\u0435\u0440", "cold", "reefer", "fridge"),
    "vacuum": ("\u0432\u0430\u043a\u0443\u0443\u043c",),
    "kmu": ("\u043a\u043c\u0443", "\u043a\u0440\u0430\u043d\u043e\u043c\u0430\u043d\u0438\u043f"),
    "platform": ("\u043f\u043b\u0430\u0442\u0444\u043e\u0440\u043c",),
    "tow": ("\u044d\u0432\u0430\u043a\u0443\u0430\u0442", "tow"),
    "garbage": ("\u043c\u0443\u0441\u043e\u0440",),
    "tank": ("\u0446\u0438\u0441\u0442\u0435\u0440\u043d", "tank"),
    "tipper": ("\u0441\u0430\u043c\u043e\u0441\u0432\u0430\u043b", "tipper"),
    "lift": ("\u043f\u043e\u0434\u044a\u0451\u043c", "lift"),
    "bunker": ("\u0431\u0443\u043d\u043a\u0435\u0440",),
}
_PRODUCT_KEYWORDS = {
    "gazelle_next": ("\u0433\u0430\u0437\u0435\u043b\u044c next", "gazelle next"),
    "gazelle_nn": ("\u0433\u0430\u0437\u0435\u043b\u044c nn", "gazelle nn"),
    "gazelle_city": ("\u0433\u0430\u0437\u0435\u043b\u044c city", "gazelle city"),
    "sobol_nn": ("\u0441\u043e\u0431\u043e\u043b\u044c nn", "sobol nn"),
    "sobol_business": ("\u0441\u043e\u0431\u043e\u043b\u044c business", "sobol business", "\u0441\u043e\u0431\u043e\u043b\u044c \u0431\u0438\u0437\u043d\u0435\u0441"),
    "gazon_next": ("\u0433\u0430\u0437\u043e\u043d next", "gazon next"),
    "valdai": ("\u0432\u0430\u043b\u0434\u0430\u0439", "valdai"),
    "sadko": ("\u0441\u0430\u0434\u043a\u043e", "sadko"),
    "vector_next": ("vector next", "\u0432\u0435\u043a\u0442\u043e\u0440 next"),
    "citymax": ("citymax", "city max", "\u0441\u0438\u0442\u0438\u043c\u0430\u043a\u0441"),
    "paz": ("\u043f\u0430\u0437", "paz"),
    "sat": ("\u0441\u0430\u0442", "sat"),
}
_PRODUCT_FAMILY_ALIASES = {
    "gazelle_next": "gazelle_next",
    "gazelle next": "gazelle_next",
    "\u0433\u0430\u0437\u0435\u043b\u044c next": "gazelle_next",
    "gazelle_nn": "gazelle_nn",
    "gazelle nn": "gazelle_nn",
    "\u0433\u0430\u0437\u0435\u043b\u044c nn": "gazelle_nn",
    "gazelle_city": "gazelle_city",
    "gazelle city": "gazelle_city",
    "\u0433\u0430\u0437\u0435\u043b\u044c city": "gazelle_city",
    "sobol_nn": "sobol_nn",
    "sobol nn": "sobol_nn",
    "\u0441\u043e\u0431\u043e\u043b\u044c nn": "sobol_nn",
    "sobol_business": "sobol_business",
    "sobol business": "sobol_business",
    "\u0441\u043e\u0431\u043e\u043b\u044c business": "sobol_business",
    "\u0441\u043e\u0431\u043e\u043b\u044c \u0431\u0438\u0437\u043d\u0435\u0441": "sobol_business",
    "gazon_next": "gazon_next",
    "gazon next": "gazon_next",
    "\u0433\u0430\u0437\u043e\u043d next": "gazon_next",
    "valdai": "valdai",
    "\u0432\u0430\u043b\u0434\u0430\u0439": "valdai",
    "sadko": "sadko",
    "\u0441\u0430\u0434\u043a\u043e": "sadko",
    "vector_next": "vector_next",
    "vector next": "vector_next",
    "\u0432\u0435\u043a\u0442\u043e\u0440 next": "vector_next",
    "citymax": "citymax",
    "city max": "citymax",
    "\u0441\u0438\u0442\u0438\u043c\u0430\u043a\u0441": "citymax",
    "paz": "paz",
    "\u043f\u0430\u0437": "paz",
    "sat": "sat",
    "\u0441\u0430\u0442": "sat",
}
_COMPETITOR_KEYWORDS = {
    "sollers_atlant": ("\u0430\u0442\u043b\u0430\u043d\u0442", "atlant", "sollers atlant", "\u0441\u043e\u043b\u043b\u0435\u0440\u0441"),
    "sollers_tr": ("tr80", "tr120", "tr180", "sollers tr", "\u0441\u043e\u043b\u043b\u0435\u0440\u0441 tr"),
    "paz": ("\u043f\u0430\u0437", "paz"),
}
_SPECIAL_CONDITION_KEYWORDS = {
    "offroad": ("\u0431\u0435\u0437\u0434\u043e\u0440\u043e\u0436", "offroad", "4x4", "4\u04454"),
    "harsh": ("\u0441\u0443\u0440\u043e\u0432", "harsh", "severe"),
    "municipal": ("\u043c\u0443\u043d\u0438\u0446\u0438\u043f", "municipal"),
    "heavy": ("\u0442\u044f\u0436\u0435\u043b", "heavy"),
}
_INTENT_DOC_KIND_BONUS = {
    "overview": {"general": 4, "configuration": 3, "comparison": 1},
    "compare": {"comparison": 6, "configuration": 2, "general": 1},
    "specs": {"configuration": 6, "general": 2, "comparison": 1},
    "financing": {"approval": 6, "tco": 3, "general": 1},
    "objection": {"comparison": 6, "tco": 3, "service": 2, "general": 1},
    "recommendation": {"configuration": 4, "comparison": 2, "service": 1, "general": 1},
    "materials": {"comparison": 3, "configuration": 3, "service": 2, "approval": 2, "general": 1},
    "next_step": {"approval": 3, "comparison": 2, "configuration": 2, "general": 1},
}
_MAX_SEARCH_TEXT_LENGTH = 24000
_MAX_RUNTIME_CHUNK_LENGTH = 1600
_CHUNK_SIZE = 1200


@dataclass
class GazRuntimeService:
    docs_root: Path
    cache_root: Path
    default_collection_id: str = "gaz"

    def __post_init__(self) -> None:
        self.docs_root = self.docs_root.resolve()
        self.cache_root = self.cache_root.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._manifest_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._material_artifact_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _collection_dir(self, collection_id: str) -> Path:
        target = self.cache_root / collection_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _manifest_path(self, collection_id: str) -> Path:
        return self._collection_dir(collection_id) / "manifest.json"

    def _material_artifacts_path(self, collection_id: str) -> Path:
        return self._collection_dir(collection_id) / "material_artifacts.json"

    def _status_path(self, collection_id: str) -> Path:
        return self._collection_dir(collection_id) / "status.json"

    def _vector_store_path(self, collection_id: str) -> Path:
        return self._collection_dir(collection_id) / "vector_store"

    @contextmanager
    def _vector_store_environment(self, collection_id: str):
        vector_store_path = self._vector_store_path(collection_id)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        previous_vector_path = os.environ.get("VECTOR_PATH")
        os.environ["VECTOR_PATH"] = str(vector_store_path)
        try:
            yield vector_store_path
        finally:
            if previous_vector_path is None:
                os.environ.pop("VECTOR_PATH", None)
            else:
                os.environ["VECTOR_PATH"] = previous_vector_path

    def _candidate_id(self, relative_path: str) -> str:
        digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]
        return f"cand_{digest}"

    def _load_manifest(self, collection_id: str) -> List[Dict[str, Any]]:
        if collection_id in self._manifest_cache:
            return self._manifest_cache[collection_id]
        manifest_path = self._manifest_path(collection_id)
        if not manifest_path.exists():
            return []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._manifest_cache[collection_id] = manifest
        return manifest

    def _load_material_artifacts(self, collection_id: str) -> Dict[str, Dict[str, Any]]:
        if collection_id in self._material_artifact_cache:
            return self._material_artifact_cache[collection_id]
        artifacts_path = self._material_artifacts_path(collection_id)
        if not artifacts_path.exists():
            return {}
        artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
        self._material_artifact_cache[collection_id] = artifacts
        return artifacts

    def _require_runtime_assets(self, collection_id: str) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        manifest = self._load_manifest(collection_id)
        artifacts = self._load_material_artifacts(collection_id)
        if not manifest or not artifacts:
            raise RuntimeError(
                f"Collection '{collection_id}' is not ready for runtime use. "
                "Run rebuild_collection() to build manifest and material artifacts first."
            )
        return manifest, artifacts

    def collection_status(self, collection_id: Optional[str] = None) -> Dict[str, Any]:
        collection = collection_id or self.default_collection_id
        manifest_path = self._manifest_path(collection)
        artifacts_path = self._material_artifacts_path(collection)
        vector_store_path = self._vector_store_path(collection)
        manifest = self._load_manifest(collection)
        artifacts = self._load_material_artifacts(collection)
        payload: Dict[str, Any] = {
            "collection_id": collection,
            "available": bool(manifest) and bool(artifacts),
            "doc_count": len(manifest),
            "docs_root": str(self.docs_root),
            "manifest_path": str(manifest_path),
            "material_artifacts_path": str(artifacts_path),
            "vector_store_path": str(vector_store_path),
            "manifest_built": manifest_path.exists(),
            "material_artifacts_built": artifacts_path.exists(),
            "rag_index_built": vector_store_path.exists() and any(vector_store_path.iterdir()),
        }
        status_path = self._status_path(collection)
        if status_path.exists():
            payload.update(json.loads(status_path.read_text(encoding="utf-8")))
        payload["available"] = bool(manifest) and bool(artifacts)
        payload["doc_count"] = len(manifest)
        payload["manifest_built"] = manifest_path.exists()
        payload["material_artifacts_built"] = artifacts_path.exists()
        payload["material_artifacts_path"] = str(artifacts_path)
        payload["vector_store_path"] = str(vector_store_path)
        payload.setdefault("rag_index_built", vector_store_path.exists() and any(vector_store_path.iterdir()))
        return payload

    def rebuild_collection(self, collection_id: Optional[str] = None, *, force: bool = False) -> Dict[str, Any]:
        collection = collection_id or self.default_collection_id
        manifest = self._build_manifest()
        logger.info("Building material artifacts for collection %s (%s documents)", collection, len(manifest))
        material_artifacts = self._build_material_artifacts(manifest)
        manifest_path = self._manifest_path(collection)
        artifacts_path = self._material_artifacts_path(collection)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts_path.write_text(json.dumps(material_artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
        self._manifest_cache[collection] = manifest
        self._material_artifact_cache[collection] = material_artifacts
        rag_index_built = self._build_rag_index(collection, manifest, material_artifacts)
        status = {
            "collection_id": collection,
            "available": bool(manifest) and bool(material_artifacts),
            "doc_count": len(manifest),
            "manifest_built": True,
            "material_artifacts_built": True,
            "rag_index_built": rag_index_built,
            "manifest_path": str(manifest_path),
            "material_artifacts_path": str(artifacts_path),
            "vector_store_path": str(self._vector_store_path(collection)),
            "force": force,
        }
        self._status_path(collection).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    def _build_manifest(self) -> List[Dict[str, Any]]:
        manifest: List[Dict[str, Any]] = []
        for path in self.docs_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue
            relative_path = path.relative_to(self.docs_root).as_posix()
            manifest.append(self._classify_entry(path, relative_path))
        manifest.sort(key=lambda item: item["relative_path"])
        return manifest

    def _classify_entry(self, path: Path, relative_path: str) -> Dict[str, Any]:
        rel_lower = relative_path.lower()
        title = path.stem
        product_families = [key for key, terms in _PRODUCT_KEYWORDS.items() if any(term in rel_lower for term in terms)]
        competitor_tags = [key for key, terms in _COMPETITOR_KEYWORDS.items() if any(term in rel_lower for term in terms)]
        body_tags = [key for key, terms in _SPECIAL_BODY_KEYWORDS.items() if any(term in rel_lower for term in terms)]
        special_conditions = [key for key, terms in _SPECIAL_CONDITION_KEYWORDS.items() if any(term in rel_lower for term in terms)]
        transport_type = "passenger" if any(term in rel_lower for term in _PASSENGER_KEYWORDS) else ("special" if body_tags else "cargo")
        doc_kind = self._detect_doc_kind(rel_lower)
        branches = sorted(set(self._derive_branches(doc_kind, transport_type, body_tags, special_conditions, competitor_tags)))
        return {
            "candidate_id": self._candidate_id(relative_path),
            "title": title,
            "relative_path": relative_path,
            "source_path": str(path),
            "extension": path.suffix.lower(),
            "doc_kind": doc_kind,
            "branches": branches,
            "product_families": product_families,
            "competitor_tags": competitor_tags,
            "body_tags": body_tags,
            "transport_type": transport_type,
            "special_conditions": special_conditions,
            "preview_snippet": title,
        }

    def _detect_doc_kind(self, rel_lower: str) -> str:
        if any(token in rel_lower for token in ("\u0441\u0440\u0430\u0432\u043d", "\u043a\u043e\u043d\u043a\u0443\u0440\u0435\u043d\u0442", "atlant", "sollers")):
            return "comparison"
        if any(token in rel_lower for token in ("\u0442\u0441\u043e", "\u044d\u043a\u043e\u043d\u043e\u043c", "tco", "\u0441\u0442\u043e\u0438\u043c")):
            return "tco"
        if any(token in rel_lower for token in ("\u0441\u0435\u0440\u0432\u0438\u0441", "\u0433\u0430\u0440\u0430\u043d\u0442", "\u0440\u0435\u043c\u043e\u043d\u0442", "\u0437\u0430\u043f\u0447\u0430\u0441\u0442", "downtime")):
            return "service"
        if any(token in rel_lower for token in ("\u043a\u043e\u043c\u043f\u043b\u0435\u043a\u0442", "\u0431\u0430\u0437\u0430", "\u043e\u043f\u0446", "\u0442\u0435\u0445", "\u0445\u0430\u0440\u0430\u043a\u0442")):
            return "configuration"
        if any(token in rel_lower for token in ("\u0444\u0438\u043d\u0430\u043d\u0441", "\u043b\u0438\u0437\u0438\u043d\u0433", "\u043a\u0440\u0435\u0434\u0438\u0442", "\u043e\u0434\u043e\u0431\u0440")):
            return "approval"
        if any(token in rel_lower for values in _SPECIAL_BODY_KEYWORDS.values() for token in values):
            return "special_body"
        return "general"

    def _derive_branches(
        self,
        doc_kind: str,
        transport_type: str,
        body_tags: List[str],
        special_conditions: List[str],
        competitor_tags: List[str],
    ) -> Iterable[str]:
        if doc_kind == "tco":
            yield "tco"
        if doc_kind == "comparison" or competitor_tags:
            yield "comparison"
        if doc_kind == "service":
            yield "service_risk"
        if doc_kind == "approval":
            yield "internal_approval"
        if doc_kind in {"configuration", "general"}:
            yield "configuration"
        if transport_type == "passenger":
            yield "passenger_route"
        if body_tags or doc_kind == "special_body":
            yield "special_body"
        if special_conditions:
            yield "special_conditions"

    def _extract_content(self, docs: Sequence[Any]) -> str:
        return "\n\n".join(
            doc.page_content for doc in docs if isinstance(getattr(doc, "page_content", None), str)
        ).strip()

    def _normalize_chunk(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _split_content(self, text: str) -> List[Dict[str, Any]]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
        paragraphs = [
            self._normalize_chunk(chunk)
            for chunk in re.split(r"\n\s*\n", normalized_text)
            if self._normalize_chunk(chunk)
        ]
        chunks: List[Dict[str, Any]] = []
        chunk_index = 0
        for paragraph in paragraphs:
            if len(paragraph) <= _CHUNK_SIZE:
                chunks.append({"index": chunk_index, "text": paragraph})
                chunk_index += 1
                continue
            for start in range(0, len(paragraph), _CHUNK_SIZE):
                piece = self._normalize_chunk(paragraph[start : start + _CHUNK_SIZE])
                if piece:
                    chunks.append({"index": chunk_index, "text": piece})
                    chunk_index += 1
        if not chunks:
            return [{"index": 0, "text": self._normalize_chunk(normalized_text[:_CHUNK_SIZE])}]
        return chunks

    def _artifact_search_text(self, chunks: Sequence[Dict[str, Any]]) -> str:
        joined = " ".join(str(chunk.get("text") or "") for chunk in chunks)
        normalized = self._normalize_chunk(joined).lower()
        return normalized[:_MAX_SEARCH_TEXT_LENGTH]

    def _build_material_artifacts(self, manifest: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        artifacts: Dict[str, Dict[str, Any]] = {}
        for entry in manifest:
            docs = load_single_document(entry["source_path"])
            content = self._extract_content(docs)
            chunks = self._split_content(content)
            preview = chunks[0]["text"][:240] if chunks else entry["title"]
            entry["preview_snippet"] = preview
            artifacts[entry["candidate_id"]] = {
                "candidate_id": entry["candidate_id"],
                "relative_path": entry["relative_path"],
                "preview_snippet": preview,
                "search_text": self._artifact_search_text(chunks),
                "chunks": chunks,
            }
        return artifacts

    def _normalize_requested_families(self, families: Sequence[str] | None) -> List[str]:
        normalized: List[str] = []
        for item in families or []:
            text = re.sub(r"\s+", " ", str(item or "").strip().lower().replace("-", " "))
            if not text:
                continue
            family = _PRODUCT_FAMILY_ALIASES.get(text) or _PRODUCT_FAMILY_ALIASES.get(text.replace(" ", "_"))
            if family and family not in normalized:
                normalized.append(family)
        return normalized

    def search_sales_materials(
        self,
        query: str,
        intent: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
        top_k: int = 5,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        collection = collection_id or self.default_collection_id
        manifest, artifacts = self._require_runtime_assets(collection)
        normalized_families = self._normalize_requested_families(families)
        scored = []
        for entry in manifest:
            artifact = artifacts.get(entry["candidate_id"]) or {}
            score = self._search_score_entry(query, intent, normalized_families, competitor, entry, artifact)
            if score <= 0:
                continue
            scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1]["title"]))
        candidates = [
            self._candidate_payload(
                entry,
                score=score,
                rationale=self._search_rationale(intent, query, normalized_families, competitor, entry),
            )
            for score, entry in scored[: max(1, min(int(top_k or 5), 6))]
        ]
        return {
            "collection_id": collection,
            "intent": intent,
            "query": query,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    def estimate_research_cost(
        self,
        query: str,
        intended_depth: str,
        intent: str,
        families: Sequence[str] | None = None,
        competitor: str = "",
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        collection = collection_id or self.default_collection_id
        manifest, artifacts = self._require_runtime_assets(collection)
        normalized_families = self._normalize_requested_families(families)
        positive_matches = 0
        max_score = 0
        for entry in manifest:
            artifact = artifacts.get(entry["candidate_id"]) or {}
            score = self._search_score_entry(query, intent, normalized_families, competitor, entry, artifact)
            if score > 0:
                positive_matches += 1
                max_score = max(max_score, score)
        depth_bonus = {"broad": 0, "bounded": 1, "justified": 3, "deep_research": 6}.get(str(intended_depth), 0)
        complexity_score = positive_matches + depth_bonus + (4 if competitor else 0)
        requires_wait = bool(intended_depth == "deep_research" or complexity_score >= 18)
        estimated_cost = "high" if complexity_score >= 18 else ("medium" if complexity_score >= 8 else "low")
        rationale = "deep_search_needed" if requires_wait else "bounded_search_should_suffice"
        return {
            "collection_id": collection,
            "query": query,
            "intent": intent,
            "intended_depth": intended_depth,
            "estimated_remaining_cost": estimated_cost,
            "positive_match_count": positive_matches,
            "max_match_score": max_score,
            "requires_hitl_wait_confirmation": requires_wait,
            "rationale": rationale,
        }

    def get_branch_pack(
        self,
        branch: str,
        slots: Dict[str, Any],
        problem_summary: str,
        top_k: int = 5,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        collection = collection_id or self.default_collection_id
        manifest, artifacts = self._require_runtime_assets(collection)
        scored = []
        for entry in manifest:
            artifact = artifacts.get(entry["candidate_id"]) or {}
            score = self._score_entry(branch, slots, problem_summary, entry, artifact)
            if score <= 0:
                continue
            scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1]["title"]))
        candidates = [
            self._candidate_payload(
                entry,
                score=score,
                rationale=self._candidate_rationale(branch, slots, entry),
                branch_relevance=branch,
            )
            for score, entry in scored[: max(1, min(top_k, 5))]
        ]
        return {"collection_id": collection, "branch": branch, "candidates": candidates}

    def _candidate_payload(
        self,
        entry: Dict[str, Any],
        *,
        score: int,
        rationale: str,
        branch_relevance: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "candidate_id": entry["candidate_id"],
            "title": entry["title"],
            "doc_kind": entry["doc_kind"],
            "rationale": rationale,
            "branch_relevance": branch_relevance,
            "preview_snippet": entry.get("preview_snippet"),
            "metadata": {
                "relative_path": entry["relative_path"],
                "score": score,
                "branches": entry["branches"],
                "product_families": entry["product_families"],
                "competitor_tags": entry["competitor_tags"],
                "body_tags": entry["body_tags"],
            },
        }

    def _tokenize(self, value: str) -> List[str]:
        return [token for token in re.findall(r"[\w-]{3,}", (value or "").lower()) if token]

    def _search_score_entry(
        self,
        query: str,
        intent: str,
        families: Sequence[str],
        competitor: str,
        entry: Dict[str, Any],
        artifact: Dict[str, Any],
    ) -> int:
        score = 0
        artifact_search_text = str(artifact.get("search_text") or "")
        haystack = " ".join(
            [
                entry["title"].lower(),
                entry["relative_path"].lower(),
                " ".join(entry["product_families"]),
                " ".join(entry["competitor_tags"]),
                " ".join(entry["body_tags"]),
                " ".join(entry["branches"]),
                entry["doc_kind"],
                artifact_search_text,
            ]
        )
        score += _INTENT_DOC_KIND_BONUS.get(intent, {}).get(entry["doc_kind"], 0)
        query_tokens = self._tokenize(query)
        if not query_tokens:
            score += 1
        for token in query_tokens:
            if token in entry["title"].lower():
                score += 4
            elif token in entry["relative_path"].lower():
                score += 3
            elif token in artifact_search_text:
                score += 2
            elif token in haystack:
                score += 1
        for family in families:
            if family in entry["product_families"]:
                score += 4
        competitor_text = str(competitor or "").lower()
        if competitor_text and any(competitor_text in tag for tag in entry["competitor_tags"]):
            score += 5
        if competitor_text and competitor_text in artifact_search_text:
            score += 2
        if intent == "overview" and entry["doc_kind"] == "general":
            score += 2
        return score

    def _search_rationale(
        self,
        intent: str,
        query: str,
        families: Sequence[str],
        competitor: str,
        entry: Dict[str, Any],
    ) -> str:
        reasons = [f"supports {intent} answer"]
        if competitor and entry["competitor_tags"]:
            reasons.append("matches competitor context")
        if families and entry["product_families"]:
            reasons.append("touches likely product families")
        if query:
            reasons.append("matches the current ask")
        return "; ".join(reasons)

    def _score_entry(self, branch: str, slots: Dict[str, Any], problem_summary: str, entry: Dict[str, Any], artifact: Dict[str, Any]) -> int:
        score = 0
        artifact_search_text = str(artifact.get("search_text") or "")
        if branch in entry["branches"]:
            score += 12
        if branch == entry["doc_kind"]:
            score += 5
        criterion = str(slots.get("decision_criterion") or "").lower()
        if criterion and (criterion in entry["doc_kind"] or criterion in entry["title"].lower() or criterion in artifact_search_text):
            score += 3
        competitor = str(slots.get("competitor") or "").lower()
        if competitor and any(competitor in tag for tag in entry["competitor_tags"]):
            score += 5
        body_type = str(slots.get("body_type") or "").lower()
        if body_type and any(body_type in tag for tag in entry["body_tags"]):
            score += 4
        transport_type = str(slots.get("transport_type") or "").lower()
        if transport_type and transport_type == entry["transport_type"]:
            score += 2
        for token in self._tokenize(problem_summary):
            if token in entry["title"].lower() or token in entry["relative_path"].lower():
                score += 1
            elif token in artifact_search_text:
                score += 1
        return score

    def _candidate_rationale(self, branch: str, slots: Dict[str, Any], entry: Dict[str, Any]) -> str:
        reasons = [f"supports {branch} reasoning"]
        if slots.get("competitor") and entry["competitor_tags"]:
            reasons.append("matches competitor context")
        if slots.get("body_type") and entry["body_tags"]:
            reasons.append("matches body type")
        if entry["transport_type"] == "passenger":
            reasons.append("fits passenger route selection")
        return "; ".join(reasons)

    def read_material(
        self,
        candidate_id: str,
        focus: str,
        max_segments: int = 3,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        collection = collection_id or self.default_collection_id
        manifest, artifacts = self._require_runtime_assets(collection)
        entry = next((item for item in manifest if item["candidate_id"] == candidate_id), None)
        if entry is None:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")
        artifact = artifacts.get(candidate_id)
        if artifact is None:
            raise RuntimeError(f"Material artifact for candidate '{candidate_id}' is missing in collection '{collection}'.")
        scored = []
        for chunk in artifact.get("chunks") or []:
            excerpt_text = str(chunk.get("text") or "")
            score = self._score_excerpt(excerpt_text, focus, entry)
            if score <= 0:
                continue
            scored.append((score, excerpt_text, int(chunk.get("index") or 0)))
        scored.sort(key=lambda item: (-item[0], item[2]))
        excerpts = []
        for score, excerpt, chunk_index in scored[: max(1, min(max_segments, 5))]:
            excerpts.append(
                {
                    "excerpt": excerpt[:_MAX_RUNTIME_CHUNK_LENGTH],
                    "relevance_reason": f"focus match score {score}",
                    "metadata": {"score": score, "chunk_index": chunk_index},
                }
            )
        return {
            "candidate_id": candidate_id,
            "title": entry["title"],
            "focus": focus,
            "excerpts": excerpts,
            "metadata": {
                "relative_path": entry["relative_path"],
                "doc_kind": entry["doc_kind"],
                "product_families": entry["product_families"],
                "artifact_chunk_count": len(artifact.get("chunks") or []),
            },
        }

    def _score_excerpt(self, excerpt: str, focus: str, entry: Dict[str, Any]) -> int:
        excerpt_lower = excerpt.lower()
        score = 1 if excerpt_lower else 0
        for term in self._tokenize(focus or ""):
            if term in excerpt_lower:
                score += 2
        for family in entry["product_families"]:
            if family.lower() in excerpt_lower:
                score += 1
        for tag in entry["body_tags"]:
            if tag in excerpt_lower:
                score += 2
        if entry["doc_kind"] in (focus or "").lower():
            score += 2
        return score

    def _build_rag_index(
        self,
        collection_id: str,
        manifest: List[Dict[str, Any]],
        material_artifacts: Dict[str, Dict[str, Any]],
    ) -> bool:
        segments: List[Segment] = []
        for entry in manifest:
            artifact = material_artifacts.get(entry["candidate_id"]) or {}
            for chunk in artifact.get("chunks") or []:
                text = str(chunk.get("text") or "").strip()
                if not text:
                    continue
                chunk_index = int(chunk.get("index") or 0)
                segments.append(
                    Segment(
                        content=text[:6000],
                        metadata={
                            "candidate_id": entry["candidate_id"],
                            "source": entry["relative_path"],
                            "doc_kind": entry["doc_kind"],
                            "branches": entry["branches"],
                            "product_families": entry["product_families"],
                            "chunk_index": chunk_index,
                        },
                        segment_id=f"{entry['candidate_id']}:{chunk_index}",
                    )
                )
        if not segments:
            raise RuntimeError("No indexable segments were produced from the GAZ material artifacts.")

        embeddings = create_embeddings_model(provider="local", model_name=config.EMBEDDING_MODEL)
        with self._vector_store_environment(collection_id):
            vector_store = create_vector_store(
                provider="chroma",
                embeddings=embeddings,
                collection_name=f"gaz_{collection_id}",
                cleanup=True,
            )
            indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
            indexer.index(segments, batch_size=32)
        return True
