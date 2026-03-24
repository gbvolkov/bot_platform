import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from rag_lib.core.domain import Segment, SegmentType

import services.kb_manager.gaz_runtime as gaz_runtime


class FakeVectorStore:
    def __init__(self, key, registry, path: Path):
        self.key = key
        self.registry = registry
        self.path = path

    @property
    def records(self):
        return self.registry[self.key]

    def delete_collection(self):
        self.registry[self.key] = {}
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "collection.marker").write_text("deleted", encoding="utf-8")

    def add_texts(self, texts, metadatas, ids):
        self.path.mkdir(parents=True, exist_ok=True)
        for text, metadata, segment_id in zip(texts, metadatas, ids):
            self.registry[self.key][segment_id] = {
                "id": segment_id,
                "text": text,
                "metadata": metadata,
            }
        (self.path / "collection.marker").write_text("indexed", encoding="utf-8")


def _install_fake_rag_stack(monkeypatch):
    vector_registry = {}

    def fake_create_embeddings_model(*args, **kwargs):
        return SimpleNamespace(name="fake-embeddings")

    def fake_create_vector_store(provider="chroma", embeddings=None, collection_name="rag_collection", connection_uri=None, cleanup=True):
        vector_path = Path(os.environ["VECTOR_PATH"])
        key = (str(vector_path), collection_name)
        if cleanup or key not in vector_registry:
            vector_registry[key] = {}
        vector_path.mkdir(parents=True, exist_ok=True)
        return FakeVectorStore(key, vector_registry, vector_path)

    def fake_create_retriever(
        vector_store,
        doc_store,
        id_key="segment_id",
        search_kwargs=None,
        search_type=None,
        score_threshold=None,
        hydration_mode=None,
        enrichment_separator="\n",
    ):
        options = dict(search_kwargs or {})
        filter_payload = dict(options.get("filter") or {})
        limit = int(options.get("k") or 4)

        class FakeRetriever:
            def invoke(self, query: str):
                tokens = [token for token in re.split(r"\W+", query.lower()) if len(token) >= 2]
                results = []
                for segment_id, record in vector_store.records.items():
                    metadata = dict(record["metadata"])
                    if any(metadata.get(name) != value for name, value in filter_payload.items()):
                        continue
                    haystack = (record["text"] + " " + " ".join(str(value) for value in metadata.values())).lower()
                    matched = sum(1 for token in tokens if token in haystack)
                    score = matched / max(1, len(tokens)) if tokens else 1.0
                    if score_threshold is not None and score < score_threshold:
                        continue
                    hydrated = doc_store.mget([segment_id])[0]
                    if hydrated is None:
                        continue
                    document = Document(id=segment_id, page_content=hydrated.page_content, metadata=dict(hydrated.metadata))
                    document.metadata[id_key] = segment_id
                    document.metadata["similarity_score"] = score
                    document.metadata["max_similarity_score"] = score
                    results.append(document)
                results.sort(key=lambda item: item.metadata["similarity_score"], reverse=True)
                return results[:limit]

        return FakeRetriever()

    monkeypatch.setattr(gaz_runtime, "create_embeddings_model", fake_create_embeddings_model)
    monkeypatch.setattr(gaz_runtime, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(gaz_runtime, "create_scored_dual_storage_retriever", fake_create_retriever)
    return vector_registry


def _install_loader_mocks(monkeypatch, content_by_name):
    def _build_loader(source_type: str, output_format: str):
        def _load(self):
            file_name = Path(self.file_path).name
            return [
                Document(
                    page_content=content_by_name[file_name],
                    metadata={"source": self.file_path, "source_type": source_type, "output_format": output_format},
                )
            ]

        return _load

    monkeypatch.setattr(gaz_runtime.PDFLoader, "load", _build_loader("pdf", "text"))
    monkeypatch.setattr(gaz_runtime.PPTXLoader, "load", _build_loader("pptx", "markdown"))
    monkeypatch.setattr(gaz_runtime.ImageLoader, "load", _build_loader("image", "markdown"))
    monkeypatch.setattr(gaz_runtime.DocXLoader, "load", _build_loader("docx", "text"))
    monkeypatch.setattr(gaz_runtime.TextLoader, "load", _build_loader("text", "text"))
    monkeypatch.setattr(gaz_runtime.HTMLLoader, "load", _build_loader("html", "html"))
    monkeypatch.setattr(gaz_runtime.JsonLoader, "load", _build_loader("json", "json"))
    monkeypatch.setattr(gaz_runtime.CSVLoader, "load", _build_loader("csv", "csv"))
    monkeypatch.setattr(gaz_runtime.ExcelLoader, "load", _build_loader("excel", "markdown"))


@pytest.fixture
def runtime_fixture(tmp_path, monkeypatch):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    cache_root = tmp_path / "cache"
    content_by_name = {
        "gazelle nn manual.pdf": "Engine 2.5 turbo diesel 150 hp. Payload 1500 kg. Urban cargo use.",
        "gazelle nn service book.pdf": "Warranty 36 months. Maintenance every 20000 km. Dealer service support.",
        "gazelle next pricing.pdf": "Total cost of ownership. Leasing scenario. Monthly payment logic.",
        "gazelle next summary.pptx": "# Slide 1: Summary\nUrban cargo direction.\n# Slide 2: Options\nBase package and city fit.",
    }
    for file_name in content_by_name:
        (docs_root / file_name).write_text("stub", encoding="utf-8")

    vector_registry = _install_fake_rag_stack(monkeypatch)
    _install_loader_mocks(monkeypatch, content_by_name)

    service = gaz_runtime.GazRuntimeService(docs_root=docs_root, cache_root=cache_root)
    status = service.rebuild_collection("gaz", force=True)
    assert status["available"] is True
    return SimpleNamespace(service=service, docs_root=docs_root, cache_root=cache_root, vector_registry=vector_registry)


def test_runtime_search_and_read_use_indexed_assets_without_raw_reread(runtime_fixture, monkeypatch):
    service = runtime_fixture.service
    manifest = service._load_manifest("gaz")
    manual_id = next(item["candidate_id"] for item in manifest if item["title"] == "gazelle nn manual")

    def fail_load(self):
        raise AssertionError("runtime attempted raw document reread")

    monkeypatch.setattr(gaz_runtime.PDFLoader, "load", fail_load)
    monkeypatch.setattr(gaz_runtime.PPTXLoader, "load", fail_load)

    result = service.search_sales_materials(
        query="engine diesel horsepower",
        intent="specs",
        families=["Gazelle NN"],
        collection_id="gaz",
    )
    assert result["candidate_count"] >= 1
    assert result["candidates"][0]["candidate_id"] == manual_id

    read_result = service.read_material(
        candidate_id=manual_id,
        focus="engine horsepower payload",
        max_segments=2,
        collection_id="gaz",
    )
    assert read_result["candidate_id"] == manual_id
    assert read_result["excerpts"]
    assert "150 hp" in read_result["excerpts"][0]["excerpt"].lower()
    assert read_result["metadata"]["artifact_chunk_count"] >= 1


def test_get_branch_pack_filters_results_by_branch(runtime_fixture):
    service = runtime_fixture.service
    manifest = service._load_manifest("gaz")
    service_book_id = next(item["candidate_id"] for item in manifest if item["title"] == "gazelle nn service book")

    result = service.get_branch_pack(
        branch="service_risk",
        slots={"decision_criterion": "service"},
        problem_summary="Need warranty and maintenance risk assessment",
        top_k=3,
        collection_id="gaz",
    )

    assert result["candidate_count"] >= 1
    assert result["candidates"][0]["candidate_id"] == service_book_id
    assert all("service_risk" in item["metadata"]["branches"] for item in result["candidates"])


def test_estimate_research_cost_uses_retrieval_statistics(runtime_fixture):
    estimate = runtime_fixture.service.estimate_research_cost(
        query="engine warranty maintenance",
        intended_depth="justified",
        intent="specs",
        families=["Gazelle NN"],
        collection_id="gaz",
    )

    assert estimate["positive_match_count"] >= 2
    assert estimate["max_match_score"] > 0
    assert estimate["estimated_remaining_cost"] >= 1.5
    assert "Retrieval surfaced" in estimate["rationale"]


def test_collection_status_reports_segment_store_and_compatibility_alias(runtime_fixture):
    status = runtime_fixture.service.collection_status("gaz")

    assert status["manifest_built"] is True
    assert status["segment_store_built"] is True
    assert status["rag_index_built"] is True
    assert status["material_artifacts_built"] is True
    assert status["segment_store_path"] == status["material_artifacts_path"]


def test_vector_metadata_contains_filter_flags_and_deterministic_segment_ids(runtime_fixture):
    service = runtime_fixture.service
    manifest = service._load_manifest("gaz")
    service_book = next(item for item in manifest if item["title"] == "gazelle nn service book")
    key = next(iter(runtime_fixture.vector_registry.keys()))
    records = runtime_fixture.vector_registry[key]
    service_records = [record for record in records.values() if record["metadata"]["candidate_id"] == service_book["candidate_id"]]

    assert service_records
    assert any(record["id"].startswith(f"{service_book['candidate_id']}:") for record in service_records)
    assert any(record["metadata"]["branch__service_risk"] is True for record in service_records)
    assert any(record["metadata"]["family__gazelle_nn"] is True for record in service_records)
    assert all("chunk_index" in record["metadata"] for record in service_records)


def test_splitter_strategy_registry_uses_expected_chain(tmp_path, monkeypatch):
    service = gaz_runtime.GazRuntimeService(docs_root=tmp_path / "docs", cache_root=tmp_path / "cache")
    monkeypatch.setattr(service, "_get_table_summarizer", lambda: object())

    class _BaseSplitter:
        def __init__(self, label, log, **kwargs):
            self.label = label
            self.log = log
            self.kwargs = kwargs

        def split_documents(self, documents):
            self.log.append((self.label, self.kwargs))
            return [Segment(content=documents[0].page_content, metadata=dict(documents[0].metadata), type=SegmentType.TEXT)]

    def make_factory(label, log):
        class _Factory:
            def __init__(self, **kwargs):
                self.impl = _BaseSplitter(label, log, **kwargs)

            def split_documents(self, documents):
                return self.impl.split_documents(documents)

        return _Factory

    log = []

    class FakeHTMLFactory(make_factory("html", log)):
        def split_documents(self, documents):
            log.append(("html", {}))
            return [
                Segment(content=documents[0].page_content, metadata=dict(documents[0].metadata), type=SegmentType.TEXT),
                Segment(content="| a | b |", metadata=dict(documents[0].metadata), type=SegmentType.TABLE),
            ]

    class FakeSentenceFactory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def split_documents(self, documents):
            log.append(("sentence", self.kwargs))
            return [Segment(content=documents[0].page_content, metadata=dict(documents[0].metadata), type=SegmentType.TEXT)]

    monkeypatch.setattr(gaz_runtime, "CSVTableSplitter", make_factory("csv", log))
    monkeypatch.setattr(gaz_runtime, "MarkdownTableSplitter", make_factory("markdown_table", log))
    monkeypatch.setattr(gaz_runtime, "SentenceSplitter", FakeSentenceFactory)
    monkeypatch.setattr(gaz_runtime, "HTMLSplitter", FakeHTMLFactory)
    monkeypatch.setattr(gaz_runtime, "JsonSplitter", make_factory("json", log))
    monkeypatch.setattr(gaz_runtime, "RegexSplitter", make_factory("regex", log))
    monkeypatch.setattr(gaz_runtime, "RecursiveCharacterTextSplitter", make_factory("recursive", log))

    document = Document(page_content="sample content", metadata={"output_format": "text"})

    service._split_documents_for_entry({"extension": ".csv"}, [document])
    service._split_documents_for_entry({"extension": ".xlsx"}, [document])
    service._split_documents_for_entry({"extension": ".pdf"}, [document])
    service._split_documents_for_entry({"extension": ".png"}, [document])
    service._split_documents_for_entry({"extension": ".html"}, [document])
    service._split_documents_for_entry({"extension": ".json"}, [document])
    service._split_documents_for_entry({"extension": ".pptx"}, [document])

    csv_kwargs = next(kwargs for label, kwargs in log if label == "csv")
    xlsx_kwargs = next(kwargs for label, kwargs in log if label == "markdown_table")
    assert csv_kwargs["max_rows_per_chunk"] == service._rag_settings.ingestion.chunk_size
    assert csv_kwargs["summarize_table"] is True
    assert xlsx_kwargs["split_table_rows"] is True
    sentence_sizes = [kwargs["chunk_size"] for label, kwargs in log if label == "sentence"]
    assert sentence_sizes.count(2400) >= 2
    assert sentence_sizes.count(1200) >= 3
    assert any(label == "html" for label, _kwargs in log)
    assert any(label == "json" for label, _kwargs in log)
    assert any(label == "regex" for label, _kwargs in log)
    assert any(label == "recursive" for label, _kwargs in log)
