from __future__ import annotations

from agents.gaz_agent import marketing_tools


class _FakeGazDocumentsClient:
    instances = []

    def __init__(self, *, base_url, collection_id, timeout_seconds):
        self.base_url = base_url
        self.collection_id = collection_id
        self.timeout_seconds = timeout_seconds
        self.calls = []
        self.instances.append(self)

    def search_sales_materials(self, **kwargs):
        self.calls.append(("search_sales_materials", kwargs))
        return {"candidates": [{"id": "m1"}]}

    def read_material(self, **kwargs):
        self.calls.append(("read_material", kwargs))
        return {"segments": [{"text": "evidence"}]}

    def get_branch_pack(self, **kwargs):
        self.calls.append(("get_branch_pack", kwargs))
        return {"materials": [{"id": "branch1"}]}

    def estimate_research_cost(self, **kwargs):
        self.calls.append(("estimate_research_cost", kwargs))
        return {"estimated_calls": 2}


def test_marketing_document_tools_return_plain_dicts(monkeypatch):
    _FakeGazDocumentsClient.instances = []
    monkeypatch.setattr(marketing_tools, "GazDocumentsClient", _FakeGazDocumentsClient)

    tools = marketing_tools.build_marketing_document_tools(
        locale="ru",
        docs_collection="gaz-test",
        docs_base_url="http://docs.local",
        timeout_seconds=3.0,
    )
    by_name = {tool.name: tool for tool in tools}

    search_result = by_name["search_marketing_materials"].invoke(
        {
            "query": "positioning",
            "intent": "landscape",
            "families": ["gazelle"],
            "competitor": "",
            "top_k": 4,
        }
    )
    read_result = by_name["read_marketing_material"].invoke(
        {"candidate_id": "m1", "focus": "value", "max_segments": 2}
    )
    pack_result = by_name["get_marketing_branch_pack"].invoke(
        {
            "branch": "delivery",
            "problem_summary": "urban delivery",
            "slots": {"body": "van"},
            "top_k": 3,
        }
    )
    estimate_result = by_name["estimate_marketing_research_cost"].invoke(
        {
            "query": "competitor comparison",
            "intended_depth": "standard",
            "intent": "comparison",
            "families": [],
            "competitor": "Ford",
        }
    )

    assert search_result["status"] == "ok"
    assert search_result["payload"] == {"candidates": [{"id": "m1"}]}
    assert read_result["payload"] == {"segments": [{"text": "evidence"}]}
    assert pack_result["payload"] == {"materials": [{"id": "branch1"}]}
    assert estimate_result["payload"] == {"estimated_calls": 2}
    assert _FakeGazDocumentsClient.instances[0].base_url == "http://docs.local"
    assert _FakeGazDocumentsClient.instances[0].collection_id == "gaz-test"
    assert [name for name, _kwargs in _FakeGazDocumentsClient.instances[0].calls] == [
        "search_sales_materials",
        "read_material",
        "get_branch_pack",
        "estimate_research_cost",
    ]


def test_marketing_document_tools_return_error_dict(monkeypatch):
    class FailingClient(_FakeGazDocumentsClient):
        def search_sales_materials(self, **kwargs):
            raise RuntimeError("service down")

    monkeypatch.setattr(marketing_tools, "GazDocumentsClient", FailingClient)

    tool = marketing_tools.build_marketing_document_tools()[0]
    result = tool.invoke({"query": "positioning"})

    assert result == {
        "status": "error",
        "tool": "search_marketing_materials",
        "error_type": "RuntimeError",
        "message": "service down",
    }
