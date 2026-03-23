from pathlib import Path

import httpx

from agents.sales_lead_agent.services.external_apis import CounterpartyClients
from agents.sales_lead_agent.settings import SalesLeadAgentSettings


def _settings(tmp_path: Path) -> SalesLeadAgentSettings:
    return SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        retention_hours=72,
        damia_api_key="token",
        scoring_base_url="https://damia.example.test",
        fssp_base_url="https://damia.example.test",
        purchase_headless=True,
        open_source_max_concurrency=4,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test",
    )


class _FakeResponse:
    def __init__(self, payload, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.content = b"{}"

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://damia.example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("bad status", request=request, response=response)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params=None):
        self.calls.append((url, params))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_counterparty_scoring_normalizes_score_and_fincoefs(tmp_path, monkeypatch):
    client = CounterpartyClients(_settings(tmp_path))
    fake_client = _FakeClient(
        [
            _FakeResponse(
                {
                    "risk_value": "0.2",
                    "risk_zone": "low",
                    "score_value": "87.5",
                    "score_zone": "green",
                    "reliability_value": "91.3",
                    "reliability_zone": "green",
                    "top_factors": [
                        {"name": "Revenue stability", "value": "1.2", "nwoe": "0.3"},
                    ],
                }
            ),
            _FakeResponse(
                [
                    {"name": "Quick ratio", "value": "1.4", "norm": "1.0", "comparison": "above"},
                ]
            ),
        ]
    )
    monkeypatch.setattr(client, "_client", lambda: fake_client)

    response = client.scoring(
        inn="7707083893",
        model="default",
        include_fincoefs=True,
    )

    assert response.status == "success"
    assert response.inn == "7707083893"
    assert response.score is not None
    assert response.score.risk_value == 0.2
    assert response.score.score_value == 87.5
    assert response.score.top_factors[0].name == "Revenue stability"
    assert response.score.top_factors[0].nwoe == 0.3
    assert response.fincoefs[0].name == "Quick ratio"
    assert response.fincoefs[0].comparison == "above"
    assert fake_client.calls[0][0].endswith("/scoring/score")
    assert fake_client.calls[1][0].endswith("/scoring/fincoefs")


def test_counterparty_scoring_returns_failed_response_on_transport_error(tmp_path, monkeypatch):
    client = CounterpartyClients(_settings(tmp_path))
    fake_client = _FakeClient([httpx.ConnectError("connect boom")])
    monkeypatch.setattr(client, "_client", lambda: fake_client)

    response = client.scoring(
        inn="7707083893",
        model=None,
        include_fincoefs=False,
    )

    assert response.status == "failed"
    assert response.inn == "7707083893"
    assert "connect boom" in (response.error or "")


def test_counterparty_fssp_normalizes_grouped_records(tmp_path, monkeypatch):
    client = CounterpartyClients(_settings(tmp_path))
    fake_client = _FakeClient(
        [
            _FakeResponse(
                [
                    {
                        "year": "2024",
                        "status": "active",
                        "subject": "tax",
                        "amount": "1520.4",
                        "count": "2",
                        "proceeding_ids": [101, "102"],
                    }
                ]
            )
        ]
    )
    monkeypatch.setattr(client, "_client", lambda: fake_client)

    response = client.fssp(
        inn="7707083893",
        from_date="2024-01-01",
        to_date="2024-12-31",
        response_format=1,
    )

    assert response.status == "success"
    assert response.inn == "7707083893"
    assert response.grouped[0].year == 2024
    assert response.grouped[0].status == "active"
    assert response.grouped[0].amount == 1520.4
    assert response.grouped[0].count == 2
    assert response.grouped[0].proceeding_ids == ["101", "102"]
    assert fake_client.calls[0][0].endswith("/fssp/isps")


def test_counterparty_fssp_returns_failed_response_on_transport_error(tmp_path, monkeypatch):
    client = CounterpartyClients(_settings(tmp_path))
    fake_client = _FakeClient([httpx.ReadTimeout("timeout boom")])
    monkeypatch.setattr(client, "_client", lambda: fake_client)

    response = client.fssp(
        inn="7707083893",
        from_date=None,
        to_date=None,
        response_format=1,
    )

    assert response.status == "failed"
    assert response.inn == "7707083893"
    assert "timeout boom" in (response.error or "")
