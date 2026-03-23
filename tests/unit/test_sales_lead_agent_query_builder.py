from agents.sales_lead_agent.schemas import PurchaseSearchRequest, SearchFilters
from agents.sales_lead_agent.services.query_builder import ProcurementQueryBuilder
from agents.sales_lead_agent.settings import SalesLeadAgentSettings


def _settings() -> SalesLeadAgentSettings:
    return SalesLeadAgentSettings(
        work_root=__import__("pathlib").Path("."),
        retention_hours=72,
        damia_api_key="",
        scoring_base_url="",
        fssp_base_url="",
        purchase_headless=True,
        open_source_max_concurrency=4,
        procurement_search_template=(
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            "?searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2%D0%B0%D0%BD"
            "&morphology=on"
            "&search-filter=%D0%94%D0%B0%D1%82%D0%B5+%D1%80%D0%B0%D0%B7%D0%BC%D0%B5%D1%89%D0%B5%D0%BD%D0%B8%D1%8F"
            "&pageNumber=1"
            "&sortDirection=false"
            "&recordsPerPage=_2"
            "&showLotsInfoHidden=false"
            "&sortBy=UPDATE_DATE"
            "&fz44=on"
            "&fz223=on"
            "&af=on"
            "&currencyIdGeneral=-1"
            "&gws=%D0%92%D1%8B%D0%B1%D0%B5%D1%80%D0%B8%D1%82%D0%B5+%D1%82%D0%B8%D0%BF+%D0%B7%D0%B0%D0%BA%D1%83%D0%BF%D0%BA%D0%B8"
        ),
    )


def test_query_builder_builds_contextualized_search_string():
    builder = ProcurementQueryBuilder(_settings())
    filters = SearchFilters(
        query_text="услуг страхования транспортных средств",
        supplier_hint="КАСКО",
    )

    value = builder.build_search_string(filters)

    assert "страхования" in value
    assert "транспортных" in value
    assert "+" in value
    assert "оказание" not in value


def test_query_builder_drops_or_and_limits_query_breadth():
    builder = ProcurementQueryBuilder(_settings())
    filters = SearchFilters(
        query_text=(
            "страхование грузов "
            "OR страхование "
            "ответственности "
            "перевозчика "
            "OR страхование "
            "транспортных "
            "средств"
        )
    )

    value = builder.build_search_string(filters)

    assert "or" not in value
    assert len(value.split("+")) <= 5


def test_query_builder_changes_only_search_string_in_template():
    builder = ProcurementQueryBuilder(_settings())
    filters = SearchFilters(
        query_text="страхование имущества",
    )

    url = builder.build_url(filters)

    assert "searchString=" in url
    assert "recordsPerPage=_2" in url
    assert "gws=%D0%92%D1%8B%D0%B1%D0%B5%D1%80%D0%B8%D1%82%D0%B5+%D1%82%D0%B8%D0%BF+%D0%B7%D0%B0%D0%BA%D1%83%D0%BF%D0%BA%D0%B8" in url
    assert "fz44=on" in url
    assert "fz223=on" in url
    assert "sortBy=UPDATE_DATE" in url


def test_purchase_search_request_schema_is_flat_for_tool_calling():
    schema = PurchaseSearchRequest.model_json_schema()

    assert "search_filters" not in schema["properties"]
    assert "query_text" in schema["properties"]
    assert "customer_name" in schema["properties"]
