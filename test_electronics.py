"""Regression checks for the team's model rankings and real catalog integration."""

import hashlib
import json
from contextlib import closing
from decimal import Decimal
from pathlib import Path

import pytest

from shopping_agent.electronics import ElectronicsCatalog
from shopping_agent.service import Agent
from shopping_agent.storage import Store

PROJECT_ROOT = Path(__file__).resolve().parent
TEAM_ROOT = PROJECT_ROOT.parent / "teammate_code" / "Recommender_Project-main"


@pytest.fixture(scope="session")
def team_root():
    if not (TEAM_ROOT / "data" / "item_catalog.csv").is_file():
        pytest.skip("The sibling Electronics recommender catalog is unavailable.")
    return TEAM_ROOT


@pytest.fixture(scope="session")
def team_catalog(team_root):
    return ElectronicsCatalog(team_root)


@pytest.fixture
def small_artifacts(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    csv = data / "item_catalog.csv"
    csv.write_text(
        "parent_asin,title,store,price,category,rating_number\n"
        "headphones-range,Wireless Headphones,Example,from 10.00,Headphones,80\n"
        "headphones-over,Wireless Headphones,Example,15.01,Headphones,70\n"
        "headphones-exact,Wireless Headphones,Example,15.00,Headphones,60\n"
        "headphones-blank,Wireless Headphones,Example,,Headphones,50\n"
        "adapter,Wireless Adapter,Example,12.00,Adapters,40\n"
        "titleless,,Example,9.99,Headphones,100\n",
        encoding="utf-8",
    )
    export = {
        "schema_version": 1,
        "catalog_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "model_sha256": {"ncf": "a" * 64, "tiger": "b" * 64},
        "rankings": {
            "accessories-tiger": [
                "adapter",
                "headphones-range",
                "headphones-over",
                "headphones-exact",
                "headphones-blank",
            ]
        },
    }
    rankings = tmp_path / "rankings.json"
    rankings.write_text(json.dumps(export), encoding="utf-8")
    return csv, rankings


def test_electronics_enforces_keyword_budget_exclusion_and_unknown_prices(
    small_artifacts,
):
    csv, rankings = small_artifacts
    catalog = ElectronicsCatalog(csv.parent.parent, rankings_path=rankings)
    persona = "accessories-tiger"

    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog)
        found = agent.submit(
            "s",
            "initial",
            "Electronics, search: wireless headphones, budget to $15",
            persona=persona,
        )
        assert found.outcome == "PARTIAL"
        assert found.personalization == "EXAMPLE_HISTORY_USED"
        assert [
            (card.id, card.price_usd, card.price_status) for card in found.verified
        ] == [("headphones-exact", Decimal("15.00"), "SATISFIED")]
        assert {card.id for card in found.pending} == {
            "headphones-range",
            "headphones-blank",
        }
        assert all(
            card.price_usd is None and card.price_status == "UNKNOWN"
            for card in found.pending
        )

        excluded = agent.submit("s", "exclude", "Exclude first", persona=persona)
        assert "headphones-exact" not in excluded.shown_ids()
        assert store.get_state("s")["excluded_ids"] == ["headphones-exact"]
        assert "headphones-over" not in excluded.shown_ids()
        assert "titleless" not in excluded.shown_ids()


def test_electronics_rejects_stale_export_and_cross_snapshot_requests(small_artifacts):
    csv, rankings = small_artifacts
    catalog = ElectronicsCatalog(csv.parent.parent, rankings_path=rankings)
    result = catalog.call(
        "get_products", {"ids": ["adapter"], "snapshot": "another-snapshot"}
    )
    assert result.status == "ERROR"
    assert result.error_code == "snapshot_mismatch"
    assert result.products == []

    csv.write_bytes(csv.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="different item_catalog"):
        ElectronicsCatalog(csv.parent.parent, rankings_path=rankings)


def test_electronics_distinguishes_absent_review_source_from_absent_product(
    small_artifacts,
):
    csv, rankings = small_artifacts
    catalog = ElectronicsCatalog(csv.parent.parent, rankings_path=rankings)
    unavailable = catalog.call("get_product_evidence", {"ids": ["adapter"]})
    assert unavailable.status == "ERROR"
    assert unavailable.error_code == "review_unavailable"
    assert unavailable.review_gaps == {}
    missing = catalog.call("get_product_evidence", {"ids": ["nonexistent"]})
    assert missing.status == "ERROR"
    assert missing.error_code == "missing_product"


def test_team_artifact_discovery_without_history_uses_catalog_search(team_catalog):
    with closing(Store(":memory:")) as store:
        result = Agent(store, team_catalog).submit(
            "s", "initial", "find electronics under $40"
        )
    assert result.data_mode == "team-electronics"
    assert result.snapshot == team_catalog.snapshot
    assert result.personalization == "NO_HISTORY"
    assert not any(item["action"] == "recommend_products" for item in result.trace)
    assert any(item["action"] == "search_products" for item in result.trace)
    assert result.verified
    assert all(
        card.category == "Electronics" and card.price_usd <= Decimal(40)
        for card in result.verified
    )


def test_team_artifact_personas_use_different_model_rankings(team_catalog):
    with closing(Store(":memory:")) as store:
        agent = Agent(store, team_catalog)
        tiger, ncf = [
            agent.submit(alias, "initial", "find electronics under $40", persona=alias)
            for alias in ("accessories-tiger", "accessories-ncf")
        ]
    assert tiger.verified[0].id == "B0BVPT6QJ4"
    assert ncf.verified[0].id == "B08XPWDSWW"
    assert tiger.verified[0].example_ranking == "accessories-tiger"
    assert ncf.verified[0].example_ranking == "accessories-ncf"
    for result in (tiger, ncf):
        assert result.personalization == "EXAMPLE_HISTORY_USED"
        assert any(
            item["action"] == "recommend_products" and item["status"] == "OK"
            for item in result.trace
        )
        assert all(card.price_usd <= Decimal(40) for card in result.verified)


def test_team_artifact_budget_edit_comparison_and_reviews(team_catalog):
    with closing(Store(":memory:")) as store:
        agent = Agent(store, team_catalog)
        persona = "accessories-tiger"
        initial = agent.submit("s", "1", "Find electronics under $40", persona=persona)
        assert [(card.id, card.price_usd) for card in initial.verified] == [
            ("B0BVPT6QJ4", Decimal("6.99")),
            ("B0B9L1N68J", Decimal("12.95")),
            ("B0BQRHFQJJ", Decimal("16.98")),
        ]
        updated = agent.submit("s", "2", "Change the budget to $15", persona=persona)
        assert initial.verified[2].id not in updated.shown_ids()
        assert updated.revision > initial.revision
        assert Decimal(store.get_state("s")["max_price_usd"]) == Decimal(15)
        assert all(card.price_usd <= Decimal(15) for card in updated.verified)

        compared = agent.submit("s", "3", "Compare first and second", persona=persona)
        assert compared.shown_ids() == updated.shown_ids()[:2]
        assert compared.price_difference_usd == Decimal("5.96")

        reviewed = agent.submit("s", "4", "Reviews for the first item", persona=persona)
        assert reviewed.shown_ids() == compared.shown_ids()[:1]
        assert reviewed.outcome == "COMPLETE"
        assert {quote.product_id for quote in reviewed.evidence} == {"B0BVPT6QJ4"}
        assert any(
            item["action"] == "get_product_evidence" and item["status"] == "OK"
            for item in reviewed.trace
        )


def test_streamlit_opens_team_catalog_and_applies_selected_persona(
    team_root, monkeypatch, tmp_path
):
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("SHOPPING_AGENT_MODE", "offline")
    monkeypatch.setenv("SHOPPING_AGENT_RECOMMENDER", str(team_root))
    monkeypatch.setenv("SHOPPING_AGENT_DB", str(tmp_path / "web.sqlite"))
    monkeypatch.delenv("SHOPPING_AGENT_SNAPSHOT", raising=False)
    monkeypatch.delenv("SHOPPING_AGENT_LIVE_MODEL", raising=False)
    monkeypatch.delenv("SHOPPING_AGENT_OLLAMA_MODEL", raising=False)

    app = streamlit_testing.AppTest.from_file(
        PROJECT_ROOT / "shopping_agent" / "ui.py", default_timeout=25
    ).run()
    assert not app.exception
    assert not app.error
    assert any("team-electronics" in caption.value for caption in app.caption)

    app.selectbox[0].set_value("accessories-tiger")
    app.chat_input[0].set_value("find electronics under $40").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "EXAMPLE_HISTORY_USED" in app.chat_message[1].text[0].value
    assert "B0BVPT6QJ4" in app.chat_message[1].text[0].value


def test_electronics_review_sample_preserves_source_and_product_scope(team_catalog):
    first = "B08XPWDSWW"
    outside_sample = "B07NY21DF6"
    observed = team_catalog.call(
        "get_product_evidence", {"ids": [first, outside_sample]}
    )
    assert observed.status == "PARTIAL"
    assert observed.missing_ids == [outside_sample]
    assert observed.review_gaps == {outside_sample: "OUTSIDE_SAMPLE"}
    assert len(observed.evidence) == 4
    for excerpt in observed.evidence:
        assert excerpt.product_id == first
        assert excerpt.source_type == "review"
        assert excerpt.id.startswith("ar23-electronics-")
        assert (
            "mcauleylab.ucsd.edu" in excerpt.source and "JSONL line" in excerpt.source
        )
    aspect = team_catalog.call(
        "get_product_evidence", {"ids": [first], "query": "waterproofing"}
    )
    assert aspect.status == "EMPTY"
    assert aspect.review_gaps == {first: "NO_ASPECT_MATCH"}

    with closing(Store(":memory:")) as store:
        agent = Agent(store, team_catalog)
        agent.submit("s", "1", "Find electronics under $40", persona="accessories-ncf")
        turn = agent.submit(
            "s", "2", "Any reviews for the first one?", persona="accessories-ncf"
        )
    assert turn.outcome == "COMPLETE"
    assert {quote.product_id for quote in turn.evidence} == {first}
    assert all(quote.snapshot == team_catalog.snapshot for quote in turn.evidence)
