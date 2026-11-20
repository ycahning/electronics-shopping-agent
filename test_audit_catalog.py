"""Independent checks for paging, catalog facts and review aspect matching."""

import csv
import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from integration.export_electronics_reviews import START_MS, export_archive
from shopping_agent.electronics import ElectronicsCatalog

TEAM_DIR = (
    Path(__file__).resolve().parent.parent
    / "teammate_code"
    / "Recommender_Project-main"
)


@pytest.fixture
def prices_catalog(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    catalog = data / "item_catalog.csv"
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("parent_asin", "title", "store", "price", "category", "rating_number")
        )
        writer.writerows(
            (f"pricy-{i}", "HDMI Cable", "Store", "6.10", "Cables", 100 + i)
            for i in range(30)
        )
        writer.writerows(
            (
                ("eligible-a", "HDMI Cable", "Store", "04.19", "Cables", "22"),
                ("eligible-b", "HDMI Cable", "Store", "5.00", "Cables", "12"),
                ("eligible-c", "HDMI Cable", "Store", "5.01", "Cables", "12"),
                ("unknown", "HDMI Cable", "Store", "from 1.00", "Cables", "1000"),
                ("unrelated", "Desk Lamp", "Store", "3.90", "Lighting", "12"),
            )
        )
    rankings = tmp_path / "rankings.json"
    rankings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
                "model_sha256": {"ncf": "a" * 64, "tiger": "b" * 64},
                "rankings": {"test-ncf": ["eligible-b", "unrelated", "eligible-a"]},
            }
        )
    )
    return ElectronicsCatalog(tmp_path, rankings_path=rankings)


def test_budget_filter_runs_before_pagination_and_keeps_unknowns_separate(
    prices_catalog,
):
    args = {
        "query": "HDMI cable",
        "category": "Electronics",
        "max_price_usd": "5.00",
        "limit": 1,
    }
    page = prices_catalog.call("search_products", args)
    assert page.ids == ["eligible-a"]
    assert page.pending_ids == ["unknown"]
    assert page.has_more is True
    page = prices_catalog.call("search_products", {**args, "offset": 1})
    assert page.ids == ["eligible-b"]
    assert page.pending_ids == []
    assert page.has_more is False
    excluded = prices_catalog.call(
        "search_products", {**args, "exclude_ids": ["eligible-a"]}
    )
    assert excluded.ids == ["eligible-b"]
    assert excluded.pending_ids == ["unknown"]
    assert excluded.has_more is False


def test_ranked_candidate_scope_uses_own_matches_and_preserves_order(prices_catalog):
    args = {
        "query": "HDMI cable",
        "category": "Electronics",
        "max_price_usd": "5.00",
        "limit": 3,
    }
    results = prices_catalog.call(
        "search_products",
        {**args, "candidate_ids": ["eligible-b", "unrelated", "eligible-a"]},
    )
    assert results.ids == ["eligible-b", "eligible-a"]
    assert results.pending_ids == []
    assert results.has_more is False


@pytest.fixture(scope="module")
def actual_catalog():
    if not (TEAM_DIR / "data" / "item_catalog.csv").exists():
        pytest.skip(
            "The original Electronics catalog is required for data regression checks."
        )
    return ElectronicsCatalog(TEAM_DIR)


def test_low_budget_real_catalog_returns_known_hdmi_cables(actual_catalog):
    result = actual_catalog.call(
        "search_products",
        {
            "query": "HDMI cable",
            "category": "Electronics",
            "max_price_usd": "5",
            "limit": 25,
        },
    )
    assert result.ids
    with (TEAM_DIR / "data" / "item_catalog.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        source = {
            row["parent_asin"]: Decimal(row["price"])
            for row in csv.DictReader(handle)
            if row["parent_asin"] in result.ids
        }
    assert set(result.ids) == source.keys()
    assert all(price <= 5 for price in source.values())


def test_real_ranked_earbuds_are_checked_by_asin_not_against_search_page(
    actual_catalog,
):
    # The teammate NCF example's first product is independently identifiable from
    # its published ASIN, title and the historical price in item_catalog.csv.
    result = actual_catalog.call(
        "search_products",
        {
            "query": "wireless earbuds",
            "category": "Electronics",
            "max_price_usd": "40",
            "candidate_ids": ["B08XPWDSWW"],
        },
    )
    assert result.ids == ["B08XPWDSWW"]


def test_source_ratings_and_title_specs_are_exposed_without_inference(actual_catalog):
    result = actual_catalog.call(
        "get_products",
        {"ids": ["B08KSJ56WR", "B09TNV7Q69", "B0BPB5Q41C", "B0BBZZRL8G"]},
    )
    ring, bnlink, four_port, conflicting = result.products
    # Expected values were read directly from the original title/rating data.
    assert ring.average_rating == bnlink.average_rating == Decimal("4.5")
    assert (ring.rating_count, bnlink.rating_count) == (1570, 379)
    assert four_port.port_count == 4
    assert four_port.spec_source == "catalog_title"
    assert four_port.usb_versions == ("3.0",)
    assert set(conflicting.usb_versions) == {"2.0", "3.0"}


def test_named_lookup_prefers_matching_product_identity(actual_catalog):
    assert actual_catalog.resolve_product("Ring Outdoor Smart Plug") == ["B08KSJ56WR"]
    assert actual_catalog.resolve_product("Nonexistent Tricorder LQ-9080") == []
    sony = actual_catalog.resolve_product("Sony WH-1000XM5 headphones")
    assert len(sony) == 2
    titles = actual_catalog.call("get_products", {"ids": sony}).products
    assert all(card.title.startswith("Sony WH-1000XM5") for card in titles)


def test_manually_checked_review_passages_do_not_claim_waterproofing(actual_catalog):
    ids = ["B08KSJ56WR", "B09TNV7Q69"]
    positive = actual_catalog.call(
        "get_product_evidence", {"ids": ids, "query": "Christmas lights"}
    )
    # Source lines were checked as holiday-light usage in the cached corpus.
    assert [(entry.product_id, entry.id) for entry in positive.evidence] == [
        ("B08KSJ56WR", "ar23-electronics-89241"),
        ("B09TNV7Q69", "ar23-electronics-18458"),
    ]
    absent = actual_catalog.call(
        "get_product_evidence", {"ids": ids, "query": "waterproof Christmas lights"}
    )
    assert absent.status == "EMPTY"
    assert absent.evidence == []
    assert absent.review_gaps == dict.fromkeys(ids, "NO_ASPECT_MATCH")


def test_full_archive_export_retains_attributed_passages_and_bounded_response(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    catalog_path = data / "item_catalog.csv"
    catalog_path.write_text(
        "parent_asin,title,store,price,category,rating_number\n"
        "product-a,Outdoor plug,Store,4.00,Plugs,1\n",
        encoding="utf-8",
    )
    unrelated = {
        "parent_asin": "not-in-catalog",
        "timestamp": START_MS,
        "text": "I used these outside for Christmas lights.",
    }
    reviews = [
        {
            "parent_asin": "product-a",
            "timestamp": START_MS + i,
            "text": (
                "The shipping package arrived neatly packed. It held a box and a plastic "
                "bag with packing material and bubble wrap for all the items. "
                f"I used this with Christmas lights outside for {i} evenings."
            ),
            "user_id": "omitted-from-index",
        }
        for i in range(1, 7)
    ]
    raw = "\n".join(json.dumps(row) for row in [unrelated, *reviews]) + "\n"
    archive = tmp_path / "full.jsonl.gz"
    archive.write_bytes(gzip.compress(raw.encode(), mtime=0))
    exported = export_archive(archive, catalog_path)
    assert (
        exported["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert exported["complete_source_rows"] == 7
    assert set(exported["excerpts"]) == {"product-a"}
    assert exported["excerpts"]["product-a"][0] == [2, reviews[0]["text"]]
    assert "omitted-from-index" not in json.dumps(exported)

    reviews_path = tmp_path / "index.json.gz"
    reviews_path.write_bytes(gzip.compress(json.dumps(exported).encode(), mtime=0))
    rankings = tmp_path / "rankings.json"
    rankings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog_sha256": exported["catalog_sha256"],
                "model_sha256": {"ncf": "a" * 64, "tiger": "b" * 64},
                "rankings": {"test-ncf": ["product-a"]},
            }
        ),
        encoding="utf-8",
    )
    catalog = ElectronicsCatalog(
        tmp_path, rankings_path=rankings, reviews_path=reviews_path
    )
    result = catalog.call(
        "get_product_evidence", {"ids": ["product-a"], "query": "Christmas lights"}
    )
    assert [entry.id for entry in result.evidence] == [
        "ar23-electronics-2",
        "ar23-electronics-3",
        "ar23-electronics-4",
        "ar23-electronics-5",
    ]
    assert result.has_more is True
