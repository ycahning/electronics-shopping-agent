"""Catalog contract and failure regressions."""

import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from shopping_agent.catalog import (
    CATEGORIES,
    CatalogResponse,
    FixtureCatalog,
    SnapshotCatalog,
)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = FixtureCatalog()

    def test_fixture_categories_have_under_over_and_missing_prices(self):
        self.assertEqual(self.catalog.mode, "fixture")
        for category in CATEGORIES:
            with self.subTest(category=category):
                ids = self.catalog.call("search_products", {"category": category}).ids
                response = self.catalog.call("get_products", {"ids": ids})
                self.assertEqual(response.status, "OK")
                self.assertEqual(response.snapshot, "fixture-v1")
                self.assertTrue(all(p.category == category for p in response.products))
                prices = [p.price for p in response.products]
                self.assertIn(None, prices)
                known = [price for price in prices if price is not None]
                self.assertLessEqual(min(known), Decimal("40.00"))
                self.assertGreater(max(known), Decimal("40.00"))

    def test_search_scope_is_bounded_and_empty_is_not_error(self):
        first = self.catalog.call(
            "search_products",
            {"category": "Appliances", "query": "small appliance", "limit": 1},
        )
        no_match = self.catalog.call(
            "search_products",
            {"category": "Appliances", "query": "no-match-xyz"},
        )
        unsupported = self.catalog.call(
            "search_products",
            {"category": "Appliances", "live_stock": True},
        )
        self.assertEqual(first.ids, ["app-kettle"])
        self.assertTrue(first.has_more)
        self.assertEqual(
            (no_match.status, no_match.ids, no_match.has_more), ("EMPTY", [], False)
        )
        self.assertEqual(unsupported.status, "ERROR")

    def test_partial_details_keep_usable_record_and_unknown_price(self):
        result = self.catalog.call("get_products", {"ids": ["app-grinder", "absent"]})
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.missing_ids, ["absent"])
        self.assertEqual([item.id for item in result.products], ["app-grinder"])
        self.assertIsNone(result.products[0].price)

    def test_evidence_remains_product_bound_and_tracks_missing_coverage(self):
        full = self.catalog.call(
            "get_product_evidence",
            {"ids": ["app-kettle", "app-toaster"], "query": "noise"},
        )
        partial = self.catalog.call(
            "get_product_evidence",
            {"ids": ["app-kettle", "app-grinder"], "query": "noise"},
        )
        self.assertEqual(full.status, "OK")
        self.assertEqual(
            {quote.product_id for quote in full.evidence}, {"app-kettle", "app-toaster"}
        )
        self.assertEqual(partial.status, "PARTIAL")
        self.assertEqual(partial.missing_ids, ["app-grinder"])
        self.assertEqual(
            {quote.product_id for quote in partial.evidence}, {"app-kettle"}
        )

    def test_persona_returns_ranked_candidates_not_budget_checks(self):
        args = {"category": "Appliances", "limit": 2}
        example = self.catalog.call(
            "recommend_products", {**args, "persona": "example-shopper"}
        )
        invalid = self.catalog.call(
            "recommend_products", {**args, "persona": "unknown-shopper"}
        )
        # Candidates remain ranked before the Agent verifies the budget.
        self.assertEqual(example.ids, ["app-blender", "app-kettle"])
        self.assertEqual(invalid.error_code, "ineligible_persona")

    def test_wrong_snapshot_and_local_deadline_fail_without_usable_data(self):
        mismatch = self.catalog.call(
            "get_products",
            {"ids": ["app-kettle"], "snapshot": "wrong-snapshot"},
        )
        with patch("shopping_agent.catalog.monotonic", side_effect=[10.0, 12.0]):
            timed_out = self.catalog.call(
                "search_products", {"category": "Appliances"}, timeout=1
            )
        self.assertEqual(mismatch.error_code, "snapshot_mismatch")
        self.assertEqual(
            (timed_out.status, timed_out.error_code, timed_out.ids),
            ("ERROR", "timeout", []),
        )
        with self.assertRaises(ValidationError):
            CatalogResponse(status="EMPTY", snapshot="fixture-v1", ids=["app-kettle"])


class SnapshotTests(unittest.TestCase):
    def test_export_requires_provenance_and_valid_relationships(self):
        payload = {
            "manifest": {
                "snapshot": "team-v1",
                "origin": "team-export",
                "categories": ["Appliances"],
                "description": "Synthetic stand-in used only to exercise the export loader",
                "source_urls": ["https://example.org/catalog"],
            },
            "products": [
                {
                    "id": "item-1",
                    "category": "Appliances",
                    "title": "Kettle",
                    "price": 12.01,
                }
            ],
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "export.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            snapshot = SnapshotCatalog(path)
            response = snapshot.call("get_products", {"ids": ["item-1"]})
            self.assertEqual(snapshot.mode, "external-export")
            self.assertEqual(response.products[0].price, Decimal("12.01"))

            payload["manifest"]["source_urls"] = []
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source URLs"):
                SnapshotCatalog(path)

            payload["manifest"]["source_urls"] = ["https://example.org/catalog"]
            payload["evidence"] = [
                {
                    "id": "e-1",
                    "product_id": "different-product",
                    "text": "Unbound claim",
                    "source": "Test",
                },
            ]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValidationError, "Evidence refers to a product absent"
            ):
                SnapshotCatalog(path)
