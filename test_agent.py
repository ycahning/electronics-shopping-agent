"""Agent end-to-end regressions with read-only synthetic adapters; no API calls."""

import json
import subprocess
import sys
import unittest
from decimal import Decimal
from pathlib import Path

from shopping_agent.catalog import CATEGORIES, CatalogResponse, Evidence, FixtureCatalog
from shopping_agent.provider import ProviderError
from shopping_agent.service import Agent
from shopping_agent.storage import Store


class RecordingCatalog(FixtureCatalog):
    def __init__(self):
        super().__init__()
        self.calls = []

    def call(self, action, args, timeout):
        self.calls.append((action, args.copy()))
        return super().call(action, args, timeout)


class ScriptedProvider:
    name = "scripted-test-provider"
    last_usage = None

    def __init__(self, actions):
        self.actions = iter(actions)

    def parse(self, text, state, timeout):
        # The conservative application parser supplies the explicit budget and category.
        return {}

    def decide(self, context, timeout):
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        return action


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.catalog = RecordingCatalog()
        self.agent = Agent(self.store, self.catalog)

    def test_four_category_discovery_and_followup_on_synthetic_data(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                initial = self.agent.submit(category, "1", f"find {category} under $40")
                self.assertEqual(initial.data_mode, "fixture")
                self.assertEqual(initial.snapshot, "fixture-v1")
                self.assertTrue(initial.verified)
                self.assertTrue(initial.pending)
                self.assertTrue(
                    all(
                        item.category == category and item.price_usd <= Decimal(40)
                        for item in initial.verified
                    )
                )
                self.assertTrue(
                    all(item.price_status == "UNKNOWN" for item in initial.pending)
                )
                update = self.agent.submit(category, "2", "budget to 30")
                self.assertGreater(update.revision, initial.revision)
                self.assertEqual(
                    Decimal(self.store.get_state(category)["max_price_usd"]), 30
                )
                self.assertTrue(
                    all(item.price_usd <= Decimal(30) for item in update.verified)
                )

    def test_old_display_cannot_replay_after_new_search_same_revision(self):
        first = self.agent.submit("s", "old", "find appliances under $60")
        newer = self.agent.submit("s", "new", "search: vacuum")
        self.assertEqual(first.revision, newer.revision)
        self.assertNotEqual(first.verified[0].id, newer.verified[0].id)
        before = len(self.catalog.calls)
        replay = self.agent.submit("s", "old", "find appliances under $60")
        self.assertEqual(replay.outcome, "TEMPORARILY_UNAVAILABLE")
        self.assertFalse(replay.verified)
        self.assertEqual(len(self.catalog.calls), before)
        comparison = self.agent.submit("s", "compare", "compare first")
        self.assertEqual(comparison.comparison[0].id, newer.verified[0].id)

    def test_legacy_session_recovers_discovery_after_an_old_detail_overwrote_it(self):
        initial = self.agent.submit("s", "initial", "find appliances under $40")
        originals = initial.shown_ids()
        self.store.begin("s", "legacy-review", "reviews for first")
        state = self.store.get_state("s")
        state.pop("followup")
        self.store.commit("s", "legacy-review", state)
        self.store.finish(
            "s",
            "legacy-review",
            initial.revision,
            {"intent": "evidence"},
            originals[:1],
        )
        assert self.store.get_state("s")["displayed_ids"] == originals[:1]
        result = self.agent.submit("s", "fixed-review", "reviews for second")
        assert result.shown_ids() == originals[1:2]
        assert self.store.get_state("s")["displayed_ids"] == originals

    def test_comparison_references_display_order_and_binds_quotes(self):
        discovery = self.agent.submit("s", "1", "find appliances under $40")
        shown = [card.id for card in discovery.verified[:2]]
        result = self.agent.submit("s", "2", "compare second and first noise")
        self.assertEqual([card.id for card in result.comparison], shown[::-1])
        self.assertEqual(result.price_difference_usd, Decimal("10.00"))
        self.assertEqual({quote.product_id for quote in result.evidence}, set(shown))
        self.assertTrue(
            all(quote.snapshot == result.snapshot for quote in result.evidence)
        )
        other = self.agent.submit("other", "1", "compare second and first")
        self.assertEqual(other.outcome, "CLARIFICATION_NEEDED")
        self.assertEqual(other.comparison, [])

    def test_unsupported_hard_requirement_is_not_unknown_price(self):
        result = self.agent.submit("s", "1", "find appliances under $40 in stock")
        self.assertEqual(result.outcome, "PARTIAL")
        self.assertEqual(result.verified, [])
        statuses = {item.id: item.price_status for item in result.pending}
        self.assertEqual(statuses["app-kettle"], "SATISFIED")
        self.assertEqual(statuses["app-grinder"], "UNKNOWN")
        self.assertNotIn("app-blender", statuses)
        self.assertIn("live stock", " ".join(result.limitations))

    def test_accepted_budget_persists_when_search_times_out(self):
        found = self.agent.submit("s", "1", "find appliances under $60")
        store = self.store
        observed_budgets = []

        class TimeoutCatalog(RecordingCatalog):
            def call(self, action, args, timeout):
                if action == "search_products":
                    observed_budgets.append(
                        Decimal(store.get_state("s")["max_price_usd"])
                    )
                    raise TimeoutError("Injected timeout")
                return super().call(action, args, timeout)

        result = Agent(store, TimeoutCatalog()).submit("s", "2", "budget to 40")
        self.assertEqual(result.outcome, "TEMPORARILY_UNAVAILABLE")
        self.assertEqual(observed_budgets, [Decimal(40), Decimal(40)])
        self.assertEqual(Decimal(store.get_state("s")["max_price_usd"]), 40)
        self.assertEqual(store.get_state("s")["displayed_ids"], found.shown_ids())
        self.assertEqual(result.verified, [])
        self.assertEqual(result.tool_attempts, 2)

    def test_duplicate_reuses_current_result_but_not_stale_recommendations(self):
        text = "find appliances under $60"
        first = self.agent.submit("s", "1", text)
        calls = list(self.catalog.calls)
        self.assertEqual(self.agent.submit("s", "1", text), first)
        self.assertEqual(self.catalog.calls, calls)

        new = self.agent.submit("s", "2", "budget to 40")
        calls_after_update = list(self.catalog.calls)
        displayed_after_update = self.store.get_state("s")["displayed_ids"]
        stale = self.agent.submit("s", "1", text)
        self.assertEqual(stale.revision, new.revision)
        self.assertEqual(stale.outcome, "TEMPORARILY_UNAVAILABLE")
        self.assertEqual(stale.verified, [])
        self.assertEqual(self.catalog.calls, calls_after_update)
        self.assertEqual(
            self.store.get_state("s")["displayed_ids"], displayed_after_update
        )

    def test_example_history_used_but_cannot_override_budget(self):
        result = self.agent.submit(
            "persona", "1", "find appliances under $40", persona="example-shopper"
        )
        self.assertEqual(result.personalization, "EXAMPLE_HISTORY_USED")
        self.assertIn(
            "recommend_products", [action for action, _ in self.catalog.calls]
        )
        # The fixture ranks the blender first, but it costs more than the accepted budget.
        self.assertNotIn("app-blender", [item.id for item in result.verified])
        self.assertEqual(result.verified[0].id, "app-kettle")

    def test_simulated_adaptive_policy_broadens_after_inadequate_observations(self):
        class ObservingProvider(ScriptedProvider):
            def decide(self, context, timeout):
                category = context["requirements"]["category"]
                query = context["query"]
                search_calls = [
                    step
                    for step in context["trace"]
                    if step["action"] == "search_products"
                ]
                if not search_calls:
                    return {
                        "action": "search_products",
                        "args": {"category": category, "query": query, "limit": 2},
                    }
                observed = {product["id"] for product in context["products"]}
                unread = [
                    item for item in context["candidate_ids"] if item not in observed
                ]
                if unread:
                    return {"action": "get_products", "args": {"ids": unread}}
                if not context["verified_ids"] and len(search_calls) == 1:
                    # Both observed prices exceeded the budget; widen the candidate pool.
                    return {
                        "action": "search_products",
                        "args": {"category": category, "query": query, "limit": 25},
                    }
                return {"action": "finish_decision"}

        result = Agent(self.store, self.catalog, provider=ObservingProvider([])).submit(
            "s", "1", "find appliances under $25"
        )
        self.assertEqual(
            [action for action, _ in self.catalog.calls],
            ["search_products", "get_products", "search_products", "get_products"],
        )
        self.assertEqual(self.catalog.calls[1][1]["ids"], ["app-kettle", "app-toaster"])
        self.assertEqual(self.catalog.calls[2][1]["limit"], 25)
        self.assertEqual([product.id for product in result.verified], ["app-mini-vac"])

    def test_unknown_model_proposed_id_never_reaches_the_tool(self):
        provider = ScriptedProvider(
            [
                {
                    "action": "search_products",
                    "args": {"category": "Appliances", "query": "", "limit": 10},
                },
                {"action": "get_products", "args": {"ids": ["unknown-id"]}},
                {"action": "get_products", "args": {"ids": ["unknown-id"]}},
            ]
        )
        result = Agent(self.store, self.catalog, provider=provider).submit(
            "s", "1", "find appliances under $40"
        )
        self.assertEqual(
            [action for action, _ in self.catalog.calls], ["search_products"]
        )
        self.assertEqual(result.verified, [])
        self.assertIn("outside the allowed scope", " ".join(result.limitations))
        self.assertEqual(
            result.model_attempts, 4
        )  # parser + search + two rejected proposals

    def test_incomplete_model_output_keeps_accepted_budget_and_stops(self):
        provider = ScriptedProvider(
            [ProviderError("MODEL_INCOMPLETE", "Injected incomplete model response")]
            * 2
        )
        result = Agent(self.store, self.catalog, provider=provider).submit(
            "s", "1", "find appliances under $40"
        )
        self.assertEqual(result.outcome, "TEMPORARILY_UNAVAILABLE")
        self.assertEqual(Decimal(self.store.get_state("s")["max_price_usd"]), 40)
        self.assertEqual(result.model_attempts, 3)
        self.assertEqual(self.catalog.calls, [])
        self.assertIn(
            "INVALID_OR_UNAVAILABLE", [item["status"] for item in result.trace]
        )

    def test_catalog_details_from_wrong_snapshot_are_not_published(self):
        class WrongSnapshotCatalog(RecordingCatalog):
            def call(self, action, args, timeout):
                result = super().call(action, args, timeout)
                if action == "get_products":
                    return result.model_copy(
                        update={"snapshot": "unrequested-snapshot"}
                    )
                return result

        result = Agent(self.store, WrongSnapshotCatalog()).submit(
            "s", "1", "find appliances under $40"
        )
        self.assertEqual(result.outcome, "TEMPORARILY_UNAVAILABLE")
        self.assertEqual(result.verified, [])
        self.assertEqual(result.pending, [])
        self.assertIn("different catalog snapshot", " ".join(result.limitations))

    def test_wrong_product_evidence_cannot_enter_comparison(self):
        class WrongEvidenceCatalog(RecordingCatalog):
            def call(self, action, args, timeout):
                if action == "get_product_evidence":
                    return CatalogResponse(
                        status="OK",
                        snapshot=self.snapshot,
                        evidence=[
                            Evidence(
                                id="misbound-evidence",
                                product_id="app-blender",
                                text="Unrelated synthetic excerpt.",
                                source="Test fault injection",
                            )
                        ],
                    )
                return super().call(action, args, timeout)

        agent = Agent(self.store, WrongEvidenceCatalog())
        agent.submit("s", "1", "find appliances under $40")
        result = agent.submit("s", "2", "compare first and second noise")
        self.assertEqual(result.outcome, "PARTIAL")
        self.assertEqual(
            [card.id for card in result.comparison], ["app-kettle", "app-toaster"]
        )
        self.assertEqual(result.evidence, [])
        self.assertIn("unrelated product", " ".join(result.limitations))


class CLITests(unittest.TestCase):
    def test_json_entry_point_is_offline_and_labels_fixture(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "shopping_agent",
                "--fixture",
                "--db",
                ":memory:",
                "--json",
                "find appliances under $40",
            ],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = json.loads(completed.stdout)
        self.assertEqual(output["data_mode"], "fixture")
        self.assertEqual(output["provider"], "offline")
        self.assertEqual(output["snapshot"], "fixture-v1")
        self.assertEqual(output["model_attempts"], 0)
        self.assertTrue(output["verified"])
