"""Replay goal-loop regressions through Streamlit, local Ollama and the catalog."""

import argparse
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from integration.verify_agent_loop import (
    BN_LINK,
    DEFAULT_SAMPLE,
    RING,
    ROOT,
    Checks,
    action_names,
    digest,
    write_report,
)

ORDINARY_QUERY = "Hello, find wireless earbuds under $40."
COMPOUND_QUERY = (
    "Find two smart plug options under $40. Then check sampled reviews for underwater "
    "use on both. If the sample lacks useful evidence, choose the "
    "cheaper of the two based on catalog price. Show both options and explain your choice."
)
EMPTY_QUERY = "Search: xzyqmissingmodel, budget $40."


class GoalChecks(Checks):
    """Check user-visible obligations rather than requiring one valid action order."""

    def profile(self, name, model="ncf", *, dataset_id=None):
        app = self.app()
        app.selectbox(key="history_profile").set_value(name).run()
        if dataset_id:
            app.text_input(key="history_user_id").set_value(dataset_id).run()
        if name != "No history":
            app.selectbox(key="ranking_model").set_value(model).run()
        return app

    def catalog_invariance(self):
        lists = []
        for name, model, dataset_id in (
            ("No history", "ncf", None),
            ("Example user A", "ncf", None),
            ("Example user B", "tiger", None),
            ("Dataset user ID", "ncf", "DSC200UNKNOWN9999"),
        ):
            turn = self.turn(
                self.profile(name, model, dataset_id=dataset_id), ORDINARY_QUERY
            )
            state = turn["diagnostics"]
            self.within_budget(turn, Decimal(40), minimum=2)
            self.check(
                "search_products" in action_names(turn)
                and "recommend_products" not in action_names(turn)
                and state["ranking_source"] is None,
                f"{name} ordinary query used catalog and no history checkpoint",
                "condition",
            )
            lists.append(
                (
                    (state["search_query"] or "").casefold(),
                    list(state["verified_prices_usd"]),
                )
            )
        self.check(
            all(record == lists[0] for record in lists),
            "Same plain search keeps the same query and ordered verified products across profiles",
            "fact",
        )

    def budget_language(self):
        app = self.profile("Example user A")
        first = self.turn(app, ORDINARY_QUERY)
        self.within_budget(first, Decimal(40))
        query = first["diagnostics"]["search_query"]
        for prompt, expected in (
            ("Could we change the price limitation to $30?", Decimal(30)),
            ("Actually, set the same search limit to $28.", Decimal(28)),
        ):
            changed = self.turn(app, prompt)
            self.within_budget(changed, expected)
            self.check(
                changed["diagnostics"]["search_query"] == query
                and "recommend_products" not in action_names(changed),
                "Budget-only edit preserves the search and does not trigger personalization",
                "condition",
            )
        observed = self.turn(
            app, "I saw one for $23 last week. Keep my current budget."
        )
        self.check(
            Decimal(observed["diagnostics"]["budget_usd"]) == 28,
            "A quoted historical price does not replace the accepted budget",
            "fact",
        )
        ambiguous = self.turn(app, "Set my price limitation to either $25 or $27.")
        self.check(
            Decimal(ambiguous["diagnostics"]["budget_usd"]) == 28
            and ambiguous["diagnostics"]["outcome"] == "CLARIFICATION_NEEDED",
            "Conflicting ceilings prompt clarification and keep the existing value",
            "fact",
        )

    def explicitly_personalized(self):
        turn = self.turn(
            self.profile("Example user A"),
            "Find TOZO T10 earbuds under $40. Search the catalog and use my purchase "
            "history to prioritize matching recommendations.",
        )
        self.within_budget(turn, Decimal(40))
        state = turn["diagnostics"]
        item = "B08XPWDSWW"
        self.check(
            {"search_products", "recommend_products"} <= set(action_names(turn))
            and state["requested_model"] == "ncf"
            and state["ranking_source"]
            in {"checkpoint_inference", "version_checked_cache"},
            "Explicit history preference consults NCF and the catalog",
        )
        self.check(
            state["verified_prices_usd"].get(item) == "21.99"
            and state["discovery_cards"].get(item, {}).get("origin") == "checkpoint",
            "A real history candidate is independently matched and published with its source",
            "fact",
        )
        self.check(
            any(
                row["action"] == "publish_answer" and row["status"] == "VERIFIED"
                for row in state["trace"]
            ),
            "The model completes the combined request with a verified finish",
            "condition",
        )

    def ordinary_comparison(self):
        turn = self.turn(
            self.profile("Example user B", "tiger"),
            f"Compare the catalog price of ASIN {RING} and {BN_LINK}. Which is cheaper?",
        )
        state = turn["diagnostics"]
        self.check(
            set(state["selected_product_ids"]) == {RING, BN_LINK}
            and state["decision"] is not None
            and state["decision"]["basis"] == "price"
            and state["decision"]["product_id"] == BN_LINK
            and state["price_difference_usd"] == "5.00",
            "Exact-ASIN comparison uses both catalog prices and chooses the verified lower price",
            "fact",
        )
        self.check(
            "recommend_products" not in action_names(turn)
            and "get_product_evidence" not in action_names(turn),
            "A catalog-price comparison needs neither history nor reviews",
            "condition",
        )

    def compound(self):
        turn = self.turn(self.app(), COMPOUND_QUERY)
        state = turn["diagnostics"]
        self.within_budget(turn, Decimal(40), minimum=2)
        prices = state["verified_prices_usd"]
        review_calls = [
            row for row in state["trace"] if row["action"] == "get_product_evidence"
        ]
        reviewed = {
            asin
            for row in review_calls
            for asin in row.get("arguments", {}).get("ids", [])
        }
        self.check(
            "search_products" in action_names(turn)
            and len(prices) == 2
            and reviewed == set(prices),
            "Review lookup covers both verified candidates inside the same search turn",
            "condition",
        )
        observed = self.seen(turn, ("get_product_evidence", *sorted(reviewed)))
        self.check(
            any(
                (call.get("response") or {}).get("action") == "finish"
                and (call.get("response") or {}).get("basis") in {"price", "reviews"}
                and any(
                    row["action"] == "get_product_evidence"
                    for row in call["context"]["TOOL_DATA"]["tool_steps"]
                )
                for call in observed
            ),
            "Actual Ollama observed the review results and proposed a decision basis",
        )
        accepted = next(
            (
                row
                for row in reversed(state["trace"])
                if row["action"] == "publish_answer"
            ),
            {},
        )
        decision = state["decision"] or accepted
        winner = decision.get("product_id")
        self.check(
            winner in reviewed and winner in turn["reply"],
            "The result publishes a choice among the scoped candidates",
            "condition",
        )
        if decision.get("basis") == "price":
            self.check(
                winner in prices
                and Decimal(prices[winner])
                == min(Decimal(prices[asin]) for asin in reviewed)
                and all(
                    row["status"] in {"OK", "PARTIAL", "EMPTY"} for row in review_calls
                ),
                "Price fallback follows a completed review check and uses exact catalog amounts",
                "fact",
            )
        else:
            self.check(
                decision.get("basis") == "reviews"
                and any(
                    citation.get("product_id") == winner
                    for citation in accepted.get("citations", [])
                ),
                "A review-led choice uses an attributed citation for the selected ASIN",
                "fact",
            )

    def empty_vs_outage(self):
        from shopping_agent.catalog import CatalogResponse
        from shopping_agent.electronics import ElectronicsCatalog

        normal = self.turn(self.app(), EMPTY_QUERY)
        self.check(
            normal["diagnostics"]["outcome"] == "NO_VERIFIED_MATCH_IN_SCOPE"
            and any(
                row["action"] == "search_products" and row["status"] == "EMPTY"
                for row in normal["diagnostics"]["trace"]
            ),
            "An actual empty response permits a scoped no-match result",
            "fact",
        )
        original = ElectronicsCatalog.call

        def unavailable(catalog, action, arguments, timeout):
            if action == "search_products":
                return CatalogResponse(
                    status="ERROR",
                    snapshot=catalog.snapshot,
                    error_code="synthetic-outage",
                    message="Catalog search is temporarily unavailable in this test.",
                )
            return original(catalog, action, arguments, timeout)

        with patch.object(ElectronicsCatalog, "call", unavailable):
            broken = self.turn(self.app(), EMPTY_QUERY)
        state = broken["diagnostics"]
        self.check(
            state["outcome"] == "TEMPORARILY_UNAVAILABLE"
            and not any(
                row["action"] == "publish_answer" and row.get("basis") == "no_match"
                for row in state["trace"]
            )
            and any(
                row.get("error_code") == "synthetic-outage" for row in state["trace"]
            ),
            "An explicitly injected catalog outage is reported as unavailable, never no-match",
            "fact",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recommender", required=True, type=lambda v: Path(v).resolve()
    )
    parser.add_argument(
        "--recommender-python", required=True, type=lambda v: Path(v).absolute()
    )
    parser.add_argument("--ollama-model", default="qwen3:4b-q4_K_M")
    parser.add_argument("--review-snapshot", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", help="Comma-separated case IDs")
    args = parser.parse_args()
    args.legacy_root = args.legacy_policy = None

    from shopping_agent.ollama import DEFAULT_OLLAMA_URL, OllamaProvider

    model = OllamaProvider(
        args.ollama_model,
        base_url=os.getenv("SHOPPING_AGENT_OLLAMA_URL", DEFAULT_OLLAMA_URL),
    )
    tags = model._request("GET", "/api/tags", 5).get("models", [])
    model_digest = next(
        (tag.get("digest") for tag in tags if tag.get("name") == args.ollama_model),
        None,
    )
    checks = GoalChecks(args)
    cases = {
        "catalog-profile-invariance": checks.catalog_invariance,
        "budget-language": checks.budget_language,
        "explicit-personalization": checks.explicitly_personalized,
        "ordinary-comparison": checks.ordinary_comparison,
        "compound-discovery-evidence-choice": checks.compound,
        "empty-vs-outage": checks.empty_vs_outage,
        "controlled-observation-ab": checks.controlled,
        "direct-and-resume": checks.greetings,
        "exact-asin-conditional-price": checks.prices_after_reviews,
        "rating-fallback": checks.alternate_rating_fallback,
        "quoted-review-decision": checks.real_setup_review,
        "original-ordinal-replay": checks.ordinal_flow,
        "alternate-ordinal-replay": lambda: checks.ordinal_flow(variant=True),
    }
    selected = set(args.only.split(",")) if args.only else set(cases)
    if selected - cases.keys():
        parser.error(f"Unknown case IDs: {', '.join(sorted(selected - cases.keys()))}")
    report = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "runner": "Streamlit AppTest with actual local Ollama and Electronics catalog",
        "controlled_fixtures": (
            "The outage case injects a synthetic catalog error. The paired observation case "
            "replaces the first review response with labeled synthetic excerpts."
        ),
        "ollama_model": args.ollama_model,
        "ollama_model_digest": model_digest,
        "catalog_sha256": digest(args.recommender / "data/item_catalog.csv"),
        "review_index_sha256": digest(args.review_snapshot),
        "recommender_asset_sha256": {
            name: digest(args.recommender / name)
            for name in (
                "recommend.py",
                "data/split.npz",
                "data/semantic_ids.npy",
                "models/ncf_mlp_best.pt",
                "models/tiger_best.pt",
            )
        },
        "source_sha256": {
            path.name: digest(path)
            for path in sorted((ROOT / "shopping_agent").glob("*.py"))
        },
        "harness_sha256": digest(Path(__file__)),
    }

    def save(results, active_case=None):
        measured = {case["id"] for case in results}
        report.update(
            cases=results,
            active_case=active_case,
            run_complete=measured == selected,
            skipped_cases=[
                {
                    "id": case,
                    "status": "skipped",
                    "reason": "Awaiting execution"
                    if case in selected
                    else "Excluded by --only",
                }
                for case in cases
                if case not in measured
            ],
        )
        write_report(args.output, report)

    with (
        TemporaryDirectory(prefix="goal-loop-acceptance-") as directory,
        patch.dict(
            os.environ,
            {
                "SHOPPING_AGENT_MODE": "final",
                "SHOPPING_AGENT_DB": f"{directory}/acceptance.sqlite3",
                "SHOPPING_AGENT_RECOMMENDER": str(args.recommender),
                "SHOPPING_AGENT_RECOMMENDER_PYTHON": str(args.recommender_python),
                "SHOPPING_AGENT_OLLAMA_MODEL": args.ollama_model,
                "SHOPPING_AGENT_REVIEW_SNAPSHOT": str(args.review_snapshot.resolve()),
                "SHOPPING_AGENT_LIVE_MODEL": "",
            },
        ),
    ):
        save([])
        results = checks.run(
            [(name, case) for name, case in cases.items() if name in selected], save
        )
    return int(any(case["status"] != "passed" for case in results))


if __name__ == "__main__":
    raise SystemExit(main())
