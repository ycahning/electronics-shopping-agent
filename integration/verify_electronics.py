"""Run the Electronics acceptance suite against real snapshots and local Ollama."""

import argparse
import gzip
import json
import os
import socket
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch

from shopping_agent.electronics import ElectronicsCatalog

from .export_electronics_reviews import export as export_reviews

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "shopping_agent" / "fixtures"
FORZA = "B07NY21DF6"
FORZA_CARD = (
    "I mean do you have any comments for this one:\n"
    "2. Electronic Arts Forza Horizon 4 [B07NY21DF6] · USD 25.00"
)


def require(check, message):
    if not check:
        raise AssertionError(message)


def tool(turn, action, status="OK"):
    return any(
        event["action"] == action and event["status"] == status
        for event in turn["trace"]
    )


def verified_prices(turn, cap):
    require(Decimal(turn["budget_usd"]) == cap, "Wrong accepted budget")
    prices = {
        asin: Decimal(price) for asin, price in turn["verified_prices_usd"].items()
    }
    require(
        prices and max(prices.values()) <= cap, "Expected verified prices within cap"
    )
    require(
        not prices.keys() & set(turn["pending_product_ids"]),
        "Pending product was shown as price-verified",
    )
    return prices


class LiveChecks:
    """Drive the real Streamlit script with AppTest and its usual Ollama HTTP client."""

    def __init__(self, catalog, model):
        self.catalog = catalog
        self.model = model

    @staticmethod
    def open():
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(
            ROOT / "shopping_agent" / "ui.py", default_timeout=110
        ).run()

    def ready(self, app):
        require(not app.exception and not app.error, repr((app.exception, app.error)))
        require(len(app.chat_input) == 1, "The chat input is missing")
        require(
            any(
                f"Ollama · {self.model}" in caption.value
                and self.catalog.snapshot in caption.value
                for caption in app.caption
            ),
            "Wrong model or data on the webpage",
        )

    def turn(self, app, request):
        app.chat_input[0].set_value(request).run()
        self.ready(app)
        turn = app.session_state.shop_diagnostics
        require(turn["provider"] == f"ollama:{self.model}", "Ollama was not used")
        require(turn["snapshot"] == self.catalog.snapshot, "Wrong catalog snapshot")
        require(tool(turn, "model_parse"), "The live model did not parse this turn")
        return turn, app.chat_message[-1].text[0].value

    def check_reviews(self, turn, reply, ids):
        require(turn["selected_product_ids"] == ids, "Selected the wrong product")
        require(tool(turn, "get_products"), "The catalog did not verify the selection")
        with gzip.open(
            ASSETS / "electronics_reviews.json.gz", "rt", encoding="utf8"
        ) as file:
            index = json.load(file)["excerpts"]
        original_sources = {
            f"ar23-electronics-{line}": (asin, excerpt)
            for asin in ids
            for line, excerpt in index.get(asin, [])
        }
        shown_sources = turn["review_sources"]
        if not turn["review_aspect"]:
            require(
                {source["id"] for source in shown_sources} == original_sources.keys(),
                "General review citations differ from the saved source",
            )
        for source in shown_sources:
            asin, excerpt = original_sources[source["id"]]
            require(asin == source["product_id"], "Wrong review owner")
            require(
                excerpt in reply and source["source"] in reply,
                "Citation missing from reply",
            )

    def original_dialogue(self):
        app = self.open()
        self.ready(app)
        require(app.selectbox[0].value is None, "No history is not the default")
        hello, _ = self.turn(app, "Hi")
        require(
            hello["intent"] == "answer" and hello["tool_attempts"] == 0,
            "Greeting did not produce a direct reply",
        )
        found, _ = self.turn(app, "Give me any electronic under $40")
        ids = found["numbered_product_ids"]
        require(len(verified_prices(found, 40)) >= 3, "Expected three priced items")
        require(
            found["search_query"] == "" and tool(found, "search_products"),
            "Expected a category browse",
        )
        require(found["personalization"] == "NO_HISTORY", "Unexpected history context")
        require(
            not any(
                event["action"] == "recommend_products" for event in found["trace"]
            ),
            "No-history called the recommender",
        )

        looked_up = []
        for request, asin in [
            ("I need all reviews for first oe", ids[0]),
            ("OK, How about seconed one", ids[1]),
            (FORZA_CARD, FORZA),
            ("Same for the third one?", ids[2]),
        ]:
            turn, reply = self.turn(app, request)
            require(turn["intent"] == "evidence", "Review intent was lost")
            require(
                turn["numbered_product_ids"] == ids,
                "A review changed the active numbering",
            )
            self.check_reviews(turn, reply, [asin])
            if turn["review_gaps"].get(asin) == "OUTSIDE_SAMPLE":
                require(
                    "review sample has no excerpts" in reply, "Missing sample notice"
                )
            looked_up.append(
                {
                    "asin": asin,
                    "sources": turn["review_sources"],
                    "gaps": turn["review_gaps"],
                }
            )
        require(
            not any("historical result" in item.value for item in app.caption),
            "The current list was marked historical",
        )
        return {
            "numbered_ids": ids,
            "pending": found["pending_product_ids"],
            "reviews": looked_up,
        }

    def different_products(self):
        app = self.open()
        self.ready(app)
        found, _ = self.turn(app, "I need a USB hub for my desk, at most $22.")
        ids = found["numbered_product_ids"]
        verified_prices(found, 22)
        require(
            {"usb", "hub"} <= set(found["search_query"].lower().split()),
            "USB hub terms were lost",
        )
        ambiguous, reply = self.turn(app, "Any reviews for that one?")
        require(
            ambiguous["outcome"] == "CLARIFICATION_NEEDED"
            and ambiguous["tool_attempts"] == 0
            and ambiguous["numbered_product_ids"] == ids
            and reply.count("?") == 1,
            "An ambiguous reference should ask one question and preserve the list",
        )
        for request, asin in [
            ("Could you pull the customer feedback for option 1?", ids[0]),
            ("Same for the second item?", ids[1]),
        ]:
            turn, reply = self.turn(app, request)
            require(turn["intent"] == "evidence", "Follow-up lost the review intent")
            require(
                turn["numbered_product_ids"] == ids, "A follow-up reordered USB hubs"
            )
            self.check_reviews(turn, reply, [asin])
        newer, _ = self.turn(app, "I need wired headphones for commuting, at most $25.")
        verified_prices(newer, 25)
        require(
            newer["numbered_product_ids"] != ids,
            "New search did not publish a new list",
        )
        followup, reply = self.turn(app, "Show reviews for item 1.")
        self.check_reviews(followup, reply, newer["numbered_product_ids"][:1])
        return {"usb_hub_ids": ids, "headphone_ids": newer["numbered_product_ids"]}

    def budget_and_tool_choice(self):
        app = self.open()
        self.ready(app)
        found, _ = self.turn(app, "I need wired headphones for commuting, at most $25.")
        verified_prices(found, 25)
        updated, _ = self.turn(app, "Could we bring it down to $15?")
        prices = verified_prices(updated, 15)
        require(
            found["search_query"] == updated["search_query"],
            "A budget edit erased the keyword",
        )
        ids = updated["numbered_product_ids"]
        compared, _ = self.turn(app, "How much do the first two differ?")
        require(compared["selected_product_ids"] == ids[:2], "Wrong products compared")
        require(compared["numbered_product_ids"] == ids, "Comparison changed the list")
        gap = abs(prices[ids[0]] - prices[ids[1]])
        require(
            Decimal(compared["price_difference_usd"]) == gap, "Wrong price difference"
        )
        require(
            compared["decision"] is None,
            "A price difference became an unsolicited recommendation",
        )
        choices = [
            event["proposal"]
            for turn in (updated, compared)
            for index, event in enumerate(turn["trace"])
            if event["action"] == "model_decide"
            and event["status"] == "SELECTED"
            and tool({"trace": turn["trace"][:index]}, "get_products")
        ]
        require(choices, "No real model tool choice followed a catalog observation")
        return {
            "query": updated["search_query"],
            "price_gap_usd": str(gap),
            "model_choices": choices,
        }

    def decision_with_evidence(self):
        app = self.open()
        self.ready(app)
        found, _ = self.turn(app, "I need a smart plug, at most $40.")
        prices = verified_prices(found, 40)
        ids = found["numbered_product_ids"][:2]
        require(
            len(ids) == 2 and set(ids) <= prices.keys(),
            "Need two priced smart plug results",
        )
        decided, reply = self.turn(
            app,
            "Compare the first two and recommend the cheaper one. Include review excerpts.",
        )
        cheaper = min(ids, key=prices.__getitem__)
        require(
            decided["decision"]
            == {"basis": "price", "status": "LOWEST_PRICE", "product_id": cheaper}
            and "Price source" in reply,
            "The price decision did not cite the cheapest eligible product",
        )
        require(
            Decimal(decided["price_difference_usd"])
            == abs(prices[ids[0]] - prices[ids[1]]),
            "The comparison did not show the correct tradeoff",
        )
        self.check_reviews(decided, reply, ids)
        require(
            {review["product_id"] for review in decided["review_sources"]} == set(ids),
            "Expected review examples for both products in the price comparison",
        )
        insufficient, reply = self.turn(
            app,
            "Which of items 1 and 2 should I pick for waterproofing? Check buyer reviews about waterproofing.",
        )
        self.check_reviews(insufficient, reply, ids)
        require(
            insufficient["decision"]
            == {
                "basis": "review_aspect",
                "status": "INSUFFICIENT_REVIEW_EVIDENCE",
                "product_id": None,
            }
            and "need comparable evidence" in reply
            and insufficient["review_gaps"] == dict.fromkeys(ids, "NO_ASPECT_MATCH"),
            "The review-based choice should explain the missing waterproofing evidence",
        )
        require(
            decided["numbered_product_ids"]
            == insufficient["numbered_product_ids"]
            == found["numbered_product_ids"],
            "Decision changed item numbers",
        )
        return {
            "catalog_prices_usd": {asin: str(prices[asin]) for asin in ids},
            "price_decision": decided["decision"],
            "reviews": decided["review_sources"],
            "aspect": insufficient["review_aspect"],
            "aspect_decision": insufficient["decision"],
            "aspect_gaps": insufficient["review_gaps"],
        }

    def history(self, alias, rankings):
        app = self.open()
        self.ready(app)
        app.selectbox[0].set_value(alias)
        found, _ = self.turn(app, "Browse all electronics under $40.")
        prices = verified_prices(found, 40)
        require(
            found["personalization"] == "EXAMPLE_HISTORY_USED"
            and tool(found, "recommend_products"),
            "The example ranking was not used",
        )
        require(
            next(iter(prices)) == rankings[alias][0],
            "Expected the first exported candidate",
        )
        first = next(iter(prices))
        reviewed, reply = self.turn(app, "Any reviews for the first one?")
        self.check_reviews(reviewed, reply, [first])
        require(
            reviewed["review_sources"], "Expected reviews for the top example candidate"
        )
        narrowed, _ = self.turn(app, "I need wired headphones, at most $15.")
        narrowed_prices = verified_prices(narrowed, 15)
        require(
            first not in narrowed_prices
            and "wired" in narrowed["search_query"].lower().split()
            and tool(narrowed, "search_products"),
            "A history candidate overrode the updated catalog query or budget",
        )
        return {
            "example": alias,
            "first_verified": first,
            "current_query": narrowed["search_query"],
        }

    def catalog_price_states(self):
        app = self.open()
        self.ready(app)
        found, _ = self.turn(app, "Give me any electronic under $40")
        verified_prices(found, 40)
        require(
            found["pending_product_ids"],
            "Expected missing-price products to stay pending",
        )
        updated, _ = self.turn(app, "Change the budget to $15")
        verified_prices(updated, 15)
        compared, reply = self.turn(
            app, f"Compare ASIN {FORZA} and ASIN B01M0TXQ8C. Keep the budget."
        )
        require(
            compared["selected_price_status"]
            == {FORZA: "VIOLATED", "B01M0TXQ8C": "UNKNOWN"}
            and "Above the budget" in reply
            and "Price unknown" in reply,
            "The webpage conflated above-budget and missing-price products",
        )
        missing, reply = self.turn(app, "Any reviews for ASIN B0DSC200ZZ?")
        require(
            missing["missing_product_ids"] == ["B0DSC200ZZ"]
            and tool(missing, "get_products", "ERROR")
            and not missing["review_gaps"]
            and "absent from the catalog" in reply,
            "Unknown ASIN was mistaken for a review gap",
        )
        return {
            "selected_price_status": compared["selected_price_status"],
            "absent_id": missing["missing_product_ids"][0],
        }

    def recovery(self, configuration, expected):
        with patch.dict(os.environ, configuration):
            app = self.open()
        require(
            not app.exception
            and not app.chat_input
            and any(
                message in error.value for message in expected for error in app.error
            ),
            f"Expected an English dependency error before offering chat, got {list(app.error)!r}",
        )
        message = app.error[0].value
        retry = next(
            button for button in app.button if button.label == "Retry connection"
        )
        retry.click().run()
        self.ready(app)
        found, _ = self.turn(app, "Show me electronics under $40.")
        verified_prices(found, 40)
        return {"error": message, "provider_after_retry": found["provider"]}

    def service_unavailable(self):
        # The reserved address has no listener. TCP may refuse or time out.
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            return self.recovery(
                {
                    "SHOPPING_AGENT_OLLAMA_URL": f"http://127.0.0.1:{port.getsockname()[1]}"
                },
                (
                    "Cannot reach the local Ollama service",
                    "The Ollama request timed out.",
                ),
            )


def run_case(cases, layer, name, check=None, *, skip_reason=""):
    start = monotonic()
    case = {"layer": layer, "name": name}
    if check is None:
        case.update(status="skipped", reason=skip_reason)
    else:
        try:
            case.update(status="passed", evidence=check())
        except Exception as error:  # noqa: BLE001 — log failure and run the remaining cases.
            case.update(status="failed", reason=f"{type(error).__name__}: {error}")
    case["elapsed_s"] = round(monotonic() - start, 3)
    cases.append(case)
    detail = f" ({case['reason']})" if "reason" in case else ""
    print(f"{case['status']:7} {layer} · {name}{detail}", flush=True)


def check_catalog(catalog, rankings, reviews):
    coverage = {}
    for alias, expected in rankings.items():
        actual = catalog.call(
            "recommend_products",
            {"persona": alias, "category": "Electronics", "limit": 25},
        )
        require(actual.ids == expected, f"Ranking order changed for {alias}")
        products = catalog.call("get_products", {"ids": actual.ids})
        require(
            [product.id for product in products.products] == expected,
            "Rankings failed the catalog join",
        )
        coverage[alias] = {
            "catalog_matches": len(products.products),
            "review_coverage": sum(asin in reviews for asin in actual.ids),
        }
    require(
        catalog.call("get_products", {"ids": [FORZA]}).status == "OK",
        "The reference ASIN is absent from the catalog",
    )
    return coverage


def compare_export(output, expected, message):
    require(output == expected, message)
    return {"exact_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommender", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:4b-q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/local/electronics-acceptance.json"),
    )
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--regenerated-rankings", type=Path)
    parser.add_argument("--review-prefix", type=Path)
    args = parser.parse_args()

    cases = []
    ranking_export = json.loads((ASSETS / "electronics_rankings.json").read_bytes())
    review_export = json.loads(
        gzip.decompress((ASSETS / "electronics_reviews.json.gz").read_bytes())
    )
    catalog = None

    def catalog_check():
        nonlocal catalog
        catalog = ElectronicsCatalog(args.recommender)
        return check_catalog(
            catalog, ranking_export["rankings"], review_export["excerpts"]
        )

    run_case(
        cases, "real_catalog", "Team ranking order and parent ASIN join", catalog_check
    )
    run_case(
        cases,
        "source_reproduction",
        "Checkpoint ranking export",
        (
            lambda: compare_export(
                json.loads(args.regenerated_rankings.read_bytes()),
                ranking_export,
                "The ranking exports differ",
            )
        )
        if args.regenerated_rankings
        else None,
        skip_reason="Provide --regenerated-rankings after running the export with the original checkpoints.",
    )
    run_case(
        cases,
        "source_reproduction",
        "Raw review excerpt export",
        (
            lambda: compare_export(
                export_reviews(
                    args.review_prefix,
                    args.recommender / "data" / "item_catalog.csv",
                    prefix_bytes=review_export["prefix_bytes"],
                ),
                review_export,
                "The review exports differ",
            )
        )
        if args.review_prefix
        else None,
        skip_reason="Provide --review-prefix to regenerate the review excerpts from source.",
    )

    if catalog is None or args.catalog_only:
        reason = (
            "Requested catalog-only run."
            if args.catalog_only
            else "The team catalog could not be opened."
        )
        run_case(cases, "live_streamlit_ollama", "Web acceptance", skip_reason=reason)
    else:
        checks = LiveChecks(catalog, args.model)
        with (
            TemporaryDirectory(prefix="electronics-acceptance-") as directory,
            patch.dict(
                os.environ,
                {
                    "SHOPPING_AGENT_DB": str(Path(directory) / "acceptance.sqlite3"),
                    "SHOPPING_AGENT_MODE": "offline",
                    "SHOPPING_AGENT_RECOMMENDER": str(args.recommender.resolve()),
                    "SHOPPING_AGENT_REVIEW_SNAPSHOT": "",
                    "SHOPPING_AGENT_SNAPSHOT": "",
                    "SHOPPING_AGENT_LIVE_MODEL": "",
                    "SHOPPING_AGENT_OLLAMA_URL": args.ollama_url,
                    "SHOPPING_AGENT_OLLAMA_MODEL": args.model,
                },
            ),
        ):
            for name, check in [
                ("Original dialogue and third item", checks.original_dialogue),
                ("Independent wording and new search", checks.different_products),
                (
                    "Budget revision and bounded model choice",
                    checks.budget_and_tool_choice,
                ),
                (
                    "Price decision and insufficient aspect evidence",
                    checks.decision_with_evidence,
                ),
                (
                    "NCF example history",
                    lambda: checks.history(
                        "accessories-ncf", ranking_export["rankings"]
                    ),
                ),
                (
                    "TIGER example history",
                    lambda: checks.history(
                        "accessories-tiger", ranking_export["rankings"]
                    ),
                ),
                (
                    "Pending price, over budget and missing ASIN",
                    checks.catalog_price_states,
                ),
                (
                    "Missing local model and retry",
                    lambda: checks.recovery(
                        {
                            "SHOPPING_AGENT_OLLAMA_MODEL": "dsc200-acceptance-deliberately-missing:latest"
                        },
                        ("is unavailable. Check ollama list",),
                    ),
                ),
                ("Unavailable Ollama address and retry", checks.service_unavailable),
            ]:
                run_case(cases, "live_streamlit_ollama", name, check)

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "requested_model": args.model,
        "catalog_snapshot": catalog.snapshot if catalog else None,
        "catalog_sha256": ranking_export["catalog_sha256"],
        "checkpoint_sha256": ranking_export["model_sha256"],
        "review_prefix_sha256": review_export["prefix_sha256"],
        "cases": cases,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    print(f"Report saved to {args.report}", flush=True)
    return int(any(case["status"] == "failed" for case in cases))


if __name__ == "__main__":
    raise SystemExit(main())
