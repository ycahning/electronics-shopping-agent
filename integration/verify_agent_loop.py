"""Drive the formal Streamlit entry point with real Ollama and Electronics data."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest.mock import patch
from zipfile import ZipFile

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "shopping_agent/fixtures/electronics_reviews.json.gz"
FROZEN_FILE = ROOT / "integration/frozen_agent_tasks.json"
BASELINE_ARCHIVE = ROOT / "reference/pre-loop-shopping-agent.zip"
RING = "B08KSJ56WR"
BN_LINK = "B09TNV7Q69"
FORZA = "B07NY21DF6"
NO_PRICE = "B07X4854MR"
COPY_CARD = (
    "I mean do you have any comments for this one:\n"
    "2. Electronic Arts Forza Horizon 4 [B07NY21DF6] · USD 25.00"
)
UNDERWATER = (
    f"Compare ASIN {RING} and ASIN {BN_LINK}. First check reviews for underwater use. "
    "If the sample lacks relevant evidence, recommend the less expensive one by catalog price and explain why."
)
SETUP_PREF = f"I'm choosing between {RING} and {BN_LINK}. Easy setup matters to me. Which choice does the evidence support?"
TOOLS = {
    "get_products",
    "get_product_evidence",
    "recommend_products",
    "resolve_product",
    "search_products",
    "browse_products",
}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def raw_tools(turn):
    return [event for event in turn["diagnostics"]["trace"] if event["action"] in TOOLS]


def action_names(turn):
    return [event["action"] for event in raw_tools(turn)]


def used_citations(turn):
    return [
        source
        for event in turn["diagnostics"]["trace"]
        if event["action"] == "publish_answer" and event["status"] == "VERIFIED"
        for source in event.get("citations", [])
    ]


def matches(reply: str, pattern: str) -> bool:
    return re.search(pattern, reply, re.IGNORECASE | re.DOTALL) is not None


class Checks:
    def __init__(self, args):
        self.args = args
        self.turns = []
        self.assertions = []
        self.model_inputs = []
        self.observations = {}
        self.on_turn = lambda: None
        path = args.recommender / "data/item_catalog.csv"
        with path.open(encoding="utf-8", newline="") as file:
            self.catalog_rows = {
                row["parent_asin"]: row for row in csv.DictReader(file)
            }
        with gzip.open(args.review_snapshot, "rt", encoding="utf8") as file:
            self.index = json.load(file)
        self.program = (
            args.legacy_root / "shopping_agent/ui.py"
            if args.legacy_root
            else ROOT / "shopping_agent/ui.py"
        )

    def check(self, condition, label, category="task"):
        self.assertions.append(
            {"assertion": label, "passed": bool(condition), "category": category}
        )
        return bool(condition)

    def abort_unless(self, condition, label):
        if not self.check(condition, label):
            raise RuntimeError(label)

    def app(self, *, ready=True):
        app = AppTest.from_file(self.program, default_timeout=160).run()
        if ready:
            self.ready(app)
        return app

    def ready(self, app):
        self.abort_unless(
            not app.error and not app.exception and len(app.chat_input) == 1,
            "Streamlit chat loaded without app errors",
        )
        self.abort_unless(
            any(
                f"Ollama · {self.args.ollama_model}" in caption.value
                and "team-electronics" in caption.value
                for caption in app.caption
            ),
            "App shows requested live model and real Electronics data",
        )

    def turn(self, app, text):
        start_at = len(self.model_inputs)
        started = monotonic()
        app.chat_input[0].set_value(text).run()
        self.ready(app)
        self.abort_unless(len(app.chat_message) >= 2, "Assistant reply was rendered")
        result = {
            "request": text,
            "reply": app.chat_message[-1].text[0].value,
            "elapsed_ms": round((monotonic() - started) * 1000, 2),
            "diagnostics": dict(app.session_state.shop_diagnostics),
            "model_inputs": self.model_inputs[start_at:],
        }
        self.turns.append(result)
        self.check(
            result["diagnostics"]["provider"] == f"ollama:{self.args.ollama_model}"
            and bool(result["model_inputs"]),
            "Turn reached actual Ollama",
        )
        self.on_turn()
        return result

    def button(self, app):
        started = monotonic()
        app.button(key="recommend_next").click().run()
        self.ready(app)
        result = {
            "request": "Clicked Recommend next items",
            "reply": app.chat_message[-1].text[0].value,
            "elapsed_ms": round((monotonic() - started) * 1000, 2),
            "diagnostics": dict(app.session_state.shop_diagnostics),
            "model_inputs": [],
        }
        self.turns.append(result)
        self.on_turn()
        return result

    def direct(self, turn, expected_pattern):
        self.check(turn["diagnostics"]["tool_attempts"] == 0, "Zero shopping tools")
        self.check(
            matches(turn["reply"], expected_pattern), "Direct reply answers request"
        )

    def within_budget(self, turn, cap, *, minimum=1, keywords=()):
        data = turn["diagnostics"]
        self.check(
            data["budget_usd"] is not None and Decimal(data["budget_usd"]) == cap,
            f"Accepted USD cap equals {cap}",
            "fact",
        )
        prices = data["verified_prices_usd"]
        self.check(len(prices) >= minimum, "Required count of verified matches")
        for asin, shown in prices.items():
            row = self.catalog_rows[asin]
            self.check(
                bool(row["price"])
                and Decimal(row["price"]) == Decimal(shown) <= cap
                and all(term in row["title"].casefold() for term in keywords),
                f"{asin} price and description independently match the catalog and constraints",
                "fact",
            )
            self.check(
                asin in turn["reply"],
                f"{asin} verified result appears in the visible reply",
            )
        self.check(
            set(prices).isdisjoint(data["pending_product_ids"]),
            "Unknown-price and verified groups are disjoint",
            "fact",
        )
        return data["numbered_product_ids"]

    def source_owners(self, turn, selected, *, controlled=None):
        by_id = {
            f"ar23-electronics-{line}": (asin, text)
            for asin in selected
            for line, text in self.index["excerpts"].get(asin, [])
        }
        if controlled:
            by_id.update(controlled)
        data = turn["diagnostics"]
        self.check(
            set(data["selected_product_ids"]) == set(selected),
            f"Tool operation selected exactly {', '.join(selected)}",
            "fact",
        )
        if data["review_sources"]:
            counts = Counter(source["product_id"] for source in data["review_sources"])
            self.check(
                max(counts.values()) <= 4
                and matches(turn["reply"], r"up to (?:4|four) excerpts per product"),
                "The four-excerpt-per-product cap holds and is visible",
                "fact",
            )
        for citation in data["review_sources"]:
            citation_id = citation["id"]
            self.check(
                citation_id in by_id
                and by_id[citation_id][0] == citation["product_id"],
                f"{citation_id} resolves to the right review author-product association",
                "fact",
            )
            self.check(
                citation["source"] in turn["reply"],
                f"{citation_id} has a visible source",
            )
            if citation_id in by_id:
                self.check(
                    by_id[citation_id][1] in turn["reply"],
                    f"{citation_id} displays the excerpt without alteration",
                    "fact",
                )

    def seen(self, turn, names, *, quoted=None):
        matches_input = []
        for call in turn["model_inputs"]:
            context = call["context"]
            # Fail on a summary-only context. Evidence text must be provided where present.
            shadow = json.dumps(context.get("TOOL_DATA", {}), ensure_ascii=False)
            if all(name in shadow for name in names) and (
                not quoted or quoted in shadow
            ):
                matches_input.append(call)
        self.check(
            bool(matches_input),
            "A subsequent actual Ollama request received "
            + ", ".join(names)
            + (" and the literal excerpt" if quoted else ""),
        )
        return matches_input

    def greetings(self):
        app = self.app()
        self.direct(self.turn(app, "Hi, introduce yourself"), r"electronic|shopping")
        self.direct(
            self.turn(
                app, "Before we get going, what can you help a shopper figure out?"
            ),
            r"search|recommend|compar|review|products",
        )
        self.direct(
            self.turn(app, "What's the difference between HDMI and DisplayPort?"),
            r"(?=.*\bHDMI\b)(?=.*DisplayPort)(?=.*display|video|monitor)",
        )
        listed = self.turn(app, "Hi there, can you find an HDMI cable under $12?")
        ids = self.within_budget(listed, 12, keywords=("hdmi", "cable"))
        self.check(
            "search_products" in action_names(listed),
            "Greeting included a shopping search",
        )
        aside = self.turn(app, "Thanks for the explanation.")
        self.direct(aside, r"welcome|anytime|glad|help|pleasure|course|sure")
        self.check(
            aside["diagnostics"]["numbered_product_ids"] == ids,
            "Interjection preserved numbering",
        )
        follow = self.turn(app, "Could you get the details for number one?")
        if ids:
            self.check(
                "get_products" in action_names(follow),
                "Resumed shopping with catalog detail",
            )
            self.check(
                follow["diagnostics"]["selected_product_ids"] == ids[:1],
                "Resumed the original selection",
            )

    def prices_after_reviews(self):
        self.check(
            self.catalog_rows[RING]["price"] == "29.99"
            and self.catalog_rows[BN_LINK]["price"] == "24.99",
            "Independent source prices match the expected observation",
            "fact",
        )
        self.check(
            all(
                asin in self.index["excerpts"]
                and all(
                    "underwater" not in text.lower()
                    for _, text in self.index["excerpts"][asin]
                )
                for asin in (RING, BN_LINK)
            ),
            "The indexed sample covers both ASINs and lacks underwater excerpts",
            "fact",
        )
        turn = self.turn(self.app(), UNDERWATER)
        self.check(
            "get_product_evidence" in action_names(turn),
            "Actually queried the review source",
        )
        observed = self.seen(turn, ("get_product_evidence", RING, BN_LINK))
        self.check(
            any(
                (call.get("response") or {}).get("action") == "finish"
                and call["response"].get("basis") == "price"
                and {
                    asin: products.get(asin, {}).get("historical_usd")
                    for asin in (RING, BN_LINK)
                }
                == {RING: "29.99", BN_LINK: "24.99"}
                and any(
                    event["action"] == "get_product_evidence"
                    and event["status"] == "EMPTY"
                    and event.get("review_gaps")
                    == {
                        RING: "NO_ASPECT_MATCH",
                        BN_LINK: "NO_ASPECT_MATCH",
                    }
                    for event in call["context"]["TOOL_DATA"]["tool_steps"]
                )
                for call in observed
                for products in [call["context"]["TOOL_DATA"]["products"]]
            ),
            "The actual post-observation Ollama response chose the price basis after receiving both price facts and empty review results",
        )
        self.check(
            turn["diagnostics"]["review_gaps"]
            == {RING: "NO_ASPECT_MATCH", BN_LINK: "NO_ASPECT_MATCH"},
            "Tool outcome distinguishes aspect gap from missing product or sample coverage",
            "fact",
        )
        data = turn["diagnostics"]
        if not self.args.legacy_root:
            self.check(
                any(
                    event["action"] == "publish_answer"
                    and event.get("basis") == "price"
                    and event.get("product_id") == BN_LINK
                    and event["status"] == "VERIFIED"
                    for event in data["trace"]
                ),
                "Application computed and published the verified cheaper ASIN",
                "fact",
            )
        self.check(
            data.get("decision", {}).get("product_id") == BN_LINK
            if data.get("decision")
            else any(
                event.get("product_id") == BN_LINK or event.get("winner") == BN_LINK
                for event in data["trace"]
                if event["action"] == "publish_answer"
            )
            or matches(
                turn["reply"],
                r"(?:(?:choose|recommend|pick|cheaper|less expensive|lower(?:-priced| price)).{0,180}(?:BN.LINK|24\.99)|(?:BN.LINK|24\.99).{0,180}(?:cheaper|less expensive|lower(?:-priced| price)))",
            ),
            "Answer actually selects BN-LINK when underwater reviews are inconclusive",
            "condition",
        )
        self.check(
            matches(turn["reply"], r"24\.99")
            and matches(turn["reply"], r"29\.99|\$5\.00"),
            "Visible catalog values support the correct fallback and optional difference",
            "fact",
        )
        self.check(
            matches(turn["reply"], r"underwater")
            and matches(
                turn["reply"],
                r"no (?:relevant|matching|evidence|review|keyword)|couldn.t|cannot|insufficien|unable|does not|didn.t|lack|unknown|inconclusive",
            ),
            "Answer explicitly leaves underwater suitability unresolved",
            "condition",
        )

    def alternate_rating_fallback(self):
        one, two = "B0BBZZRL8G", "B0BPB5Q41C"
        turn = self.turn(
            self.app(),
            f"Between USB hubs {one} and {two}, look for buyer experience on Raspberry Pi 5. "
            "If the sample can't say which is better for that, pick the one with the higher average catalog rating.",
        )
        self.seen(turn, ("get_product_evidence", one, two))
        self.check(
            any(
                event["action"] == "get_product_evidence"
                and matches(
                    event.get("arguments", {}).get("query", ""), r"raspberry|\bpi\b"
                )
                for event in raw_tools(turn)
            )
            and not any(
                matches(text, r"raspberry|\bpi\s*5\b")
                for asin in (one, two)
                for _, text in self.index["excerpts"].get(asin, [])
            ),
            "Tool searched for the requested Raspberry Pi aspect, which lacks independent sample support",
            "condition",
        )
        self.check(
            turn["diagnostics"]["review_gaps"]
            == {one: "NO_ASPECT_MATCH", two: "OUTSIDE_SAMPLE"},
            "Correctly distinguishes reviewed product with no matching aspect from one outside the sample",
            "fact",
        )
        rating = turn["diagnostics"].get("decision") or {}
        self.check(
            rating.get("product_id") == one
            and rating.get("basis") == "rating"
            and self.catalog_rows[one]["average_rating"] == "4.1"
            and self.catalog_rows[two]["average_rating"] == "4.0"
            and matches(turn["reply"], r"(?=.*4\.1)(?=.*catalog average rating)"),
            "Explains the different rating-based fallback with independently checked catalog data",
            "condition",
        )

    def real_setup_review(self):
        line_number = 19276
        citation_id = f"ar23-electronics-{line_number}"
        indexed = {line: text for line, text in self.index["excerpts"].get(BN_LINK, [])}
        excerpt = indexed.get(line_number)
        self.abort_unless(
            excerpt and "easy to set up" in excerpt.casefold(),
            "Independent source excerpt identifies a buyer's ease-of-setup observation",
        )
        turn = self.turn(
            self.app(),
            f"Easy setup matters to me. Does a buyer report support shortlisting ASIN {BN_LINK} for easy setup?",
        )
        self.source_owners(turn, [BN_LINK])
        self.check(
            "get_product_evidence" in action_names(turn),
            "Actual Electronics review adapter was queried",
        )
        observed = self.seen(
            turn, ("get_product_evidence", citation_id), quoted=excerpt
        )
        self.check(
            any(
                (call.get("response") or {}).get("action") == "finish"
                and call["response"].get("basis") == "reviews"
                and any(
                    claim["citation"] == citation_id and claim["product_id"] == BN_LINK
                    for claim in call["response"].get("claims", [])
                )
                for call in observed
            ),
            "Actual observed excerpt leads the live model to a review-based finish",
        )
        self.check(
            any(
                citation["id"] == citation_id and citation["product_id"] == BN_LINK
                for citation in used_citations(turn)
            ),
            "Same-product real quote is verified in the final answer",
            "fact",
        )
        self.check(
            matches(turn["reply"], r"easy (?:to set up|setup)")
            and matches(turn["reply"], r"a buyer|one buyer")
            and matches(
                turn["reply"], r"shortlist|choose|support|sampled buyer experience"
            ),
            "Visible evidence-based conclusion describes a buyer's experience and limits the claim",
            "condition",
        )

    def ordinal_flow(self, *, variant=False):
        app = self.app()
        if variant:
            opening = "Good evening. Would you help me look at HDMI cables priced at $14 or less?"
            turns = (
                "Would you pull the customer feedback for item no. 1?",
                "Do the same for item #2, please.",
                f"Actually, show reviews for this listing instead: {BN_LINK}",
                "What about number three on the list?",
            )
            target = BN_LINK
            cap = 14
        else:
            self.direct(
                self.turn(app, "Hi"), r"\b(?:hello|hi|hey|welcome)\b|help|assist"
            )
            opening = "Give me any electronic under $40"
            turns = (
                "I need all reviews for first oe",
                "OK, How about seconed one",
                COPY_CARD,
                "What about reviews for the third one?",
            )
            target = FORZA
            cap = 40
        listing = self.turn(app, opening)
        ids = self.within_budget(listing, cap, minimum=3)
        self.abort_unless(len(ids) >= 3, "Numbered list has three ASINs")
        for message, target_asin in zip(
            turns, (ids[0], ids[1], target, ids[2]), strict=True
        ):
            response = self.turn(app, message)
            data = response["diagnostics"]
            self.check(
                data["numbered_product_ids"] == ids,
                "Published numbering remains stable",
            )
            self.check(
                "get_product_evidence" in action_names(response),
                "Actually queried reviews",
            )
            self.source_owners(response, [target_asin])
            if target_asin in self.index["excerpts"]:
                self.check(
                    data["review_sources"], "Sampled item returned attributed excerpts"
                )
            else:
                self.check(
                    data["review_gaps"].get(target_asin) == "OUTSIDE_SAMPLE"
                    and matches(response["reply"], r"sample|cover"),
                    "Valid product is correctly explained as outside review sample",
                    "fact",
                )

    def ambiguous(self):
        app = self.app()
        ids = self.within_budget(
            self.turn(
                app, "I need HDMI cables for up to $14. Please list some options."
            ),
            14,
            minimum=2,
        )
        self.abort_unless(
            len(ids) > 1, "The reference has multiple plausible candidates"
        )
        clarification = self.turn(app, "What do shoppers say about that one?")
        self.check(
            clarification["diagnostics"]["outcome"] == "CLARIFICATION_NEEDED"
            and "?" in clarification["reply"]
            and "get_product_evidence" not in action_names(clarification),
            "Ambiguous reference prompts useful clarification before querying evidence",
        )
        resumed = self.turn(app, "Number two.")
        self.check(
            "get_product_evidence" in action_names(resumed),
            "Follow-up resumes pending review goal",
        )
        self.source_owners(resumed, [ids[1]])
        self.check(
            resumed["diagnostics"]["numbered_product_ids"] == ids,
            "Clarification does not replace list",
        )
        new = self.turn(app, "Start a fresh search for wired headphones at most $21.")
        changed = self.within_budget(new, 21)
        self.check(
            changed
            and changed != ids
            and new["diagnostics"]["search_query"]
            != clarification["diagnostics"]["search_query"],
            "Explicit new search publishes a new numbered list",
        )

    def missing_and_recovery(self):
        from shopping_agent.electronics import ElectronicsCatalog

        app = self.app()
        missing = self.turn(app, "Any customer review comments for ASIN X0ZZZZZZZZ?")
        self.check(
            "X0ZZZZZZZZ" in missing["diagnostics"]["missing_product_ids"]
            or matches(
                missing["reply"],
                r"X0ZZZZZZZZ.*(?:catalog|not found|absent|couldn.t)|(?:catalog|not found|absent|couldn.t).*X0ZZZZZZZZ",
            ),
            "Unknown product is distinguished from a review sample gap",
            "fact",
        )
        request = f"Show sampled customer reviews for ASIN {BN_LINK}."
        with patch.object(
            ElectronicsCatalog,
            "_evidence",
            side_effect=TimeoutError("deliberate test fault"),
        ):
            failed = self.turn(app, request)
        self.check(
            "get_product_evidence" in action_names(failed)
            and matches(failed["reply"], r"unavailable|could not|timeout|failed|unable")
            and not failed["diagnostics"]["review_sources"],
            "Injected review outage is reported without fabricated evidence",
        )
        recovered = self.turn(app, request)
        self.check(
            bool(recovered["diagnostics"]["review_sources"]),
            "Real reviews return after removing injected outage",
        )
        self.source_owners(recovered, [BN_LINK])

    def provider_recovery(self):
        with patch.dict(
            os.environ,
            {"SHOPPING_AGENT_OLLAMA_MODEL": "dsc200-uninstalled-test:latest"},
        ):
            missing = self.app(ready=False)
            errors = [message.value for message in missing.error]
            self.observations["missing_model_errors"] = errors
            self.check(
                errors and matches(" ".join(errors), r"model|tag|install|download"),
                "Real Ollama tags detect requested model is missing",
            )
        with socket.socket() as unlistening:
            unlistening.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{unlistening.getsockname()[1]}"
            with patch.dict(os.environ, {"SHOPPING_AGENT_OLLAMA_URL": url}):
                failed = self.app(ready=False)
                errors = [message.value for message in failed.error]
                self.observations["unreachable_url_errors"] = errors
                self.check(
                    errors
                    and matches(
                        " ".join(errors),
                        r"ollama.*(?:running|connect|unavailable|server|reach|service|timed out)|(?:connect|unavailable|reach).*ollama",
                    ),
                    "Real unreachable loopback endpoint is reported as unavailable or timed out",
                )
        self.direct(
            self.turn(self.app(), "Hi, introduce yourself"), r"electronic|shopping"
        )

    def ranking_context(self):
        app = self.app()
        self.check(
            app.selectbox(key="history_profile").value == "No history",
            "No-history profile is the default",
        )
        no_history = self.turn(
            app,
            "What are the next five products you recommend from my purchase history?",
        )
        self.check(
            matches(no_history["reply"], r"history")
            and not no_history["diagnostics"]["numbered_product_ids"],
            "No history does not silently substitute a personalized ranking",
        )
        app.selectbox(key="history_profile").set_value("Dataset user ID").run()
        app.text_input(key="history_user_id").set_value("DSC200UNKNOWN9999").run()
        unknown = self.button(app)
        self.check(
            not unknown["diagnostics"]["numbered_product_ids"]
            and matches(
                unknown["reply"],
                r"no (?:usable |available |interaction |dataset )*history|not found|unknown",
            ),
            "Unknown history ID is reported by deterministic UI ranking",
        )
        app.selectbox(key="history_profile").set_value("Example user A").run()
        app.selectbox(key="ranking_model").set_value("ncf").run()
        ranked = self.turn(
            app,
            "What are the top 4 next Electronics items for me based on my history?",
        )
        data = ranked["diagnostics"]
        self.check(
            "recommend_products" in action_names(ranked)
            and data["requested_model"] == "ncf"
            and data["requested_n"] == 4
            and len(data["numbered_product_ids"]) == 4
            and matches(ranked["reply"], r"history|NCF"),
            "Live model routes conversational history request with requested N",
        )

    def controlled(self):
        from shopping_agent.catalog import Evidence
        from shopping_agent.electronics import ElectronicsCatalog

        original = ElectronicsCatalog._evidence
        sample = [
            (
                "The Ring setup was difficult. Pairing failed three times, and installation took an hour.",
                "The BN-LINK power strip was easy to set up. I plugged it in and finished setup in two minutes.",
            ),
            (
                "The Ring package was a dark color, and I received a cardboard box.",
                "The BN-LINK power strip was delivered in a large blue carton.",
            ),
        ]
        found = []
        for label, pair in zip(("sufficient", "irrelevant"), sample, strict=True):
            targeted_calls = []
            replacement = {
                f"controlled-{i}": (asin, text)
                for i, (asin, text) in enumerate(
                    zip((RING, BN_LINK), pair, strict=True), 1
                )
            }

            def first_only(
                catalog,
                args,
                *,
                calls=targeted_calls,
                test_label=label,
                rows=replacement,
            ):
                response = original(catalog, args)
                calls.append(dict(args))
                if len(calls) != 1 or set(args.get("ids", [])) != {
                    RING,
                    BN_LINK,
                }:
                    return response
                return response.model_copy(
                    update={
                        "status": "OK",
                        "evidence": [
                            Evidence(
                                id=source_id,
                                product_id=asin,
                                text=text,
                                source=f"Controlled first review response ({test_label}) {source_id}",
                            )
                            for source_id, (asin, text) in rows.items()
                        ],
                        "missing_ids": [],
                        "review_gaps": {},
                        "has_more": False,
                    }
                )

            app = self.app()
            with patch.object(ElectronicsCatalog, "_evidence", first_only):
                turn = self.turn(app, SETUP_PREF)
            turn["controlled_first_review"] = {
                "condition": label,
                "synthetic_excerpts": replacement,
                "evidence_requests": targeted_calls,
                "later_evidence_calls": "unchanged real dataset adapter",
            }
            model_messages = self.seen(turn, ("get_product_evidence",), quoted=pair[0])
            self.check(
                bool(targeted_calls)
                and set(targeted_calls[0].get("ids", [])) == {RING, BN_LINK},
                f"{label} run naturally queried both products before test observation",
            )
            if label == "sufficient":
                self.check(
                    matches(
                        turn["reply"],
                        r"(?:BN.LINK.{0,120}(?:easy|easier|recommend|two minutes)|(?:recommend|choose|easy|easier).{0,120}BN.LINK)",
                    ),
                    "Sufficient observation supports BN-LINK on easy setup",
                    "condition",
                )
                self.source_owners(turn, [RING, BN_LINK], controlled=replacement)
                self.check(
                    any(
                        source["id"] == "controlled-2"
                        for source in used_citations(turn)
                    ),
                    "Sufficient answer cites the synthetic excerpt it used",
                )
            else:
                self.check(
                    not any(
                        source["id"].startswith("controlled-")
                        for source in used_citations(turn)
                    ),
                    "Irrelevant packaging snippets are not cited as setup evidence",
                    "fact",
                )
            found.append((turn, model_messages))
        (sufficient, left), (irrelevant, right) = found
        first_tool = "get_product_evidence"
        left_tools, right_tools = action_names(sufficient), action_names(irrelevant)
        if first_tool in left_tools and first_tool in right_tools:
            self.check(
                left_tools[: left_tools.index(first_tool) + 1]
                == right_tools[: right_tools.index(first_tool) + 1],
                "The same tool path led to both controlled observations",
            )
            self.check(
                left_tools[left_tools.index(first_tool) + 1 :]
                != right_tools[right_tools.index(first_tool) + 1 :],
                "Observed evidence alone led to different actual subsequent tool execution",
            )
        if left and right:
            left_obs = left[0]["context"]
            right_obs = right[0]["context"]
            shared_keys = (
                "user_request",
                "accepted_constraints",
                "available_actions",
                "requested_review_aspect",
                "selected_asins",
            )
            self.check(
                all(
                    key in left_obs
                    and key in right_obs
                    and left_obs[key] == right_obs[key]
                    for key in shared_keys
                )
                and bool(left_obs["available_actions"])
                and sufficient["controlled_first_review"]["evidence_requests"][:1]
                == irrelevant["controlled_first_review"]["evidence_requests"][:1],
                "Exact goal, accepted state, allowable actions and initial tool arguments match",
            )

    def frozen(self, spec):
        app = self.app()
        turns = [self.turn(app, message) for message in spec["turns"]]
        last = turns[-1]
        data = last["diagnostics"]
        name = spec["id"]
        if name == "capabilities":
            self.direct(last, r"electronics|recommend|compare|review|product")
        if name == "adapt-budget":
            first = turns[0]
            ids = self.within_budget(first, 14, keywords=("hdmi", "cable"))
            self.check(
                turns[1]["diagnostics"]["tool_attempts"] == 0,
                "Interjection needs no tools",
            )
            self.check(
                turns[1]["diagnostics"]["numbered_product_ids"] == ids,
                "Interjection preserves list",
            )
            self.within_budget(last, 7, keywords=("hdmi", "cable"))
            self.check(
                data["search_query"] == first["diagnostics"]["search_query"],
                "Updated cap keeps original search keywords",
                "condition",
            )
        if name == "evidence-choice":
            self.check(
                "get_product_evidence" in action_names(last), "Review source consulted"
            )
            self.source_owners(last, [RING, BN_LINK])
            matching = {
                f"ar23-electronics-{line}": excerpt
                for line, excerpt in self.index["excerpts"][BN_LINK]
                if "easy to set up" in excerpt.casefold()
            }
            self.check(
                any(
                    source["id"] in matching and source["product_id"] == BN_LINK
                    for source in data["review_sources"]
                )
                and matches(last["reply"], r"easy to set up|easier setup|easy setup"),
                "Independent excerpt inspection supports a carefully scoped BN-LINK easy-setup finding",
                "fact",
            )
            self.check(
                matches(
                    last["reply"],
                    r"sample|limited|one reviewer|one excerpt|lack|no (?:review|evidence|match)",
                ),
                "Review conclusion conveys sample limits",
                "condition",
            )
            if not self.args.legacy_root and matching:
                self.seen(
                    last,
                    ("get_product_evidence", BN_LINK),
                    quoted=next(iter(matching.values())),
                )
                self.check(
                    any(
                        citation["id"] in matching and citation["product_id"] == BN_LINK
                        for citation in used_citations(last)
                    ),
                    "Published decision explicitly cites the source it used for easy setup",
                    "fact",
                )
        if name == "rating-fallback":
            self.check(
                "get_product_evidence" in action_names(last),
                "Review source consulted before rating fallback",
            )
            self.check(
                matches(last["reply"], r"30|temperature|cold")
                and matches(
                    last["reply"],
                    r"couldn.t|can.t|cannot|insufficien|lack|no |unknown|inconclusive|unverified",
                ),
                "Cold reliability remains unknown",
                "condition",
            )
            self.check(
                matches(last["reply"], r"4\.7")
                and matches(last["reply"], r"4\.5")
                and (
                    (data.get("decision") or {}).get("product_id") == FORZA
                    or matches(
                        last["reply"],
                        r"(?:(?:choose|pick|recommend|higher).{0,120}(?:Forza|B07NY21DF6)|(?:Forza|B07NY21DF6).{0,120}(?:higher|prefer|choice))",
                    )
                ),
                "Selects Forza explicitly by verified average ratings, 4.7 vs 4.5",
                "condition",
            )
        if name == "missing-price":
            self.check(
                self.catalog_rows[NO_PRICE]["price"] == ""
                and matches(last["reply"], r"25\.00")
                and matches(
                    last["reply"],
                    r"missing|unknown|no (?:catalog |listed )?price|does not (?:have|list)|unavailable|couldn't",
                ),
                "Identifies independently verified known and missing prices",
                "fact",
            )
            self.check(
                matches(
                    last["reply"],
                    r"can.t|cannot|could not|unable|unknown|inconclusive|not enough|only known",
                ),
                "Leaves price comparison unresolved",
                "condition",
            )
            self.check(
                "get_product_evidence" not in action_names(last),
                "No irrelevant review lookup",
            )

    def run(self, cases, save):
        from shopping_agent.ollama import OllamaProvider
        from shopping_agent.service import Agent

        original_chat = OllamaProvider._chat

        def record(provider, instructions, context, schema, timeout, **options):
            attempt = {
                "context": json.loads(context) if isinstance(context, str) else context,
                "response_schema": schema.__name__,
            }
            self.model_inputs.append(attempt)
            try:
                proposal = original_chat(
                    provider, instructions, context, schema, timeout, **options
                )
            except Exception as error:
                attempt["error"] = type(error).__name__
                raise
            attempt["response"] = proposal.model_dump(mode="json")
            return proposal

        original_init = Agent.__init__

        def fixed(instance, *args, **kwargs):
            original_init(instance, *args, **{**kwargs, "policy": "fixed"})

        legacy = (
            patch.object(Agent, "__init__", fixed)
            if self.args.legacy_policy == "fixed"
            else nullcontext()
        )
        results = []
        with patch.object(OllamaProvider, "_chat", record), legacy:
            for name, case in cases:
                self.turns = []
                self.assertions = []
                self.observations = {}
                failure = None

                def record_active(case_name=name):
                    save(
                        results,
                        active_case={
                            "id": case_name,
                            "turns": self.turns,
                            "assertions": self.assertions,
                        },
                    )

                self.on_turn = record_active
                try:
                    case()
                except Exception as error:  # noqa: BLE001 — retain failed-run records
                    failure = f"{type(error).__name__}: {error}"
                results.append(
                    {
                        "id": name,
                        "status": "passed"
                        if not failure and all(a["passed"] for a in self.assertions)
                        else "failed",
                        "error": failure,
                        "assertions": self.assertions,
                        "observations": self.observations,
                        "turns": self.turns,
                    }
                )
                save(results)
                print(f"{name}: {results[-1]['status']}", flush=True)
                for description in [
                    failure,
                    *(a["assertion"] for a in self.assertions if not a["passed"]),
                ]:
                    if description:
                        print(f"  {description}", flush=True)
        return results


def metrics(results):
    turns = [turn for case in results for turn in case["turns"]]
    assertions = [assertion for case in results for assertion in case["assertions"]]
    unexpected = 0
    repeats = 0
    for case in results:
        for turn in case["turns"]:
            exercised = action_names(turn)
            if case["id"] in {"capabilities", "adapt-budget", "missing-price"}:
                unexpected += exercised.count("get_product_evidence")
            if case["id"] == "capabilities" or turn["request"] == "Got it, thanks.":
                unexpected += len(
                    [action for action in exercised if action != "get_product_evidence"]
                )
            signatures = [
                (
                    event["action"],
                    json.dumps(event["arguments"], sort_keys=True, default=str),
                )
                for event in raw_tools(turn)
                if "arguments" in event
            ]
            repeats += sum(n - 1 for n in Counter(signatures).values())
    return {
        "tasks_completed": sum(case["status"] == "passed" for case in results),
        "tasks_measured": len(results),
        "omitted_or_incorrect_tested_conditions": sum(
            a["category"] == "condition" and not a["passed"] for a in assertions
        ),
        "failed_fact_assertions": sum(
            a["category"] == "fact" and not a["passed"] for a in assertions
        ),
        "obviously_unnecessary_case_specific_tool_calls": unexpected,
        "identical_tool_calls_where_arguments_logged": repeats,
        "clarifications": sum(
            t["diagnostics"]["outcome"] == "CLARIFICATION_NEEDED" for t in turns
        ),
        "tool_calls": sum(t["diagnostics"]["tool_attempts"] for t in turns),
        "model_calls": sum(t["diagnostics"]["model_attempts"] for t in turns),
        "wall_latency_ms_sum": round(sum(t["elapsed_ms"] for t in turns), 2),
        "wall_latency_ms_per_turn": [t["elapsed_ms"] for t in turns],
    }


def run_legacy(args, policy="fixed"):
    with TemporaryDirectory(prefix="dsc200-baseline-") as directory:
        archive = Path(directory) / "legacy"
        with ZipFile(BASELINE_ARCHIVE) as bundle:
            bundle.extractall(archive)
        output = Path(directory) / "baseline-results.json"
        command = [
            sys.executable,
            "-B",
            __file__,
            "--recommender",
            str(args.recommender),
            "--recommender-python",
            str(args.recommender_python),
            "--ollama-model",
            args.ollama_model,
            "--review-snapshot",
            str(args.review_snapshot),
            "--legacy-root",
            str(archive),
            "--legacy-policy",
            policy,
            "--output",
            str(output),
        ]
        if args.only:
            command += ["--only", args.only]
        completed = subprocess.run(
            command,
            env={**os.environ, "PYTHONPATH": f"{archive}{os.pathsep}{ROOT}"},
            check=False,
        )
        if not output.is_file():
            raise RuntimeError(
                f"Frozen legacy comparator exited {completed.returncode}"
            )
        return json.loads(output.read_text(encoding="utf-8"))


def write_report(path, report):
    updated = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    updated.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    updated.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recommender", required=True, type=lambda v: Path(v).resolve()
    )
    parser.add_argument(
        "--recommender-python", required=True, type=lambda v: Path(v).absolute()
    )
    parser.add_argument("--ollama-model", default="qwen3:4b-q4_K_M")
    parser.add_argument(
        "--review-snapshot", default=DEFAULT_SAMPLE, type=lambda v: Path(v).resolve()
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only", help="Comma-separated case IDs")
    parser.add_argument("--compare-fixed", action="store_true")
    parser.add_argument(
        "--before-repair",
        action="store_true",
        help="Replay the underwater task against the frozen original and record the outcome",
    )
    parser.add_argument("--legacy-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--legacy-policy", choices=["fixed", "original"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if (args.before_repair or args.compare_fixed) and not BASELINE_ARCHIVE.is_file():
        parser.error(
            "The archived comparator needs the authorized archive at "
            "reference/pre-loop-shopping-agent.zip."
        )
    if args.before_repair:
        args.only = "underwater-price-fallback"
        report = run_legacy(args, policy="original")
        report["legacy_archive_sha256"] = digest(BASELINE_ARCHIVE)
        write_report(args.output, report)
        return int(any(case["status"] != "passed" for case in report["cases"]))

    from shopping_agent.ollama import DEFAULT_OLLAMA_URL, OllamaProvider
    from shopping_agent.provider import ProviderError

    specs = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))["tasks"]
    installed = OllamaProvider(
        args.ollama_model,
        base_url=os.getenv("SHOPPING_AGENT_OLLAMA_URL", DEFAULT_OLLAMA_URL),
    )
    try:
        tags = installed._request("GET", "/api/tags", 5).get("models", [])
        model_digest = next(
            (row.get("digest") for row in tags if row.get("name") == args.ollama_model),
            None,
        )
    except ProviderError:
        model_digest = None
    with TemporaryDirectory(prefix="dsc200-acceptance-") as directory:
        env = {
            "SHOPPING_AGENT_MODE": "final",
            "SHOPPING_AGENT_DB": f"{directory}/acceptance.sqlite3",
            "SHOPPING_AGENT_RECOMMENDER": str(args.recommender),
            "SHOPPING_AGENT_RECOMMENDER_PYTHON": str(args.recommender_python),
            "SHOPPING_AGENT_OLLAMA_MODEL": args.ollama_model,
            "SHOPPING_AGENT_REVIEW_SNAPSHOT": str(args.review_snapshot),
            "SHOPPING_AGENT_LIVE_MODEL": "",
        }
        with patch.dict(os.environ, env):
            checks = Checks(args)
            cases = [
                ("frozen/" + spec["id"], lambda s=spec: checks.frozen(s))
                for spec in specs
            ]
            if args.legacy_policy != "fixed":
                cases = [
                    ("zero-tools-and-interjection", checks.greetings),
                    ("underwater-price-fallback", checks.prices_after_reviews),
                    ("alternate-rating-fallback", checks.alternate_rating_fallback),
                    ("controlled-observation-ab", checks.controlled),
                    ("actual-setup-review-decision", checks.real_setup_review),
                    ("exact-five-turn-dialogue", checks.ordinal_flow),
                    (
                        "alternate-ordinal-dialogue",
                        lambda: checks.ordinal_flow(variant=True),
                    ),
                    ("ambiguity-and-fresh-search", checks.ambiguous),
                    ("review-outage-and-missing-product", checks.missing_and_recovery),
                    ("missing-model-outage-and-recovery", checks.provider_recovery),
                    ("history-context", checks.ranking_context),
                    *cases,
                ]
            all_case_ids = [name for name, _ in cases]
            if args.only:
                requested = set(args.only.split(","))
                cases = [(name, case) for name, case in cases if name in requested]
                if not cases:
                    parser.error("No requested scenarios match this runner")
            selected_ids = {name for name, _ in cases}
            report = {
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "policy": "legacy " + args.legacy_policy
                if args.legacy_policy
                else "model-led loop",
                "streamlit_entry": "shopping_agent/ui.py (AppTest)",
                "acceptance_harness_sha256": digest(Path(__file__)),
                "ollama_model": args.ollama_model,
                "ollama_model_digest": model_digest,
                "application_source_sha256": {
                    str(path.relative_to(checks.program.parent)): digest(path)
                    for path in sorted(checks.program.parent.glob("*.py"))
                },
                "catalog_sha256": digest(args.recommender / "data/item_catalog.csv"),
                "review_index_sha256": digest(args.review_snapshot),
                "frozen_scenarios_sha256": digest(FROZEN_FILE),
                "selected_scenarios": args.only or "all",
                "controlled_test": (
                    "First evidence-tool response uses explicitly labeled test excerpts. "
                    "Later responses use the normal dataset. Ollama is real."
                ),
            }

            def save(results, active_case=None):
                frozen_results = [
                    {**case, "id": case["id"].removeprefix("frozen/")}
                    for case in results
                    if case["id"].startswith("frozen/")
                ]
                measured = {case["id"] for case in results}
                report.update(
                    cases=results,
                    active_case=active_case,
                    run_complete=measured == selected_ids
                    and (
                        not args.compare_fixed
                        or not frozen_results
                        or "fixed_comparator" in report
                    ),
                    skipped_cases=[
                        {
                            "id": name,
                            "status": "skipped",
                            "reason": "Awaiting execution"
                            if name in selected_ids
                            else "Excluded by --only",
                        }
                        for name in all_case_ids
                        if name not in measured
                    ],
                    frozen_only_metrics=metrics(frozen_results),
                )
                write_report(args.output, report)

            save([])
            results = checks.run(cases, save)

    frozen_ids = [case["id"] for case in results if case["id"].startswith("frozen/")]
    if args.compare_fixed and frozen_ids:
        baseline_args = argparse.Namespace(
            **{**vars(args), "only": ",".join(frozen_ids)}
        )
        report["fixed_comparator"] = run_legacy(baseline_args)
        report["fixed_comparator_archive_sha256"] = digest(BASELINE_ARCHIVE)
        save(results)
    return int(any(case["status"] != "passed" for case in results))


if __name__ == "__main__":
    raise SystemExit(main())
