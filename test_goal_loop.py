"""Acceptance units for the tool boundary and mixed-turn publication."""

from contextlib import closing
from decimal import Decimal

import pytest

from shopping_agent.catalog import CatalogResponse, Evidence, Product
from shopping_agent.contracts import RecommendationRequest
from shopping_agent.presentation import chat_lines
from shopping_agent.service import Agent
from shopping_agent.storage import Store
from shopping_agent.workflow import Limits

FIRST = "B012345671"
SECOND = "B012345672"
RANKED = "B012345673"
REVIEW_ID = "synthetic-setup-problem"
REVIEW_TEXT = "Setup was difficult and the switch never connected."
DISCOVER = {"action": "search_products", "task": "discover", "query": "smart plug"}
COMPOUND = (
    "Find two smart plug options under $40. Check reviews for underwater use on both. "
    "If the sample lacks useful evidence, pick the cheaper by catalog price."
)


class FixtureCatalog:
    """Small labeled Electronics fixture with injectable source failures."""

    categories = ("Electronics",)
    mode = "team-electronics"
    snapshot = "goal-loop-unit-fixture"

    def __init__(
        self, *, details_fail=False, reviews_fail=False, excerpts=None, review_gap=None
    ):
        self.details_fail = details_fail
        self.reviews_fail = reviews_fail
        self.excerpts = excerpts
        self.review_gap = review_gap
        self.calls = []
        self.products = {
            asin: Product(id=asin, category="Electronics", title=title, price=price)
            for asin, title, price in (
                (FIRST, "Fixture Smart Plug A", "14.00"),
                (SECOND, "Fixture Smart Plug B", "18.00"),
                (RANKED, "Unrelated History Charger", "20.00"),
            )
        }

    def call(self, action, args, timeout):
        assert timeout > 0
        self.calls.append((action, args))
        if action == "search_products":
            ids = [] if args["query"] == "film scanner" else [FIRST, SECOND]
            if "candidate_ids" in args:
                ids = [asin for asin in args["candidate_ids"] if asin in ids]
            return CatalogResponse(
                status="OK" if ids else "EMPTY", snapshot=self.snapshot, ids=ids
            )
        if action == "get_products":
            if self.details_fail:
                return self.outage()
            return CatalogResponse(
                status="OK",
                snapshot=self.snapshot,
                products=[self.products[asin] for asin in args["ids"]],
            )
        if action == "get_product_evidence":
            if self.reviews_fail:
                return self.outage()
            if self.excerpts is not None:
                return CatalogResponse(
                    status="OK",
                    snapshot=self.snapshot,
                    evidence=[
                        Evidence(
                            id=citation,
                            product_id=asin,
                            text=quote,
                            source=self.snapshot,
                        )
                        for citation, asin, quote in self.excerpts
                        if asin in args["ids"]
                    ],
                )
            if self.review_gap or "underwater" in args.get("query", ""):
                return CatalogResponse(
                    status="EMPTY",
                    snapshot=self.snapshot,
                    missing_ids=args["ids"],
                    review_gaps={
                        asin: self.review_gap or "NO_ASPECT_MATCH"
                        for asin in args["ids"]
                    },
                )
            assert FIRST in args["ids"]
            return CatalogResponse(
                status="OK",
                snapshot=self.snapshot,
                evidence=[
                    Evidence(
                        id=REVIEW_ID,
                        product_id=FIRST,
                        text=REVIEW_TEXT,
                        source="Synthetic unit-test review",
                    )
                ],
            )
        raise AssertionError(action)

    def outage(self):
        return CatalogResponse(
            status="ERROR",
            snapshot=self.snapshot,
            error_code="fixture-outage",
            message="Injected fixture outage",
        )


class ScriptedPlanner:
    name = "ollama:goal-loop-fixture"
    local_intents = True

    def __init__(self, *proposals):
        self.proposals = iter(proposals)
        self.contexts = []

    def decide(self, context, timeout):
        assert timeout > 0
        self.contexts.append(context)
        return next(self.proposals)


class FixtureRanker:
    def __init__(self):
        self.calls = []

    def recommend(self, user_id, model, n, timeout):
        assert timeout > 0
        self.calls.append((user_id, model, n))
        return [RANKED, FIRST]


def run(catalog, planner, text, *, recommendation=None, ranker=None, limits=None):
    with closing(Store(":memory:")) as store:
        result = Agent(
            store, catalog, provider=planner, recommender=ranker, limits=limits
        ).submit("unit-session", "turn", text, recommendation=recommendation)
        state = store.get_state("unit-session")
        return result, state, "\n".join(chat_lines(result, state))


@pytest.mark.parametrize("profile", [None, "sample-ncf", "sample-tiger", "unknown-id"])
def test_plain_search_keeps_history_out_even_if_planner_proposes_it(profile):
    planner = ScriptedPlanner(
        DISCOVER,
        {"action": "recommend_products"},
        {"action": "search_products", "scope": "ranked"},
        {"action": "get_products"},
        {"action": "finish", "basis": "results"},
    )
    ranker = FixtureRanker()
    request = (
        RecommendationRequest(
            user_id=profile, model="tiger" if profile == "sample-tiger" else "ncf", n=2
        )
        if profile
        else None
    )
    result, state, _ = run(
        FixtureCatalog(),
        planner,
        "Find smart plug options under $40.",
        recommendation=request,
        ranker=ranker,
    )
    assert not ranker.calls
    assert all(
        "recommend_products" not in observation["available_actions"]
        for observation in planner.contexts
    )
    assert [card.id for card in result.verified] == [FIRST, SECOND]
    assert state["displayed_ids"] == [FIRST, SECOND]
    assert result.history_mode == "catalog_only"


def test_explicit_history_preference_uses_checkpoint_as_one_source():
    catalog = FixtureCatalog()
    planner = ScriptedPlanner(
        DISCOVER,
        {"action": "recommend_products"},
        {"action": "search_products", "scope": "ranked"},
        {"action": "get_products"},
        {"action": "finish", "basis": "results"},
    )
    ranker = FixtureRanker()
    result, _, _ = run(
        catalog,
        planner,
        "Find smart plug options under $40 and use my purchase history to prioritize matching recommendations.",
        recommendation=RecommendationRequest(user_id="known", model="ncf", n=2),
        ranker=ranker,
    )
    assert ranker.calls == [("known", "ncf", 2)]
    assert [card.id for card in result.verified] == [FIRST, SECOND]
    assert result.verified[0].model_rank == 2
    assert result.verified[1].model_rank is None
    assert result.history_mode == "history_preferred"


def test_pending_eligible_candidate_prevents_no_match_finish():
    planner = ScriptedPlanner(
        DISCOVER,
        {"action": "get_products", "ids": [FIRST]},
        {"action": "finish", "basis": "no_match"},
        {"action": "get_products", "ids": [SECOND]},
        {"action": "finish", "basis": "results"},
    )
    result, _, _ = run(
        FixtureCatalog(), planner, "Find two smart plug options under $40."
    )
    assert "no_match" not in planner.contexts[2]["available_final_bases"]
    assert any(
        row.get("status") == "REJECTED" and row.get("selected") == "finish"
        for row in result.trace
    )
    assert [card.id for card in result.verified] == [FIRST, SECOND]


def test_repeated_step_preserves_verified_partial_result_when_budget_ends():
    detail = {"action": "get_products", "ids": [FIRST]}
    planner = ScriptedPlanner(DISCOVER, detail, detail, detail)
    result, state, reply = run(
        FixtureCatalog(),
        planner,
        "Find two smart plug options under $40.",
        limits=Limits(model_attempts=4),
    )
    assert result.outcome == "PARTIAL"
    assert [card.id for card in result.verified] == [FIRST]
    assert SECOND in result.unchecked_candidate_ids
    assert state["displayed_ids"] == [FIRST]
    assert FIRST in reply
    assert sum(row["action"] == "get_products" for row in result.trace) == 1


def test_review_only_followup_preserves_the_numbered_list():
    planner = ScriptedPlanner(
        DISCOVER,
        {"action": "get_products"},
        {"action": "finish", "basis": "results"},
        {"action": "get_product_evidence", "task": "evidence", "aspect": ""},
        {"action": "finish", "basis": "review_excerpts"},
        {"action": "search_products", "task": "discover", "query": "second one"},
        {"action": "get_product_evidence", "task": "evidence", "aspect": ""},
        {"action": "finish", "basis": "review_excerpts"},
    )
    catalog = FixtureCatalog(
        excerpts=[
            (REVIEW_ID, FIRST, REVIEW_TEXT),
            ("second-quote", SECOND, "I paired it with the app on my first try."),
        ]
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=planner)
        agent.submit("session", "list", "Find two smart plugs under $40.")
        first = agent.submit("session", "review-one", "Show me reviews for option one.")
        assert first.selected_ids() == [FIRST]
        assert not first.comparison
        assert store.get_state("session")["followup"]["ids"] == [FIRST]
        second = agent.submit("session", "review-two", "How about the seconed one?")
        assert second.selected_ids() == [SECOND]
        assert {quote.product_id for quote in second.evidence} == {SECOND}
        assert store.get_state("session")["displayed_ids"] == [FIRST, SECOND]
    assert [action for action, _ in catalog.calls] == [
        "search_products",
        "get_products",
        "get_product_evidence",
        "get_product_evidence",
    ]


def test_review_only_outside_sample_is_a_valid_scoped_result():
    planner = ScriptedPlanner(
        {"action": "get_product_evidence", "task": "evidence", "aspect": ""},
        {"action": "finish", "basis": "insufficient_reviews"},
    )
    result, state, reply = run(
        FixtureCatalog(review_gap="OUTSIDE_SAMPLE"),
        planner,
        f"Could you show comments for this product: {FIRST}?",
    )
    assert result.outcome == "PARTIAL"
    assert result.selected_ids() == [FIRST]
    assert result.review_gaps == {FIRST: "OUTSIDE_SAMPLE"}
    assert state["followup"]["ids"] == [FIRST]
    assert f"The review sample has no excerpts for {FIRST}." in reply
    assert "temporarily unavailable" not in reply.casefold()


def test_detail_failure_does_not_allow_scoped_no_match():
    planner = ScriptedPlanner(
        DISCOVER,
        {"action": "get_products"},
        {"action": "finish", "basis": "no_match"},
        {"action": "finish", "basis": "unavailable"},
    )
    result, _, _ = run(
        FixtureCatalog(details_fail=True), planner, "Find smart plugs under $40."
    )
    assert result.outcome == "TEMPORARILY_UNAVAILABLE"
    assert set(result.unchecked_candidate_ids) == {FIRST, SECOND}
    assert all(
        row.get("basis") != "no_match"
        for row in result.trace
        if row["action"] == "publish_answer"
    )


def test_compound_discovery_publishes_list_and_qualified_price_choice():
    planner = ScriptedPlanner(
        {**DISCOVER, "review_goal": "underwater use"},
        {"action": "get_products"},
        {
            "action": "get_product_evidence",
            "aspect": "underwater use",
            "ids": [FIRST, SECOND],
        },
        {"action": "finish", "basis": "price"},
    )
    result, state, reply = run(FixtureCatalog(), planner, COMPOUND)
    assert [card.id for card in result.verified] == [FIRST, SECOND]
    assert state["displayed_ids"] == [FIRST, SECOND]
    assert result.intent == "discover"
    assert result.decision is not None
    assert result.decision.product_id == FIRST
    assert result.decision.basis == "price"
    assert result.price_difference_usd == Decimal("4.00")
    assert result.review_gaps == {FIRST: "NO_ASPECT_MATCH", SECOND: "NO_ASPECT_MATCH"}
    assert all(asin in reply for asin in (FIRST, SECOND))


def test_topical_mentions_do_not_block_the_shoppers_evidence_fallback():
    catalog = FixtureCatalog(
        excerpts=[
            ("first-note", FIRST, "I never tried underwater use."),
            ("second-note", SECOND, "I cannot speak about underwater use."),
        ]
    )
    planner = ScriptedPlanner(
        {**DISCOVER, "review_goal": "underwater use"},
        {"action": "get_products"},
        {"action": "get_product_evidence"},
        {"action": "finish", "basis": "price"},
    )
    result, _, reply = run(catalog, planner, COMPOUND)
    last_context = planner.contexts[-1]
    assert last_context["TOOL_DATA"]["review_scope"]["topic_matching_product_ids"] == [
        FIRST,
        SECOND,
    ]
    assert {"price", "reviews"} <= set(last_context["available_final_bases"])
    assert result.decision is not None
    assert result.decision.product_id == FIRST
    assert "leave the requested use unresolved" in reply


def test_compound_review_outage_does_not_authorize_conditional_price_fallback():
    planner = ScriptedPlanner(
        {**DISCOVER, "review_goal": "underwater use"},
        {"action": "get_products"},
        {
            "action": "get_product_evidence",
            "aspect": "underwater use",
            "ids": [FIRST, SECOND],
        },
        {"action": "finish", "basis": "price"},
    )
    result, _, reply = run(
        FixtureCatalog(reviews_fail=True),
        planner,
        COMPOUND,
        limits=Limits(model_attempts=4),
    )
    assert result.outcome == "PARTIAL"
    assert result.decision is None
    assert [card.id for card in result.verified] == [FIRST, SECOND]
    assert "unavailable" in reply.casefold() or "failed" in reply.casefold()


def test_same_asin_reverse_meaning_paraphrase_cannot_reach_the_reply():
    reversing = "Setup was simple and the switch connected."
    planner = ScriptedPlanner(
        {"action": "get_products", "task": "evidence", "aspect": "setup"},
        {"action": "get_product_evidence", "aspect": "setup"},
        {
            "action": "finish",
            "basis": "reviews",
            "claims": [
                {"product_id": FIRST, "citation": REVIEW_ID, "observation": reversing},
            ],
        },
        {
            "action": "finish",
            "basis": "reviews",
            "claims": [
                {
                    "product_id": FIRST,
                    "citation": REVIEW_ID,
                    "observation": REVIEW_TEXT,
                },
            ],
        },
    )
    result, _, reply = run(
        FixtureCatalog(), planner, f"What do reviews say about setup for ASIN {FIRST}?"
    )
    assert REVIEW_TEXT in reply
    assert reversing not in reply
    assert [quote.id for quote in result.evidence] == [REVIEW_ID]


@pytest.mark.parametrize("relevant", [True, False])
def test_review_first_keeps_the_comparative_goal_and_checks_relevance(relevant):
    second_quote = (
        "Setup took two minutes and the device paired immediately."
        if relevant
        else "The package arrived in a large brown cardboard box."
    )
    excerpts = [("second-review", SECOND, second_quote)]
    first_quote = REVIEW_TEXT if relevant else "The delivery arrived in the evening."
    excerpts.append((REVIEW_ID, FIRST, first_quote))
    followup = []
    if relevant:
        finish = {
            "action": "finish",
            "basis": "reviews",
            "choice": SECOND,
            "claims": [
                {
                    "product_id": SECOND,
                    "citation": "second-review",
                    "observation": second_quote,
                }
            ],
        }
    else:
        followup = [{"action": "get_product_evidence", "aspect": "set up"}]
        finish = {"action": "finish", "basis": "insufficient_reviews"}
    planner = ScriptedPlanner(
        {"action": "get_product_evidence", "task": "compare", "aspect": "setup"},
        {"action": "get_products"},
        *followup,
        finish,
    )
    result, _, reply = run(
        FixtureCatalog(excerpts=excerpts),
        planner,
        f"I'm choosing between {FIRST} and {SECOND}. Easy setup matters to me. Which choice does the evidence support?",
    )
    assert planner.contexts[1]["task"] == "compare"
    assert "review_excerpts" not in planner.contexts[1]["available_final_bases"]
    assert {card.id for card in result.comparison} == {FIRST, SECOND}
    published = next(row for row in result.trace if row["action"] == "publish_answer")
    if relevant:
        assert published["basis"] == "reviews"
        assert published["product_id"] == SECOND
        assert second_quote in reply
    else:
        assert result.decision is None
        assert published["basis"] == "insufficient_reviews"
        assert second_quote not in reply


@pytest.mark.parametrize("catalog_fact_required", [False, True])
def test_qualitative_finish_only_skips_unneeded_catalog_facts(catalog_fact_required):
    finish = {
        "action": "finish",
        "basis": "reviews",
        "choice": SECOND,
        "claims": [{"product_id": SECOND, "citation": "second-quote"}],
    }
    proposals = [
        {"action": "get_product_evidence", "task": "compare", "aspect": "setup"},
        finish,
    ]
    request = (
        f"Which of {FIRST} and {SECOND} seems easier to set up from buyer reviews?"
    )
    if catalog_fact_required:
        request += " Keep the price under $40."
        proposals.extend(({"action": "get_products"}, finish))
    catalog = FixtureCatalog(
        excerpts=[
            (REVIEW_ID, FIRST, REVIEW_TEXT),
            ("second-quote", SECOND, "Setup was easy and the app connected."),
        ]
    )
    planner = ScriptedPlanner(*proposals)
    result, _, reply = run(catalog, planner, request)
    assert result.selected_ids() == [FIRST, SECOND]
    assert "Setup was easy and the app connected." in reply
    assert ("reviews" in planner.contexts[1]["available_final_bases"]) == (
        not catalog_fact_required
    )
    assert bool(result.comparison) == catalog_fact_required
    assert any(
        event["action"] == "publish_answer" and event.get("product_id") == SECOND
        for event in result.trace
    )
