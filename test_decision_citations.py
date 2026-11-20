"""Offline contract checks for model claims and application-issued citations."""

from contextlib import closing
from unittest.mock import patch

import pytest

from shopping_agent.catalog import SnapshotData, _SnapshotReader
from shopping_agent.presentation import chat_lines
from shopping_agent.service import Agent
from shopping_agent.storage import Store

FIRST = "B012345671"
SECOND = "B012345672"
UNPRICED = "B012345673"
QUOTE_ID = "fixture-setup-review"
QUOTE = (
    "I found this smart switch easy to set up by following the pairing instructions."
)
SOURCE = "Simulated review sample, line 1"


@pytest.fixture
def catalog():
    snapshot = SnapshotData.model_validate(
        {
            "manifest": {
                "snapshot": "citation-test",
                "origin": "synthetic",
                "categories": ["Electronics"],
                "description": "Simulated switches and a single attributed review",
            },
            "products": [
                {"id": asin, "title": title, "price": price, "category": "Electronics"}
                for asin, title, price in (
                    (FIRST, "Switch A", "14.00"),
                    (SECOND, "Switch B", "18.00"),
                    (UNPRICED, "Switch without a price", None),
                )
            ],
            "evidence": [
                {"id": QUOTE_ID, "product_id": SECOND, "text": QUOTE, "source": SOURCE}
            ],
        }
    )
    return _SnapshotReader(snapshot, mode="team-electronics")


class ScriptedModel:
    name = "ollama:simulated-citation"
    local_intents = True

    def __init__(self, bad_claim):
        self.calls = []
        self.replies = iter(
            [
                {"action": "get_products", "task": "compare", "aspect": "setup"},
                {"action": "get_product_evidence", "aspect": "set up"},
                {"action": "finish", "basis": "reviews", "claims": [bad_claim]},
                {
                    "action": "finish",
                    "basis": "reviews",
                    "choice": SECOND,
                    "claims": [
                        {
                            "product_id": SECOND,
                            "citation": QUOTE_ID,
                            "observation": "the switch was easy to set up",
                        }
                    ],
                },
            ]
        )

    def decide(self, observation, timeout):
        assert timeout > 0
        self.calls.append(observation)
        return next(self.replies)


@pytest.mark.parametrize(
    ("bad_claim", "feedback"),
    [
        pytest.param(
            {
                "product_id": FIRST,
                "citation": QUOTE_ID,
                "observation": "the switch was easy to set up",
            },
            "same product",
            id="cross-product-citation",
        ),
        pytest.param(
            {
                "product_id": SECOND,
                "citation": "fabricated-source",
                "observation": "the switch was easy to set up",
            },
            "same product",
            id="fabricated-citation",
        ),
        pytest.param(
            {
                "product_id": SECOND,
                "citation": QUOTE_ID,
                "observation": "the switch is in stock and costs $2",
            },
            "qualitative",
            id="unverified-number-and-stock",
        ),
    ],
)
def test_application_rejects_untrusted_claim_and_accepts_attributed_correction(
    catalog, bad_claim, feedback
):
    model = ScriptedModel(bad_claim)
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        result = agent.submit(
            "simulated",
            "1",
            f"Between ASIN {FIRST} and ASIN {SECOND}, which is easier to set up?",
        )
        reply = "\n".join(chat_lines(result, store.get_state("simulated")))
    details = model.calls[1]["TOOL_DATA"]["products"]
    assert {asin: details[asin]["historical_usd"] for asin in (FIRST, SECOND)} == {
        FIRST: "14.00",
        SECOND: "18.00",
    }
    reviews = model.calls[2]["TOOL_DATA"]["review_excerpts"]
    assert reviews == [
        {"id": QUOTE_ID, "product_id": SECOND, "text": QUOTE, "source": SOURCE}
    ]
    assert feedback in model.calls[3]["feedback"]
    rejection = next(row for row in result.trace if row["status"] == "REJECTED")
    assert rejection["selected"] == "finish"
    published = [row for row in result.trace if row["action"] == "publish_answer"]
    assert len(published) == 1
    assert published[0]["product_id"] == SECOND
    assert published[0]["citations"] == [
        {"id": QUOTE_ID, "product_id": SECOND, "source": SOURCE}
    ]
    assert "A buyer of " + SECOND in reply and QUOTE in reply
    assert (
        bad_claim["citation"] != "fabricated-source" or "fabricated-source" not in reply
    )
    assert "$2" not in reply and "in stock" not in reply


def test_observation_can_change_the_next_tool_under_identical_permissions(catalog):
    original = catalog._product_evidence
    raw_runs = {}
    for observation in ("relevant", "irrelevant"):
        contexts = []
        evidence_args = []

        def first_source(args, deadline, *, arm=observation, calls=evidence_args):
            response = original(args, deadline)
            calls.append(args)
            if arm == "irrelevant" and len(calls) == 1:
                watered = response.evidence[0].model_copy(
                    update={
                        "id": "fixture-packaging-review",
                        "text": "The pairing logo appeared on the cardboard box.",
                    }
                )
                return response.model_copy(update={"evidence": [watered]})
            return response

        class ObservationModel:
            name = "ollama:simulated-observation"
            local_intents = True

            def decide(self, context, timeout, *, seen=contexts):
                assert timeout > 0
                seen.append(context)
                step = len(seen)
                if step == 1:
                    return {
                        "action": "get_products",
                        "task": "compare",
                        "aspect": "setup",
                    }
                if step == 2:
                    return {"action": "get_product_evidence", "aspect": "pairing"}
                evidence = context["TOOL_DATA"]["review_excerpts"]
                if not any("easy" in quote["text"] for quote in evidence):
                    return {"action": "get_product_evidence", "aspect": "easy set up"}
                return {
                    "action": "finish",
                    "basis": "reviews",
                    "choice": SECOND,
                    "claims": [
                        {
                            "product_id": SECOND,
                            "citation": QUOTE_ID,
                            "observation": "setup was easy using the pairing instructions",
                        }
                    ],
                }

        with (
            patch.object(catalog, "_product_evidence", first_source),
            closing(Store(":memory:")) as store,
        ):
            result = Agent(store, catalog, provider=ObservationModel()).submit(
                "paired-simulation",
                "1",
                f"I want the simpler switch to set up. My options are {FIRST} and {SECOND}.",
            )
        raw_runs[observation] = (contexts, evidence_args, result)

    enough, plenty_calls, enough_result = raw_runs["relevant"]
    lacking, sparse_calls, lacking_result = raw_runs["irrelevant"]
    for key in (
        "available_actions",
        "accepted_constraints",
        "user_request",
        "selected_asins",
    ):
        assert enough[2][key] == lacking[2][key]
    assert plenty_calls == sparse_calls[:1]
    assert len(plenty_calls) == 1
    assert len(sparse_calls) == 2
    assert sparse_calls[1]["query"] == "easy set up"
    for result in (enough_result, lacking_result):
        published = next(
            row for row in result.trace if row["action"] == "publish_answer"
        )
        assert published["product_id"] == SECOND
        assert published["citations"][0]["id"] == QUOTE_ID


def test_catalog_prices_have_three_states_and_no_match_preserves_numbers(catalog):
    class DiscoveryModel:
        name = "ollama:simulated-discovery"
        local_intents = True
        replies = iter(
            [
                {"action": "search_products", "task": "discover", "query": "switch"},
                {"action": "get_products"},
                {"action": "finish", "basis": "results"},
                {
                    "action": "search_products",
                    "task": "discover",
                    "query": "film scanner",
                },
                {"action": "finish", "basis": "no_match"},
            ]
        )

        def decide(self, _context, timeout):
            assert timeout > 0
            return next(self.replies)

    search = catalog._candidate_ids

    def with_service_price_groups(action, args, deadline):
        response = search(action, args, deadline)
        if action == "search_products" and args.get("query") == "switch":
            # The real Electronics adapter reports separate price groups.
            return response.model_copy(
                update={
                    "ids": [FIRST],
                    "pending_ids": [UNPRICED],
                    "above_budget_ids": [SECOND],
                }
            )
        return response

    with (
        patch.object(catalog, "_candidate_ids", with_service_price_groups),
        closing(Store(":memory:")) as store,
    ):
        agent = Agent(store, catalog, provider=DiscoveryModel())
        found = agent.submit(
            "budget-example", "1", "Find switches with a budget of $15."
        )
        initial_state = store.get_state("budget-example")
        absent = agent.submit(
            "budget-example", "2", "Start a new search for a film scanner."
        )
        restored = store.get_state("budget-example")
    assert [card.id for card in found.verified] == [FIRST]
    assert [card.id for card in found.above_budget] == [SECOND]
    assert [card.id for card in found.pending] == [UNPRICED]
    assert "USD 14.00" in "\n".join(chat_lines(found, initial_state))
    assert not absent.shown_ids()
    assert (
        initial_state["displayed_ids"] == restored["displayed_ids"] == [FIRST, UNPRICED]
    )
