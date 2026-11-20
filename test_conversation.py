"""Offline regressions for LLM-guided turns and grounded conversation state."""

from contextlib import closing
from decimal import Decimal

import pytest

from shopping_agent.catalog import SnapshotData, _SnapshotReader
from shopping_agent.presentation import chat_lines
from shopping_agent.provider import ProviderError
from shopping_agent.service import Agent
from shopping_agent.storage import Store


class ConversationModel:
    name = "ollama:test-local"
    local_intents = True

    def __init__(self, *turns):
        self.turns = iter(turns)
        self.contexts = []
        self.last_usage = {"input_tokens": 40, "output_tokens": 20}

    def parse(self, text, state, timeout):
        self.contexts.append(state)
        answer = next(self.turns)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _catalog(products, evidence=()):
    return _SnapshotReader(
        SnapshotData.model_validate(
            {
                "manifest": {
                    "snapshot": "electronics-chat-test",
                    "origin": "synthetic",
                    "categories": ["Electronics"],
                    "description": "Small chat regression catalog",
                },
                "products": [
                    {
                        "id": pid,
                        "title": title,
                        "price": price,
                        "category": "Electronics",
                    }
                    for pid, title, price in products
                ],
                "evidence": list(evidence),
            }
        ),
        mode="test",
    )


@pytest.fixture
def catalog():
    return _catalog(
        [
            ("a", "Wireless Headphones Studio", "24.90"),
            ("b", "Wireless Headphones Travel", "12.95"),
            ("c", "Wireless Headphones Basic", "6.99"),
            ("d", "HDMI Cable", "1.00"),
        ]
    )


def test_local_conversation_keeps_keyword_budget_and_product_references(catalog):
    provider = ConversationModel(
        {"clarification": "Hi! What kind of electronics are you looking for?"},
        {"query": "wireless headphones"},
        {},
        {"intent": "compare", "references": [1, 2], "priority": "price"},
        {"intent": "evidence", "references": [1]},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=provider)
        ready = agent.submit("s", "1", "Hello there!")
        initial = agent.submit("s", "2", "I need wireless headphones for at most $40.")
        updated = agent.submit("s", "3", "Could we bring it down to $20?")
        after_update = store.get_state("s")
        compared = agent.submit("s", "4", "How much do the first two differ?")
        evidence = agent.submit("s", "5", "Any reviews of the first one?")

    assert ready.question == "Hi! What kind of electronics are you looking for?"
    assert ready.tool_attempts == 0
    assert [card.id for card in initial.verified] == ["a", "b", "c"]
    assert [card.id for card in updated.verified] == ["b", "c"]
    assert after_update["search_query"] == "wireless headphones"
    assert Decimal(after_update["max_price_usd"]) == 20
    assert provider.contexts[2]["search_query"] == "wireless headphones"
    assert provider.contexts[1]["available_categories"] == ("Electronics",)
    assert compared.shown_ids() == ["b", "c"]
    assert compared.price_difference_usd == Decimal("5.96")
    assert compared.decision is None
    assert evidence.outcome == "PARTIAL"
    assert evidence.shown_ids() == ["b"]
    assert not evidence.evidence
    for result in (ready, initial, updated, compared, evidence):
        assert result.model_attempts == 1
        assert {"action": "model_parse", "status": "OK"} in result.trace
        assert not any(row["action"] == "model_decide" for row in result.trace)
    assert "wireless headphones" in " ".join(chat_lines(updated, after_update))
    assert "5.96" in " ".join(chat_lines(compared, {}))
    assert "matching review excerpt" in " ".join(chat_lines(evidence, {}))


def test_local_model_cannot_invent_budget_and_failed_turn_does_not_fallback(catalog):
    provider = ConversationModel(
        {"query": "wireless headphones"},
        {"operations": [{"field": "max_price_usd", "op": "set", "value": "999"}]},
        ProviderError("OLLAMA_UNAVAILABLE", "Cannot reach Ollama."),
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=provider)
        agent.submit("s", "1", "Wireless headphones under $15")
        result = agent.submit("s", "2", "Show me again")
        failed = agent.submit("s", "3", "Change the budget to $10")
        state = store.get_state("s")

    assert [card.id for card in result.verified] == ["b", "c"]
    assert Decimal(state["max_price_usd"]) == 15
    assert failed.outcome == "TEMPORARILY_UNAVAILABLE"
    assert failed.tool_attempts == 0
    assert failed.trace == [{"action": "model_parse", "status": "OLLAMA_UNAVAILABLE"}]


def test_local_can_clear_search_and_keep_budget(catalog):
    provider = ConversationModel(
        {"query": "wireless headphones"},
        {"clear_query": True},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=provider)
        agent.submit("s", "1", "I'm looking for wireless headphones, at most $15.")
        result = agent.submit(
            "s", "2", "Clear the keyword and browse the whole category"
        )
        state = store.get_state("s")
    assert state["search_query"] == ""
    assert Decimal(state["max_price_usd"]) == 15
    assert "d" in result.shown_ids()


def test_model_can_choose_a_bounded_followup_after_seeing_details(catalog):
    class FollowupModel(ConversationModel):
        def choose_followup(self, observation, timeout):
            assert observation["last_tool"]["action"] == "get_products"
            assert observation["products_with_known_price"] == 2
            assert observation["available_actions"] == ["retrieve_reviews", "finish"]
            return "finish"

    model = FollowupModel({"query": "wireless headphones"}, {"intent": "compare"})
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        agent.submit("s", "1", "I need wireless headphones for at most $30")
        result = agent.submit("s", "2", "How much do the first two differ?")

    assert result.price_difference_usd == Decimal("11.95")
    assert {
        "action": "model_decide",
        "status": "SELECTED",
        "proposal": "finish",
    } in result.trace
    assert not any(event["action"] == "get_product_evidence" for event in result.trace)


def test_sequential_reviews_keep_original_numbering_and_accept_pasted_asin():
    first, second, third = "B00GLSRBLQ", "B07NY21DF6", "B0BHQJBM3M"
    catalog = _catalog(
        [
            (first, "Wearable kit", "18.61"),
            (second, "Forza Horizon 4", "25.00"),
            (third, "USB hub", "35.97"),
        ],
        [
            {
                "id": "review-1",
                "product_id": second,
                "text": "The game runs smoothly.",
                "source": "Test excerpt",
            }
        ],
    )
    provider = ConversationModel(
        {"query": "wearable"},
        {"query": "electronic"},
        {"intent": "evidence", "references": [1]},
        {"intent": "compare", "references": [2]},
        {"intent": "evidence", "references": [1]},
        {"intent": "evidence", "references": []},
        {"query": "USB hub"},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=provider)
        agent.submit("s", "1", "Find wearable gear under $30")
        initial = agent.submit("s", "2", "Give me any electronic under $40")
        first_review = agent.submit("s", "3", "I need all reviews for first oe")
        assert store.get_state("s")["displayed_ids"] == [first, second, third]
        second_review = agent.submit("s", "4", "OK, How about seconed one")
        pasted = agent.submit(
            "s",
            "5",
            f"I mean do you have any comments for this one:\n1. Forza Horizon 4 [{second}] · USD 25.00",
        )
        followup = agent.submit("s", "6", "Any reviews for this one?")
        stable_state = store.get_state("s")
        new_search = agent.submit("s", "7", "What about a USB hub under $40?")
        final_state = store.get_state("s")

    assert initial.shown_ids() == [first, second, third]
    assert first_review.missing_review_ids == [first]
    assert first_review.missing_product_ids == []
    for result in (second_review, pasted, followup):
        assert result.intent == "evidence"
        assert result.shown_ids() == [second]
        assert {quote.product_id for quote in result.evidence} == {second}
        assert any(
            row.startswith("2. Forza Horizon 4")
            for row in chat_lines(result, stable_state)
        )
    assert provider.contexts[2]["search_query"] == ""
    assert stable_state["displayed_ids"] == [first, second, third]
    assert new_search.shown_ids() == [third]
    assert final_state["followup"] is None
    assert final_state["displayed_ids"] == [third]


def test_asin_without_a_search_checks_catalog_and_distinguishes_review_coverage():
    covered, uncovered, unknown = "B012345678", "B012345679", "B012345670"
    catalog = _catalog(
        [(covered, "DX Headphones", "20.00"), (uncovered, "DX Speakers", "25.00")]
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog)
        missing_reviews = agent.submit("s", "1", f"Any reviews for ASIN {uncovered}?")
        missing_product = agent.submit("s", "2", f"Any reviews for [{unknown}]?")
        state = store.get_state("s")

    assert missing_reviews.shown_ids() == [uncovered]
    assert missing_reviews.missing_review_ids == [uncovered]
    assert missing_product.outcome == "NO_VERIFIED_MATCH_IN_SCOPE"
    assert missing_product.missing_product_ids == [unknown]
    assert not missing_product.missing_review_ids
    assert all(
        action["action"] != "get_product_evidence" for action in missing_product.trace
    )
    assert state["displayed_ids"] == []
