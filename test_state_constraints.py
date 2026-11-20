from decimal import Decimal

import pytest

from shopping_agent.contracts import Session
from shopping_agent.state import (
    BUDGET_QUESTION,
    apply_patch,
    grounded_budget_proposal,
    history_usage,
    offline_parse,
)


@pytest.mark.parametrize(
    "text",
    [
        "Could we change the price limitation to $30?",
        "Set the limit to $30.",
        "Actually, set the same search limit to $30.",
        "I will not spend over $30.",
        "I paid under $18 last week. My price ceiling is $30.",
        "Find earbuds under $30 a pair.",
    ],
)
def test_explicit_ceiling_edits_budget_and_keeps_query(text):
    state = Session(category="Electronics", max_price_usd=40, search_query="earbuds")
    updated = apply_patch(state, offline_parse(text, state, conversational=True))
    assert updated.max_price_usd == Decimal(30)
    assert updated.search_query == "earbuds"


@pytest.mark.parametrize(
    "text",
    [
        "I saw earbuds under $30 last week.",
        "Do not change the price limitation to $30.",
        "Keep the same search. No budget changes.",
        "Find a hub with at most 8 ports.",
        "Not at most $30.",
    ],
)
def test_price_mentions_negation_and_measurements_preserve_budget(text):
    state = Session(category="Electronics", max_price_usd=40, search_query="earbuds")
    updated = apply_patch(state, offline_parse(text, state, conversational=True))
    assert updated.max_price_usd == Decimal(40)


@pytest.mark.parametrize(
    "text",
    [
        "Set the budget to $30, or up to $40.",
        "Set the budget to $30 or $40.",
        "Set the budget to 30–40 dollars.",
        "Clear the budget. Set the price limit to $30.",
        "Maybe set the budget to $30.",
    ],
)
def test_ambiguous_budget_requires_confirmation_before_any_write(text):
    state = Session(category="Electronics", max_price_usd=40, search_query="earbuds")
    proposed = offline_parse(text, state, conversational=True)
    assert proposed.clarification == BUDGET_QUESTION
    assert apply_patch(state, proposed) == state
    pending = state.model_copy(update={"pending_question": BUDGET_QUESTION})
    assert apply_patch(pending, offline_parse("$30.", pending)).max_price_usd == 30


def test_semantic_budget_proposal_requires_a_literal_ceiling_and_exact_money():
    text = "The most I am comfortable spending is $30."
    proposed = grounded_budget_proposal(text, "30", text)
    assert proposed is not None and proposed.value == "30"
    assert grounded_budget_proposal(text, "25", text) is None
    assert grounded_budget_proposal(text, "30", "The budget is $30") is None
    assert grounded_budget_proposal(text, "30", "$30") is None
    text_with_alternatives = "The most I am comfortable spending is $30 or $40."
    assert (
        grounded_budget_proposal(text_with_alternatives, "30", text_with_alternatives)
        is None
    )
    assert (
        grounded_budget_proposal("I saw it under $30 yesterday", "30", "under $30")
        is None
    )
    assert (
        grounded_budget_proposal(
            "Do not set the limit to $30", "30", "set the limit to $30"
        )
        is None
    )
    assert (
        grounded_budget_proposal(
            "The most I can spend is CAD $30", "30", "The most I can spend is CAD $30"
        )
        is None
    )


@pytest.mark.parametrize(
    ("text", "mode"),
    [
        ("Find wireless earbuds under $40", "catalog_only"),
        ("I looked at my purchase history yesterday", "catalog_only"),
        ("How has this product's price history changed?", "catalog_only"),
        ("Do not use my history. Find wireless earbuds.", "catalog_only"),
        ("Find wireless earbuds without using my purchase history", "catalog_only"),
        ("Find wireless earbuds using my purchase history", "history_preferred"),
        ("Start with catalog search, then check my history", "history_preferred"),
        ("Only use candidates from my history", "history_only"),
        ("Use my shopping history as the only source", "history_only"),
    ],
)
def test_history_scope_is_grounded_in_the_user_message(text, mode):
    assert history_usage(text) == mode
