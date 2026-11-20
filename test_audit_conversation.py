"""Independent regression inputs for product references and accepted requirements."""

from contextlib import closing
from decimal import Decimal

import pytest

from shopping_agent.contracts import Session
from shopping_agent.service import Agent
from shopping_agent.state import CHOICE_CRITERION_QUESTION, offline_parse
from shopping_agent.storage import Store
from test_conversation import ConversationModel, _catalog


def test_explicit_product_name_cannot_be_replaced_by_an_ordinal():
    catalog = _catalog(
        [
            ("hub-a", "USB hub Compact", "15.00"),
            ("hub-b", "USB hub Travel", "19.00"),
            ("sony", "Sony WH-1000XM5 Headphones", "165.00"),
        ],
        [
            {
                "id": "hub-review",
                "product_id": "hub-a",
                "text": "Fits my desk.",
                "source": "Hub review",
            },
            {
                "id": "sony-review",
                "product_id": "sony",
                "text": "Quiet on my commute.",
                "source": "Sony review",
            },
        ],
    )
    name_calls = []

    def resolve_name(name, timeout):
        name_calls.append(name.casefold())
        return ["sony"] if "sony wh-1000xm5" in name.casefold() else ["hub-a", "hub-b"]

    catalog.resolve_product = resolve_name
    model = ConversationModel(
        {"query": "USB hub"},
        {"intent": "evidence", "references": [1]},
        {"intent": "evidence", "references": [1], "product_names": ["GoPro Hero 9"]},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        agent.submit("s", "search", "Find a USB hub under $22")
        sony = agent.submit(
            "s", "sony", "What do reviewers say about the Sony WH-1000XM5 headphones?"
        )
        ambiguous = agent.submit("s", "ambiguous", "Any reviews for GoPro Hero 9?")

    assert sony.shown_ids() == ["sony"]
    assert [quote.id for quote in sony.evidence] == ["sony-review"]
    assert ambiguous.outcome == "CLARIFICATION_NEEDED"
    assert ambiguous.shown_ids() == []
    assert ambiguous.evidence == []
    assert name_calls == ["sony wh-1000xm5 headphones", "gopro hero 9"]


def test_short_clarification_reply_resumes_the_original_review_aspect():
    catalog = _catalog(
        [("a", "USB hub Compact", "15.00"), ("b", "USB hub Travel", "19.00")],
        [
            {
                "id": "heat-b",
                "product_id": "b",
                "text": "A little heat.",
                "source": "Travel review",
            }
        ],
    )
    model = ConversationModel(
        {"query": "USB hub"},
        {"intent": "evidence", "query": "heat"},
        {"intent": "compare", "references": [1]},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        initial = agent.submit("s", "1", "Find a USB hub under $25")
        asked = agent.submit("s", "2", "Any reviews about heat for that one?")
        resumed = agent.submit("s", "3", "The second one.")
        state = store.get_state("s")
    assert initial.shown_ids() == ["a", "b"]
    assert asked.outcome == "CLARIFICATION_NEEDED"
    assert resumed.intent == "evidence"
    assert resumed.shown_ids() == ["b"]
    assert resumed.review_aspect == "heat"
    assert [quote.id for quote in resumed.evidence] == ["heat-b"]
    assert state["pending_reference"] is None


@pytest.mark.parametrize("amount", ["$22 max", "Max $22", "at most $22", "$22 or less"])
def test_money_and_port_counts_go_to_distinct_fields(amount):
    patch = offline_parse(
        f"Find a USB hub with at least six USB ports, {amount}. No USB 2.0 hubs.",
        Session(category="Electronics"),
        conversational=True,
    )
    assert {(op.field, op.value) for op in patch.operations} == {
        ("max_price_usd", "22")
    }
    assert {(r.kind, r.value) for r in patch.requirements} == {
        ("min_ports", "6"),
        ("excluded_usb_version", "2.0"),
    }


def test_maximum_port_count_and_usb_c_specific_count_do_not_become_a_budget():
    patch = offline_parse(
        "Find a USB hub with at most 8 USB ports and at least 2 USB-C ports, under $35.",
        Session(category="Electronics"),
        conversational=True,
    )
    assert {(op.field, op.value) for op in patch.operations} == {
        ("max_price_usd", "35")
    }
    assert {(r.kind, r.value) for r in patch.requirements} == {
        ("max_ports", "8"),
        ("property", "at least 2 usb-c ports"),
    }


def test_budget_edit_preserves_requirements_and_new_product_search_releases_them():
    catalog = _catalog(
        [
            ("hub", "USB Hub for Notebook", "15.00"),
            ("plug", "Smart Plug", "17.00"),
        ]
    )
    model = ConversationModel(
        {"query": "USB hub"},
        {},
        {"query": "smart plug"},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        hub = agent.submit(
            "s", "1", "Find a USB hub under $25. It must work with my MacBook Air M2."
        )
        updated = agent.submit("s", "2", "Change the budget to $20.")
        hub_state = store.get_state("s")
        plug = agent.submit("s", "3", "Now I need a smart plug.")
        plug_state = store.get_state("s")
    assert [r["kind"] for r in hub_state["requirements"]] == ["compatibility"]
    assert Decimal(hub_state["max_price_usd"]) == 20
    assert hub.verified == updated.verified == []
    assert [card.id for card in plug.verified] == ["plug"]
    assert plug_state["requirements"] == []
    assert Decimal(plug_state["max_price_usd"]) == 20


def test_negated_price_never_becomes_purchase_priority():
    catalog = _catalog([("a", "Smart Plug A", "25.00"), ("b", "Smart Plug B", "29.00")])
    model = ConversationModel(
        {"query": "smart plug"},
        {"intent": "compare", "references": [1, 2], "priority": "price"},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        agent.submit("s", "1", "Find a smart plug under $40")
        result = agent.submit(
            "s",
            "2",
            "Compare the first two. I do not want the cheaper one. Which should I pick to automate outdoor lights?",
        )
    assert result.shown_ids() == ["a", "b"]
    assert result.decision is None or result.decision.basis != "price"


def test_history_request_without_selected_user_does_not_search():
    catalog = _catalog([("a", "Smart Plug A", "25.00")])
    model = ConversationModel({"query": "shopping history"})
    with closing(Store(":memory:")) as store:
        result = Agent(store, catalog, provider=model).submit(
            "s", "1", "Based on my shopping history, recommend my next 5 products."
        )
    assert result.intent == "recommend"
    assert result.question and "dataset user" in result.question
    assert result.verified == []
    assert all(event["action"] != "search_products" for event in result.trace)


def test_direct_help_and_rephrase_keep_pending_shopping_context():
    catalog = _catalog(
        [("a", "USB hub Compact", "15.00"), ("b", "USB hub Travel", "19.00")],
        [
            {
                "id": "heat-b",
                "product_id": "b",
                "text": "A little heat.",
                "source": "Travel review",
            }
        ],
    )
    intro = "I help you find electronics and compare sourced catalog facts."
    model = ConversationModel(
        {"query": "USB hub"},
        {"intent": "evidence", "query": "heat"},
        {"intent": "answer", "answer": intro},
        {"intent": "answer", "answer": "I help you shop for electronics."},
        {"intent": "compare", "references": [2]},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        agent.submit("s", "1", "Find a USB hub under $25")
        agent.submit("s", "2", "Any reviews about heat for that one?")
        waiting = store.get_state("s")
        first = agent.submit("s", "3", "Hi, introduce yourself")
        shorter = agent.submit("s", "4", "Can you put it in one sentence?")
        assert store.get_state("s") == waiting
        resumed = agent.submit("s", "5", "The second one.")

    assert first.answer == intro
    assert shorter.answer == "I help you shop for electronics."
    for outcome in (first, shorter):
        assert outcome.outcome == "COMPLETE"
        assert outcome.intent == "answer"
        assert outcome.tool_attempts == 0 and not outcome.shown_ids()
    assert model.contexts[3]["recent_turns"][-1] == {
        "user": "Hi, introduce yourself",
        "assistant": intro,
    }
    assert resumed.intent == "evidence"
    assert [quote.id for quote in resumed.evidence] == ["heat-b"]


def test_general_concept_can_answer_without_a_catalog_comparison():
    catalog = _catalog([("a", "HDMI cable", "12.00")])
    model = ConversationModel(
        {
            "intent": "answer",
            "answer": "HDMI and DisplayPort are digital display interfaces.",
        }
    )
    with closing(Store(":memory:")) as store:
        result = Agent(store, catalog, provider=model).submit(
            "s", "1", "What is the difference between HDMI and DisplayPort?"
        )
    assert result.intent == "answer"
    assert result.tool_attempts == 0
    assert result.answer and "interfaces" in result.answer


def test_model_cannot_replace_an_explicit_lookup_with_an_ungrounded_direct_answer():
    catalog = _catalog(
        [("a", "USB hub Compact", "15.00")],
        [
            {
                "id": "actual-review",
                "product_id": "a",
                "text": "Easy to carry.",
                "source": "Travel review",
            }
        ],
    )
    model = ConversationModel(
        {"query": "USB hub"},
        {"intent": "answer", "answer": "It received perfect scores everywhere."},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        found = agent.submit("s", "1", "Hello, find a USB hub under $25.")
        review = agent.submit("s", "2", "Can you pull reviews for option 1?")
    assert found.intent == "discover" and found.shown_ids() == ["a"]
    assert review.intent == "evidence" and review.answer is None
    assert [quote.id for quote in review.evidence] == ["actual-review"]


def test_missing_choice_criterion_passes_through_help_to_a_short_answer():
    class ChoiceModel(ConversationModel):
        def choose_action(self, observation, timeout):
            choices = observation["available_actions"]
            return "clarify_priority" if "clarify_priority" in choices else choices[0]

    catalog = _catalog(
        [("a", "USB hub Compact", "15.00"), ("b", "USB hub Travel", "19.00")]
    )
    model = ChoiceModel(
        {"query": "USB hub"},
        {"intent": "compare", "priority": "review_aspect", "query": "buy"},
        {"intent": "answer", "answer": "I help compare electronics."},
        {"clarification": "Which product do you mean?"},
    )
    with closing(Store(":memory:")) as store:
        agent = Agent(store, catalog, provider=model)
        agent.submit("s", "1", "Find a USB hub under $25")
        waiting = agent.submit(
            "s", "2", "Compare the first two. Which one should I buy?"
        )
        answer = agent.submit("s", "3", "Can you explain your role?")
        decision = agent.submit("s", "4", "Price")
    assert waiting.question == CHOICE_CRITERION_QUESTION
    assert answer.intent == "answer" and answer.tool_attempts == 0
    assert decision.shown_ids() == ["a", "b"]
    assert decision.decision is not None and decision.decision.basis == "price"
    assert decision.decision.product_id == "a"
