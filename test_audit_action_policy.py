"""Exercise the local agent's authorized menus and choice ordering."""

import pytest

from shopping_agent.catalog import SnapshotData, _SnapshotReader
from shopping_agent.contracts import Patch, Session
from shopping_agent.workflow import Limits, Run, build_result, execute


class ScriptedChooser:
    local_intents = True

    def __init__(self, chooser):
        self.chooser = chooser
        self.observations = []

    def choose_action(self, observation, timeout):
        assert timeout > 0
        self.observations.append(observation)
        return self.chooser(observation)


@pytest.fixture
def sample():
    # The ranked item falls outside the first ten generic keyword results.
    products = [
        {
            "id": identifier,
            "category": "Electronics",
            "title": "Wireless Earbuds",
            "price": price,
        }
        for identifier, price in [
            *[(f"catalog-{i}", 23) for i in range(1, 12)],
            ("ranked-1", 20),
        ]
    ]
    return _SnapshotReader(
        SnapshotData.model_validate(
            {
                "manifest": {
                    "snapshot": "planner-catalog",
                    "origin": "synthetic",
                    "categories": ["Electronics"],
                    "description": "Planner test inputs",
                },
                "products": products,
                "recommendations": {"history": {"Electronics": ["ranked-1"]}},
                "evidence": [
                    {
                        "id": "review-1",
                        "product_id": "catalog-1",
                        "text": "I use these for evening walks.",
                        "source": "Independent planner test excerpt",
                    }
                ],
            }
        ),
        mode="team-electronics",
    )


def _run(sample, provider, *, request, patch=None, refs=None, history=False):
    run = Run(
        state=Session(
            snapshot=sample.snapshot,
            category="Electronics",
            persona="history" if history else None,
            max_price_usd=40,
        ),
        patch=patch or Patch(query="wireless earbuds"),
        refs=refs or [],
        request=request,
        catalog=sample,
        provider=provider,
        policy="adaptive",
        limits=Limits(),
    )
    execute(run)
    return run


@pytest.mark.parametrize("first", ["search_products", "recommend_products"])
def test_same_menu_supports_different_first_tools_and_preserves_ranks(sample, first):
    def choose(data):
        options = data["available_actions"]
        if not data["observed_tools"]:
            assert set(options) == {"search_products", "recommend_products"}
            return first
        # Verify generic candidates before asking for history on the catalog-first
        # route. The workflow must still verify ranked details before finishing.
        for choice in (
            "finish",
            "match_ranked_candidates",
            "get_products",
            "recommend_products",
            "search_products",
        ):
            if choice in options:
                return choice
        return options[0]

    provider = ScriptedChooser(choose)
    run = _run(
        sample,
        provider,
        request="I want wireless earbuds under $40. Use history and check the catalog.",
        history=True,
    )
    trace = [
        row
        for row in run.trace
        if row["action"] in {"search_products", "recommend_products", "get_products"}
    ]
    assert trace[0]["action"] == first
    assert build_result(run).verified[0].id == "ranked-1"
    assert run.checked_rank_query
    assert "ranked-1" in run.requested_details
    assert any(
        observation["observed_tools"] for observation in provider.observations[1:]
    )


def test_model_cannot_finish_before_verifying_possible_matches(sample):
    provider = ScriptedChooser(lambda _data: "finish")
    run = _run(sample, provider, request="Find wireless earbuds under $40.")
    fallback = next(row for row in run.trace if row["action"] == "model_decide")
    assert fallback["status"] == "FALLBACK"
    assert fallback["proposal"] == "get_products"
    assert "finish" not in fallback["available_actions"]
    assert fallback["observed_tools"][0]["action"] == "search_products"
    assert len(build_result(run).verified) == 3


def test_explicit_review_lookup_cannot_be_skipped(sample):
    provider = ScriptedChooser(lambda _data: "finish")
    run = _run(
        sample,
        provider,
        patch=Patch(intent="compare"),
        refs=["catalog-1", "catalog-2"],
        request="Compare the first two and include buyer reviews.",
    )
    assert not provider.observations
    assert [entry.id for entry in build_result(run).evidence] == ["review-1"]
    assert [
        row["action"] for row in run.trace if row["status"] in {"OK", "PARTIAL"}
    ] == [
        "get_products",
        "get_product_evidence",
    ]


def test_missing_criterion_can_trigger_clarification_after_inspection(sample):
    provider = ScriptedChooser(lambda _data: "clarify_priority")
    run = _run(
        sample,
        provider,
        patch=Patch(intent="compare"),
        refs=["catalog-1", "catalog-2"],
        request="Which of the first two should I buy?",
    )
    assert "What matters most" in build_result(run).question
    assert provider.observations[0]["selected_detail_count"] == 2
    assert set(provider.observations[0]["available_actions"]) == {
        "get_product_evidence",
        "clarify_priority",
    }
