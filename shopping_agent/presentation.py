"""Render the model's grounded answer with trusted application facts and sources."""

from .contracts import Card, Result


def _product_prefix(result: Result, product_id: str, ordinal: int) -> str:
    if result.intent in {"discover", "recommend"} or result.reference_ids is None:
        return f"{ordinal}."
    if product_id in result.reference_ids:
        return f"{result.reference_ids.index(product_id) + 1}."
    return "-"


def _decision_lines(result: Result) -> list[str]:
    decision = result.decision
    if decision is None:
        return []
    if decision.status == "LOWEST_PRICE":
        card = next(
            card for card in result.comparison if card.id == decision.product_id
        )
        return [
            (
                f"For the lowest catalog price, choose {card.title} [{card.id}] "
                f"at ${card.price_usd:.2f}."
            ),
            f"Price source · Catalog snapshot {card.catalog_source}",
        ]
    if decision.status == "INSUFFICIENT_REVIEW_EVIDENCE":
        aspect = result.review_aspect or "the requested quality"
        return [
            (
                f"I would need comparable evidence for {aspect} across the selected products "
                "to recommend one on that basis."
            )
        ]
    if decision.status == "REVIEW_TRADEOFFS":
        coverage = {
            q.product_id for q in result.evidence if q.kind == "review_observation"
        }
        sampled = ", ".join(c.id for c in result.comparison if c.id in coverage)
        message = f"The review observations below address {result.review_aspect} for {sampled}."
        if len(coverage) < len(result.comparison):
            return [
                message,
                "A comparison on this criterion also needs relevant evidence for the other selected products.",
            ]
        return [
            message,
            "The excerpts describe individual experiences. Use them to weigh the options for your situation.",
        ]
    if decision.status in {"HIGHEST_RATING", "RATING_TIE"}:
        if decision.status == "HIGHEST_RATING":
            card = next(
                card for card in result.comparison if card.id == decision.product_id
            )
            return [
                f"On catalog average rating, choose {card.title} [{card.id}] at {card.average_rating:g}/5.",
                f"Rating source · Catalog snapshot {card.catalog_source}",
            ]
        return ["The highest eligible catalog average ratings are tied."]
    return [
        {
            "PRICE_TIE": "The lowest eligible catalog prices are tied.",
            "INSUFFICIENT_PRICES": "I need comparable catalog prices for the selected items to choose on price.",
            "INSUFFICIENT_RATINGS": "I need comparable catalog ratings and any required prices to choose on average rating.",
            "UNVERIFIED_REQUIREMENTS": "I need to verify the outstanding requirements before selecting a product.",
            "NO_BUDGET_MATCH": "The selected products with known prices are above the budget.",
        }[decision.status]
    ]


def _review_gap_lines(result: Result) -> list[str]:
    messages = []
    outside_sample = []
    no_match = []
    unknown = []
    for product_id in result.missing_review_ids:
        gap = result.review_gaps.get(product_id)
        if gap == "OUTSIDE_SAMPLE":
            outside_sample.append(product_id)
        elif gap == "NO_ASPECT_MATCH":
            no_match.append(product_id)
        else:
            unknown.append(product_id)
    if outside_sample:
        messages.append(
            "The review sample has no excerpts for " + ", ".join(outside_sample) + "."
        )
    if no_match:
        aspect = result.review_aspect
        messages.append(
            f'The sampled excerpts for {", ".join(no_match)} have no keyword match for "{aspect}".'
        )
    if unknown:
        messages.append(
            "No matching review excerpt was returned for " + ", ".join(unknown) + "."
        )
    return messages


def _rank_note(result: Result, card: Card) -> str:
    if card.model_rank is not None and result.recommendation_model:
        return f" · {result.recommendation_model.upper()} rank {card.model_rank}"
    if card.example_ranking:
        return f" · Example ranking {card.example_ranking}"
    if result.intent == "discover" and result.recommendation_model:
        return " · Catalog search"
    return ""


def _checks(card: Card) -> list[str]:
    return [
        f"   {check.requirement} · {check.status.lower()}"
        + (" (catalog title)" if check.source == "catalog_title" else "")
        for check in card.conditions
    ]


def _rating(card: Card) -> str:
    if card.average_rating is None:
        return "Average rating unavailable"
    count = "" if card.rating_count is None else f" from {card.rating_count:,} ratings"
    return f"Catalog average rating {card.average_rating:g}/5{count}"


def lines(result: Result) -> list[str]:
    output = [
        (
            f"{result.outcome} | data={result.data_mode} snapshot={result.snapshot} "
            f"revision={result.revision} policy={result.policy} model={result.provider}"
        ),
        f"Personalization: {result.personalization}",
    ]
    if result.question:
        output.append(result.question)
    if result.answer:
        output.append(result.answer)
    ordinal = 0
    for heading, cards in (
        (
            "Model ranking" if result.intent == "recommend" else "Verified",
            result.verified,
        ),
        ("Pending verification", result.pending),
        ("Comparison", result.comparison),
    ):
        if cards:
            output.append(heading + ":")
        for card in cards:
            ordinal += 1
            price = (
                "unknown price"
                if card.price_usd is None
                else f"USD {card.price_usd:.2f}"
            )
            ranking = _rank_note(result, card)
            output.append(
                f"  {_product_prefix(result, card.id, ordinal)} {card.title} [{card.id}] — {price} "
                f"({card.price_status}; catalog {card.catalog_source}{ranking})"
            )
            if result.decision and result.decision.basis == "rating":
                output.append(f"    {_rating(card)}")
            output.extend(_checks(card))
    if result.price_difference_usd is not None:
        output.append(
            f"Catalog price difference: USD {result.price_difference_usd:.2f}"
        )
    output.extend(_decision_lines(result))
    output.extend(_review_gap_lines(result))
    for quote in result.evidence:
        output.append(
            f"  {quote.kind} [{quote.id} / {quote.product_id} / {quote.source}]: {quote.text}"
        )
    output.extend(f"Limitation: {item}" for item in result.limitations)
    return output


def chat_lines(result: Result, state: dict) -> list[str]:
    """Render a conversation from verified application data."""
    if result.question:
        return [result.question]
    if result.answer:
        return [result.answer]
    if result.outcome == "TEMPORARILY_UNAVAILABLE":
        return [
            "I couldn't get new results this turn. Check the diagnostics and try again.",
            *result.limitations,
        ]

    output: list[str] = list(result.model_reply)
    if result.intent == "recommend" and result.recommendation_model:
        output.append(
            f"{result.recommendation_model.upper()} next-item ranking · Requested {result.requested_n} items"
        )
        output.append(
            "Ranked from dataset history. Shopping constraints apply to catalog search."
        )
    elif result.intent == "discover":
        query = state.get("search_query")
        budget = state.get("max_price_usd")
        if query:
            output.append(f"I looked for {query}.")
        if budget is not None:
            output.append(f"The budget cap is ${budget}.")
        if not result.verified:
            if (
                result.unchecked_candidate_ids
                or result.pending
                or result.outcome == "PARTIAL"
            ):
                output.append("Some candidate checks are still unresolved.")
            elif result.outcome == "NO_VERIFIED_MATCH_IN_SCOPE":
                output.append("No verified matches in the checked scope.")
        if result.decision_ids and result.price_difference_usd is not None:
            output.append(
                f"The two shortlisted catalog prices differ by ${result.price_difference_usd:.2f}."
            )
    elif result.intent == "compare" and result.price_difference_usd is not None:
        output.append(
            f"The two catalog prices differ by ${result.price_difference_usd:.2f}."
        )
    output.extend(_decision_lines(result))
    output.extend(_review_gap_lines(result))
    if result.ranking_source:
        source = {
            "checkpoint_inference": "Original checkpoint · local inference",
            "version_checked_cache": "Original checkpoint · version-checked cache",
            "precomputed_example": "Saved ranking · precomputed example",
        }[result.ranking_source]
        output.append(source)

    ordinal = 0
    for heading, cards in (
        (
            "Ranked products" if result.intent == "recommend" else "Verified products",
            result.verified,
        ),
        ("Pending verification", result.pending),
        ("Selected products", result.comparison if result.intent != "discover" else []),
    ):
        if cards:
            output.extend(("", heading))
        for card in cards:
            ordinal += 1
            price = (
                f"USD {card.price_usd:.2f}"
                if card.price_usd is not None
                else "Price unknown"
            )
            ranking = _rank_note(result, card)
            budget_status = (
                " · Above the budget" if card.price_status == "VIOLATED" else ""
            )
            output.append(
                f"{_product_prefix(result, card.id, ordinal)} {card.title} [{card.id}] · {price}{ranking}{budget_status}"
            )
            if result.decision and result.decision.basis == "rating":
                output.append(f"   {_rating(card)}")
            output.extend(_checks(card))
    if result.evidence:
        output.extend(("", "Attributed review excerpts"))
    if result.data_mode == "team-electronics" and (
        result.evidence or result.missing_review_ids
    ):
        output.append("This review sample can return up to four excerpts per product.")
    for evidence in result.evidence:
        output.append(
            f'{evidence.product_id} — "{evidence.text}" [{evidence.id} · {evidence.source}]'
        )
    if result.above_budget:
        output.extend(("", "Above the budget"))
        for card in result.above_budget:
            output.append(f"- {card.title} [{card.id}] · USD {card.price_usd:.2f}")
    explained = {
        "Pending products have unverified mandatory conditions.",
        "No verified match in the current search scope; catalog coverage is not guaranteed.",
    }
    if result.missing_review_ids:
        explained.add(
            "Requested review evidence was not available for every selected product."
        )
    output.extend(note for note in result.limitations if note not in explained)
    return output
