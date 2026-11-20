"""Shared execution gate and fixed/adaptive decision loop.

Only the application calls the catalog. Model proposals and catalog data are
treated as inputs and never become trusted display text automatically.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from time import monotonic
from typing import TypedDict

from pydantic import TypeAdapter

from .catalog import MAX_EVIDENCE_PER_PRODUCT, CanonicalID, CatalogResponse
from .contracts import (
    Action,
    Card,
    ConditionCheck,
    Decision,
    DecisionPriority,
    DiscoveryGoal,
    EvidenceQuote,
    Patch,
    RecommendationRequest,
    Result,
    ReviewGap,
    Session,
)
from .state import CHOICE_CRITERION_QUESTION, HistoryUsage, asks_for_reviews


@dataclass(frozen=True)
class Limits:
    model_attempts: int = 8
    tool_attempts: int = 10
    seconds: float = 90
    tool_timeout: float = 12
    page: int = 25
    candidates: int = 100
    recommendations: int = 3
    evidence: int = 12


@dataclass
class Run:
    state: Session
    patch: Patch
    refs: list[str]
    catalog: object
    provider: object | None
    policy: str
    limits: Limits
    request: str = ""
    start: float = field(default_factory=monotonic)
    model_attempts: int = 0
    tool_attempts: int = 0
    candidates: list[str] = field(default_factory=list)
    matched_query_ids: set[str] = field(default_factory=set)
    ranked_ids: set[str] = field(default_factory=set)
    requested_details: set[str] = field(default_factory=set)
    products: dict = field(default_factory=dict)
    requested_evidence: set[str] = field(default_factory=set)
    missing_products: set[str] = field(default_factory=set)
    missing_reviews: set[str] = field(default_factory=set)
    review_gaps: dict[str, ReviewGap] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    seen: set[tuple] = field(default_factory=set)
    searched: int = 0
    recommended: bool = False
    failures: int = 0
    discovery_ok: bool = False
    has_more: bool = False
    question: str | None = None
    personalization: str = "NO_HISTORY"
    done: bool = False
    repairs: int = 0
    recommender: object | None = None
    recommendation: RecommendationRequest | None = None
    model_order: list[str] = field(default_factory=list)
    checked_rank_query: bool = False
    search_offset: int = 0
    recent_turns: list[dict[str, str]] = field(default_factory=list)
    above_budget_ids: set[str] = field(default_factory=set)
    ranking_source: str | None = None
    agent_loop: bool = False
    history_mode: HistoryUsage = "catalog_only"
    discovery_goal: DiscoveryGoal | None = None
    catalog_search_ok: bool = False
    ranked_search_ok: bool = False
    ranking_ok: bool = False
    reviewed_ids: set[str] = field(default_factory=set)
    topical_citation_ids: set[str] | None = None
    decision_ids: list[str] = field(default_factory=list)
    final_decision: Decision | None = None
    duplicate_rejections: int = 0

    def remaining(self) -> float:
        return self.limits.seconds - (monotonic() - self.start)

    def note(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)


def _conditions(run: Run, product) -> list[ConditionCheck]:
    checks = [
        ConditionCheck(requirement=text, status="UNKNOWN")
        for text in run.state.unsupported
    ]
    for item in run.state.requirements:
        result = None
        ports = getattr(product, "port_count", None)
        versions = getattr(product, "usb_versions", ())
        if ports is not None and item.kind in {"min_ports", "max_ports", "exact_ports"}:
            bound = int(item.value)
            result = {
                "min_ports": ports >= bound,
                "max_ports": ports <= bound,
                "exact_ports": ports == bound,
            }[item.kind]
        elif item.value in versions:
            if item.kind == "required_usb_version":
                result = True
            elif item.kind == "excluded_usb_version":
                result = False
        checks.append(
            ConditionCheck(
                requirement=item.text,
                status="UNKNOWN"
                if result is None
                else "SATISFIED"
                if result
                else "VIOLATED",
                source=getattr(product, "spec_source", None)
                if result is not None
                else None,
            )
        )
    return checks


def _card(run: Run, product) -> Card:
    raw_ranking = run.patch.intent == "recommend"
    budget = None if raw_ranking else run.state.max_price_usd
    price = product.price
    if budget is None:
        status = "NOT_REQUIRED"
    elif price is None:
        status = "UNKNOWN"
    else:
        status = "SATISFIED" if price <= budget else "VIOLATED"
    return Card(
        id=product.id,
        category=product.category,
        title=product.title,
        price_usd=price,
        price_status=status,
        catalog_source=run.catalog.snapshot,
        example_ranking=run.state.persona if product.id in run.ranked_ids else None,
        model_rank=run.model_order.index(product.id) + 1
        if product.id in run.model_order
        else None,
        average_rating=getattr(product, "average_rating", None),
        rating_count=getattr(product, "rating_count", None),
        conditions=[] if raw_ranking else _conditions(run, product),
    )


def _eligible_candidate(run: Run, product_id: str) -> bool:
    return not run.patch.query or product_id in run.matched_query_ids


def _verified(run: Run) -> list[Card]:
    cards: list[Card] = []
    limit = (
        run.discovery_goal.count if run.discovery_goal else run.limits.recommendations
    )
    for product_id in run.candidates:
        if product_id not in run.products or not _eligible_candidate(run, product_id):
            continue
        card = _card(run, run.products[product_id])
        if card.price_status in {"SATISFIED", "NOT_REQUIRED"} and all(
            condition.status == "SATISFIED" for condition in card.conditions
        ):
            cards.append(card)
            if len(cards) >= limit:
                break
    return cards


def unchecked_candidates(run: Run) -> list[str]:
    return [
        asin
        for asin in run.candidates
        if _eligible_candidate(run, asin) and asin not in run.products
    ]


def _policy(run: Run) -> Action:
    """Deterministic offline policy, also the reasonable fixed comparator."""

    def make(action: str, **args) -> Action:
        return Action(action=action, args=args)

    category = run.state.category
    if run.patch.intent == "recommend":
        if not run.recommended:
            if run.recommendation is None:
                return make(
                    "ask_user",
                    question="Choose a dataset user in History context for next-item recommendations. You can also ask to search the catalog.",
                )
            return make(
                "recommend_products", category=category, limit=run.recommendation.n
            )
        missing = [i for i in run.model_order if i not in run.requested_details]
        return make("get_products", ids=missing) if missing else make("finish_decision")
    if run.patch.intent != "discover":
        missing = [i for i in run.refs if i not in run.requested_details]
        if missing:
            return make("get_products", ids=missing)
        if run.patch.priority == "rating" and not asks_for_reviews(run.request):
            return make("finish_decision")
        missing = [
            i for i in run.refs if i in run.products and i not in run.requested_evidence
        ]
        if missing:
            return make("get_product_evidence", ids=missing, query=run.patch.query)
        return make("finish_decision")
    if (run.state.persona or run.recommendation) and not run.recommended:
        return make(
            "recommend_products",
            category=category,
            persona=run.state.persona,
            limit=run.recommendation.n if run.recommendation else 10,
        )
    if (
        run.catalog.mode == "team-electronics"
        and run.patch.query
        and run.ranked_ids
        and not run.checked_rank_query
    ):
        return make(
            "search_products",
            category=category,
            query=run.patch.query,
            limit=len(run.ranked_ids),
            candidate_ids=[asin for asin in run.candidates if asin in run.ranked_ids],
        )
    if not run.searched:
        return make(
            "search_products",
            category=category,
            query=run.patch.query,
            limit=10 if run.policy == "adaptive" else 25,
        )
    missing = [i for i in run.candidates if i not in run.requested_details]
    if missing:
        return make("get_products", ids=missing[:25])
    if (
        run.policy == "adaptive"
        and run.searched == 1
        and run.has_more
        and len(_verified(run)) < 3
    ):
        page = (
            {"limit": 15, "offset": run.search_offset}
            if run.catalog.mode == "team-electronics"
            else {"limit": 25}
        )
        return make("search_products", category=category, query=run.patch.query, **page)
    return make("finish_decision")


def _context(run: Run) -> dict:
    return {
        "requirements": run.state.model_dump(
            mode="json", exclude={"displayed_ids", "pending_question"}
        ),
        "intent": run.patch.intent,
        "query": run.patch.query,
        "selected_ids": run.refs,
        "candidate_ids": run.candidates[:25],
        "products": [
            _card(run, p).model_dump(mode="json")
            for p in list(run.products.values())[:25]
        ],
        # Raw review text is not needed to decide tool order. Never feed it as instructions.
        "evidence": [
            {"id": e.id, "product_id": e.product_id} for e in run.evidence.values()
        ],
        "trace": run.trace[-12:],
        "verified_ids": [c.id for c in _verified(run)],
        "remaining_tools": run.limits.tool_attempts - run.tool_attempts,
    }


def _normalize(run: Run, action: Action) -> dict:
    name, args = action.action, action.args
    allowed_keys = {
        "search_products": {"category", "query", "limit", "candidate_ids", "offset"},
        "recommend_products": {"category", "persona", "limit"},
        "get_products": {"ids"},
        "get_product_evidence": {"ids", "query"},
    }
    if set(args) - allowed_keys[name]:
        raise ValueError("Unexpected tool argument.")
    if name in {"search_products", "recommend_products"}:
        if run.patch.intent not in {"discover", "recommend"}:
            raise ValueError(
                "Comparison and evidence use the displayed product references."
            )
        limit = args.get("limit", 10)
        if type(limit) is not int or not 1 <= limit <= run.limits.page:
            raise ValueError("Invalid candidate limit.")
        if args.get("category") != run.state.category:
            raise ValueError("Wrong category.")
        if name == "recommend_products":
            if run.recommendation is not None:
                return {"category": run.state.category, "limit": run.recommendation.n}
            if not run.state.persona or args.get("persona") != run.state.persona:
                raise ValueError("No authorized example persona.")
            return {
                "category": run.state.category,
                "persona": run.state.persona,
                "limit": limit,
            }
        if args.get("query", "") != run.patch.query:
            raise ValueError("Search must retain the accepted query.")
        search = {
            "category": run.state.category,
            "query": run.patch.query,
            "limit": limit,
        }
        if run.catalog.mode == "team-electronics":
            search["exclude_ids"] = run.state.excluded_ids
            search["max_price_usd"] = run.state.max_price_usd
            if "candidate_ids" in args:
                ids = [asin for asin in run.candidates if asin in run.ranked_ids]
                if args["candidate_ids"] != ids or not ids:
                    raise ValueError(
                        "Ranked candidate matching must use the observed IDs."
                    )
                search["candidate_ids"] = ids
            elif "offset" in args:
                if args["offset"] != run.search_offset:
                    raise ValueError("Unexpected search page.")
                search["offset"] = run.search_offset
        return search
    ids = args.get("ids")
    allowed = set(
        run.refs if run.patch.intent in {"compare", "evidence"} else run.candidates
    )
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= 25
        or any(not isinstance(i, str) or i not in allowed for i in ids)
    ):
        raise ValueError(
            "Tool IDs must come from observed candidates or displayed references."
        )
    data = {"ids": list(dict.fromkeys(ids))}
    if name == "get_product_evidence":
        if len(ids) > 3 or any(i not in run.products for i in ids):
            raise ValueError(
                "Fetch trusted details before requesting evidence for up to three products."
            )
        query = args.get("query")
        if query is None:
            query = run.patch.query
        if query != run.patch.query:
            raise ValueError("Evidence aspect is outside the requested scope.")
        data["query"] = query
    return data


def _observe(run: Run, name: str, args: dict, response: CatalogResponse) -> None:
    if response.snapshot != run.catalog.snapshot:
        run.failures += 1
        run.note("Rejected a response from a different catalog snapshot.")
        return
    if (
        name in {"get_products", "get_product_evidence"}
        and run.patch.intent != "discover"
        and response.status in {"PARTIAL", "ERROR"}
    ):
        is_product_gap = (
            name == "get_products" or response.error_code == "missing_product"
        )
        missing = (
            set(args["ids"]) & set(response.missing_ids) if is_product_gap else set()
        )
        if missing:
            run.missing_products.update(missing)
            if response.status == "ERROR" and response.error_code == "missing_product":
                return
    if response.status == "ERROR":
        run.failures += 1
        run.note(
            "Review evidence is temporarily unavailable."
            if name == "get_product_evidence"
            else response.message
            if name == "recommend_products" and run.recommendation and response.message
            else f"{name} was unavailable for this run."
        )
        return
    expected = (
        "ids"
        if name in {"search_products", "recommend_products"}
        else "products"
        if name == "get_products"
        else "evidence"
    )
    if any(
        getattr(response, field)
        for field in {"ids", "products", "evidence"} - {expected}
    ):
        run.failures += 1
        run.note(f"{name} returned the wrong record type.")
        return
    if response.status == "PARTIAL" and name != "get_product_evidence":
        run.note(f"{name} returned partial data.")
    if name in {"search_products", "recommend_products"}:
        if (name == "search_products" or not run.patch.query) and (
            response.status != "PARTIAL" or response.ids
        ):
            run.discovery_ok = True
        if name == "search_products" and "candidate_ids" not in args:
            run.catalog_search_ok = response.status in {"OK", "EMPTY"}
            run.has_more = response.has_more is True
            run.search_offset += len(response.ids)
        elif name == "search_products":
            run.ranked_search_ok = response.status in {"OK", "EMPTY"}
        else:
            run.ranking_ok = response.status in {"OK", "EMPTY"}
        # Exclusions and result cap are enforced before detail requests.
        observed = [
            i
            for i in response.ids[: args["limit"]]
            if run.patch.intent == "recommend" or i not in run.state.excluded_ids
        ]
        if name == "search_products":
            if run.agent_loop and "candidate_ids" not in args:
                run.above_budget_ids.update(
                    asin
                    for asin in response.above_budget_ids[:2]
                    if asin not in run.state.excluded_ids
                )
            observed.extend(
                asin
                for asin in getattr(response, "pending_ids", [])[:3]
                if asin not in run.state.excluded_ids
            )
            if "candidate_ids" in args:
                observed = [asin for asin in observed if asin in args["candidate_ids"]]
            run.matched_query_ids.update(observed)
        else:
            run.ranked_ids.update(observed)
            if run.recommendation:
                run.model_order = list(response.ids)
        # A model may search the catalog first. Ranking matches retain model
        # order in the combined output regardless of acquisition order.
        combined = (
            [*observed, *run.candidates]
            if name == "recommend_products"
            else [*run.candidates, *observed]
        )
        run.candidates = list(dict.fromkeys(combined))[: run.limits.candidates]
    elif name == "get_products":
        for product in response.products:
            if product.id not in args["ids"] or product.category != run.state.category:
                run.note(
                    "Rejected product details outside the requested IDs or category."
                )
                continue
            prior = run.products.get(product.id)
            if prior is not None and prior != product:
                run.note("Rejected conflicting product data in the same snapshot.")
                continue
            run.products[product.id] = product
    else:
        run.reviewed_ids.update(
            set(args["ids"])
            & (
                set(response.missing_ids)
                | {item.product_id for item in response.evidence}
            )
        )
        if response.status in {"EMPTY", "PARTIAL"}:
            missing = set(args["ids"]) & set(response.missing_ids)
            run.missing_reviews.update(missing)
            run.review_gaps.update(
                {
                    pid: gap
                    for pid, gap in response.review_gaps.items()
                    if pid in missing
                }
            )
        per_product = Counter(item.product_id for item in run.evidence.values())
        for evidence in response.evidence:
            if evidence.product_id not in args["ids"] or (
                evidence.product_id not in run.products and not run.agent_loop
            ):
                run.note("Rejected evidence attached to an unrelated product.")
                continue
            if evidence.id in run.evidence:
                if run.evidence[evidence.id] != evidence:
                    run.note("Rejected conflicting evidence identity.")
                continue
            if per_product[evidence.product_id] >= MAX_EVIDENCE_PER_PRODUCT:
                continue
            if len(run.evidence) < run.limits.evidence:
                run.evidence[evidence.id] = evidence
                per_product[evidence.product_id] += 1


_RANKING_IDS = TypeAdapter(list[CanonicalID])


def _call_tool(run: Run, name: str, args: dict) -> CatalogResponse:
    if (
        name == "recommend_products"
        and run.agent_loop
        and run.history_mode == "catalog_only"
    ):
        return CatalogResponse(
            status="ERROR",
            snapshot=run.catalog.snapshot,
            error_code="HISTORY_NOT_REQUESTED",
            message="This turn has not requested dataset history.",
        )
    if name != "recommend_products" or run.recommendation is None:
        return CatalogResponse.model_validate(
            run.catalog.call(
                name,
                {**args, "snapshot": run.catalog.snapshot},
                min(run.remaining(), run.limits.tool_timeout),
            )
        )
    from .recommendation import RecommenderError

    request = run.recommendation
    try:
        if run.recommender is None:
            raise RecommenderError(
                "RECOMMENDER_UNAVAILABLE",
                "Configure the teammate model environment to run next-item inference.",
            )
        arguments = (
            request.user_id,
            request.model,
            request.n,
            min(run.remaining(), 45),
        )
        with_source = getattr(run.recommender, "recommend_with_provenance", None)
        if with_source is None:
            ids = _RANKING_IDS.validate_python(run.recommender.recommend(*arguments))
        else:
            ranking = with_source(*arguments)
            ids = _RANKING_IDS.validate_python(list(ranking.ids))
            run.ranking_source = ranking.source
        if len(ids) > request.n or len(set(ids)) != len(ids):
            raise RecommenderError(
                "INVALID_RANKING", "The model returned an invalid ranking."
            )
    except RecommenderError as error:
        return CatalogResponse(
            status="ERROR",
            snapshot=run.catalog.snapshot,
            error_code=error.code,
            message=str(error),
        )
    return CatalogResponse(
        status="OK" if ids else "EMPTY",
        snapshot=run.catalog.snapshot,
        ids=ids,
        ranking_source=run.ranking_source,
    )


def _action_key(name: str, args: dict) -> tuple:
    return (
        name,
        tuple(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in sorted(args.items())
        ),
    )


def _execute(run: Run, action: Action) -> None:
    name = action.action
    if name == "finish_decision":
        run.done = True
        return
    if name == "ask_user":
        # Do not publish unconstrained model text as a product/evidence claim.
        question = action.args.get("question")
        if (
            not isinstance(question, str)
            or not 1 <= len(question) <= 240
            or "\n" in question
        ):
            _invalid(run, "Rejected an invalid clarification proposal.")
            return
        run.question = question
        run.done = True
        return
    try:
        args = _normalize(run, action)
    except (ValueError, TypeError):
        _invalid(run, "Rejected a tool proposal outside the allowed scope.")
        return
    execute_tool(run, name, args)


def execute_tool(run: Run, name: str, args: dict) -> None:
    """Execute an already validated tool request and log its scoped observation."""
    key = _action_key(name, args)
    if key in run.seen:
        run.note("Stopped after a repeated action without progress.")
        run.done = True
        return
    run.seen.add(key)
    if name == "search_products":
        if "candidate_ids" in args:
            run.checked_rank_query = True
        else:
            if "offset" not in args:
                run.search_offset = 0
            run.searched += 1
    elif name == "recommend_products":
        run.recommended = True
    elif name == "get_products":
        run.requested_details.update(args["ids"])
    else:
        run.requested_evidence.update(args["ids"])
    for attempt in range(2):
        if run.tool_attempts >= run.limits.tool_attempts or run.remaining() <= 0:
            run.note("Tool or elapsed-time budget reached.")
            run.done = True
            return
        run.tool_attempts += 1
        start = monotonic()
        try:
            response = _call_tool(run, name, args)
            if (
                response.snapshot == run.catalog.snapshot
                and response.error_code == "timeout"
            ):
                raise TimeoutError
        except TimeoutError:
            run.trace.append(
                {
                    "action": name,
                    "status": "TIMEOUT",
                    "attempt": attempt + 1,
                    "latency_ms": round((monotonic() - start) * 1000, 2),
                }
            )
            if attempt == 0:
                continue
            run.failures += 1
            run.note(
                "The review lookup timed out."
                if name == "get_product_evidence"
                else f"{name} timed out."
            )
            return
        except Exception:  # noqa: BLE001 — isolate untrusted integration failures
            # Do not expose unexpected backend exceptions or credentials to the model/UI.
            run.trace.append({"action": name, "status": "ADAPTER_ERROR"})
            run.failures += 1
            run.note(
                "Review evidence is temporarily unavailable."
                if name == "get_product_evidence"
                else f"{name} failed or returned an invalid contract."
            )
            return
        run.trace.append(
            {
                "action": name,
                "status": response.status,
                "count": len(response.ids)
                + len(response.products)
                + len(response.evidence),
                "latency_ms": round((monotonic() - start) * 1000, 2),
                **({"error_code": response.error_code} if response.error_code else {}),
                **({"scope": "ranked_candidates"} if "candidate_ids" in args else {}),
                "arguments": args,
                "returned_ids": [
                    *response.ids,
                    *response.pending_ids,
                    *(p.id for p in response.products),
                ],
                "pending_ids": response.pending_ids,
                "above_budget_ids": response.above_budget_ids,
                "has_more": response.has_more,
                "review_gaps": response.review_gaps,
                "missing_ids": response.missing_ids,
                "evidence": [
                    {"id": e.id, "product_id": e.product_id, "source": e.source}
                    for e in response.evidence
                ],
                **(
                    {"ranking_source": response.ranking_source}
                    if response.ranking_source
                    else {}
                ),
            }
        )
        if response.snapshot == run.catalog.snapshot and response.ranking_source:
            run.ranking_source = response.ranking_source
        _observe(run, name, args, response)
        return


def _invalid(run: Run, note: str) -> None:
    run.note(note)
    run.repairs += 1
    if run.repairs > 1 or run.provider is None:
        run.done = True


def _choose_local_followup(run: Run, scheduled: Action) -> Action:
    """Offer the local model choices only after a usable tool observation."""
    choose = getattr(run.provider, "choose_followup", None)
    if (
        choose is None
        or run.policy != "adaptive"
        or run.model_attempts >= run.limits.model_attempts
    ):
        return scheduled
    verified_count = len(_verified(run)) if run.patch.intent == "discover" else 0
    if scheduled.action == "search_products" and run.searched and verified_count:
        followup = "expand_search"
    elif (
        scheduled.action == "get_product_evidence"
        and run.patch.intent == "compare"
        and not run.patch.query
        and run.patch.priority != "review_aspect"
        and not asks_for_reviews(run.request)
    ):
        followup = "retrieve_reviews"
    else:
        return scheduled
    options = {followup: scheduled, "finish": Action(action="finish_decision")}
    run.model_attempts += 1
    try:
        choice = choose(
            {
                "request": run.request,
                "intent": run.patch.intent,
                "available_actions": list(options),
                "verified_count": verified_count,
                "has_more": run.has_more,
                "products_with_known_price": sum(
                    p.price is not None for p in run.products.values()
                ),
                "last_tool": run.trace[-1] if run.trace else None,
            },
            min(20, run.remaining()),
        )
        if choice not in options:
            raise ValueError("Unscheduled follow-up")
    except Exception:  # noqa: BLE001 — complete the safe planned action on provider failure
        run.trace.append(
            {"action": "model_decide", "status": "FALLBACK", "proposal": followup}
        )
        return scheduled
    finally:
        usage = getattr(run.provider, "last_usage", None)
        if usage:
            run.usage.append(dict(usage))
    run.trace.append(
        {"action": "model_decide", "status": "SELECTED", "proposal": choice}
    )
    return options[choice]


def _discovery_actions(run: Run) -> dict[str, Action]:
    options: dict[str, Action] = {}
    search_args = {"category": run.state.category, "query": run.patch.query}
    history_pending = bool(
        (run.state.persona or run.recommendation) and not run.recommended
    )
    if history_pending:
        options["recommend_products"] = Action(
            action="recommend_products",
            args={
                "category": run.state.category,
                "persona": run.state.persona,
                "limit": run.recommendation.n if run.recommendation else 10,
            },
        )
    ranked = [pid for pid in run.candidates if pid in run.ranked_ids]
    match_ranked = bool(
        run.catalog.mode == "team-electronics"
        and run.patch.query
        and ranked
        and not run.checked_rank_query
    )
    if match_ranked:
        options["match_ranked_candidates"] = Action(
            action="search_products",
            args={**search_args, "candidate_ids": ranked, "limit": len(ranked)},
        )
    open_items = [
        pid
        for pid in run.candidates
        if pid not in run.requested_details and _eligible_candidate(run, pid)
    ]
    if open_items:
        options["get_products"] = Action(
            action="get_products", args={"ids": open_items[:25]}
        )

    # The required history and ranked matching cannot be skipped. Start or
    # expand a catalog search if there is still room for verified matches.
    enough = len(_verified(run)) >= run.limits.recommendations
    catalog_requested = bool(re.search(r"\bcatalog(?:ue)?\b", run.request.casefold()))
    if not run.searched and (not enough or history_pending or catalog_requested):
        options["search_products"] = Action(
            action="search_products", args={**search_args, "limit": 10}
        )
    elif not enough and run.has_more and 0 < run.searched < 3:
        if run.catalog.mode == "team-electronics":
            page = {"limit": 10, "offset": run.search_offset}
        else:
            page = {"limit": run.limits.page}
        action = Action(action="search_products", args={**search_args, **page})
        if _action_key(action.action, _normalize(run, action)) not in run.seen:
            options["expand_search"] = action
    waiting = (
        history_pending
        or match_ranked
        or any(pid in run.ranked_ids for pid in open_items)
        or (
            not enough
            and any(
                step in options
                for step in ("get_products", "search_products", "expand_search")
            )
        )
        or (catalog_requested and not run.searched)
    )
    if not waiting or not options:
        options["finish"] = Action(action="finish_decision")
    return options


def _local_actions(run: Run) -> dict[str, Action]:
    if run.patch.intent == "discover":
        return _discovery_actions(run)
    if run.patch.intent == "recommend":
        step = _policy(run)
        return {step.action: step}
    missing = [pid for pid in run.refs if pid not in run.requested_details]
    if missing:
        return {"get_products": Action(action="get_products", args={"ids": missing})}
    evidence_ids = [
        pid
        for pid in run.refs
        if pid in run.products and pid not in run.requested_evidence
    ]
    explicit_evidence = (
        run.patch.intent == "evidence"
        or bool(run.patch.query)
        or run.patch.priority == "review_aspect"
        or asks_for_reviews(run.request)
    )
    options: dict[str, Action] = {}
    if evidence_ids and (explicit_evidence or run.patch.priority != "rating"):
        options["get_product_evidence"] = Action(
            action="get_product_evidence",
            args={"ids": evidence_ids, "query": run.patch.query},
        )
    if explicit_evidence and options:
        return options
    if (
        run.patch.intent == "compare"
        and run.patch.priority is None
        and re.search(
            r"\b(?:which|what)\b[^.?!]{0,90}\b(?:better|best|buy|pick|choose|recommend)\b",
            run.request.casefold(),
        )
    ):
        options["clarify_priority"] = Action(
            action="ask_user",
            args={"question": CHOICE_CRITERION_QUESTION},
        )
    else:
        options["finish"] = Action(action="finish_decision")
    return options


def _choose_local_action(run: Run) -> Action:
    options = _local_actions(run)
    default = next(iter(options))
    if len(options) == 1 or run.model_attempts >= run.limits.model_attempts:
        return options[default]
    tools = [
        row
        for row in run.trace
        if row["action"]
        in {
            "recommend_products",
            "search_products",
            "get_products",
            "get_product_evidence",
        }
    ][-5:]
    context = {
        "request": run.request,
        "intent": run.patch.intent,
        "query": run.patch.query,
        "budget_usd": str(run.state.max_price_usd)
        if run.state.max_price_usd is not None
        else None,
        "priority": run.patch.priority,
        "has_history": bool(run.state.persona or run.recommendation),
        "available_actions": list(options),
        "verified_count": len(_verified(run)),
        "target_verified_count": run.limits.recommendations,
        "has_more": run.has_more,
        "selected_detail_count": sum(pid in run.products for pid in run.refs),
        "products_with_known_price": sum(
            p.price is not None for p in run.products.values()
        ),
        "observed_tools": tools,
        "recent_turns": getattr(run, "recent_turns", [])[-3:],
    }
    run.model_attempts += 1
    status = "SELECTED"
    try:
        choice = run.provider.choose_action(context, min(20, run.remaining()))
        if choice not in options:
            raise ValueError("Unavailable tool selection")
    except Exception:  # noqa: BLE001 — use an authorized option if the model call fails
        choice, status = default, "FALLBACK"
    finally:
        usage = getattr(run.provider, "last_usage", None)
        if usage:
            run.usage.append(dict(usage))
    run.trace.append(
        {
            "action": "model_decide",
            "status": status,
            "proposal": choice,
            "available_actions": list(options),
            "observed_tools": tools,
        }
    )
    return options[choice]


def _step(run: Run) -> None:
    if run.patch.intent == "answer":
        run.done = True
        return
    if run.remaining() <= 0 or run.tool_attempts >= run.limits.tool_attempts:
        run.note("Tool or elapsed-time budget reached.")
        run.done = True
        return
    if (
        run.provider is not None
        and run.patch.intent != "recommend"
        and run.policy == "adaptive"
        and not getattr(run.provider, "local_intents", False)
    ):
        if run.model_attempts >= run.limits.model_attempts:
            run.note("Model attempt budget reached.")
            run.done = True
            return
        run.model_attempts += 1
        try:
            action = Action.model_validate(
                run.provider.decide(_context(run), min(20, run.remaining()))
            )
        except Exception:  # noqa: BLE001 — isolate untrusted integration failures
            run.trace.append(
                {"action": "model_decide", "status": "INVALID_OR_UNAVAILABLE"}
            )
            _invalid(run, "Model decision was invalid or unavailable.")
            return
        finally:
            usage = getattr(run.provider, "last_usage", None)
            if usage:
                run.usage.append(dict(usage))
        run.trace.append(
            {"action": "model_decide", "status": "PROPOSED", "proposal": action.action}
        )
    elif run.policy == "adaptive" and callable(
        getattr(run.provider, "choose_action", None)
    ):
        action = _choose_local_action(run)
    else:
        action = _choose_local_followup(run, _policy(run))
    _execute(run, action)


class LoopState(TypedDict):
    run: Run


@lru_cache(maxsize=1)
def _compiled_graph():
    from langgraph.graph import END, START, StateGraph

    def step(state: LoopState) -> LoopState:
        current = state["run"]
        _step(current)
        return {"run": current}

    def route(state: LoopState) -> str:
        return "finish" if state["run"].done else "step"

    graph = StateGraph(LoopState)
    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_conditional_edges("step", route, {"finish": END, "step": "step"})
    return graph.compile()


def execute(run: Run) -> None:
    """StateGraph is the default; direct loop is the dependency fallback."""
    try:
        graph = _compiled_graph()
    except ModuleNotFoundError as error:
        if error.name != "langgraph":
            raise
        run.trace.append({"action": "engine", "status": "DIRECT_FALLBACK"})
        while not run.done:
            _step(run)
        return

    run.trace.append({"action": "engine", "status": "STATEGRAPH"})
    graph.invoke({"run": run}, config={"recursion_limit": 64})


def _decision(
    run: Run,
    comparison: list[Card],
    quotes: list[EvidenceQuote],
    *,
    priority: DecisionPriority | None = None,
    refs: list[str] | None = None,
) -> Decision | None:
    if priority is None:
        if run.patch.intent != "compare":
            return None
        priority = run.patch.priority
    if run.question or priority is None:
        return None
    if priority == "review_aspect":
        supported = bool(run.patch.query) and any(
            q.kind == "review_observation" for q in quotes
        )
        return Decision(
            basis=priority,
            status="REVIEW_TRADEOFFS" if supported else "INSUFFICIENT_REVIEW_EVIDENCE",
        )
    if any(
        check.status == "UNKNOWN" for card in comparison for check in card.conditions
    ):
        return Decision(basis=priority, status="UNVERIFIED_REQUIREMENTS")
    metric = "average_rating" if priority == "rating" else "price_usd"
    unavailable = (
        "INSUFFICIENT_RATINGS" if priority == "rating" else "INSUFFICIENT_PRICES"
    )
    if (
        len(comparison) < 2
        or len(comparison) != len(run.refs if refs is None else refs)
        or any(
            getattr(card, metric) is None or card.price_status == "UNKNOWN"
            for card in comparison
        )
    ):
        return Decision(basis=priority, status=unavailable)
    eligible = [
        card
        for card in comparison
        if card.price_status != "VIOLATED"
        and all(check.status == "SATISFIED" for check in card.conditions)
    ]
    if not eligible:
        status = (
            "NO_BUDGET_MATCH"
            if all(c.price_status == "VIOLATED" for c in comparison)
            else "UNVERIFIED_REQUIREMENTS"
        )
        return Decision(basis=priority, status=status)
    choose = max if priority == "rating" else min
    best = choose(getattr(card, metric) for card in eligible)
    winners = [card for card in eligible if getattr(card, metric) == best]
    if len(winners) > 1:
        return Decision(
            basis=priority, status="RATING_TIE" if priority == "rating" else "PRICE_TIE"
        )
    return Decision(
        basis=priority,
        status="HIGHEST_RATING" if priority == "rating" else "LOWEST_PRICE",
        product_id=winners[0].id,
    )


def build_result(run: Run) -> Result:
    is_ranking = run.patch.intent == "recommend"
    answer = run.patch.answer if run.patch.intent == "answer" else None
    verified = _verified(run) if run.patch.intent == "discover" else []
    pending: list[Card] = []
    comparison: list[Card] = []
    if run.state.unsupported and run.patch.intent not in {"recommend", "answer"}:
        run.note(
            "Unsupported mandatory conditions: " + ", ".join(run.state.unsupported)
        )
    if run.patch.intent == "discover":
        limit = (
            run.discovery_goal.count
            if run.discovery_goal
            else run.limits.recommendations
        )
        for i in run.candidates:
            if not _eligible_candidate(run, i):
                continue
            if i not in run.products:
                if not run.agent_loop or i in run.requested_details:
                    run.note("Some requested candidate details were unavailable.")
                elif len(verified) < limit:
                    run.note("Some catalog candidates still need verification.")
                continue
            card = _card(run, run.products[i])
            statuses = {card.price_status, *(check.status for check in card.conditions)}
            if "UNKNOWN" in statuses and "VIOLATED" not in statuses:
                pending.append(card)
        comparison = [
            _card(run, run.products[i]) for i in run.decision_ids if i in run.products
        ]
    elif is_ranking:
        for index, asin in enumerate(run.model_order, start=1):
            product = run.products.get(asin)
            if product is not None:
                verified.append(_card(run, product))
            else:
                run.note("Some ranked products have no available catalog details.")
                verified.append(
                    Card(
                        id=asin,
                        category="Electronics",
                        title="Catalog details unavailable",
                        model_rank=index,
                        price_usd=None,
                        price_status="NOT_REQUIRED",
                        catalog_source=run.catalog.snapshot,
                    )
                )
        if (
            run.recommendation
            and run.recommended
            and not run.failures
            and len(verified) < run.recommendation.n
        ):
            run.note(
                f"The model returned {len(verified)} of the requested {run.recommendation.n} items."
            )
    else:
        comparison = [
            _card(run, run.products[i]) for i in run.refs if i in run.products
        ]
        if run.missing_products:
            run.note(
                "These product IDs are absent from the catalog: "
                + ", ".join(sorted(run.missing_products))
                + "."
            )
        needs_details = (
            not run.agent_loop
            or run.requested_details
            or (run.patch.intent == "compare" and not run.decision_ids)
        )
        if needs_details and len(comparison) + len(run.missing_products) < len(
            run.refs
        ):
            run.note("Some selected product details were unavailable.")
        if any(c.price_status == "UNKNOWN" for c in comparison):
            run.note("Some selected product prices could not be verified.")
    displayed_scope = {c.id for c in [*comparison, *verified, *pending]}
    displayed_scope.update(run.decision_ids)
    if run.agent_loop and run.patch.intent == "evidence":
        displayed_scope.update(run.refs)
    quotes = [
        EvidenceQuote(
            id=e.id,
            product_id=e.product_id,
            text=e.text[:900],
            source=e.source,
            snapshot=run.catalog.snapshot,
            kind="review_observation" if e.source_type == "review" else "catalog_fact",
        )
        for e in run.evidence.values()
        if e.product_id in displayed_scope
        and (run.topical_citation_ids is None or e.id in run.topical_citation_ids)
    ]
    if run.patch.intent == "evidence" or (
        run.patch.intent == "compare"
        and (
            (run.patch.query and run.patch.priority != "rating")
            or asks_for_reviews(run.request)
        )
    ):
        if {q.product_id for q in quotes} != set(run.refs):
            run.note(
                "Requested review evidence was not available for every selected product."
            )
        if run.patch.query:
            run.note(
                "Review excerpts are observations; relevance and generality require human judgment."
            )
    if pending:
        run.note("Pending products have unverified mandatory conditions.")
    if run.failures:
        run.note(
            "One or more data dependencies failed; a failure is not a no-match result."
        )
    if run.recommendation or run.state.persona:
        label = "MODEL" if run.recommendation else "EXAMPLE"
        if {card.id for card in [*verified, *pending[:3]]} & run.ranked_ids:
            run.personalization = label + "_HISTORY_USED"
        elif run.ranked_ids:
            run.personalization = label + "_CANDIDATES_ONLY"
            if verified:
                run.note("The displayed matches came from catalog search.")
        elif run.recommended:
            run.personalization = label + "_HISTORY_UNAVAILABLE"
        else:
            run.personalization = label + "_HISTORY_NOT_REQUESTED"
    decision = run.final_decision or _decision(run, comparison, quotes)
    useful = bool(verified or pending or comparison or quotes)
    if run.question:
        outcome = "CLARIFICATION_NEEDED"
    elif answer:
        outcome = "COMPLETE"
    elif useful:
        insufficient = decision and decision.status in {
            "INSUFFICIENT_PRICES",
            "INSUFFICIENT_REVIEW_EVIDENCE",
            "INSUFFICIENT_RATINGS",
            "UNVERIFIED_REQUIREMENTS",
        }
        outcome = "PARTIAL" if run.limitations or insufficient else "COMPLETE"
    elif (
        run.missing_products == set(run.refs)
        and run.missing_products
        and not run.failures
    ):
        outcome = "NO_VERIFIED_MATCH_IN_SCOPE"
    elif (
        run.agent_loop
        and run.patch.intent in {"compare", "evidence"}
        and run.decision_ids
        and run.refs
        and set(run.refs) <= run.reviewed_ids
        and not run.failures
    ):
        outcome = "PARTIAL"
    elif run.failures or not run.discovery_ok:
        outcome = "TEMPORARILY_UNAVAILABLE"
    elif run.state.unsupported or run.limitations:
        outcome = "PARTIAL"
    else:
        outcome = "NO_VERIFIED_MATCH_IN_SCOPE"
        run.note(
            "No verified match in the current search scope; catalog coverage is not guaranteed."
        )
    difference = None
    if len(comparison) == 2:
        a, b = (c.price_usd for c in comparison)
        if a is not None and b is not None:
            difference = abs(a - b)
    missing_reviews = run.missing_reviews - {q.product_id for q in quotes}
    return Result(
        outcome=outcome,
        revision=run.state.revision,
        snapshot=run.catalog.snapshot,
        data_mode=run.catalog.mode,
        policy=run.policy,
        provider=getattr(run.provider, "name", "offline"),
        intent=run.patch.intent,
        reference_ids=run.state.displayed_ids
        if run.patch.intent in {"compare", "evidence"}
        else None,
        missing_product_ids=sorted(run.missing_products),
        missing_review_ids=sorted(missing_reviews),
        review_gaps={
            pid: run.review_gaps[pid]
            for pid in sorted(missing_reviews & run.review_gaps.keys())
        },
        review_aspect=(run.discovery_goal.review_aspect or "")
        if run.discovery_goal
        else run.patch.query
        if run.patch.intent != "discover"
        else "",
        verified=verified,
        pending=pending[:3],
        comparison=comparison,
        price_difference_usd=difference,
        decision=decision,
        evidence=quotes,
        ranking_source=run.ranking_source,
        history_mode=run.history_mode,
        decision_ids=run.decision_ids,
        unchecked_candidate_ids=unchecked_candidates(run)
        if run.patch.intent == "discover"
        else [],
        limitations=run.limitations,
        question=run.question,
        answer=answer,
        trace=run.trace,
        personalization=run.personalization,
        recommendation_model=run.recommendation.model if run.recommendation else None,
        requested_n=run.recommendation.n if is_ranking and run.recommendation else None,
        model_attempts=run.model_attempts,
        tool_attempts=run.tool_attempts,
        latency_ms=round((monotonic() - run.start) * 1000, 2),
        usage=run.usage,
    )
