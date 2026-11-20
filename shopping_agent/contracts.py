"""Schemas for untrusted requests, accepted state, and trusted presentation."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Electronics",
]
DecisionPriority = Literal["price", "rating", "review_aspect"]
ReviewGap = Literal["OUTSIDE_SAMPLE", "NO_ASPECT_MATCH"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Operation(Record):
    field: Literal["category", "max_price_usd", "excluded_id", "unsupported"]
    op: Literal["set", "add", "remove", "clear"]
    value: str | None = Field(default=None, max_length=200)


class HardRequirement(Record):
    kind: Literal[
        "min_ports",
        "max_ports",
        "exact_ports",
        "required_usb_version",
        "excluded_usb_version",
        "compatibility",
        "property",
    ]
    value: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200)


class Patch(Record):
    operations: list[Operation] = Field(default_factory=list, max_length=12)
    intent: Literal["discover", "recommend", "compare", "evidence", "answer"] = (
        "discover"
    )
    references: list[int] = Field(default_factory=list, max_length=3)
    product_names: list[str] = Field(default_factory=list, max_length=3)
    requirements: list[HardRequirement] = Field(default_factory=list, max_length=12)
    query: str = Field(default="", max_length=160)
    clear_query: bool = False
    clarification: str | None = Field(default=None, max_length=240)
    answer: str | None = Field(default=None, max_length=700)
    priority: DecisionPriority | None = None


class Action(Record):
    action: Literal[
        "search_products",
        "recommend_products",
        "get_products",
        "get_product_evidence",
        "ask_user",
        "finish_decision",
    ]
    args: dict = Field(default_factory=dict)


class ProductFollowup(Record):
    intent: Literal["compare", "evidence"]
    ids: list[str] = Field(max_length=3)
    query: str = Field(default="", max_length=160)
    priority: DecisionPriority | None = None


class PendingReference(Record):
    intent: Literal["compare", "evidence"]
    query: str
    priority: DecisionPriority | None = None
    listed_ids: list[str]
    snapshot: str | None


class Session(Record):
    revision: int = Field(default=0, ge=0)
    category: Category | None = None
    max_price_usd: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    excluded_ids: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    requirements: list[HardRequirement] = Field(default_factory=list, max_length=12)
    displayed_ids: list[str] = Field(default_factory=list)
    snapshot: str | None = None
    persona: str | None = None
    pending_question: str | None = None
    pending_reference: PendingReference | None = None
    pending_goal: str | None = None
    search_query: str = Field(default="", max_length=160)
    followup: ProductFollowup | None = None


class RecommendationRequest(Record):
    user_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    model: Literal["ncf", "tiger"]
    n: int = Field(ge=1, le=25, strict=True)


class DiscoveryGoal(Record):
    """Requested outputs for discovery, independent of which tool runs first."""

    count: int = Field(ge=1, le=3)
    review_aspect: str | None = None
    criterion: Literal["price", "rating", "price_difference"] | None = None
    fallback: Literal["price", "rating"] | None = None


class ConditionCheck(Record):
    requirement: str
    status: Literal["SATISFIED", "VIOLATED", "UNKNOWN"]
    source: str | None = None


class Card(Record):
    id: str
    category: Category
    title: str
    price_usd: Decimal | None
    price_status: Literal["SATISFIED", "VIOLATED", "UNKNOWN", "NOT_REQUIRED"]
    catalog_source: str
    example_ranking: str | None = None
    model_rank: int | None = None
    average_rating: Decimal | None = None
    rating_count: int | None = None
    conditions: list[ConditionCheck] = Field(default_factory=list)


class EvidenceQuote(Record):
    id: str
    product_id: str
    text: str
    source: str
    snapshot: str
    kind: Literal["review_observation", "catalog_fact"] = "review_observation"


class Decision(Record):
    basis: DecisionPriority
    status: Literal[
        "LOWEST_PRICE",
        "PRICE_TIE",
        "INSUFFICIENT_PRICES",
        "UNVERIFIED_REQUIREMENTS",
        "NO_BUDGET_MATCH",
        "INSUFFICIENT_REVIEW_EVIDENCE",
        "REVIEW_TRADEOFFS",
        "INSUFFICIENT_RATINGS",
        "HIGHEST_RATING",
        "RATING_TIE",
    ]
    product_id: str | None = None


class Result(Record):
    outcome: Literal[
        "COMPLETE",
        "PARTIAL",
        "CLARIFICATION_NEEDED",
        "NO_VERIFIED_MATCH_IN_SCOPE",
        "TEMPORARILY_UNAVAILABLE",
    ]
    revision: int
    snapshot: str
    data_mode: str
    policy: str
    provider: str
    intent: Literal["discover", "compare", "evidence", "recommend", "answer"] = (
        "discover"
    )
    reference_ids: list[str] | None = None
    missing_product_ids: list[str] = Field(default_factory=list)
    missing_review_ids: list[str] = Field(default_factory=list)
    review_gaps: dict[str, ReviewGap] = Field(default_factory=dict)
    review_aspect: str = ""
    verified: list[Card] = Field(default_factory=list)
    pending: list[Card] = Field(default_factory=list)
    comparison: list[Card] = Field(default_factory=list)
    price_difference_usd: Decimal | None = None
    decision: Decision | None = None
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    model_reply: list[str] = Field(default_factory=list)
    above_budget: list[Card] = Field(default_factory=list)
    ranking_source: str | None = None
    history_mode: Literal["catalog_only", "history_preferred", "history_only"] = (
        "catalog_only"
    )
    decision_ids: list[str] = Field(default_factory=list)
    unchecked_candidate_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    question: str | None = None
    answer: str | None = None
    trace: list[dict] = Field(default_factory=list)
    personalization: str = "NO_HISTORY"
    recommendation_model: Literal["ncf", "tiger"] | None = None
    requested_n: int | None = None
    model_attempts: int = 0
    tool_attempts: int = 0
    latency_ms: float = 0
    usage: list[dict] = Field(default_factory=list)

    def selected_ids(self) -> list[str]:
        """User-selected or decision ASINs, including reviews without detail cards."""
        return self.decision_ids or [card.id for card in self.comparison]

    def shown_ids(self) -> list[str]:
        """The IDs rendered in this turn, including any selected details."""
        return list(
            dict.fromkeys(
                card.id for card in [*self.verified, *self.pending, *self.comparison]
            )
        )
