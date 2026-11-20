"""Read-only catalog, recommendation, and evidence contracts."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from html import unescape
from importlib.resources import files
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .contracts import Category, ReviewGap
from .recommendation import RankingSource

# These are the categories provided by the bundled synthetic fixture.
CATEGORIES = ("All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing")
Status = Literal["OK", "EMPTY", "PARTIAL", "ERROR"]
CanonicalID = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
]
ShortText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)
]
Excerpt = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=900)
]
USDPrice = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
MAX_PAGE = 25
MAX_EVIDENCE = 12
MAX_EVIDENCE_PER_PRODUCT = 4
MAX_EXPORT_BYTES = 8 * 1024 * 1024
SEARCH_FIELDS = {"max_price_usd", "price_state", "offset", "candidate_ids"}


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _web_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
    ):
        raise ValueError("Expected an absolute HTTP(S) URL without credentials")
    return value


class Product(_Contract):
    id: CanonicalID
    category: Category
    title: ShortText
    # Prices are observed USD values. Missing prices stay unknown.
    price: USDPrice | None = None
    currency: Literal["USD"] = "USD"
    url: str | None = None
    keywords: tuple[ShortText, ...] = ()
    average_rating: Decimal | None = Field(
        default=None, ge=0, le=5, allow_inf_nan=False
    )
    rating_count: int | None = Field(default=None, ge=0)
    port_count: int | None = Field(default=None, gt=0)
    usb_versions: tuple[str, ...] = ()
    spec_source: ShortText | None = None

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        return None if value is None else _web_url(value)


class Evidence(_Contract):
    id: CanonicalID
    product_id: CanonicalID
    text: Excerpt
    source: ShortText
    source_type: Literal["review", "catalog"] = "review"


class CatalogResponse(_Contract):
    status: Status
    snapshot: CanonicalID
    ids: list[CanonicalID] = Field(default_factory=list)
    pending_ids: list[CanonicalID] = Field(default_factory=list)
    above_budget_ids: list[CanonicalID] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    has_more: bool | None = None
    message: ShortText | None = None
    error_code: ShortText | None = None
    missing_ids: list[CanonicalID] = Field(default_factory=list)
    review_gaps: dict[CanonicalID, ReviewGap] = Field(default_factory=dict)
    ranking_source: RankingSource | None = None

    @model_validator(mode="after")
    def truthful_status(self) -> CatalogResponse:
        has_records = bool(
            self.ids
            or self.pending_ids
            or self.above_budget_ids
            or self.products
            or self.evidence
        )
        if self.status in {"EMPTY", "ERROR"} and has_records:
            raise ValueError("EMPTY and ERROR cannot carry usable records")
        if self.status in {"OK", "PARTIAL"} and not has_records:
            raise ValueError("OK and PARTIAL must carry usable records")
        if self.review_gaps.keys() - set(self.missing_ids):
            raise ValueError("Review gaps must refer to missing evidence")
        if set(self.pending_ids) & set(self.ids):
            raise ValueError("Verified and pending candidates must be distinct")
        if set(self.above_budget_ids) & (set(self.ids) | set(self.pending_ids)):
            raise ValueError("Above-budget candidates must be distinct")
        return self


class Catalog(Protocol):
    """Injectable tool boundary. The adapter enforces its own I/O timeout."""

    snapshot: str
    mode: str

    def call(self, action: str, args: dict, timeout: float) -> CatalogResponse: ...


class SnapshotManifest(_Contract):
    """A manifest describes a snapshot and its declared provenance."""

    snapshot: CanonicalID
    origin: Literal["synthetic", "team-export"]
    categories: tuple[Category, ...] = Field(min_length=1, max_length=5)
    description: ShortText
    source_urls: tuple[str, ...] = ()
    product_id_field: ShortText = "id"
    context_provenance: ShortText | None = None

    @field_validator("source_urls")
    @classmethod
    def valid_sources(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_web_url(value) for value in values)


class SnapshotData(_Contract):
    manifest: SnapshotManifest
    products: list[Product] = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    recommendations: dict[CanonicalID, dict[Category, list[CanonicalID]]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def check_relationships(self) -> SnapshotData:
        products = {product.id: product for product in self.products}
        if len(products) != len(self.products):
            raise ValueError("Duplicate product IDs")
        if len({entry.id for entry in self.evidence}) != len(self.evidence):
            raise ValueError("Duplicate evidence IDs")
        if len(set(self.manifest.categories)) != len(self.manifest.categories):
            raise ValueError("Duplicate category declarations")
        if set(self.manifest.categories) != {
            product.category for product in self.products
        }:
            raise ValueError("Manifest categories must match product categories")
        if any(entry.product_id not in products for entry in self.evidence):
            raise ValueError("Evidence refers to a product absent from this snapshot")
        for by_category in self.recommendations.values():
            for category, ids in by_category.items():
                if category not in self.manifest.categories:
                    raise ValueError(
                        "Recommendation category is absent from this snapshot"
                    )
                if len(set(ids)) != len(ids):
                    raise ValueError("Duplicate recommendation IDs in a ranking")
                if any(
                    pid not in products or products[pid].category != category
                    for pid in ids
                ):
                    raise ValueError(
                        "Recommendation IDs must exist and match their category"
                    )
        return self


def _words(value: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", value.casefold()))


_REVIEW_GLUE = frozenset(
    [
        "a",
        "an",
        "and",
        "about",
        "are",
        "as",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "please",
        "say",
        "tell",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "with",
        "would",
        "you",
    ]
)
_REVIEW_EQUIVALENTS = {
    "lights": "light",
    "lighting": "light",
    "batteries": "battery",
    "chargers": "charger",
    "charges": "charge",
    "charging": "charge",
    "cancelling": "cancellation",
    "canceling": "cancellation",
    "waterproofed": "waterproof",
    "waterproofing": "waterproof",
    "weatherproofing": "weatherproof",
}
_REVIEW_COMPOUNDS = re.compile(r"\b(water|weather)[\s-]+(proof(?:ing|ed)?)\b")
_REVIEW_SETUP = re.compile(r"\bset[\s-]+up\b")
_REVIEW_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|<br\s*/?>|\r?\n", re.IGNORECASE)


def _review_terms(value: str) -> set[str]:
    normalized = _REVIEW_COMPOUNDS.sub(r"\1\2", unescape(value).casefold())
    normalized = _REVIEW_SETUP.sub("setup", normalized)
    return {
        _REVIEW_EQUIVALENTS.get(word, word)
        for word in _words(normalized) - _REVIEW_GLUE
    }


def _review_matches(text: str, terms: set[str]) -> bool:
    """Require all aspect terms in one passage. Keep the source quote verbatim."""
    return not terms or any(
        terms <= _review_terms(passage) for passage in _REVIEW_BOUNDARY.split(text)
    )


def _search_filters(
    args: dict,
) -> tuple[Decimal | None, Literal["any", "known", "unknown"], int, list[str] | None]:
    value = args.get("max_price_usd")
    budget = None
    if value is not None:
        if type(value) not in (str, int, Decimal) or len(str(value)) > 40:
            raise ValueError("max_price_usd must be a nonnegative USD amount.")
        try:
            budget = Decimal(value)
        except InvalidOperation as error:
            raise ValueError(
                "max_price_usd must be a nonnegative USD amount."
            ) from error
        if (
            not budget.is_finite()
            or not Decimal(0) <= budget <= Decimal("9999999999.99")
            or budget.as_tuple().exponent < -9
        ):
            raise ValueError("max_price_usd must be a finite nonnegative USD amount.")
    price_state = args.get("price_state", "any")
    if price_state not in ("any", "known", "unknown"):
        raise ValueError("price_state must be any, known, or unknown.")
    offset = args.get("offset", 0)
    if type(offset) is not int or not 0 <= offset <= 1000:
        raise ValueError("offset must be an integer in 0..1000.")
    scope = (
        _id_list(args["candidate_ids"], "candidate_ids", required=True)
        if "candidate_ids" in args
        else None
    )
    return budget, price_state, offset, scope


def _bounded_text(value: object, field: str, cap: int = 300) -> str:
    if not isinstance(value, str) or len(value) > cap:
        raise ValueError(f"{field} must be a string up to {cap} characters")
    return value.strip()


def _bounded_limit(value: object, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"limit must be an integer in 1..{maximum}")
    return value


def _id_list(value: object, field: str, *, required: bool) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_PAGE
        or (required and not value)
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", item)
            for item in value
        )
    ):
        raise ValueError(f"{field} must contain valid product IDs (at most {MAX_PAGE})")
    # Deduplicate while preserving request order.
    return list(dict.fromkeys(value))


class _SnapshotReader:
    def __init__(self, data: SnapshotData, mode: str) -> None:
        self.manifest = data.manifest
        self.categories = data.manifest.categories
        self.snapshot = data.manifest.snapshot
        self.mode = mode
        self._products = {product.id: product for product in data.products}
        self._by_category = {category: [] for category in self.manifest.categories}
        for product in data.products:
            self._by_category[product.category].append(product)
        self._product_terms = {
            # Filter category separately from product keywords.
            p.id: _words(" ".join((p.title, *p.keywords)))
            for p in data.products
        }
        self._evidence: dict[str, list[Evidence]] = {}
        for entry in data.evidence:
            self._evidence.setdefault(entry.product_id, []).append(entry)
        self._rankings = data.recommendations

    @property
    def personas(self) -> tuple[str, ...]:
        return tuple(self._rankings)

    def _response(self, status: Status, **data: object) -> CatalogResponse:
        return CatalogResponse(status=status, snapshot=self.snapshot, **data)

    def _error(self, code: str, message: str) -> CatalogResponse:
        return self._response("ERROR", error_code=code, message=message)

    def call(self, action: str, args: dict, timeout: float = 3.0) -> CatalogResponse:
        # This adapter reads a bounded in-memory snapshot.
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            return self._error("timeout", "No positive finite timeout is available.")
        if not isinstance(args, dict):
            return self._error("invalid_arguments", "Tool arguments must be an object.")
        if args.get("snapshot", self.snapshot) != self.snapshot:
            return self._error(
                "snapshot_mismatch",
                "Requested catalog snapshot does not match this adapter.",
            )
        deadline = monotonic() + timeout
        try:
            if action in {"search_products", "recommend_products"}:
                response = self._candidate_ids(action, args, deadline)
            elif action == "get_products":
                self._allowed(args, {"ids"})
                response = self._product_details(
                    _id_list(args.get("ids"), "ids", required=True)
                )
            elif action == "get_product_evidence":
                self._allowed(args, {"ids", "query", "limit"})
                response = self._product_evidence(args, deadline)
            else:
                return self._error(
                    "unsupported_action", "That tool is not supported by this catalog."
                )
            self._check_deadline(deadline)
            return response
        except TimeoutError as exc:
            return self._error("timeout", str(exc))
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if monotonic() >= deadline:
            raise TimeoutError("The local catalog call exhausted its time budget.")

    @staticmethod
    def _allowed(args: dict, allowed: set[str]) -> None:
        if set(args) - allowed - {"snapshot"}:
            raise ValueError(
                "Unsupported tool argument; no filter was silently ignored"
            )

    def _candidate_ids(
        self, action: str, args: dict, deadline: float
    ) -> CatalogResponse:
        required = "query" if action == "search_products" else "persona"
        allowed = {"category", "limit", "exclude_ids", required}
        self._allowed(
            args, allowed | SEARCH_FIELDS if action == "search_products" else allowed
        )
        category = _bounded_text(args.get("category"), "category", cap=64)
        if category not in self._by_category:
            return self._error(
                "unsupported_category", "Category is unavailable in this snapshot."
            )
        limit = _bounded_limit(args.get("limit", MAX_PAGE), MAX_PAGE)
        excluded = set(
            _id_list(args.get("exclude_ids", []), "exclude_ids", required=False)
        )
        if action == "search_products":
            budget, price_state, offset, scope = _search_filters(args)
            query = _words(_bounded_text(args.get("query", ""), "query"))
            scored = []
            products = (
                [self._products[pid] for pid in scope if pid in self._products]
                if scope is not None
                else self._by_category[category]
            )
            for product in products:
                self._check_deadline(deadline)
                if product.id in excluded or product.category != category:
                    continue
                price = product.price
                if price is None and price_state == "known":
                    continue
                if price is not None and (
                    price_state == "unknown" or (budget is not None and price > budget)
                ):
                    continue
                # Token overlap ranks search results.
                score = len(query & self._product_terms[product.id])
                if score or not query:
                    scored.append((score, product))
            if scope is None:
                # Stable ties retain snapshot order.
                scored.sort(key=lambda pair: -pair[0])
                if budget is not None:
                    scored.sort(key=lambda pair: pair[1].price is None)
            ids = [product.id for _, product in scored][offset:]
        else:
            persona = _bounded_text(args.get("persona"), "persona", cap=128)
            if persona not in self._rankings:
                return self._error(
                    "ineligible_persona",
                    "No authorized example history is available for that persona.",
                )
            ids = [
                pid
                for pid in self._rankings[persona].get(category, [])
                if pid not in excluded
            ]
        return self._response(
            "OK" if ids else "EMPTY", ids=ids[:limit], has_more=len(ids) > limit
        )

    def _product_details(self, ids: list[str]) -> CatalogResponse:
        found = [self._products[pid] for pid in ids if pid in self._products]
        missing = [pid for pid in ids if pid not in self._products]
        if missing and not found:
            return self._response(
                "ERROR",
                error_code="missing_product",
                missing_ids=missing,
                message="No requested product is in this snapshot.",
            )
        if missing:
            return self._response(
                "PARTIAL",
                products=found,
                missing_ids=missing,
                message="Some requested products are missing.",
            )
        return self._response("OK", products=found)

    def _product_evidence(self, args: dict, deadline: float) -> CatalogResponse:
        ids = _id_list(args.get("ids"), "ids", required=True)
        query = _review_terms(_bounded_text(args.get("query", ""), "query"))
        limit = _bounded_limit(args.get("limit", MAX_EVIDENCE), MAX_EVIDENCE)
        by_product: dict[str, list[Evidence]] = {}
        capped = False
        absent = [pid for pid in ids if pid not in self._products]
        for pid in ids:
            matching: list[Evidence] = []
            for entry in self._evidence.get(pid, []):
                self._check_deadline(deadline)
                if _review_matches(entry.text, query):
                    if len(matching) == MAX_EVIDENCE_PER_PRODUCT:
                        capped = True
                        break
                    matching.append(entry)
            by_product[pid] = matching
        # Interleave excerpts across products before applying the overall limit.
        matched = [
            by_product[pid][i]
            for i in range(MAX_EVIDENCE_PER_PRODUCT)
            for pid in ids
            if i < len(by_product[pid])
        ]
        found = matched[:limit]
        represented = {entry.product_id for entry in found}
        missing = [pid for pid in ids if pid not in represented]
        if not found and absent:
            return self._response(
                "ERROR",
                missing_ids=missing,
                error_code="missing_product",
                message="No evidence found and at least one requested product is absent.",
            )
        if not found:
            return self._response("EMPTY", missing_ids=missing, has_more=False)
        message = (
            "Some requested products have no matching evidence." if missing else None
        )
        return self._response(
            "PARTIAL" if missing else "OK",
            evidence=found,
            missing_ids=missing,
            has_more=capped or len(matched) > limit,
            message=message,
        )


class FixtureCatalog(_SnapshotReader):
    """Four-category synthetic catalog for offline regression testing."""

    def __init__(self) -> None:
        path = files("shopping_agent").joinpath("fixtures", "catalog.json")
        data = SnapshotData.model_validate_json(path.read_bytes())
        if data.manifest.origin != "synthetic" or set(data.manifest.categories) != set(
            CATEGORIES
        ):
            raise ValueError(
                "Bundled fixture must be synthetic and contain all four categories"
            )
        super().__init__(data, mode="fixture")


class SnapshotCatalog(_SnapshotReader):
    """Read a small local catalog export with an explicit source manifest."""

    def __init__(self, path: str | Path) -> None:
        with Path(path).open("rb") as source:
            payload = source.read(MAX_EXPORT_BYTES + 1)
        if len(payload) > MAX_EXPORT_BYTES:
            raise ValueError("Snapshot export exceeds the 8 MiB local-adapter limit")
        # Preserve decimal prices through JSON parsing and validation.
        data = SnapshotData.model_validate(json.loads(payload, parse_float=Decimal))
        if data.manifest.origin != "team-export" or not data.manifest.source_urls:
            raise ValueError(
                "External exports require team-export origin and declared source URLs"
            )
        if data.recommendations and not data.manifest.context_provenance:
            raise ValueError(
                "External recommendation rankings require declared context provenance"
            )
        if data.manifest.snapshot == "fixture-v1":
            raise ValueError(
                "An external export cannot use the bundled fixture's snapshot ID"
            )
        super().__init__(data, mode="external-export")


def load_catalog(
    *,
    snapshot: str | None = None,
    recommender: str | None = None,
    review_snapshot: str | None = None,
) -> Catalog:
    if snapshot and recommender:
        raise ValueError("Choose one catalog source: snapshot or recommender.")
    if review_snapshot and not recommender:
        raise ValueError(
            "A separate review snapshot requires the Electronics recommender catalog."
        )
    if recommender:
        from .electronics import ElectronicsCatalog

        return ElectronicsCatalog(recommender, reviews_path=review_snapshot)
    return SnapshotCatalog(snapshot) if snapshot else FixtureCatalog()
