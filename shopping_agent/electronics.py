"""Read-only access to the team's Electronics metadata and exported model rankings."""

import csv
import gzip
import hashlib
import io
import math
import re
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from threading import TIMEOUT_MAX, Lock
from time import monotonic
from typing import Annotated, Literal
from weakref import finalize

from pydantic import BaseModel, Field, StringConstraints, TypeAdapter

from .catalog import (
    MAX_EVIDENCE,
    MAX_EVIDENCE_PER_PRODUCT,
    MAX_PAGE,
    SEARCH_FIELDS,
    CanonicalID,
    CatalogResponse,
    Evidence,
    Product,
    _bounded_limit,
    _bounded_text,
    _id_list,
    _review_matches,
    _review_terms,
    _search_filters,
)

_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_Ranking = Annotated[list[CanonicalID], Field(min_length=1, max_length=MAX_PAGE)]
_ID = TypeAdapter(CanonicalID)
_PRICE = re.compile(r"[0-9]+(?:\.[0-9]{1,2})?")
_WORD = re.compile(r"[^\W_]+")
_PORTS = re.compile(r"\b([1-9][0-9]?)[\s\-‐‑–]+ports?\b", re.IGNORECASE)
_USB_VERSION = re.compile(r"\busb(?:\s+hub)?[\s\-]*([1-4]\.[0-9])\b", re.IGNORECASE)
_REQUIRED_COLUMNS = {
    "parent_asin",
    "title",
    "store",
    "price",
    "category",
    "rating_number",
}


class _RankingExport(BaseModel):
    schema_version: Literal[1]
    catalog_sha256: _SHA256
    model_sha256: dict[Literal["ncf", "tiger"], _SHA256] = Field(
        min_length=2, max_length=2
    )
    rankings: dict[CanonicalID, _Ranking] = Field(min_length=1)


class _ReviewExport(BaseModel):
    schema_version: Literal[1]
    source: str
    catalog_sha256: _SHA256
    prefix_bytes: int | None = Field(default=None, gt=0)
    archive_sha256: _SHA256 | None = None
    timestamp_interval_ms: tuple[int, int] | None = None
    excerpts: dict[CanonicalID, list[tuple[int, str]]]


def _catalog_rows(
    contents: bytes,
) -> Iterator[
    tuple[str, str, str, str, str | None, int | None, int | None, str | None]
]:
    with io.TextIOWrapper(io.BytesIO(contents), encoding="utf-8", newline="") as source:
        rows = csv.DictReader(source, strict=True)
        try:
            header = rows.fieldnames or ()
            if not _REQUIRED_COLUMNS.issubset(header) or len(set(header)) != len(
                header
            ):
                raise ValueError(
                    "Electronics catalog CSV needs unique required column names."
                )
            for row in rows:
                if None in row or any(row[name] is None for name in _REQUIRED_COLUMNS):
                    raise ValueError(
                        f"Electronics catalog CSV has the wrong field count at line {rows.line_num}."
                    )
                title = row["title"].strip()
                if not title:
                    continue
                try:
                    asin = _ID.validate_python(row["parent_asin"])
                    rating_count = (
                        int(row["rating_number"])
                        if row["rating_number"].strip()
                        else None
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Electronics catalog CSV has an invalid ASIN or rating count at line {rows.line_num}."
                    ) from error
                if rating_count is not None and not 0 <= rating_count < 2**63:
                    raise ValueError(
                        f"Electronics catalog CSV has an out-of-range rating count at line {rows.line_num}."
                    )
                price = row["price"].strip()
                # Blank prices and phrases like "from 36.94" do not represent a budget-verifiable amount.
                cents = (
                    int(Decimal(price) * 100)
                    if len(price) <= 15 and _PRICE.fullmatch(price)
                    else None
                )
                average = (row.get("average_rating") or "").strip()
                if not _PRICE.fullmatch(average) or Decimal(average) > 5:
                    average = None
                yield (
                    asin,
                    title,
                    row["store"],
                    row["category"],
                    price if cents is not None else None,
                    cents,
                    rating_count,
                    average,
                )
        except csv.Error as error:
            raise ValueError(
                f"Electronics catalog CSV is malformed near line {rows.line_num}."
            ) from error


class ElectronicsCatalog:
    """Indexes one immutable CSV snapshot and serves precomputed example histories."""

    mode = "team-electronics"
    categories = ("Electronics",)

    def __init__(
        self,
        recommender_root: str | Path,
        *,
        rankings_path: str | Path | None = None,
        reviews_path: str | Path | None = None,
    ):
        catalog_path = Path(recommender_root) / "data" / "item_catalog.csv"
        contents = catalog_path.read_bytes()
        catalog_digest = hashlib.sha256(contents).hexdigest()
        self.catalog_sha256 = catalog_digest
        ranking_source = (
            Path(rankings_path)
            if rankings_path is not None
            else files("shopping_agent").joinpath(
                "fixtures", "electronics_rankings.json"
            )
        )
        ranking_bytes = ranking_source.read_bytes()
        export = _RankingExport.model_validate_json(ranking_bytes)
        if export.catalog_sha256 != catalog_digest:
            raise ValueError(
                "Electronics rankings were made from a different item_catalog.csv."
            )

        self._reviews: dict[str, list[tuple[int, str]]] = {}
        self._review_source = ""
        self.review_sample = "The current catalog has no bundled review sample."
        evidence_bytes = b""
        if reviews_path is not None or rankings_path is None:
            evidence_source = (
                Path(reviews_path)
                if reviews_path is not None
                else (
                    files("shopping_agent").joinpath(
                        "fixtures", "electronics_reviews.json.gz"
                    )
                )
            )
            evidence_bytes = evidence_source.read_bytes()
            reviews = _ReviewExport.model_validate_json(gzip.decompress(evidence_bytes))
            if reviews.catalog_sha256 != catalog_digest:
                raise ValueError(
                    "The Electronics review sample belongs to a different catalog."
                )
            self._reviews, self._review_source = reviews.excerpts, reviews.source
            sample = f"Review sample covers {len(reviews.excerpts):,} catalog products"
            if reviews.prefix_bytes:
                sample += f" from the first {reviews.prefix_bytes / 2**20:g} MiB of the source archive"
            elif reviews.archive_sha256:
                sample += " from the local full source archive"
            if reviews.timestamp_interval_ms:
                start, stop = reviews.timestamp_interval_ms
                first = datetime.fromtimestamp(start / 1000, UTC).date()
                last = datetime.fromtimestamp((stop - 1) / 1000, UTC).date()
                sample += f", filtered to {first} through {last}"
            self.review_sample = (
                f"{sample}. Up to {MAX_EVIDENCE_PER_PRODUCT} excerpts per product."
            )
        content_digest = hashlib.sha256(ranking_bytes + evidence_bytes).hexdigest()
        self.snapshot = f"electronics-v3-{catalog_digest[:16]}-{content_digest[:16]}"
        self.personas = tuple(export.rankings)
        self._rankings = export.rankings
        self._lock = Lock()
        try:
            db = sqlite3.connect(":memory:", check_same_thread=False)
        except sqlite3.Error as error:
            raise ValueError(
                "The Electronics search index could not be opened."
            ) from error
        try:
            db.executescript(
                """
                CREATE TABLE products (
                    asin TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    store TEXT NOT NULL,
                    category_path TEXT NOT NULL,
                    price TEXT,
                    price_cents INTEGER,
                    rating_count INTEGER,
                    average_rating TEXT
                );
                CREATE VIRTUAL TABLE product_search USING fts5(
                    title, store, category_path, content='products', tokenize='unicode61'
                );
                """
            )
            with db:
                inserted = db.executemany(
                    "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    _catalog_rows(contents),
                ).rowcount
                if not inserted:
                    raise ValueError(
                        "Electronics catalog CSV has no products with a title."
                    )
                db.execute("CREATE INDEX browse ON products(rating_count DESC, asin)")
                db.execute("CREATE INDEX title_exact ON products(title COLLATE NOCASE)")
                db.execute(
                    "INSERT INTO product_search(product_search) VALUES ('rebuild')"
                )
            for alias, ids in self._rankings.items():
                if len(set(ids)) != len(ids) or len(self._lookup(db, ids)) != len(ids):
                    raise ValueError(
                        f"Electronics ranking {alias!r} contains duplicate or unlistable IDs."
                    )
            db.execute("PRAGMA query_only = ON")
        except sqlite3.IntegrityError as error:
            db.close()
            raise ValueError(
                "Electronics catalog CSV contains a duplicate ASIN."
            ) from error
        except sqlite3.Error as error:
            db.close()
            if "no such module: fts5" in str(error).lower():
                raise ValueError(
                    "The Electronics catalog needs a Python SQLite build with FTS5."
                ) from error
            raise ValueError(
                "The Electronics search index could not be initialized."
            ) from error
        except Exception:
            db.close()
            raise
        self._db = db
        # Streamlit can share the adapter until the cached resource is released.
        finalize(self, db.close)

    def _response(self, status, **fields) -> CatalogResponse:
        return CatalogResponse(status=status, snapshot=self.snapshot, **fields)

    def _error(self, code: str, message: str, **fields) -> CatalogResponse:
        return self._response("ERROR", error_code=code, message=message, **fields)

    def _timeout(self) -> CatalogResponse:
        return self._error(
            "timeout", "The Electronics catalog call exhausted its time budget."
        )

    def call(self, action: str, args: dict, timeout: float = 3.0) -> CatalogResponse:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            return self._timeout()
        if not isinstance(args, dict):
            return self._error("invalid_arguments", "Tool arguments must be an object.")
        if args.get("snapshot", self.snapshot) != self.snapshot:
            return self._error(
                "snapshot_mismatch", "Requested Electronics snapshot is unavailable."
            )

        deadline = monotonic() + timeout
        wait = max(0.0, min(deadline - monotonic(), TIMEOUT_MAX))
        if not self._lock.acquire(timeout=wait):
            return self._timeout()
        try:
            if monotonic() >= deadline:
                return self._timeout()
            # The lock makes the single connection and its per-call deadline safe
            # to use from different Streamlit sessions.
            self._db.set_progress_handler(lambda: monotonic() >= deadline, 1000)
            response = self._dispatch(action, args)
            return self._timeout() if monotonic() >= deadline else response
        except ValueError as error:
            return self._error("invalid_arguments", str(error))
        except sqlite3.Error as error:
            if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                return self._timeout()
            return self._error(
                "catalog_unavailable", "The Electronics index could not be read."
            )
        finally:
            self._db.set_progress_handler(None, 0)
            self._lock.release()

    def _dispatch(self, action: str, args: dict) -> CatalogResponse:
        if action in {"search_products", "recommend_products"}:
            selector = "query" if action == "search_products" else "persona"
            allowed = {"category", "limit", "exclude_ids", selector}
            self._allowed(
                args,
                allowed | SEARCH_FIELDS if action == "search_products" else allowed,
            )
            return self._candidates(action, args)
        if action == "resolve_product":
            self._allowed(args, {"name"})
            name = _bounded_text(args.get("name"), "name")
            ids = self._resolve_product(name)
            return self._response("OK" if ids else "EMPTY", ids=ids)
        if action == "get_products":
            self._allowed(args, {"ids"})
            return self._details(_id_list(args.get("ids"), "ids", required=True))
        if action == "get_product_evidence":
            self._allowed(args, {"ids", "query", "limit"})
            return self._evidence(args)
        return self._error(
            "unsupported_action", "That tool is not supported by this catalog."
        )

    @staticmethod
    def _allowed(args: dict, allowed: set[str]) -> None:
        if set(args) - allowed - {"snapshot"}:
            raise ValueError(
                "Unsupported tool argument; no filter was silently ignored"
            )

    def _candidates(self, action: str, args: dict) -> CatalogResponse:
        category = _bounded_text(args.get("category"), "category", cap=64)
        if category not in self.categories:
            return self._error(
                "unsupported_category", "Category is unavailable in this snapshot."
            )
        limit = _bounded_limit(args.get("limit", MAX_PAGE), MAX_PAGE)
        excluded = set(
            _id_list(args.get("exclude_ids", []), "exclude_ids", required=False)
        )
        if action == "recommend_products":
            alias = _bounded_text(args.get("persona"), "persona", cap=128)
            if alias not in self._rankings:
                return self._error(
                    "ineligible_persona",
                    "No authorized example history is available for that persona.",
                )
            ids = [asin for asin in self._rankings[alias] if asin not in excluded]
            return self._response(
                "OK" if ids else "EMPTY",
                ids=ids[:limit],
                has_more=len(ids) > limit,
                ranking_source="precomputed_example",
            )
        query = _bounded_text(args.get("query", ""), "query")
        budget, price_state, offset, scope = _search_filters(args)
        # A separate unknown-price bucket keeps unpriced hits from consuming the
        # page of priced matches. Scope queries preserve model rank instead.
        separate_pending = budget is not None and price_state == "any" and scope is None
        ids = self._search(
            query,
            excluded,
            limit,
            offset=offset,
            budget=budget,
            price_state="known" if separate_pending else price_state,
            scope=scope,
        )
        pending = (
            self._search(query, excluded, 3, price_state="unknown")[:3]
            if separate_pending and offset == 0
            else []
        )
        above = (
            self._search(query, excluded, 2, budget=budget, price_state="above")[:2]
            if separate_pending and offset == 0
            else []
        )
        return self._response(
            "OK" if ids or pending or above else "EMPTY",
            ids=ids[:limit],
            pending_ids=pending,
            above_budget_ids=above,
            has_more=len(ids) > limit,
        )

    def _search(
        self,
        query: str,
        excluded: set[str],
        limit: int,
        *,
        offset: int = 0,
        budget: Decimal | None = None,
        price_state: str = "any",
        scope: list[str] | None = None,
        title_only: bool = False,
    ) -> list[str]:
        terms = dict.fromkeys(_WORD.findall(query))
        if query and not terms:
            return []
        conditions = []
        parameters = []
        source = "products"
        sort = "products.rating_count DESC, products.asin"
        if terms:
            match = " AND ".join(f'"{term}"' for term in terms)
            source = (
                "product_search JOIN products ON products.rowid = product_search.rowid"
            )
            conditions.append("product_search MATCH ?")
            # Named-product lookups also anchor the first token in the title.
            # "Case compatible with Sony ..." must not become a Sony match.
            parameters.append(f"title : (^{match})" if title_only else match)
            sort = f"bm25(product_search, 5.0, 2.0, 1.0), {sort}"
        if excluded:
            placeholders = ", ".join("?" for _ in excluded)
            conditions.append(f"products.asin NOT IN ({placeholders})")
            parameters.extend(sorted(excluded))
        if scope is not None:
            placeholders = ", ".join("?" for _ in scope)
            conditions.append(f"products.asin IN ({placeholders})")
            parameters.extend(scope)
        if price_state == "known":
            conditions.append("products.price_cents IS NOT NULL")
        elif price_state == "unknown":
            conditions.append("products.price_cents IS NULL")
        if budget is not None and price_state != "unknown":
            numerator, denominator = budget.as_integer_ratio()
            if price_state == "above":
                conditions.append("products.price_cents > ?")
            else:
                conditions.append(
                    "(products.price_cents IS NULL OR products.price_cents <= ?)"
                )
            parameters.append(numerator * 100 // denominator)
        where = " AND ".join(conditions) or "1 = 1"
        if scope is not None:
            rows = self._db.execute(
                f"SELECT products.asin FROM {source} WHERE {where}",
                parameters,
            )
            matched = {asin for (asin,) in rows}
            return [asin for asin in scope if asin in matched][
                offset : offset + limit + 1
            ]
        rows = self._db.execute(
            f"SELECT products.asin FROM {source} WHERE {where} ORDER BY {sort} LIMIT ? OFFSET ?",
            (*parameters, limit + 1, offset),
        )
        return [asin for (asin,) in rows]

    def resolve_product(self, name: str, timeout: float = 3.0) -> list[str]:
        """Find up to two matches so callers can report unresolved ambiguity."""
        result = self.call("resolve_product", {"name": name}, timeout)
        if result.error_code == "timeout":
            raise TimeoutError(result.message)
        if result.status == "ERROR":
            raise ValueError(result.message)
        return result.ids

    def _resolve_product(self, name: str) -> list[str]:
        exact = [
            asin
            for (asin,) in self._db.execute(
                "SELECT asin FROM products WHERE title = ? COLLATE NOCASE ORDER BY asin LIMIT 2",
                (name,),
            )
        ]
        if exact or len(set(_WORD.findall(name.casefold()))) < 2:
            return exact
        return self._search(name, set(), 2, title_only=True)[:2]

    @staticmethod
    def _lookup(
        db: sqlite3.Connection, ids: list[str]
    ) -> dict[str, tuple[str, str | None, str | None, int | None]]:
        placeholders = ", ".join("?" for _ in ids)
        return {
            asin: (title, price, average_rating, rating_count)
            for asin, title, price, average_rating, rating_count in db.execute(
                f"SELECT asin, title, price, average_rating, rating_count FROM products WHERE asin IN ({placeholders})",
                ids,
            )
        }

    def _details(self, ids: list[str]) -> CatalogResponse:
        rows = self._lookup(self._db, ids)
        missing = [asin for asin in ids if asin not in rows]
        if not rows:
            return self._error(
                "missing_product",
                "No requested product is in this snapshot.",
                missing_ids=missing,
            )
        products = []
        for asin in ids:
            if asin not in rows:
                continue
            title, price, average_rating, rating_count = rows[asin]
            versions = tuple(dict.fromkeys(_USB_VERSION.findall(title)))
            lower_title = title.casefold()
            counts = (
                set(_PORTS.findall(title))
                if "usb" in lower_title and "hub" in lower_title
                else set()
            )
            products.append(
                Product(
                    id=asin,
                    category="Electronics",
                    title=title[:300],
                    price=price,
                    average_rating=average_rating,
                    rating_count=rating_count,
                    port_count=int(next(iter(counts))) if len(counts) == 1 else None,
                    usb_versions=versions,
                    spec_source="catalog_title" if counts or versions else None,
                )
            )
        return self._response(
            "PARTIAL" if missing else "OK", products=products, missing_ids=missing
        )

    def _evidence(self, args: dict) -> CatalogResponse:
        ids = _id_list(args.get("ids"), "ids", required=True)
        terms = _review_terms(_bounded_text(args.get("query", ""), "query"))
        limit = _bounded_limit(args.get("limit", MAX_EVIDENCE), MAX_EVIDENCE)
        rows = self._lookup(self._db, ids)
        missing = [asin for asin in ids if asin not in rows]
        if missing:
            return self._error(
                "missing_product",
                "At least one requested product is absent.",
                missing_ids=missing,
            )
        if not self._review_source:
            return self._error(
                "review_unavailable",
                "The review sample is unavailable for this catalog.",
                missing_ids=ids,
            )
        matches = {
            asin: [
                (line, text)
                for line, text in self._reviews.get(asin, ())
                if _review_matches(text, terms)
            ]
            for asin in ids
        }
        review_gaps = {
            asin: "NO_ASPECT_MATCH" if self._reviews.get(asin) else "OUTSIDE_SAMPLE"
            for asin, entries in matches.items()
            if not entries
        }
        capped = any(
            len(entries) > MAX_EVIDENCE_PER_PRODUCT for entries in matches.values()
        )
        ordered = [
            (asin, *matches[asin][index])
            for index in range(MAX_EVIDENCE_PER_PRODUCT)
            for asin in ids
            if index < len(matches[asin])
        ]
        selected = ordered[:limit]
        represented = {asin for asin, _, _ in selected}
        missing = [asin for asin in ids if asin not in represented]
        if not selected:
            return self._response(
                "EMPTY", missing_ids=missing, review_gaps=review_gaps, has_more=False
            )
        return self._response(
            "PARTIAL" if missing else "OK",
            evidence=[
                Evidence(
                    id=f"ar23-electronics-{line}",
                    product_id=asin,
                    text=text,
                    source=f"{self._review_source} · JSONL line {line}",
                    source_type="review",
                )
                for asin, line, text in selected
            ],
            missing_ids=missing,
            review_gaps=review_gaps,
            has_more=capped or len(ordered) > limit,
        )
