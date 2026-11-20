"""Turn lifecycle: persist accepted edits before any catalog acquisition."""

import re
from time import monotonic

from pydantic import ValidationError

from .catalog import CATEGORIES
from .contracts import (
    Operation,
    Patch,
    PendingReference,
    ProductFollowup,
    RecommendationRequest,
    Result,
    Session,
)
from .provider import ProviderError
from .state import (
    BUDGET_QUESTION,
    DISCOVERY_HELP,
    EXPLICIT_SEARCH,
    apply_patch,
    catalog_keywords,
    contextualize_followup,
    extract_asins,
    grounded_comparison_aspect,
    grounded_product_names,
    grounded_shopping_intent,
    history_count,
    history_request,
    offline_parse,
    prioritizes_price,
    prioritizes_rating,
    resolve_references,
    resume_comparison,
    resume_reference,
    structured_requirements,
    wants_category_browse,
    without_pasted_cards,
)
from .workflow import Limits, Run, build_result, execute


class Agent:
    def __init__(
        self,
        store,
        catalog,
        *,
        provider=None,
        policy="adaptive",
        limits=None,
        recommender=None,
    ):
        if policy not in {"fixed", "adaptive"}:
            raise ValueError("Policy must be fixed or adaptive.")
        self.store, self.catalog, self.provider = store, catalog, provider
        self.recommender = recommender
        self.categories = getattr(catalog, "categories", ())
        self.conversational = getattr(provider, "local_intents", False)
        self.policy = policy
        self.limits = limits or Limits()

    def _run(
        self, state: Session, patch: Patch | None = None, refs: list[str] | None = None
    ) -> Run:
        return Run(
            state,
            patch or Patch(),
            refs or [],
            self.catalog,
            self.provider,
            self.policy,
            self.limits,
            recommender=self.recommender,
        )

    def _simple(self, state: Session, message: str) -> Result:
        run = self._run(state)
        run.note(message)
        return build_result(run)

    def _parse(self, text: str, run: Run) -> Patch:
        offline = offline_parse(text, run.state, conversational=self.conversational)
        if self.provider is None:
            return offline
        if run.model_attempts >= self.limits.model_attempts or run.remaining() <= 0:
            raise ValueError("No model-call budget for parsing.")
        run.model_attempts += 1
        context = run.state.model_dump(mode="json")
        if self.conversational:
            context["available_categories"] = self.categories or CATEGORIES
            if run.recent_turns:
                context["recent_turns"] = run.recent_turns
        try:
            supplied = Patch.model_validate(
                self.provider.parse(
                    text,
                    context,
                    min(45 if self.conversational else 20, run.remaining()),
                )
            )
        finally:
            usage = getattr(self.provider, "last_usage", None)
            if usage:
                run.usage.append(dict(usage))
        if self.conversational:
            run.trace.append({"action": "model_parse", "status": "OK"})
            return self._local_patch(text, offline, supplied, run.state)
        if offline.clarification and (
            "budget" in offline.clarification.lower()
            or "usd" in offline.clarification.lower()
            or "one product category" in offline.clarification
        ):
            return offline
        # A live model can interpret the task, but cannot invent or weaken supported hard
        # edits. Only explicit phrases recognized by the application may change hard state.
        hard = [op for op in supplied.operations if op.field != "unsupported"]
        grounded = [op for op in offline.operations if op.field != "unsupported"]

        def semantic(op):
            if op.field == "max_price_usd" and op.op == "set" and op.value is not None:
                from decimal import Decimal

                return op.field, op.op, Decimal(op.value).normalize()
            return op.field, op.op, op.value

        if hard and set(map(semantic, hard)) != set(map(semantic, grounded)):
            return Patch(
                clarification="Confirm the exact category or USD budget using an explicit request."
            )
        supported_unsupported = [
            op for op in offline.operations if op.field == "unsupported"
        ]
        reported = [
            op
            for op in supplied.operations
            if op.field == "unsupported" and op.op == "add"
        ]
        if any(op.op == "clear" for op in supported_unsupported):
            reported = []
        # An unrecognized discovery qualifier cannot silently become a verified match.
        query = supplied.query or offline.query
        if (
            offline.clarification
            and not grounded
            and not query
            and not reported
            and supplied.intent == "discover"
            and not supplied.clarification
        ):
            return offline
        return supplied.model_copy(
            update={
                "operations": [*grounded, *supported_unsupported, *reported],
                "query": query,
            }
        )

    def _local_patch(
        self, text: str, offline: Patch, supplied: Patch, state: Session
    ) -> Patch:
        """Interpret the conversation while deriving hard edits from explicit input."""
        if supplied.intent == "answer":
            required = grounded_shopping_intent(text, offline, supplied.product_names)
            if required is None:
                return Patch(intent="answer", answer=supplied.answer)
            if required == "discover" and not (
                offline.query or wants_category_browse(text) or supplied.query
            ):
                return Patch(
                    clarification="What electronics would you like me to look up?"
                )
            supplied = supplied.model_copy(update={"intent": required, "answer": None})
        intent = offline.intent if offline.intent != "discover" else supplied.intent
        if offline.clarification not in (None, DISCOVERY_HELP) and not (
            offline.clarification == BUDGET_QUESTION and intent != "discover"
        ):
            return offline
        operations = list(offline.operations)
        requirements = offline.requirements
        if intent == "discover" or re.search(
            r"\b(?:must|required|only if)\b", text.casefold()
        ):
            requirements = structured_requirements(
                text, [req.text for req in supplied.requirements]
            )
        if intent != "discover":
            hard_qualifier = re.search(
                r"\b(?:must|required|only if)\b", text.casefold()
            )
            usd_edit = re.search(r"\$|\b(?:usd|dollars?|budget)\b", text.casefold())
            operations = [
                op
                for op in operations
                if not (
                    op.field == "unsupported" and op.op == "add" and not hard_qualifier
                )
                and not (op.field == "max_price_usd" and not usd_edit)
            ]
        if (
            state.category is None
            and len(self.categories) == 1
            and not any(op.field == "category" for op in operations)
        ):
            operations.append(
                Operation(field="category", op="set", value=self.categories[0])
            )
        explicit_query = EXPLICIT_SEARCH.search(text.lower())
        query = offline.query if explicit_query else supplied.query or offline.query
        priority = None
        if intent == "compare":
            if not grounded_comparison_aspect(text, query):
                query = ""
            if prioritizes_rating(text):
                priority = "rating"
            elif prioritizes_price(text):
                priority = "price"
            elif supplied.priority == "review_aspect" and query:
                priority = supplied.priority
        if not explicit_query and query.casefold().strip() in {
            "electronic",
            "electronics",
            "electronic products",
            "electronic devices",
            "electronic items",
        }:
            query = ""
        category_browse = wants_category_browse(text)
        clear_query = category_browse or bool(
            not explicit_query
            and re.search(
                r"\b(?:clear|remove|drop) (?:the )?(?:keyword|search term|query)",
                text.casefold(),
            )
        )
        return supplied.model_copy(
            update={
                "operations": operations[:12],
                "intent": intent,
                "references": offline.references or supplied.references,
                "product_names": (
                    grounded_product_names(text, supplied.product_names)
                    if intent != "discover"
                    else []
                ),
                "requirements": requirements,
                "query": (
                    ""
                    if category_browse
                    else catalog_keywords(query)
                    if intent == "discover"
                    else query
                ),
                "priority": priority,
                "clear_query": clear_query,
                "clarification": (
                    None
                    if offline.operations or offline.intent != "discover"
                    else supplied.clarification
                ),
            }
        )

    def _resolve_products(self, run: Run, text: str, asins: list[str]) -> list[str]:
        """Resolve names in the trusted catalog before retrieving any evidence."""
        names = [name for name in run.patch.product_names if not extract_asins(name)]
        if not names:
            return resolve_references(
                run.state,
                run.patch,
                text=text,
                asins=asins,
                strict=self.categories == ("Electronics",),
            )
        lookup = getattr(self.catalog, "resolve_product", None)
        if lookup is None:
            raise ValueError(
                "Paste the exact ASIN or select a displayed product number."
            )
        resolved = list(asins)
        for name in names:
            run.tool_attempts += 1
            matches = lookup(name, min(run.remaining(), self.limits.tool_timeout))
            run.trace.append(
                {
                    "action": "resolve_product",
                    "status": "OK"
                    if len(matches) == 1
                    else "AMBIGUOUS"
                    if matches
                    else "EMPTY",
                }
            )
            if len(matches) != 1:
                raise ValueError(
                    "Which exact product do you mean? Paste its ASIN or use a displayed product number."
                )
            resolved.extend(matches)
        unique = list(dict.fromkeys(resolved))
        if len(unique) > 3:
            raise ValueError("Choose up to three products for this request.")
        return unique

    def submit(
        self,
        session_id: str,
        submission_id: str,
        text: str,
        *,
        persona: str | None = None,
        recommendation: RecommendationRequest | None = None,
        next_items: bool = False,
    ) -> Result:
        text = text.strip()
        if not text or len(text) > 1200:
            return self._simple(
                Session(), "Enter a request from 1 to 1,200 characters."
            )
        disposition, payload = self.store.begin(session_id, submission_id, text)
        if disposition == "BUSY":
            state = Session.model_validate(self.store.get_state(session_id))
            return self._simple(state, "This session already has an active request.")
        if disposition == "DONE":
            state = Session.model_validate(self.store.get_state(session_id))
            if (
                not payload
                or "run_state" in payload
                or payload.get("revision") != state.revision
            ):
                return self._simple(
                    state,
                    "This earlier submission is interrupted or stale; send a new request.",
                )
            cached = Result.model_validate(payload)
            expected = (
                cached.shown_ids()
                if cached.intent in {"discover", "recommend"}
                else cached.reference_ids
            )
            # A different list can be published without changing requirements.
            if (
                cached.shown_ids() or cached.selected_ids()
            ) and expected != state.displayed_ids:
                return self._simple(
                    state,
                    "A newer display superseded this earlier result; send a new request.",
                )
            return cached
        state = Session.model_validate(payload or {})
        run = self._run(state)
        run.request = text
        run.recommendation = recommendation
        if next_items:
            run.history_mode = "history_only"
        started = monotonic()
        try:
            if self.conversational:
                run.recent_turns = self.store.recent_turns(session_id)
            if persona is not None and persona not in getattr(
                self.catalog, "personas", ()
            ):
                raise ValueError(
                    "Choose one of the catalog's explicit example personas."
                )
            # A new catalog invalidates item IDs. A history selection affects the next discovery.
            if state.snapshot != self.catalog.snapshot or state.persona != persona:
                changes = {
                    "revision": state.revision + 1,
                    "snapshot": self.catalog.snapshot,
                    "persona": persona,
                    "pending_question": None,
                    "pending_reference": None,
                }
                if state.snapshot != self.catalog.snapshot:
                    changes.update(
                        excluded_ids=[],
                        displayed_ids=[],
                        followup=None,
                        pending_goal=None,
                    )
                state = state.model_copy(update=changes)
                self.store.commit(
                    session_id, submission_id, state.model_dump(mode="json")
                )
            elif payload and "followup" not in payload:
                prior = self.store.last_discovery(
                    session_id, state.revision, state.snapshot
                )
                if prior is not None:
                    state = state.model_copy(
                        update={
                            "displayed_ids": Result.model_validate(prior).shown_ids()
                        }
                    )
                self.store.commit(
                    session_id, submission_id, state.model_dump(mode="json")
                )
            run.state = state
            if (
                self.policy == "adaptive"
                and self.categories == ("Electronics",)
                and callable(getattr(self.provider, "decide", None))
                and not next_items
            ):
                from .decision_loop import converse

                def save(accepted):
                    nonlocal state
                    self.store.commit(
                        session_id, submission_id, accepted.model_dump(mode="json")
                    )
                    state = run.state = accepted

                result = converse(self, run, save)
                return self._publish(session_id, submission_id, run, result)
            asins = extract_asins(text) if self.categories == ("Electronics",) else []
            parse_text = without_pasted_cards(text) if asins else text
            patch = (
                Patch(intent="recommend")
                if next_items
                else self._parse(parse_text or "Show the product", run)
            )
            patch = contextualize_followup(parse_text, patch, state, asins)
            patch = resume_reference(parse_text, patch, state, asins)
            if not asins and not next_items:
                patch, selected = resume_comparison(parse_text, patch, state)
                asins = selected or asins
            if next_items or history_request(parse_text, patch):
                count = history_count(parse_text) if not next_items else None
                patch = Patch(intent="recommend")
                if count is not None:
                    if not 1 <= count <= 25:
                        patch = Patch(
                            intent="recommend",
                            clarification="Choose a recommendation count from 1 to 25.",
                        )
                    elif recommendation is not None:
                        run.recommendation = recommendation.model_copy(
                            update={"n": count}
                        )
                if state.category is None and len(self.categories) == 1:
                    patch = patch.model_copy(
                        update={
                            "operations": [
                                Operation(
                                    field="category", op="set", value=self.categories[0]
                                )
                            ]
                        }
                    )
            if (
                asins
                and not state.category
                and not patch.clarification
                and not any(op.field == "category" for op in patch.operations)
            ):
                patch = patch.model_copy(
                    update={
                        "operations": [
                            *patch.operations,
                            Operation(field="category", op="set", value="Electronics"),
                        ]
                    }
                )
            run.patch = patch
            pending_reference = None
            if patch.clarification:
                run.question = patch.clarification
            elif patch.intent != "answer":
                updated = apply_patch(state, patch)
                if (
                    self.categories
                    and updated.category
                    and updated.category not in self.categories
                ):
                    run.question = (
                        "This catalog covers " + ", ".join(self.categories) + "."
                    )
                else:
                    if self.conversational and patch.intent == "discover":
                        query = (
                            ""
                            if patch.clear_query
                            else patch.query or updated.search_query
                        )
                        if query != updated.search_query:
                            updated = updated.model_copy(
                                update={
                                    "search_query": query,
                                    "revision": max(
                                        updated.revision, state.revision + 1
                                    ),
                                }
                            )
                        run.patch = patch.model_copy(update={"query": query})
                    state = updated
                    run.state = state
                    if not state.category:
                        run.question = (
                            "Which category should I use? Choose from "
                            + ", ".join(self.categories or CATEGORIES)
                            + "."
                        )
                    else:
                        try:
                            run.refs = self._resolve_products(run, parse_text, asins)
                        except ValueError as error:
                            run.question = str(error)
                            if patch.intent in {"compare", "evidence"}:
                                pending_reference = PendingReference(
                                    intent=patch.intent,
                                    query=patch.query,
                                    priority=patch.priority,
                                    listed_ids=state.displayed_ids,
                                    snapshot=state.snapshot,
                                )
            if patch.intent != "answer":
                state = state.model_copy(
                    update={
                        "pending_question": run.question,
                        "pending_reference": pending_reference,
                    }
                )
                run.state = state
                # This commit is deliberately above execute(): timeouts cannot undo it.
                self.store.commit(
                    session_id, submission_id, state.model_dump(mode="json")
                )
                if not run.question:
                    execute(run)
                    if run.question:
                        state = state.model_copy(
                            update={"pending_question": run.question}
                        )
                        run.state = state
                        self.store.commit(
                            session_id, submission_id, state.model_dump(mode="json")
                        )
        except ProviderError as error:
            run.trace.append({"action": "model_parse", "status": error.code})
            run.note(str(error))
        except (ValueError, ValidationError):
            run.note(
                "The requested edit or model response was invalid; no unvalidated result was published."
            )
        except Exception:  # noqa: BLE001 — do not expose raw integration errors
            # The interface deliberately exposes no raw provider/adapter errors to the user.
            run.note("The application or an upstream service failed during this run.")
        run.start = started
        result = build_result(run)
        return self._publish(session_id, submission_id, run, result)

    def _publish(
        self, session_id: str, submission_id: str, run: Run, result: Result
    ) -> Result:
        displayed_ids = None
        followup = None
        if result.intent in {"discover", "recommend"} and result.outcome not in {
            "CLARIFICATION_NEEDED",
            "TEMPORARILY_UNAVAILABLE",
        }:
            if result.shown_ids() or (
                not run.agent_loop
                and result.intent == "discover"
                and run.discovery_ok
                and not run.failures
            ):
                displayed_ids = result.shown_ids()
        elif result.intent in {"compare", "evidence"} and (
            selected_ids := result.selected_ids()
        ):
            followup = ProductFollowup(
                intent=result.intent,
                ids=selected_ids,
                query=run.patch.query,
                priority=run.patch.priority,
            ).model_dump(mode="json")
        if not self.store.finish(
            session_id,
            submission_id,
            run.state.revision,
            result.model_dump(mode="json"),
            displayed_ids,
            followup=followup,
        ):
            return self._simple(
                Session.model_validate(self.store.get_state(session_id)),
                "A newer state superseded this result.",
            )
        return result
