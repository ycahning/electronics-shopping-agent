"""Thin Streamlit renderer. Run with python -m streamlit run shopping_agent/ui.py."""

import os
from uuid import uuid4

import streamlit as st

from shopping_agent.catalog import load_catalog
from shopping_agent.contracts import RecommendationRequest
from shopping_agent.presentation import chat_lines, lines
from shopping_agent.provider import ProviderError, configured_provider
from shopping_agent.recommendation import (
    EXAMPLE_USERS,
    USER_ID,
    LiveRecommender,
    RecommenderError,
)
from shopping_agent.service import Agent
from shopping_agent.storage import Store


@st.cache_resource(max_entries=4)
def application(
    snapshot: str,
    recommender: str,
    review_snapshot: str,
    db: str,
    implementation: tuple[int, int],
):
    catalog = load_catalog(
        snapshot=snapshot,
        recommender=recommender,
        review_snapshot=review_snapshot or None,
    )
    return Store(db), catalog


@st.cache_resource
def model_recommender(root: str, python: str, catalog_sha256: str):
    return LiveRecommender(root, python, catalog_sha256=catalog_sha256)


st.set_page_config(page_title="Shopping decision assistant")
st.title("Electronics Shopping Decision Assistant")
mode = os.getenv("SHOPPING_AGENT_MODE", "final").strip().lower()
model = os.getenv("SHOPPING_AGENT_LIVE_MODEL", "")
ollama_model = os.getenv("SHOPPING_AGENT_OLLAMA_MODEL", "")
recommender_root = os.getenv("SHOPPING_AGENT_RECOMMENDER", "")
if mode not in {"final", "offline", "fixture"}:
    st.error("Set SHOPPING_AGENT_MODE to final, offline, or fixture.")
    st.stop()
if mode in {"final", "offline"} and not recommender_root:
    st.error(
        "Set SHOPPING_AGENT_RECOMMENDER to the Electronics recommender directory. See the README for startup commands."
    )
    st.stop()
if mode == "final" and not ollama_model:
    st.error(
        "Set SHOPPING_AGENT_OLLAMA_MODEL to an installed Ollama model tag. See the README for startup commands."
    )
    st.stop()
if mode == "fixture":
    st.warning("Developer fixture mode with synthetic catalog data.")
    model = ollama_model = recommender_root = ""

try:
    store, catalog = application(
        "" if mode == "fixture" else os.getenv("SHOPPING_AGENT_SNAPSHOT", ""),
        recommender_root,
        "" if mode == "fixture" else os.getenv("SHOPPING_AGENT_REVIEW_SNAPSHOT", ""),
        os.getenv("SHOPPING_AGENT_DB", ".shopping-agent.sqlite3"),
        # An in-browser rerun can reload these modules while Streamlit keeps resources.
        (id(Store), id(load_catalog)),
    )
except (OSError, ValueError) as error:
    st.error(f"Cannot open the shopping catalog. {error}")
    st.stop()
inference = None
if mode == "final":
    try:
        inference = model_recommender(
            recommender_root,
            os.getenv("SHOPPING_AGENT_RECOMMENDER_PYTHON", ""),
            catalog.catalog_sha256,
        )
    except RecommenderError as error:
        st.sidebar.warning(str(error))
agent = st.session_state.get("shop_agent")
if (
    type(agent) is not Agent
    or agent.store is not store
    or agent.catalog is not catalog
    or agent.recommender is not inference
):
    try:
        provider = configured_provider(ollama_model=ollama_model, live_model=model)
    except ProviderError as error:
        st.error(str(error))
        if ollama_model:
            st.markdown(
                "Start Ollama in another terminal and check the model tag with `ollama list`."
            )
            st.code(
                "ollama serve\n\n# In another terminal\nollama list", language="bash"
            )
            st.button("Retry connection")
        st.stop()
    st.session_state.shop_agent = Agent(
        store, catalog, provider=provider, recommender=inference
    )
agent = st.session_state.shop_agent
backend = f"Ollama · {ollama_model}" if ollama_model else model or "Offline rules"
st.caption(f"Data: {agent.catalog.mode} / {agent.catalog.snapshot} · Model: {backend}")
if catalog.mode == "team-electronics":
    st.caption(f"Catalog prices are historical USD values. {catalog.review_sample}")

persona = None
recommendation = None
next_items = False
if mode == "final":
    st.sidebar.caption(
        "The selected profile is available for requests that explicitly use dataset history. Ordinary shopping searches use the catalog."
    )
    profile = st.sidebar.selectbox(
        "History profile",
        ["No history", *EXAMPLE_USERS, "Dataset user ID"],
        key="history_profile",
        disabled=inference is None,
    )
    model_choice = st.sidebar.selectbox(
        "Recommendation model",
        ["ncf", "tiger"],
        format_func=str.upper,
        key="ranking_model",
        disabled=profile == "No history" or inference is None,
    )
    n = st.sidebar.number_input(
        "Number of recommendations",
        min_value=1,
        max_value=25,
        value=5,
        key="ranking_n",
        disabled=profile == "No history" or inference is None,
    )
    user_id = EXAMPLE_USERS.get(profile, "")
    if profile == "Dataset user ID":
        user_id = st.sidebar.text_input(
            "Dataset user ID", max_chars=128, key="history_user_id", type="password"
        ).strip()
        if user_id and USER_ID.fullmatch(user_id) is None:
            st.sidebar.warning(
                "Use a dataset ID containing only letters, digits, underscores or hyphens."
            )
    if inference and USER_ID.fullmatch(user_id):
        recommendation = RecommendationRequest(user_id=user_id, model=model_choice, n=n)
    next_items = st.sidebar.button(
        "Recommend next items",
        key="recommend_next",
        disabled=recommendation is None,
    )
else:
    st.sidebar.caption(
        "Example rankings in this mode come from saved data. Use final mode for checkpoint inference."
    )
    history_labels = {
        None: "No history",
        "accessories-ncf": "NCF · saved example history",
        "accessories-tiger": "TIGER · saved example history",
    }
    persona = st.sidebar.selectbox(
        "History context",
        [None, *getattr(agent.catalog, "personas", ())],
        format_func=lambda v: history_labels.get(v, f"Example: {v}"),
    )
    profile = history_labels.get(persona, "No history")
if "shop_session" not in st.session_state:
    st.session_state.shop_session = uuid4().hex
    st.session_state.shop_history = []
example = "electronics" if agent.catalog.mode == "team-electronics" else "appliances"
placeholder = (
    "Try: I need wireless headphones for under $40."
    if agent.conversational and example == "electronics"
    else f"Try: find {example} under $40"
)
text = st.chat_input(placeholder, max_chars=1200)
if next_items:
    text = f"Show {n} next-item recommendations from my history."
if text:
    # Only a widget submission creates an ID. Read the new revision before rendering history.
    with st.spinner("Working on your request"):
        result = agent.submit(
            st.session_state.shop_session,
            uuid4().hex,
            text,
            persona=persona,
            recommendation=recommendation,
            next_items=next_items,
        )
    accepted = store.get_state(st.session_state.shop_session)
    reply = chat_lines(result, accepted) if agent.conversational else lines(result)
    for role, content in (("user", text), ("assistant", "\n".join(reply))):
        shown = (
            result.shown_ids()
            if role == "assistant" and result.verified + result.pending
            else []
        )
        st.session_state.shop_history.append((role, content, result.revision, shown))
    st.session_state.shop_diagnostics = {
        "provider": result.provider,
        "snapshot": result.snapshot,
        "data_mode": result.data_mode,
        "outcome": result.outcome,
        "intent": result.intent,
        "personalization": result.personalization,
        "history_mode": result.history_mode,
        "decision_ids": result.decision_ids,
        "unchecked_candidate_ids": result.unchecked_candidate_ids,
        "history_profile": profile,
        "requested_model": result.recommendation_model,
        "requested_n": result.requested_n,
        "ranking_source": result.ranking_source,
        "pending_goal": accepted.get("pending_goal"),
        "search_query": accepted.get("search_query"),
        "budget_usd": accepted.get("max_price_usd"),
        "accepted_conditions": accepted.get("requirements", []),
        "unsupported_conditions": accepted.get("unsupported", []),
        "numbered_product_ids": accepted.get("displayed_ids", []),
        "selected_product_ids": result.selected_ids(),
        "verified_prices_usd": {
            card.id: str(card.price_usd) for card in result.verified
        },
        "pending_product_ids": [card.id for card in result.pending],
        "above_budget_ids": [card.id for card in result.above_budget],
        "selected_price_status": {
            card.id: card.price_status for card in result.comparison
        },
        "selected_ratings": {
            card.id: {
                "average": str(card.average_rating)
                if card.average_rating is not None
                else None,
                "count": card.rating_count,
            }
            for card in result.comparison
        },
        "discovery_cards": {
            card.id: {
                "origin": "checkpoint"
                if card.model_rank is not None
                else ("saved ranking" if card.example_ranking else "catalog"),
                "model_rank": card.model_rank,
                "conditions": [check.model_dump() for check in card.conditions],
            }
            for card in [*result.verified, *result.pending]
        },
        "price_difference_usd": (
            str(result.price_difference_usd)
            if result.price_difference_usd is not None
            else None
        ),
        "missing_product_ids": result.missing_product_ids,
        "review_gaps": result.review_gaps,
        "review_aspect": result.review_aspect,
        "review_sources": [
            {"id": quote.id, "product_id": quote.product_id, "source": quote.source}
            for quote in result.evidence
            if quote.kind == "review_observation"
        ],
        "decision": result.decision.model_dump() if result.decision else None,
        "trace": result.trace,
        "latency_ms": result.latency_ms,
        "model_attempts": result.model_attempts,
        "tool_attempts": result.tool_attempts,
        "usage": result.usage or None,
    }

current = store.get_state(st.session_state.shop_session)
for role, content, revision, shown in st.session_state.shop_history:
    with st.chat_message(role):
        if shown and shown != current.get("displayed_ids", []):
            st.caption(
                "Earlier search — historical result. Item numbers refer to the latest search list."
            )
        elif shown and revision != current.get("revision", 0):
            st.caption(
                "This list used earlier requirements. Its item numbers are still available."
            )
        st.text(content)
if "shop_diagnostics" in st.session_state:
    with st.expander("Run diagnostics"):
        st.json(st.session_state.shop_diagnostics)
