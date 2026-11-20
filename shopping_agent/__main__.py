"""Local one-shot and multi-turn entry point."""

import argparse
import json
import os
from uuid import uuid4

from .catalog import load_catalog
from .contracts import RecommendationRequest
from .presentation import chat_lines, lines
from .provider import ProviderError, configured_provider
from .recommendation import EXAMPLE_USERS, LiveRecommender, RecommenderError
from .service import Agent
from .storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded shopping decision assistant.")
    parser.add_argument("request", nargs="*")
    parser.add_argument("--db", default=".shopping-agent.sqlite3")
    parser.add_argument("--session", default=None)
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--snapshot", help="Team-provided JSON export.")
    data_source.add_argument(
        "--recommender", help="Path to the team's Electronics recommender directory."
    )
    data_source.add_argument(
        "--fixture", action="store_true", help="Run the developer's synthetic catalog."
    )
    parser.add_argument(
        "--review-snapshot",
        help="Optional Electronics review excerpt export (.json.gz).",
    )
    parser.add_argument("--policy", choices=("adaptive", "fixed"), default="adaptive")
    history = parser.add_mutually_exclusive_group()
    history.add_argument(
        "--persona", help="Read a saved example ranking. Omit for no-history mode."
    )
    history.add_argument("--dataset-user-id", help="Use an ID with dataset history.")
    history.add_argument(
        "--history-profile",
        choices=tuple(EXAMPLE_USERS),
        help="Use an example dataset user.",
    )
    parser.add_argument(
        "--recommender-python",
        default=os.getenv("SHOPPING_AGENT_RECOMMENDER_PYTHON", ""),
        help="Python with the original recommender's dependencies.",
    )
    parser.add_argument("--ranker", choices=("ncf", "tiger"), default="ncf")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--next-items",
        action="store_true",
        help="Return one model-only next-item ranking.",
    )
    provider_options = parser.add_mutually_exclusive_group()
    provider_options.add_argument(
        "--live-model", help="Opt in to provider calls, using OPENAI_API_KEY."
    )
    provider_options.add_argument(
        "--ollama-model",
        help="Use the installed model tag from the local Ollama service.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit structured result and traces."
    )
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help="Mark stale runs at exclusive single-instance startup; never replay.",
    )
    args = parser.parse_args()
    user_id = (
        args.dataset_user_id
        if args.dataset_user_id is not None
        else (EXAMPLE_USERS.get(args.history_profile))
    )
    if args.next_items and user_id is None:
        parser.error("--next-items requires --history-profile or --dataset-user-id.")
    if user_id is not None and not args.recommender:
        parser.error("Dataset history requires --recommender.")
    inference = recommendation = None
    try:
        provider = configured_provider(
            ollama_model=args.ollama_model or "", live_model=args.live_model or ""
        )
        catalog = load_catalog(
            snapshot=args.snapshot,
            recommender=args.recommender,
            review_snapshot=args.review_snapshot,
        )
        if user_id is not None:
            recommendation = RecommendationRequest(
                user_id=user_id.strip(), model=args.ranker, n=args.top_n
            )
            inference = LiveRecommender(
                args.recommender,
                args.recommender_python,
                catalog_sha256=catalog.catalog_sha256,
            )
    except (OSError, ValueError, ProviderError, RecommenderError) as error:
        parser.error(str(error))
    if args.persona and args.persona not in getattr(catalog, "personas", ()):
        parser.error("--persona must be one of the configured catalog's examples.")
    store = Store(args.db)
    try:
        if args.recover_interrupted:
            store.recover_interrupted()
        agent = Agent(
            store, catalog, provider=provider, policy=args.policy, recommender=inference
        )
        session = args.session or uuid4().hex

        def run(text: str, *, next_items: bool = False) -> None:
            result = agent.submit(
                session,
                uuid4().hex,
                text,
                persona=args.persona,
                recommendation=recommendation,
                next_items=next_items,
            )
            if args.json:
                print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False))
            elif agent.conversational:
                print("\n".join(chat_lines(result, store.get_state(session))))
            else:
                print("\n".join(lines(result)))

        if args.request or args.next_items:
            text = (
                " ".join(args.request)
                or f"Show {args.top_n} next-item recommendations from my history."
            )
            run(text, next_items=args.next_items)
            return
        print(
            f"Session {session} | data={catalog.mode} | snapshot={catalog.snapshot} "
            f"| model={getattr(provider, 'name', 'offline rules')}"
        )
        while True:
            try:
                text = input("\nshop> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text in {"/quit", "/exit"}:
                break
            if text:
                run(text)
    finally:
        store.close()


if __name__ == "__main__":
    main()
