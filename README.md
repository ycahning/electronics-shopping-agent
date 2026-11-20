# Electronics Conversational Shopping Decision Assistant

A DSC 200 final project for budget-aware shopping over a historical Electronics catalog. A local Ollama model proposes one action at a time. Python checks prices, product identity, evidence provenance, and which actions are allowed. The Streamlit interface shows the answers and the tool decisions behind them.

## Run the source checkout

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root. The lockfile pins the app environment. The checkpoint integration uses a separate Python environment.

```bash
uv sync --locked --all-extras --dev
SHOPPING_AGENT_MODE=fixture uv run --no-sync python -m streamlit run \
  shopping_agent/ui.py --server.address 127.0.0.1 --server.port 8503
```

Open [Streamlit on port 8503](http://127.0.0.1:8503). In fixture mode, try `Find appliances under $60`. This mode uses invented products and deterministic rules to demonstrate the interface, budget filtering, and turn handling. Follow the [Electronics asset setup](docs/ASSETS.md) to run the live Ollama conversation against the project catalog and, optionally, the original NCF and TIGER checkpoints.

In live mode, try:

- `Find wireless earbuds under $40.`
- `Could we change the price limitation to $30?`
- `What do reviews say about number two?`
- `Find two smart plugs under $40. Check reviews for underwater use. If the sample lacks relevant evidence, choose the cheaper one by catalog price.`

Expand **Run diagnostics** to inspect the model's actions, checked product IDs, and observed review coverage. Item numbers refer to the most recently published list in the current session.

## Architecture

1. **Streamlit and session state.** The app gives each browser session a local ID. SQLite stores accepted shopping constraints, the active numbered list, and completed turns by that ID.
2. **Model-led shopping loop.** Ollama reads the current state and tool observations, then proposes a structured action. The service validates it and either executes a tool, asks a clarifying question, or presents an answer.
3. **Electronics retrieval and personalization.** The catalog uses SQLite FTS5 and BM25. A user can explicitly ask to start from dataset history. The NCF or TIGER adapter invokes the original checkpoint in its own environment and hands back ordered parent ASINs.
4. **Verified presentation.** Python resolves item references, verifies prices and citation ownership, and preserves the scope of review claims. An empty review query, a missing price, and an unavailable source get different treatments.

Historical prices and sampled reviews provide bounded evidence. Keyword matches can include accessories and other adjacent product types. Review excerpts reflect individual experiences. A normal browser refresh starts a new chat session.

## Tests and validation

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python -m pytest -q -rs
```

GitHub Actions runs the source-only suite. Tests that require the original catalog skip when the team assets are absent. The [validation notes](docs/VALIDATION.md) describe the real-catalog, checkpoint, and Ollama results and provide commands for replaying them.

## Project credits

This course project integrates an Agent application with recommender models trained and evaluated by the project team. The external asset guide records version fingerprints and links to the original dataset publication. [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html)
