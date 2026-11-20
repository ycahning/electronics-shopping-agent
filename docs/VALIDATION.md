# Validation

## Offline checks

The source checkout exercises catalog contracts, budget validation, session storage, widget behavior, and agent decisions using scripted providers and synthetic fixtures. On a clean local source checkout, Ruff passed and pytest reported 128 passed, 10 skipped, and 15 subtests passed. All skips require the original Electronics catalog. GitHub Actions runs the same source-only checks on Ubuntu.

## Recorded local results from the team version

| Layer | Recorded result |
| --- | --- |
| Application tests | 144 tests and 15 subtests passed in the working tree. A clean staged course distribution passed 138 tests and 15 subtests. |
| Checkpoints | Nine integration checks passed. Four user, model, and Top-N combinations matched the original NCF and TIGER functions in ASIN order. |
| Live Ollama | With `qwen3:4b-q4_K_M`, the full initial pass completed 10 of 13 scenarios. The evidence and ordinal retest passed 7 of 8, and the final targeted policy retest passed 3 of 3. The runs cover different revisions and scopes. |
| Actual browser | A local user turn completed a discovery, review check, and price fallback in the Streamlit interface. The app reported the review sample's gap and used the verified historical prices. |

The controlled observation test supplied two different first-review responses to the same model and task. When the first response held relevant excerpts, the model completed its answer after a single review call and presented the contrasting evidence. When the first result held only packaging comments, the model made a second lookup and used a relevant excerpt from the real source. A separate real-review run made a specific shortlist recommendation.

For the GitHub source release, a separate extraction of the committed source was tested with the authorized assets supplied locally. The compound discovery and price fallback passed again with the installed Ollama model, and all nine checkpoint and catalog integration checks passed.

## Replay with authorized assets

Follow the [local asset setup](ASSETS.md). For a first check, run this targeted real-catalog scenario with the same Ollama tag.

```bash
mkdir -p artifacts/local
uv run --no-sync python -m integration.verify_goal_loop \
  --recommender "$SHOPPING_AGENT_RECOMMENDER" \
  --recommender-python "$SHOPPING_AGENT_RECOMMENDER_PYTHON" \
  --review-snapshot "$SHOPPING_AGENT_REVIEW_SNAPSHOT" \
  --only compound-discovery-evidence-choice \
  --output artifacts/local/goal-loop.json
```

Remove `--only` to run the larger suite. Configure both example user IDs first.

The checkpoint check expects both catalog-bound exports at the default fixture paths and both named example users in your shell.

```bash
uv run --no-sync python -m integration.verify_checkpoints \
  --recommender "$SHOPPING_AGENT_RECOMMENDER" \
  --recommender-python "$SHOPPING_AGENT_RECOMMENDER_PYTHON" \
  --report artifacts/local/checkpoints.json
```

The project evaluates original checkpoint parity and end-to-end assistant behavior separately. Reproducing the original model training requires the team's training environment and data.

The historical comparison flags `--before-repair` and `--compare-fixed` in `integration.verify_agent_loop` use an external baseline. Place the authorized baseline ZIP at the ignored local path `reference/pre-loop-shopping-agent.zip` before using those flags.
