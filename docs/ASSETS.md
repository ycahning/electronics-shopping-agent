# Local Electronics assets

The source checkout includes a synthetic test fixture. For the live Electronics app, use a team-provided bundle you are authorized to use and keep it outside this repository. The bundle contains the original recommender source and checkpoints, the serving catalog, and a catalog-bound ranking export and review sample. Dataset maintainers provide data for research and explain their licensing position in the [Amazon Reviews 2023 dataset discussion](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1).

## Prepare the app

Install [Ollama](https://ollama.com/download), run `ollama serve` if the service is down, and pull `qwen3:4b-q4_K_M`. From the repository root, set the two paths below to the corresponding directories in the extracted team bundle.

```bash
export SHOPPING_AGENT_RECOMMENDER="/path/to/team-bundle/teammate_code/Recommender_Project-main"
bundle_fixtures="/path/to/team-bundle/shopping-agent/shopping_agent/fixtures"

cp "$bundle_fixtures/electronics_rankings.json" shopping_agent/fixtures/
export SHOPPING_AGENT_REVIEW_SNAPSHOT="$bundle_fixtures/electronics_reviews.json.gz"

export SHOPPING_AGENT_MODE=final
export SHOPPING_AGENT_OLLAMA_MODEL=qwen3:4b-q4_K_M
uv run --no-sync python -m streamlit run shopping_agent/ui.py \
  --server.address 127.0.0.1 --server.port 8503
```

The app validates both exports against the SHA-256 of `data/item_catalog.csv`. To run the tests that load the fixtures via their default paths, copy the authorized `electronics_reviews.json.gz` into `shopping_agent/fixtures/` and put the original recommender under the sibling path `../teammate_code/Recommender_Project-main`. Git ignores both fixture exports.

The browser can search and compare catalog items while the checkpoint environment is unavailable. For next-item inference, use the separate CPU environment tested on macOS Apple Silicon.

```bash
uv venv --python 3.14.6 .venv-recommender
uv pip install --python .venv-recommender/bin/python \
  -r integration/recommender-requirements.txt
export SHOPPING_AGENT_RECOMMENDER_PYTHON="$PWD/.venv-recommender/bin/python"
```

Restart Streamlit after configuring the environment. Supply an authorized dataset identifier in the sidebar to request history-based ranks. To enable the two named example profiles, set `SHOPPING_AGENT_EXAMPLE_USER_A` and `SHOPPING_AGENT_EXAMPLE_USER_B` in your shell before starting the app or the integration checks. Use the corresponding identifiers from the authorized team materials when reproducing the historical example runs. [Environment variable examples](../.env.example)

## Asset fingerprints from the team run

| File | SHA-256 |
| --- | --- |
| `data/item_catalog.csv` | `38f306adc03238ffbfe0347724a19f76ab3e95d7951c4cf0238fde7dad1751c6` |
| `models/ncf_mlp_best.pt` | `df57143bf4dab4f1e29469a6ff99c33b821348e87c995b2dd4499970d5f69d7c` |
| `models/tiger_best.pt` | `816ddbba32e65453b059d83495c58bae68d08f5778a16ce56bfcc3c8798ac9fc` |
| `electronics_rankings.json` | `5a1fcbe4b102d07f1602f38bc8a312fe54355d3b6452333520f023db1cde9c5d` |
| `electronics_reviews.json.gz` | `5d0c317f0f9ffb9257f00dc4afbbafdc316d15da7fc10e3943074acfd4b46a44` |

The [ranking exporter](../integration/export_electronics_rankings.py) produces a catalog-bound ranking JSON from an authorized dataset ID and original checkpoints. The [review exporter](../integration/export_electronics_reviews.py) accepts a local compressed archive or prefix and writes a catalog-bound review snapshot. Consult the [original dataset page](https://amazon-reviews-2023.github.io/main.html#grouped-by-category) for the underlying Electronics research data.
