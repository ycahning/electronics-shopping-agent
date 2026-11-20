"""Export a versioned ranking from the teammate's trained models.

Run this script with the recommender's Python interpreter. The shopping agent
only reads the resulting JSON and does not import the model stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "shopping_agent"
    / "fixtures"
    / "electronics_rankings.json"
)
MODEL_FILES = {
    "tiger": "models/tiger_best.pt",
    "ncf": "models/ncf_mlp_best.pt",
}
CATALOG_FILE = "data/item_catalog.csv"
INPUT_FILES = (
    "data/Electronics_1M.csv",
    "data/semantic_ids.npy",
    "data/split.npz",
    "ncf_mlp.py",
    "recommend.py",
    "tiger.py",
)
RANKING_LIMIT = 25
PRODUCT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_ranking(table, *, product_ids, seen_ids, model: str) -> list[str]:
    if len(table) > RANKING_LIMIT:
        raise ValueError(f"{model} returned more than {RANKING_LIMIT} rows.")
    ranked_ids = []
    accepted = set()
    for row in table.itertuples(index=False):
        item_id = row.parent_asin
        if (
            not isinstance(item_id, str)
            or PRODUCT_ID.fullmatch(item_id) is None
            or item_id not in product_ids
            or item_id in seen_ids
        ):
            raise ValueError(f"{model} returned an invalid or previously seen item.")
        if item_id not in accepted and row.title.strip():
            ranked_ids.append(item_id)
            accepted.add(item_id)
    if not ranked_ids:
        raise ValueError(f"{model} returned no listable recommendations.")
    return ranked_ids


def export_payload(recommender_dir: Path, user_id: str) -> dict:
    input_files = (CATALOG_FILE, *MODEL_FILES.values(), *INPUT_FILES)
    hashes = {name: sha256(recommender_dir / name) for name in input_files}
    # The original demo resolves its imports, data, and checkpoints from its CWD.
    # Keep that working-directory requirement inside this one-shot process.
    workdir = Path.cwd()
    original_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(recommender_dir))
    try:
        os.chdir(recommender_dir)
        import recommend

        if (
            recommend.catalog["item_idx"].tolist() != list(range(recommend.num_items))
            or recommend.catalog["parent_asin"].tolist() != recommend.item_ids.tolist()
        ):
            raise ValueError("Catalog rows do not match the recommender's item index.")
        try:
            _, start, end = recommend.find_user(user_id)
        except KeyError:
            raise ValueError(
                "The supplied user ID is absent from the dataset."
            ) from None
        if start == end:
            raise ValueError("The supplied user has no dataset history.")
        product_ids = frozenset(recommend.item_ids)
        seen_ids = frozenset(recommend.item_ids[recommend.hist_item[start:end]])
        rankings = {}
        for model, infer in (
            ("tiger", recommend.recommend_tiger),
            ("ncf", recommend.recommend_ncf),
        ):
            rankings[f"accessories-{model}"] = checked_ranking(
                infer(user_id, n=RANKING_LIMIT),
                product_ids=product_ids,
                seen_ids=seen_ids,
                model=model,
            )
        runtime = {
            "python": platform.python_version(),
            "device": str(recommend.device),
            **{
                name: version(name)
                for name in ("numpy", "pandas", "torch", "transformers")
            },
        }
    finally:
        os.chdir(workdir)
        sys.path.pop(0)
        sys.dont_write_bytecode = original_bytecode_setting
    if any(sha256(recommender_dir / name) != hashes[name] for name in input_files):
        raise ValueError(
            "A source artifact changed during inference. Rerun the export."
        )
    return {
        "schema_version": 1,
        "catalog_sha256": hashes[CATALOG_FILE],
        "model_sha256": {model: hashes[name] for model, name in MODEL_FILES.items()},
        "input_sha256": {name: hashes[name] for name in INPUT_FILES},
        "runtime": runtime,
        "rankings": rankings,
    }


def replace_json(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        tmp_path.replace(output)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommender-dir", type=Path, required=True)
    parser.add_argument(
        "--user-id",
        required=True,
        help="Dataset user ID from the teammate README Top-N demo.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    recommender_dir = args.recommender_dir.resolve()
    output = args.output.resolve()
    source_files = (CATALOG_FILE, *MODEL_FILES.values(), *INPUT_FILES)
    if output in {(recommender_dir / path).resolve() for path in source_files}:
        parser.error("Output must be distinct from the recommender's source files.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        payload = export_payload(recommender_dir, args.user_id)
        replace_json(output, payload)
    except (OSError, ImportError, ValueError) as error:
        parser.exit(1, f"Ranking export failed: {error}\n")
    print(f"Saved {output}")
    for profile, ids in payload["rankings"].items():
        print(f"  {profile}  {len(ids)} products")


if __name__ == "__main__":
    main()
