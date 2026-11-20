"""Verify the untouched checkpoint functions, adapter, catalog and review joins."""

import argparse
import csv
import gzip
import json
import os
import subprocess
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from shopping_agent.electronics import ElectronicsCatalog
from shopping_agent.recommendation import (
    EXAMPLE_USERS,
    LiveRecommender,
    RecommenderError,
)

from .export_electronics_rankings import (
    CATALOG_FILE,
    INPUT_FILES,
    MODEL_FILES,
    replace_json,
    sha256,
)
from .verify_electronics import require, run_case

FIXTURES = Path(__file__).resolve().parents[1] / "shopping_agent" / "fixtures"
CASES = (
    ("Example user A", "ncf", 5),
    ("Example user A", "tiger", 10),
    ("Example user B", "ncf", 10),
    ("Example user B", "tiger", 5),
)
ASSETS = (
    *INPUT_FILES,
    *MODEL_FILES.values(),
    CATALOG_FILE,
    "README.md",
    "results.json",
    "config.json",
)
ORIGINAL = """
import json
import platform
import sys
from importlib.metadata import version

sys.path.insert(0, sys.argv[1])
import recommend

assert recommend.catalog['parent_asin'].tolist() == recommend.item_ids.tolist()
assert recommend.catalog['item_idx'].tolist() == list(range(recommend.num_items))
rows = []
for user, model, n in json.load(sys.stdin):
    index, start, end = recommend.find_user(user)
    assert start < end
    infer = {'ncf': recommend.recommend_ncf, 'tiger': recommend.recommend_tiger}[model]
    table = infer(user, n=n)
    ids = table['parent_asin'].tolist()
    assert table['rank'].tolist() == list(range(1, len(ids) + 1))
    seen = set(recommend.item_ids[recommend.hist_item[start:end]])
    assert seen.isdisjoint(ids)
    rows.append({'ids': ids, 'history_interactions': int(end - start)})

tiger = recommend.tiger
print(json.dumps({
    'runtime': {
        'python': platform.python_version(), 'device': str(recommend.device),
        **{name: version(name) for name in ('numpy', 'pandas', 'torch', 'transformers')},
    },
    'data': {
        'users': len(recommend.user_ids), 'items': len(recommend.item_ids),
        'interactions': len(recommend.hist_item),
        'interaction_columns': recommend.pd.read_csv('data/Electronics_1M.csv', nrows=0).columns.tolist(),
        'catalog_columns': recommend.catalog.columns.tolist(),
        'train_candidate_items': len(recommend.train_item_pool),
        'split_fields': recommend.split.files,
        'semantic_id_shape': list(tiger.semantic_ids.shape),
        'tiger_max_history': tiger.MAX_HIST,
        'tiger_beam_size': recommend.BEAM_SIZE,
    },
    'rows': rows,
}))
"""


def hash_assets(root):
    return {asset: sha256(root / asset) for asset in ASSETS}


def original_rankings(root, python):
    requests = [(EXAMPLE_USERS[profile], model, n) for profile, model, n in CASES]
    # Load the unmodified module and invoke the original functions directly in
    # a separate process. The module writes its logs under the process CWD.
    with TemporaryDirectory(prefix="dsc200-checkpoint-oracle-") as scratch:
        for directory in ("data", "models"):
            (Path(scratch) / directory).symlink_to(
                root / directory, target_is_directory=True
            )
        process = subprocess.run(
            [str(python), "-B", "-c", ORIGINAL, str(root)],
            input=json.dumps(requests),
            cwd=scratch,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
            },
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            "Original checkpoint process failed: " + process.stderr.strip()[-500:]
        )
    return json.loads(process.stdout.splitlines()[-1])


def check_ranking(runner, catalog, reviews, profile, model, n, expected):
    started = monotonic()
    fresh = runner.recommend_with_provenance(EXAMPLE_USERS[profile], model, n, 45)
    fresh_elapsed = monotonic() - started
    started = monotonic()
    cached = runner.recommend_with_provenance(EXAMPLE_USERS[profile], model, n, 45)
    cache_elapsed = monotonic() - started
    ids = list(fresh.ids)
    require(
        ids == expected["ids"] and len(ids) == n, "Adapter changed IDs or rank order"
    )
    require(fresh.source == "checkpoint_inference", "First request did not infer")
    require(
        cached.ids == fresh.ids and cached.source == "version_checked_cache",
        "Cache differs",
    )
    products = catalog.call("get_products", {"ids": ids})
    require(
        [product.id for product in products.products] == ids,
        "Catalog join changed rank order",
    )
    return {
        "ranked_asins": ids,
        "history_interactions": expected["history_interactions"],
        "titled_catalog_matches": len(products.products),
        "review_covered_items": sum(asin in reviews for asin in ids),
        "first_source": fresh.source,
        "repeat_source": cached.source,
        "first_elapsed_s": round(fresh_elapsed, 4),
        "repeat_elapsed_s": round(cache_elapsed, 4),
    }


def check_saved_examples(catalog, exports, hashes):
    require(exports["catalog_sha256"] == hashes[CATALOG_FILE], "Catalog hash differs")
    for model, asset in MODEL_FILES.items():
        require(
            exports["model_sha256"][model] == hashes[asset], "Checkpoint hash differs"
        )
    for asset in INPUT_FILES:
        require(exports["input_sha256"][asset] == hashes[asset], "Input hash differs")
    counts = {}
    for persona, recorded in exports["rankings"].items():
        ranking = catalog.call(
            "recommend_products",
            {"persona": persona, "category": "Electronics", "limit": 25},
        )
        require(ranking.ids == recorded, "Saved rank order changed")
        require(
            ranking.ranking_source == "precomputed_example", "Missing saved provenance"
        )
        counts[persona] = len(recorded)
    return {"source": "precomputed_example", "ranks": counts}


def check_review_join(root, catalog, review_export):
    with (root / CATALOG_FILE).open(newline="", encoding="utf-8") as source:
        titles = {
            row["parent_asin"] for row in csv.DictReader(source) if row["title"].strip()
        }
    reviews = review_export["excerpts"]
    require(
        review_export["catalog_sha256"] == catalog.catalog_sha256,
        "Review catalog hash differs",
    )
    require(reviews.keys() <= titles, "Review ASIN has no titled product")
    lines = [line for entries in reviews.values() for line, _ in entries]
    require(
        len(set(lines)) == len(lines), "Review source lines have conflicting owners"
    )
    cap = review_export["complete_source_rows"]
    require(all(0 < line <= cap for line in lines), "Invalid review source line")
    return {
        "titled_catalog_items": len(titles),
        "review_covered_items": len(reviews),
        "review_excerpts": len(lines),
        "complete_source_rows": cap,
        "review_source": review_export["source"],
        "prefix_bytes": review_export["prefix_bytes"],
        "prefix_sha256": review_export["prefix_sha256"],
        "timestamp_interval_ms": review_export["timestamp_interval_ms"],
    }


def check_unknown(runner):
    try:
        runner.recommend("DSC200UNKNOWN9999", "ncf", 5, 45)
    except RecommenderError as error:
        require(error.code == "NO_HISTORY", "Unexpected unknown-user error")
        return {"error_code": error.code, "message": str(error)}
    raise AssertionError("Unknown ID received a ranking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommender", type=Path, required=True)
    parser.add_argument("--recommender-python", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if missing_profiles := {profile for profile, *_ in CASES} - EXAMPLE_USERS.keys():
        parser.error(
            f"Configure SHOPPING_AGENT_EXAMPLE_USER_A and SHOPPING_AGENT_EXAMPLE_USER_B "
            f"before testing. Missing profiles: {', '.join(sorted(missing_profiles))}."
        )
    root = args.recommender.resolve()
    report_path = args.report.resolve()
    if report_path.is_relative_to(root):
        parser.error("Write the report outside the original teammate project.")
    python = args.recommender_python.absolute()
    hashes = hash_assets(root)
    ranking_path = FIXTURES / "electronics_rankings.json"
    review_path = FIXTURES / "electronics_reviews.json.gz"
    exports = json.loads(ranking_path.read_text(encoding="utf-8"))
    with gzip.open(review_path, "rt", encoding="utf-8") as source:
        review_export = json.load(source)
    catalog = ElectronicsCatalog(root)
    cases = []
    original = {}

    def check_original():
        observed = original_rankings(root, python)
        require(len(observed["rows"]) == len(CASES), "Original output missed a request")
        original.update(observed)
        return {"requests": len(CASES), "catalog_item_index_aligned": True}

    run_case(
        cases,
        "checkpoint_integration",
        "Original checkpoint functions and inputs",
        check_original,
    )
    runner = None
    if original:
        runner = LiveRecommender(
            root, str(python), catalog_sha256=catalog.catalog_sha256
        )
    for index, (profile, model, n) in enumerate(CASES):
        check = None
        if runner:
            check = partial(
                check_ranking,
                runner,
                catalog,
                review_export["excerpts"],
                profile,
                model,
                n,
                original["rows"][index],
            )
        run_case(
            cases,
            "checkpoint_integration",
            f"{profile} · {model.upper()} · Top {n}",
            check,
            skip_reason="Original checkpoint comparison is unavailable.",
        )
    run_case(
        cases,
        "checkpoint_integration",
        "Unknown dataset user",
        partial(check_unknown, runner) if runner else None,
        skip_reason="Checkpoint runtime is unavailable.",
    )
    run_case(
        cases,
        "catalog_integration",
        "Versioned example rankings",
        lambda: check_saved_examples(catalog, exports, hashes),
    )
    run_case(
        cases,
        "catalog_integration",
        "Catalog and review parent ASIN join",
        lambda: check_review_join(root, catalog, review_export),
    )

    def unchanged():
        require(
            hash_assets(root) == hashes,
            "Original project assets changed during verification",
        )
        return {"original_files_rehashed": len(hashes)}

    run_case(cases, "checkpoint_integration", "Original artifacts unchanged", unchanged)
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "catalog_snapshot": catalog.snapshot,
        "sha256": {
            **hashes,
            **{path.name: sha256(path) for path in (ranking_path, review_path)},
        },
        "runtime": original.get("runtime"),
        "data": original.get("data"),
        "historical_training_metrics": json.loads(
            (root / "results.json").read_text(encoding="utf-8")
        )["summary"],
        "cases": cases,
    }
    replace_json(report_path, result)
    return int(any(case["status"] != "passed" for case in cases))


if __name__ == "__main__":
    raise SystemExit(main())
