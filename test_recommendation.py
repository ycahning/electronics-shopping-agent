"""Subprocess contract tests for the checkpoint inference adapter."""

import hashlib
import sys

import pytest

from shopping_agent import recommendation


@pytest.fixture
def original_files(tmp_path):
    for asset in recommendation._ASSETS:
        file = tmp_path / asset
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    (tmp_path / "recommend.py").write_text(
        """
from pathlib import Path

class Column(list):
    def tolist(self):
        return list(self)

item_ids = Column(["B000000001", "B000000002", "B000000003"])
catalog = {"parent_asin": item_ids}
Path("logs").mkdir()

def find_user(user_id):
    if user_id not in ("USER_A", "USER_B"):
        raise KeyError(user_id)
    return (0, 0, 1)

def recommend_ncf(user_id, n):
    ids = list(item_ids)
    if user_id == "USER_B":
        ids.reverse()
    return {"parent_asin": Column(ids[:n])}

def recommend_tiger(user_id, n):
    ids = [item_ids[1], item_ids[0], item_ids[2]]
    return {"parent_asin": Column(ids[:n])}
""",
        encoding="utf-8",
    )
    return tmp_path


def test_checkpoint_worker_passes_user_model_and_n_and_isolates_logs(original_files):
    runner = recommendation.LiveRecommender(
        original_files,
        sys.executable,
        catalog_sha256=hashlib.sha256(b"").hexdigest(),
    )
    assert runner.recommend("USER_A", "ncf", 1, 10) == ["B000000001"]
    assert runner.recommend("USER_B", "ncf", 2, 10) == ["B000000003", "B000000002"]
    ranked = runner.recommend_with_provenance("USER_A", "tiger", 2, 10)
    assert ranked.ids == ("B000000002", "B000000001")
    assert ranked.source == "checkpoint_inference"
    cached = runner.recommend_with_provenance("USER_A", "tiger", 2, 10)
    assert cached.ids == ranked.ids
    assert cached.source == "version_checked_cache"
    ids = runner.recommend("USER_A", "tiger", 2, 10)
    ids.clear()
    assert runner.recommend("USER_A", "tiger", 2, 10) == ["B000000002", "B000000001"]
    assert not (original_files / "logs").exists()
    assert not (original_files / "__pycache__").exists()


def test_checkpoint_worker_reports_unknown_user_and_source_changes(original_files):
    runner = recommendation.LiveRecommender(original_files, sys.executable)
    with pytest.raises(recommendation.RecommenderError) as error:
        runner.recommend("MISSING_USER", "tiger", 5, 10)
    assert error.value.code == "NO_HISTORY"
    assert "MISSING_USER" not in str(error.value)
    runner.recommend("USER_A", "ncf", 1, 10)
    (original_files / "models/ncf_mlp_best.pt").write_bytes(b"new checkpoint")
    with pytest.raises(recommendation.RecommenderError) as error:
        runner.recommend("USER_A", "ncf", 1, 10)
    assert error.value.code == "RECOMMENDER_CHANGED"
