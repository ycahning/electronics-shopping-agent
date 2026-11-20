"""Run the original NCF and TIGER checkpoints in their own Python environment."""

import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import monotonic
from typing import Literal

USER_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
EXAMPLE_USERS = {
    label: user_id
    for label, variable in (
        ("Example user A", "SHOPPING_AGENT_EXAMPLE_USER_A"),
        ("Example user B", "SHOPPING_AGENT_EXAMPLE_USER_B"),
    )
    if USER_ID.fullmatch(user_id := os.getenv(variable, "").strip())
}
_ASIN = re.compile(r"[A-Z0-9]{10}")
_ASSETS = (
    "recommend.py",
    "ncf_mlp.py",
    "tiger.py",
    "data/Electronics_1M.csv",
    "data/item_catalog.csv",
    "data/semantic_ids.npy",
    "data/split.npz",
    "models/ncf_mlp_best.pt",
    "models/tiger_best.pt",
)
_WORKER = Path(__file__).resolve()
RankingSource = Literal[
    "checkpoint_inference", "version_checked_cache", "precomputed_example"
]


@dataclass(frozen=True)
class RankingResult:
    ids: tuple[str, ...]
    source: RankingSource


class RecommenderError(Exception):
    """A safe, actionable error for the shopping agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _validate_request(user_id: str, model: str, n: int) -> None:
    if not isinstance(user_id, str) or USER_ID.fullmatch(user_id) is None:
        raise RecommenderError("INVALID_REQUEST", "Enter a valid dataset user ID.")
    if model not in ("ncf", "tiger") or type(n) is not int or not 1 <= n <= 25:
        raise RecommenderError(
            "INVALID_REQUEST", "Choose NCF or TIGER and request 1 to 25 items."
        )


class LiveRecommender:
    """Run bounded local inference and cache exact requests for unchanged assets."""

    def __init__(
        self, root: str | Path, python: str, *, catalog_sha256: str | None = None
    ):
        self.root = Path(root).resolve()
        executable = shutil.which(os.path.expanduser(python)) if python else None
        if executable is None:
            raise RecommenderError(
                "RECOMMENDER_UNAVAILABLE",
                "Set SHOPPING_AGENT_RECOMMENDER_PYTHON to the recommender's Python executable.",
            )
        # Retain the environment's own executable path, including a venv symlink.
        self.python = os.path.abspath(executable)
        self._versions = self._signature()
        if catalog_sha256:
            try:
                with (self.root / "data/item_catalog.csv").open("rb") as source:
                    actual_hash = file_digest(source, "sha256").hexdigest()
            except OSError:
                raise RecommenderError(
                    "RECOMMENDER_UNAVAILABLE", "Cannot read the recommender's catalog."
                ) from None
            if actual_hash != catalog_sha256:
                raise RecommenderError(
                    "RECOMMENDER_CHANGED",
                    "The recommendation and shopping catalogs differ. Restart the app with the same catalog.",
                )
            self._ensure_unchanged()
        self._lock = Lock()
        self._cache: OrderedDict[tuple[str, str, int], tuple[str, ...]] = OrderedDict()

    def _signature(self) -> tuple[tuple[int, int, int, int], ...]:
        files = []
        for artifact in _ASSETS:
            path = self.root / artifact
            try:
                stat = path.stat()
                if not path.is_file():
                    raise FileNotFoundError
            except OSError:
                raise RecommenderError(
                    "RECOMMENDER_UNAVAILABLE",
                    f"The original recommender is missing {artifact}. Check SHOPPING_AGENT_RECOMMENDER.",
                ) from None
            files.append(
                (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            )
        return tuple(files)

    def _ensure_unchanged(self) -> None:
        if self._signature() != self._versions:
            raise RecommenderError(
                "RECOMMENDER_CHANGED",
                "The recommender files changed. Restart the app to load a consistent snapshot.",
            )

    def recommend(self, user_id: str, model: str, n: int, timeout: float) -> list[str]:
        """Return the original model's ranked parent ASINs for this request."""
        return list(self.recommend_with_provenance(user_id, model, n, timeout).ids)

    def recommend_with_provenance(
        self, user_id: str, model: str, n: int, timeout: float
    ) -> RankingResult:
        """Return ranks and the source of this exact request under the cache lock."""
        _validate_request(user_id, model, n)
        deadline = monotonic() + timeout
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or not self._lock.acquire(timeout=timeout)
        ):
            raise RecommenderError(
                "RECOMMENDER_TIMEOUT", "Recommendation inference timed out. Try again."
            )
        try:
            self._ensure_unchanged()
            key = (user_id, model, n)
            cached = self._cache.get(key)
            source = "version_checked_cache"
            if cached is None:
                source = "checkpoint_inference"
                cached = tuple(self._run(user_id, model, n, deadline - monotonic()))
                self._ensure_unchanged()
                self._cache[key] = cached
                if len(self._cache) > 64:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            return RankingResult(cached, source)
        finally:
            self._lock.release()

    def _run(self, user_id: str, model: str, n: int, timeout: float) -> list[str]:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        env.setdefault("OMP_NUM_THREADS", "4")
        try:
            result = subprocess.run(
                [self.python, "-B", str(_WORKER)],
                input=json.dumps({"user_id": user_id, "model": model, "n": n}),
                cwd=self.root,
                env=env,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired:
            raise RecommenderError(
                "RECOMMENDER_TIMEOUT", "Recommendation inference timed out. Try again."
            ) from None
        except OSError:
            raise RecommenderError(
                "RECOMMENDER_UNAVAILABLE",
                "Cannot start the recommender's Python executable.",
            ) from None
        try:
            payload = json.loads(result.stdout)
        except ValueError:
            payload = {}
        if isinstance(payload, dict) and payload.get("error") == "NO_HISTORY":
            raise RecommenderError(
                "NO_HISTORY",
                "No interaction history was found for this dataset user ID.",
            )
        if isinstance(payload, dict) and payload.get("error") == "INVALID_RANKING":
            raise RecommenderError(
                "INVALID_RANKING",
                "The checkpoint rankings could not be matched to the catalog.",
            )
        if result.returncode or not isinstance(payload, dict):
            raise RecommenderError(
                "RECOMMENDER_UNAVAILABLE",
                "The checkpoint worker could not run. Check the recommender Python dependencies and model files.",
            )
        ids = payload.get("ids")
        if (
            not isinstance(ids, list)
            or len(ids) > n
            or any(
                not isinstance(item, str) or _ASIN.fullmatch(item) is None
                for item in ids
            )
            or len(set(ids)) != len(ids)
        ):
            raise RecommenderError(
                "INVALID_RANKING", "The worker returned an invalid ranking."
            )
        return ids


def _original_inference(root: Path, user_id: str, model: str, n: int) -> list[str]:
    _validate_request(user_id, model, n)
    sys.path.insert(0, str(root))
    with TemporaryDirectory(prefix="shopping-checkpoint-") as workspace:
        # The teammate modules resolve data and models from their working directory
        # and create a logs directory at import. Keep those writes in scratch.
        for name in ("data", "models"):
            (Path(workspace) / name).symlink_to(root / name, target_is_directory=True)
        os.chdir(workspace)
        try:
            import recommend

            if recommend.catalog["parent_asin"].tolist() != recommend.item_ids.tolist():
                raise RecommenderError(
                    "INVALID_RANKING", "Catalog and item IDs differ."
                )
            try:
                _, start, end = recommend.find_user(user_id)
            except KeyError:
                raise RecommenderError(
                    "NO_HISTORY", "Dataset history is absent."
                ) from None
            if start == end:
                raise RecommenderError("NO_HISTORY", "Dataset history is absent.")
            infer = (
                recommend.recommend_ncf if model == "ncf" else recommend.recommend_tiger
            )
            return infer(user_id, n=n)["parent_asin"].tolist()
        finally:
            os.chdir(root)
            sys.path.pop(0)


def _worker_main() -> int:
    try:
        request = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            ids = _original_inference(Path.cwd(), **request)
    except RecommenderError as error:
        print(json.dumps({"error": error.code}))
        return 1
    except Exception:  # noqa: BLE001 — keep third-party errors out of user-facing output.
        return 1
    print(json.dumps({"ids": ids}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
