"""SQLite authority for accepted requirements and one run per genuine submission."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Literal

BeginStatus = Literal["NEW", "DONE", "BUSY"]


class StaleRunError(RuntimeError):
    """An inactive run attempted to commit state, or its revision went backward."""


def _encode(value: dict) -> str:
    # Persist JSON-ready domain data, never a Python pickle or implicit str(value).
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _revision(state: dict) -> int:
    value = state.get("revision", 0)
    if type(value) is not int or value < 0:
        raise ValueError("State revision must be a non-negative integer")
    return value


def _diagnostic(run_state: str, message: str) -> dict:
    return {
        "outcome": "TEMPORARILY_UNAVAILABLE",
        "limitations": [message],
        "run_state": run_state,
    }


class Store:
    """Storage only: orchestration must make external calls outside transactions.

    The application supplies session IDs and unique submission IDs. The same
    submission ID never starts a second run; later deliberate submissions get
    new IDs even if their text is identical. This is not external-call replay.
    """

    def __init__(self, path: str | Path):
        path_string = str(path)
        if path_string != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        # A single locked connection also supports :memory: for isolated tests.
        self._db = sqlite3.connect(
            path_string, isolation_level=None, check_same_thread=False, timeout=5
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS runs (
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                submission_id TEXT NOT NULL,
                user_text TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('ACTIVE','DONE','INTERRUPTED','STALE')),
                result_json TEXT,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TEXT,
                PRIMARY KEY (session_id, submission_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_run_per_session
                ON runs(session_id) WHERE status = 'ACTIVE';
            """
        )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        # BEGIN IMMEDIATE serializes claim/commit/publication across connections.
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    @staticmethod
    def _state(db: sqlite3.Connection, session_id: str) -> dict:
        row = db.execute(
            "SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return json.loads(row["state_json"]) if row is not None else {}

    @staticmethod
    def _run(db: sqlite3.Connection, session_id: str, submission_id: str):
        return db.execute(
            """SELECT user_text, status, result_json FROM runs
               WHERE session_id = ? AND submission_id = ?""",
            (session_id, submission_id),
        ).fetchone()

    @staticmethod
    def _retire(
        db: sqlite3.Connection,
        session_id: str,
        submission_id: str,
        status: str,
        result: dict,
    ) -> None:
        db.execute(
            """UPDATE runs SET status = ?, result_json = ?, finished_at = CURRENT_TIMESTAMP
               WHERE session_id = ? AND submission_id = ? AND status = 'ACTIVE'""",
            (status, _encode(result), session_id, submission_id),
        )

    def get_state(self, session_id: str) -> dict:
        with self._lock:
            return self._state(self._db, session_id)

    def recent_turns(self, session_id: str) -> list[dict[str, str]]:
        """Give the local model a small amount of context from this session."""
        with self._lock:
            rows = self._db.execute(
                """SELECT user_text, result_json FROM runs
                   WHERE session_id = ? AND status = 'DONE'
                     AND json_extract(result_json, '$.outcome') IN
                         ('COMPLETE', 'PARTIAL', 'CLARIFICATION_NEEDED', 'NO_VERIFIED_MATCH_IN_SCOPE')
                   ORDER BY rowid DESC LIMIT 3""",
                (session_id,),
            ).fetchall()
        turns = []
        for row in reversed(rows):
            result = json.loads(row["result_json"])
            assistant = (
                result.get("answer")
                or result.get("question")
                or " ".join(result.get("model_reply", []))
                or f"Handled a {result.get('intent', 'shopping')} request."
            )
            turns.append({"user": row["user_text"][:320], "assistant": assistant[:460]})
        return turns

    def begin(
        self, session_id: str, submission_id: str, user_text: str
    ) -> tuple[BeginStatus, dict | None]:
        """Claim one active run and return state, a prior outcome, or a busy signal."""
        with self._write() as db:
            prior = self._run(db, session_id, submission_id)
            if prior is not None:
                if prior["user_text"] != user_text:
                    raise ValueError(
                        "Submission ID was already used with different text"
                    )
                if prior["status"] == "ACTIVE":
                    return "BUSY", None
                return "DONE", json.loads(prior["result_json"])

            active = db.execute(
                "SELECT 1 FROM runs WHERE session_id = ? AND status = 'ACTIVE'",
                (session_id,),
            ).fetchone()
            if active is not None:
                return "BUSY", None

            db.execute(
                "INSERT OR IGNORE INTO sessions(session_id) VALUES (?)", (session_id,)
            )
            db.execute(
                """INSERT INTO runs(session_id, submission_id, user_text, status)
                   VALUES (?, ?, ?, 'ACTIVE')""",
                (session_id, submission_id, user_text),
            )
            return "NEW", self._state(db, session_id)

    def commit(self, session_id: str, submission_id: str, state: dict) -> None:
        """Persist accepted state before calling downstream tools."""
        revision = _revision(state)
        encoded = _encode(state)
        with self._write() as db:
            run = self._run(db, session_id, submission_id)
            if run is None or run["status"] != "ACTIVE":
                raise StaleRunError("Only the active run can commit state")
            if revision < _revision(self._state(db, session_id)):
                raise StaleRunError("Cannot overwrite a newer requirements revision")
            db.execute(
                "UPDATE sessions SET state_json = ? WHERE session_id = ?",
                (encoded, session_id),
            )

    def finish(
        self,
        session_id: str,
        submission_id: str,
        revision: int,
        result: dict,
        displayed_ids: list[str] | None = None,
        *,
        followup: dict | None = None,
    ) -> bool:
        """Atomically publish only the active run's result for the current revision."""
        if type(revision) is not int or revision < 0:
            raise ValueError("Result revision must be a non-negative integer")
        with self._write() as db:
            run = self._run(db, session_id, submission_id)
            if run is None or run["status"] != "ACTIVE":
                return False
            state = self._state(db, session_id)
            if _revision(state) != revision:
                self._retire(
                    db,
                    session_id,
                    submission_id,
                    "STALE",
                    _diagnostic(
                        "STALE", "Result discarded because the requirements changed."
                    ),
                )
                return False
            if displayed_ids is not None:
                state["displayed_ids"] = displayed_ids
                if "followup" in state:
                    state["followup"] = None
            elif followup is not None:
                state["followup"] = followup
            if displayed_ids is not None or followup is not None:
                db.execute(
                    "UPDATE sessions SET state_json = ? WHERE session_id = ?",
                    (_encode(state), session_id),
                )
            self._retire(db, session_id, submission_id, "DONE", result)
            return True

    def last_discovery(
        self, session_id: str, revision: int, snapshot: str
    ) -> dict | None:
        """Find the published list for a session saved before follow-ups had separate state."""
        with self._lock:
            row = self._db.execute(
                """SELECT result_json FROM runs
                   WHERE session_id = ? AND status = 'DONE'
                     AND json_extract(result_json, '$.revision') = ?
                     AND json_extract(result_json, '$.snapshot') = ?
                     AND json_extract(result_json, '$.intent') = 'discover'
                     AND json_extract(result_json, '$.outcome') IN
                         ('COMPLETE', 'PARTIAL', 'NO_VERIFIED_MATCH_IN_SCOPE')
                   ORDER BY rowid DESC LIMIT 1""",
                (session_id, revision, snapshot),
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def recover_interrupted(self) -> int:
        """Call explicitly at single-instance startup, when no worker can be active.

        A previously accepted state remains saved. This never repeats a tool call.
        """
        diagnostic = _diagnostic(
            "INTERRUPTED", "Previous run was interrupted and was not replayed."
        )
        with self._write() as db:
            return db.execute(
                """UPDATE runs SET status = 'INTERRUPTED', result_json = ?,
                       finished_at = CURRENT_TIMESTAMP WHERE status = 'ACTIVE'""",
                (_encode(diagnostic),),
            ).rowcount
