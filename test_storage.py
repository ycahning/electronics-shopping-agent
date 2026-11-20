"""Storage regression tests exercise state, publication and restart boundaries."""

import tempfile
import unittest
from pathlib import Path

from shopping_agent.storage import StaleRunError, Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "state.sqlite3"
        self.store = Store(self.path)
        self.addCleanup(self.store.close)

    def test_recent_turns_use_only_this_sessions_latest_completed_work(self):
        for i in range(4):
            self.store.begin("s", str(i), f"Question {i}")
            self.store.finish("s", str(i), 0, {"outcome": "COMPLETE", "answer": str(i)})
        self.store.begin("s", "current", "Still running")
        self.store.begin("other", "one", "A different session")
        self.store.finish("other", "one", 0, {"outcome": "COMPLETE", "answer": "Other"})
        self.assertEqual(
            self.store.recent_turns("s"),
            [{"user": f"Question {i}", "assistant": str(i)} for i in range(1, 4)],
        )

    def test_cross_connection_claim_and_duplicate_result(self):
        peer = Store(self.path)
        self.addCleanup(peer.close)
        self.assertEqual(self.store.begin("s", "a", "under $60"), ("NEW", {}))
        self.assertEqual(peer.begin("s", "a", "under $60"), ("BUSY", None))
        self.assertEqual(peer.begin("s", "b", "under $40"), ("BUSY", None))
        # Other sessions may submit while this session has a turn in progress.
        self.assertEqual(peer.begin("t", "a", "under $70"), ("NEW", {}))
        state = {"revision": 1, "max_price_usd": "60.00"}
        self.store.commit("s", "a", state)
        result = {"outcome": "COMPLETE", "verified_ids": ["A"]}
        self.assertTrue(self.store.finish("s", "a", 1, result, ["A"]))
        self.assertEqual(peer.begin("s", "a", "under $60"), ("DONE", result))
        self.assertFalse(peer.finish("s", "a", 1, {"outcome": "PARTIAL"}, ["B"]))
        self.assertEqual(peer.get_state("s"), {**state, "displayed_ids": ["A"]})
        with self.assertRaises(ValueError):
            peer.begin("s", "a", "unrelated text")
        # A deliberate later submission with identical text is a new event.
        self.assertEqual(peer.begin("s", "c", "under $60")[0], "NEW")

    def test_stale_publish_cannot_replace_visible_mapping(self):
        self.store.begin("s", "a", "under $60")
        self.store.commit("s", "a", {"revision": 1})
        self.store.finish("s", "a", 1, {"outcome": "COMPLETE"}, ["A"])
        _, state = self.store.begin("s", "b", "under $40")
        self.store.commit("s", "b", {**state, "revision": 2, "max_price_usd": "40"})
        with self.assertRaises(StaleRunError):
            self.store.commit("s", "b", {**state, "revision": 1})
        self.assertFalse(self.store.finish("s", "b", 1, {"outcome": "COMPLETE"}, ["B"]))
        self.assertEqual(self.store.get_state("s")["displayed_ids"], ["A"])
        self.assertEqual(self.store.get_state("s")["max_price_usd"], "40")
        status, result = self.store.begin("s", "b", "under $40")
        self.assertEqual(status, "DONE")
        self.assertEqual(result["run_state"], "STALE")
        self.assertEqual(self.store.begin("s", "c", "try again")[0], "NEW")

    def test_explicit_restart_preserves_accepted_change_without_replay(self):
        self.store.begin("s", "a", "under $40")
        state = {"revision": 1, "max_price_usd": "40"}
        self.store.commit("s", "a", state)
        reopened = Store(self.path)
        self.addCleanup(reopened.close)
        # Simply constructing a second Store must not interrupt a live run.
        self.assertEqual(reopened.begin("s", "a", "under $40"), ("BUSY", None))
        self.assertEqual(reopened.recover_interrupted(), 1)
        self.assertEqual(reopened.recover_interrupted(), 0)
        status, result = reopened.begin("s", "a", "under $40")
        self.assertEqual(status, "DONE")
        self.assertEqual(result["run_state"], "INTERRUPTED")
        self.assertEqual(reopened.get_state("s"), state)
        self.assertEqual(reopened.begin("s", "b", "try again"), ("NEW", state))
        with self.assertRaises(StaleRunError):
            self.store.commit("s", "a", {"revision": 2})
        self.assertFalse(self.store.finish("s", "a", 1, {"outcome": "COMPLETE"}, ["B"]))
