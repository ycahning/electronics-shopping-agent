"""Offline Streamlit widget lifecycle checks (requires the project's ui extra)."""

import importlib.util
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from shopping_agent.catalog import FixtureCatalog


@unittest.skipUnless(
    importlib.util.find_spec("streamlit"), "Install the ui extra for AppTest."
)
class StreamlitUITests(unittest.TestCase):
    def test_rerender_does_not_repeat_submission_and_marks_older_results(self):
        from streamlit.testing.v1 import AppTest

        calls = []
        actual_call = FixtureCatalog.call

        def record_call(catalog, action, args, timeout):
            calls.append(action)
            return actual_call(catalog, action, args, timeout)

        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "ui-test.sqlite3"
            app_path = Path(__file__).resolve().parent / "shopping_agent" / "ui.py"
            env = {
                "SHOPPING_AGENT_MODE": "fixture",
                "SHOPPING_AGENT_LIVE_MODEL": "",
                "SHOPPING_AGENT_SNAPSHOT": "",
                "SHOPPING_AGENT_DB": str(db),
            }
            with (
                patch.dict(os.environ, env),
                patch.object(FixtureCatalog, "call", record_call),
            ):
                app = AppTest.from_file(app_path, default_timeout=15).run()
                self.assertFalse(app.exception)
                self.assertEqual(calls, [])

                app.chat_input[0].set_value("find appliances under $60").run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.chat_message), 2)
                self.assertIn("search_products", calls)
                self.assertIn("get_products", calls)

                after_first_submit = list(calls)
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(calls, after_first_submit)
                self.assertEqual(len(app.chat_message), 2)

                app.chat_input[0].set_value("budget to 40").run()
                self.assertFalse(app.exception)
                self.assertGreater(len(calls), len(after_first_submit))
                after_second_submit = list(calls)
                self.assertTrue(
                    any("historical result" in c.value for c in app.caption)
                )
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.chat_message), 4)
                self.assertEqual(calls, after_second_submit)
                self.assertEqual(
                    sum(
                        "historical result" in caption.value for caption in app.caption
                    ),
                    1,
                )
                with closing(sqlite3.connect(db)) as connection:
                    self.assertEqual(
                        connection.execute("SELECT count(*) FROM runs").fetchone(), (2,)
                    )

    def test_final_mode_reports_missing_configuration(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parent / "shopping_agent" / "ui.py"
        with patch.dict(
            os.environ,
            {
                "SHOPPING_AGENT_MODE": "final",
                "SHOPPING_AGENT_RECOMMENDER": "",
                "SHOPPING_AGENT_OLLAMA_MODEL": "",
            },
        ):
            app = AppTest.from_file(app_path, default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.chat_input)
            self.assertIn("SHOPPING_AGENT_RECOMMENDER", app.error[0].value)
