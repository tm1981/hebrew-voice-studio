"""GPU engines say while they load, and why a load failed, so the page and the log can show it.

Run from the project folder: python -m unittest discover tests
"""
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import OnDemand  # noqa: E402


class FakeEngine(OnDemand):
    label = "Fake"
    LOG_EVERY = 0.05

    def __init__(self, fail=False):
        self.lock = threading.RLock()
        self.fail = fail
        self.seen_loading = None

    def _load(self):
        self.seen_loading = self.loading_since is not None
        threading.Event().wait(0.12)  # long enough for a "still loading" line
        if self.fail:
            raise RuntimeError("out of GPU memory")

    def _unload(self):
        pass


class LoadingStateTest(unittest.TestCase):
    def test_loading_is_shown_while_it_loads_and_cleared_after(self):
        engine = FakeEngine()
        with engine.lock:
            engine.ensure_loaded()
        self.assertTrue(engine.seen_loading)
        self.assertTrue(engine.loaded)
        self.assertIsNone(engine.loading_since)
        self.assertIsNone(engine.load_error)

    def test_a_failed_load_keeps_why(self):
        engine = FakeEngine(fail=True)
        with self.assertRaises(RuntimeError), engine.lock:
            engine.ensure_loaded()
        self.assertFalse(engine.loaded)
        self.assertIsNone(engine.loading_since)
        self.assertEqual(engine.load_error, "out of GPU memory")

        engine.fail = False  # a later load that works clears it
        with engine.lock:
            engine.ensure_loaded()
        self.assertIsNone(engine.load_error)


if __name__ == "__main__":
    unittest.main()
