"""Before a GPU engine loads, idle engines make room for it; when that isn't enough, the reason is kept.

Run from the project folder: python -m unittest discover tests
"""
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import OnDemand  # noqa: E402
from gpu_room import make_room  # noqa: E402


class Gpu:
    """24 GB; each loaded engine takes its size."""

    def __init__(self, *engines):
        self.engines = engines

    def free(self):
        return 24 - sum(e.size for e in self.engines if e.loaded)


class Fake(OnDemand):
    def __init__(self, name, size, idle_seconds=600, loaded=True, fail=False):
        self.name = self.label = name
        self.size, self.loaded, self.fail = size, loaded, fail
        self.last_used = time.monotonic() - idle_seconds
        self.lock = threading.RLock()

    def _load(self):
        if self.fail:
            raise RuntimeError("worker exited with code 1")

    def _unload(self):
        pass


def quiet(*_, **__):
    pass


class MakeRoomTest(unittest.TestCase):
    def test_nothing_is_unloaded_when_it_fits(self):
        old, new = Fake("old", 5), Fake("new", 10, loaded=False)
        self.assertIsNone(make_room(new, new.size, Gpu(old).free, [old], set(), quiet))
        self.assertTrue(old.loaded)

    def test_the_longest_idle_engines_go_first_and_only_as_many_as_needed(self):
        oldest, older, recent = Fake("oldest", 8, 900), Fake("older", 8, 600), Fake("recent", 6, 60)
        new = Fake("new", 10, loaded=False)
        gpu = Gpu(oldest, older, recent)  # 2 GB free
        self.assertIsNone(make_room(new, new.size, gpu.free, [recent, older, oldest], set(), quiet))
        self.assertEqual([oldest.loaded, older.loaded, recent.loaded], [False, False, True])

    def test_kept_busy_and_just_used_engines_stay_and_the_note_says_so(self):
        kept, busy, reading = Fake("kept", 8), Fake("busy", 8), Fake("reading", 6, idle_seconds=1)
        new = Fake("new", 10, loaded=False)
        held = threading.Event()
        release = threading.Event()

        def hold():
            with busy.lock:
                held.set()
                release.wait(5)

        threading.Thread(target=hold, daemon=True).start()
        held.wait(5)
        try:
            note = make_room(new, new.size, Gpu(kept, busy, reading).free, [kept, busy, reading], {"kept"}, quiet)
        finally:
            release.set()
        self.assertTrue(all(e.loaded for e in (kept, busy, reading)))
        self.assertEqual(note, "It needs about 10 GB of GPU memory and 2.0 GB is free "
                               "(kept is kept loaded; busy is in use; reading is in use).")

    def test_a_load_that_fails_for_lack_of_room_says_why(self):
        kept = Fake("kept", 20)
        new = Fake("new", 10, loaded=False, fail=True)
        OnDemand.before_load = lambda e: make_room(e, e.size, Gpu(kept).free, [kept], {"kept"}, quiet)
        self.addCleanup(setattr, OnDemand, "before_load", None)
        with self.assertRaisesRegex(RuntimeError, "needs about 10 GB of GPU memory and 4.0 GB is free"), new.lock:
            new.ensure_loaded()
        self.assertIn("kept is kept loaded", new.load_error)

    def test_a_broken_memory_check_never_stops_a_load(self):
        new = Fake("new", 10, loaded=False)

        def broken(engine):
            raise OSError("nvidia-smi not found")

        OnDemand.before_load = broken
        self.addCleanup(setattr, OnDemand, "before_load", None)
        with new.lock:
            new.ensure_loaded()
        self.assertTrue(new.loaded)


if __name__ == "__main__":
    unittest.main()
