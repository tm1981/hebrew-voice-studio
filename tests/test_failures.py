"""Engine failures reach the user as a reason, a dead worker starts again, and a bad upload
can't lose a voice.

Run from the project folder: python -m unittest discover tests
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engines  # noqa: E402
from engines import RemoteEngine  # noqa: E402


class Answer:
    def __init__(self, status_code=200, text="", body=None):
        self.status_code, self.text, self.body, self.content = status_code, text, body, b""

    def json(self):
        if self.body is None:
            raise ValueError("not JSON")
        return self.body


def worker(posts):
    """A RemoteEngine whose worker answers with `posts` in turn (an exception is raised)."""
    engine = object.__new__(RemoteEngine)
    engine.name = engine.label = "Fake"
    engine.url, engine.lock, engine.process = "http://worker", threading.RLock(), None
    engine.starts = engine.stops = 0

    def load():
        engine.starts += 1

    def unload():
        engine.stops += 1

    engine._load, engine._unload = load, unload
    answers = iter(posts)

    def post(*_, **__):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return engine, mock.patch.object(engines.requests, "post", post)


class WorkerFailureTest(unittest.TestCase):
    def test_a_dead_worker_is_started_again_once(self):
        engine, patched = worker([requests.ConnectionError("gone"), Answer()])
        engine.loaded = True  # it died while the server thought it was loaded
        with patched:
            self.assertEqual(engine._ask_worker({}).status_code, 200)
        self.assertEqual((engine.stops, engine.starts), (1, 1))
        self.assertTrue(engine.loaded)

    def test_a_worker_that_dies_twice_says_so(self):
        engine, patched = worker([requests.ConnectionError("gone"), requests.ConnectionError("gone")])
        with patched, self.assertRaisesRegex(RuntimeError, "stopped while reading"):
            engine._ask_worker({})
        self.assertFalse(engine.loaded)  # so the next reading starts it fresh

    def test_the_reason_is_read_from_json_or_plain_text(self):
        self.assertEqual(RemoteEngine._detail(Answer(400, body={"detail": "No voice 'x'"})), "No voice 'x'")
        self.assertEqual(RemoteEngine._detail(Answer(500, text="Internal Server Error")), "Internal Server Error")
        self.assertEqual(RemoteEngine._detail(Answer(500)), "HTTP 500")
        self.assertIn("not enough GPU memory", RemoteEngine._detail(Answer(500, body={"detail": "CUDA out of memory. Tried"})))


class VoiceFilesTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        patches = [mock.patch.object(engines, "VOICES_DIR", root), mock.patch.object(engines, "PREPARED_DIR", root / ".cache")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.folder.cleanup)
        self.root = root

    def test_a_replaced_voice_comes_back_when_the_new_recording_fails(self):
        (self.root / "dana.wav").write_bytes(b"old")
        stash = engines.stash_voice_files("dana")
        self.assertEqual(engines.custom_voice_names(), [])
        (self.root / "dana.mp3").write_bytes(b"new, unreadable")
        engines.restore_voice_files("dana", stash)
        self.assertEqual((self.root / "dana.wav").read_bytes(), b"old")
        self.assertFalse((self.root / "dana.mp3").exists())
        self.assertFalse(stash.exists())

    def test_a_voice_folder_is_stashed_and_dropped_when_the_new_one_is_good(self):
        (self.root / "dana").mkdir()
        (self.root / "dana" / "clip1.wav").write_bytes(b"old")
        (self.root / "dana two.wav").write_bytes(b"someone else")
        stash = engines.stash_voice_files("dana")
        self.assertTrue((stash / "dana" / "clip1.wav").exists())
        self.assertEqual(engines.custom_voice_names(), ["dana two"])
        engines.discard_voice_stash(stash)
        self.assertFalse(stash.exists())

    def test_a_new_voice_has_nothing_to_stash(self):
        self.assertIsNone(engines.stash_voice_files("new"))
        engines.restore_voice_files("new", None)

    def test_a_voice_an_engine_cannot_use_is_not_left_listed(self):
        (self.root / "dana.wav").write_bytes(b"x")

        class Engine:
            label = "Fake"
            voices = {"default": 1, "dana": None}  # listed while the engine loads

            def add_voice(self, name, path):
                raise RuntimeError("unreadable")

        with mock.patch.object(engines, "prepared_clip", lambda name: Path(name)):
            engines.load_custom_voices(Engine())
        self.assertEqual(list(Engine.voices), ["default"])


if __name__ == "__main__":
    unittest.main()
