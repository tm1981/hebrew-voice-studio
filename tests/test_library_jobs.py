"""The library's list, sizes, pins and pruning; document jobs that are saved, retried and resumed.

Run from the project folder: python -m unittest discover tests
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from documents import Jobs  # noqa: E402
from library import Library  # noqa: E402


def speech(text="שלום", seconds=0.5, rate=24000):
    return {"kind": "speech", "text": text, "voice": "lightblue/noa", "audio": np.zeros(int(rate * seconds), dtype=np.float32),
            "sample_rate": rate}


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.library = Library(Path(self.folder.name))

    def save(self, title="a", key=None, **more):
        return self.library.save(title=title, kind="reading", text=title, voice="lightblue/noa", settings={},
                                 items=[speech(), {"kind": "pause", "seconds": 1.0}, speech()], generation_seconds=1.0,
                                 key=key, **more)

    def age(self, meta, days):
        """Makes a reading `days` days old, as the files on disk would say."""
        path = Path(self.folder.name) / meta["id"] / "meta.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["created"] = time.time() - days * 86400
        path.write_text(json.dumps(data), encoding="utf-8")
        self.library.reindex()

    def test_the_list_knows_a_reading_from_the_moment_it_is_saved(self):
        meta = self.save("first")
        (item,) = self.library.list()
        self.assertEqual((item["id"], item["title"], item["sentences"], item["pinned"]), (meta["id"], "first", 2, False))
        self.assertEqual(item["bytes"], sum(f.stat().st_size for f in (Path(self.folder.name) / meta["id"]).iterdir()))
        self.assertEqual(self.library.size(), item["bytes"])
        self.library.delete(meta["id"])
        self.assertEqual((self.library.list(), self.library.size()), ([], 0))

    def test_the_list_is_the_same_after_reading_the_folder_again(self):
        self.save("one"), self.save("two", key="k")
        before = self.library.list()
        again = Library(Path(self.folder.name))
        self.assertEqual(again.list(), before)
        self.assertEqual(again.find("k")["title"], "two")

    def test_the_whole_reading_is_kept_as_flac_and_reads_back(self):
        import soundfile as sf

        meta = self.save()
        path = self.library.full_path(meta["id"])
        self.assertEqual(path.name, "full.flac")
        audio, rate = sf.read(path, dtype="float32")
        self.assertEqual((rate, len(audio)), (24000, 24000 * 2))  # 0.5 s + 1 s pause + 0.5 s

    def test_a_reading_saved_before_flac_still_plays_and_moves_over_when_it_changes(self):
        import soundfile as sf

        meta = self.save()
        folder = Path(self.folder.name) / meta["id"]
        (folder / "full.flac").unlink()
        sf.write(folder / "full.wav", np.zeros(100, dtype=np.float32), 24000)
        self.assertEqual(self.library.full_path(meta["id"]).name, "full.wav")
        self.library.replace_sentence(meta["id"], 0, "חדש", np.zeros(2400, dtype=np.float32), 24000)
        self.assertEqual(self.library.full_path(meta["id"]).name, "full.flac")
        self.assertFalse((folder / "full.wav").exists())

    def test_one_damaged_entry_does_not_break_the_list(self):
        self.save("good")
        bad = Path(self.folder.name) / "bad"
        bad.mkdir()
        (bad / "meta.json").write_text('{"id": "bad", "title": "no created or items"}', encoding="utf-8")
        self.library.reindex()
        self.assertEqual([i["title"] for i in self.library.list()], ["good"])

    def test_old_readings_are_deleted_and_pinned_ones_kept(self):
        old, pinned, new = self.save("old"), self.save("old but pinned"), self.save("new")
        self.library.pin(pinned["id"], True)
        self.age(old, 40), self.age(pinned, 40)
        self.assertEqual(self.library.prunable(30), [old["id"]])
        self.assertEqual(self.library.prune(30), [old["id"]])
        self.assertEqual(sorted(i["title"] for i in self.library.list()), ["new", "old but pinned"])
        self.assertFalse((Path(self.folder.name) / old["id"]).exists())

    def test_a_pin_survives_a_sentence_being_read_again(self):
        meta = self.save()
        self.library.pin(meta["id"], True)
        self.library.replace_sentence(meta["id"], 0, "חדש", np.zeros(2400, dtype=np.float32), 24000)
        self.assertTrue(self.library.list()[0]["pinned"])
        self.library.pin(meta["id"], False)
        self.assertFalse(self.library.list()[0]["pinned"])


class JobsTest(unittest.TestCase):
    CHAPTERS = [("one", "text 1"), ("two", "text 2"), ("three", "text 3")]

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "jobs.json"
        self.read_log = []
        self.fail = {}  # chapter title -> how many times it still fails

    def read(self, title, settings, chapter, text, group):
        self.read_log.append(chapter)
        if self.fail.get(chapter, 0) > 0:
            self.fail[chapter] -= 1
            raise RuntimeError("out of GPU memory")
        return {"id": f"{group}-{chapter}"}

    def wait(self, jobs, job, *statuses):
        deadline = time.time() + 5
        while job["status"] not in statuses and time.time() < deadline:
            time.sleep(0.01)
        self.assertIn(job["status"], statuses)

    def test_a_job_reads_every_chapter_and_is_saved(self):
        jobs = Jobs(self.path, self.read)
        job = jobs.start("book", self.CHAPTERS, {"voice": "lightblue/noa"})
        self.wait(jobs, job, "done")
        self.assertEqual((job["done"], len(job["items"]), self.read_log), (3, 3, ["one", "two", "three"]))
        saved = Jobs(self.path, self.read)
        self.assertEqual(saved.get(job["id"])["status"], "done")
        self.assertNotIn(job["id"], saved.work)  # a finished job keeps no chapter text

    def test_a_chapter_that_fails_once_is_read_again(self):
        self.fail = {"two": 1}
        jobs = Jobs(self.path, self.read)
        job = jobs.start("book", self.CHAPTERS)
        self.wait(jobs, job, "done")
        self.assertEqual(self.read_log, ["one", "two", "two", "three"])

    def test_a_chapter_that_fails_twice_stops_the_job_and_it_can_go_on_from_there(self):
        self.fail = {"two": 2}
        jobs = Jobs(self.path, self.read)
        job = jobs.start("book", self.CHAPTERS)
        self.wait(jobs, job, "failed")
        self.assertEqual((job["done"], job["error"]), (1, "two: out of GPU memory"))
        jobs.resume(job["id"])
        self.wait(jobs, job, "done")
        self.assertEqual(self.read_log, ["one", "two", "two", "two", "three"])  # "one" isn't read again
        self.assertEqual(len(job["items"]), 3)

    def test_a_job_the_server_stopped_under_is_interrupted_and_resumes_with_its_settings(self):
        self.path.write_text(json.dumps({
            "jobs": [{"id": "j1", "title": "book", "status": "running", "done": 1, "total": 3, "chapters": ["one", "two", "three"],
                      "items": ["x"], "error": None, "started": time.time()}],
            "work": {"j1": {"chapters": [list(c) for c in self.CHAPTERS], "settings": {"voice": "chatterbox/default"}}}}),
            encoding="utf-8")
        seen = []
        jobs = Jobs(self.path, lambda title, settings, chapter, text, group: seen.append((title, settings["voice"], chapter)) or {"id": chapter})
        job = jobs.get("j1")
        self.assertEqual(job["status"], "interrupted")
        jobs.resume("j1")
        self.wait(jobs, job, "done")
        self.assertEqual(seen, [("book", "chatterbox/default", "two"), ("book", "chatterbox/default", "three")])

    def test_a_running_or_finished_job_cannot_be_resumed(self):
        jobs = Jobs(self.path, self.read)
        job = jobs.start("book", self.CHAPTERS)
        self.wait(jobs, job, "done")
        with self.assertRaises(ValueError):
            jobs.resume(job["id"])

    def test_a_damaged_jobs_file_is_not_fatal(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(Jobs(self.path, self.read).jobs, {})


if __name__ == "__main__":
    unittest.main()
