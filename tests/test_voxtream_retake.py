"""voxtream2-he reads a sentence again when the reading drags, and keeps the best of the takes.

Run from the project folder: python -m unittest discover tests
"""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workers"))
try:
    import torch  # noqa: F401
    import voxtream_worker
except ImportError:  # an env without the worker's packages
    voxtream_worker = None

RATE = 24000
TEXT = "היום נדבר על מזג האוויר בירושלים."  # 27 letters: a good reading is 2 to 3 seconds


def backend(seconds_per_take):
    """A worker whose model returns readings of these lengths, one per take."""
    b = voxtream_worker.VoxtreamBackend()
    b.sample_rate, b.takes = RATE, 0
    b.voices, b.prompts = {"v": "v.wav"}, {("v", True): object()}
    b.phonemizer = types.SimpleNamespace(phonemize=lambda text, language: text)
    lengths = iter(seconds_per_take)

    def generate_stream(**_):
        b.takes += 1
        yield (0.1 * np.ones(int(RATE * next(lengths)), dtype=np.float32), None)

    b.resets = 0

    def reset():
        b.resets += 1

    b.generator = types.SimpleNamespace(
        generate_stream=generate_stream, config=types.SimpleNamespace(max_audio_length_ms=60000),
        model=types.SimpleNamespace(sink_attention=types.SimpleNamespace(reset=reset)))
    return b


@unittest.skipIf(voxtream_worker is None, "needs torch")
class RetakeTest(unittest.TestCase):
    def seconds(self, audio):
        return round(len(audio) / RATE, 1)

    def test_a_reading_at_a_normal_pace_is_kept(self):
        b = backend([2.5])
        self.assertEqual((self.seconds(b.synthesize(TEXT, "v", 1.0)), b.takes), (2.5, 1))

    def test_a_reading_that_drags_is_read_again(self):
        b = backend([20.0, 2.5])
        self.assertEqual((self.seconds(b.synthesize(TEXT, "v", 1.0)), b.takes), (2.5, 2))

    def test_when_every_take_drags_the_shortest_is_kept(self):
        b = backend([20.0, 9.0, 14.0])
        self.assertEqual((self.seconds(b.synthesize(TEXT, "v", 1.0)), b.takes), (9.0, 3))

    def test_the_leftovers_of_earlier_readings_are_cleared_before_every_take(self):
        b = backend([20.0, 2.5, 2.5])
        b.synthesize(TEXT, "v", 1.0)
        b.synthesize(TEXT, "v", 1.0)
        self.assertEqual((b.takes, b.resets), (3, 3))

    def test_a_take_may_not_run_longer_than_a_good_reading_could(self):
        b = backend([2.5, 2.5])
        b.synthesize(TEXT, "v", 1.0)
        self.assertEqual(b.generator.config.max_audio_length_ms, 9000)  # 27 letters at 4.5 a second, plus 3 s
        b.synthesize(" ".join([TEXT] * 20), "v", 1.0)
        self.assertEqual(b.generator.config.max_audio_length_ms, 60000)  # the library's own limit

    def test_a_few_words_are_not_judged_by_pace(self):
        b = backend([4.0])
        self.assertEqual((self.seconds(b.synthesize("כן, בטח.", "v", 1.0)), b.takes), (4.0, 1))


if __name__ == "__main__":
    unittest.main()
