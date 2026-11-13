"""Changing a reading's pace keeps its pitch and its sound.

Run from the project folder: python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speed import change_speed  # noqa: E402

RATE = 24000


def voice_like(seconds=2.0, pitch=120.0):
    """A buzzy tone with harmonics whose loudness rises and falls like syllables."""
    t = np.arange(int(RATE * seconds)) / RATE
    tone = sum(np.sin(2 * np.pi * pitch * h * t) / h for h in range(1, 12))
    return (0.2 * tone * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def strongest_pitch(audio):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return np.fft.rfftfreq(len(audio), 1 / RATE)[int(np.argmax(spectrum))]


class ChangeSpeedTest(unittest.TestCase):
    def test_the_length_follows_the_speed(self):
        audio = voice_like()
        for speed in (0.8, 0.95, 1.05, 1.25, 1.5):
            self.assertEqual(len(change_speed(audio, RATE, speed)), round(len(audio) / speed), speed)

    def test_speed_one_and_very_short_audio_come_back_untouched(self):
        audio = voice_like()
        self.assertIs(change_speed(audio, RATE, 1.0), audio)
        short = audio[:1000]
        np.testing.assert_array_equal(change_speed(short, RATE, 1.3), short)

    def test_the_pitch_stays_where_it_was(self):
        audio = voice_like(pitch=120.0)
        for speed in (0.85, 1.1, 1.4):
            self.assertAlmostEqual(strongest_pitch(change_speed(audio, RATE, speed)), 120.0, delta=2.0, msg=speed)

    def test_a_steady_tone_stays_steady(self):
        """The pieces are joined in step with the wave: no hollow, echoing dips where they overlap.
        (A phase vocoder passes this too on a pure tone; on speech it is what smears.)"""
        t = np.arange(RATE * 2) / RATE
        tone = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
        for speed in (0.9, 1.1, 1.3):
            out = change_speed(tone, RATE, speed)[RATE // 10:-RATE // 10]
            level = np.sqrt(np.convolve(out**2, np.ones(480) / 480, mode="valid"))  # loudness over 20 ms
            self.assertLess(level.max() / level.min(), 1.15, speed)

    def test_loudness_and_silence_are_kept(self):
        audio = voice_like()
        out = change_speed(audio, RATE, 1.2)
        self.assertAlmostEqual(float(np.sqrt((out**2).mean())), float(np.sqrt((audio**2).mean())), delta=0.01)
        self.assertEqual(float(np.abs(change_speed(np.zeros(RATE, np.float32), RATE, 1.2)).max()), 0.0)
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
