"""Splitting text into speech and pauses: [pause] markers, dialogue, and line breaks.

Run from the project folder: python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from script import LINE_PAUSE, PARAGRAPH_PAUSE, line_pieces, parse  # noqa: E402


class LinePiecesTest(unittest.TestCase):
    def test_one_line_has_no_extra_pause(self):
        self.assertEqual(line_pieces("שלום. מה שלומך?"), [("שלום. מה שלומך?", 0.0)])

    def test_new_line_and_blank_line(self):
        text = "Chapter one\nIt was late. We left.\n\nThe next day\r\nwe came back."
        self.assertEqual(line_pieces(text), [
            ("Chapter one", LINE_PAUSE),
            ("It was late. We left.", PARAGRAPH_PAUSE),
            ("The next day", LINE_PAUSE),
            ("we came back.", 0.0),
        ])

    def test_several_blank_lines_are_one_paragraph_break(self):
        self.assertEqual(line_pieces("א.\n \n\n\t\nב."), [("א.", PARAGRAPH_PAUSE), ("ב.", 0.0)])

    def test_breaks_at_the_ends_add_nothing(self):
        self.assertEqual(line_pieces("\n\nא.\n\n"), [("א.", 0.0)])
        self.assertEqual(line_pieces(" \n "), [])


class ParseTest(unittest.TestCase):
    def test_pause_marker(self):
        self.assertEqual(parse("שלום [הפסקה 2] עולם", "v"), [("speech", "v", "שלום"), ("pause", 2.0), ("speech", "v", "עולם")])

    def test_dialogue_keeps_one_speakers_lines_together(self):
        segments = parse("נועה: שלום\nנועה: מה נשמע?\nדני: טוב", "v", dialogue=True, speakers={"נועה": "a", "דני": "b"})
        self.assertEqual(segments, [("speech", "a", "שלום\nמה נשמע?"), ("speech", "b", "טוב")])


if __name__ == "__main__":
    unittest.main()
