"""Splitting text into speech and pauses: [pause] markers, dialogue, and line breaks.

Run from the project folder: python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from script import LINE_PAUSE, PARAGRAPH_PAUSE, line_pieces, parse, word_map  # noqa: E402
from text_prep import normalize  # noqa: E402


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


class WordMapTest(unittest.TestCase):
    def words(self, typed):
        return [(p, r) for p, r in word_map(typed, normalize(typed).split()) if not p.isspace()]

    def test_unchanged_words_map_one_to_one(self):
        self.assertEqual(self.words("שלום לכולם, מה נשמע?"),
                         [("שלום", (0, 0)), ("לכולם,", (1, 1)), ("מה", (2, 2)), ("נשמע?", (3, 3))])

    def test_a_price_marks_both_typed_words_for_all_three_spoken(self):
        self.assertEqual(self.words("הכרטיס עולה 250 ₪ היום."),
                         [("הכרטיס", (0, 0)), ("עולה", (1, 1)), ("250", (2, 4)), ("₪", (2, 4)), ("היום.", (5, 5))])

    def test_a_phone_number_is_one_typed_word_for_ten_spoken(self):
        self.assertEqual(self.words("Call 054-123-4567 now."), [("Call", (0, 0)), ("054-123-4567", (1, 10)), ("now.", (11, 11))])

    def test_spaces_and_line_breaks_are_kept_unmarked(self):
        pieces = word_map("א ב\nג", ["א", "ב", "ג"])
        self.assertEqual("".join(p for p, _ in pieces), "א ב\nג")
        self.assertEqual([r for p, r in pieces if p.isspace()], [None, None])


class WordTimingHelpersTest(unittest.TestCase):
    def test_alignment_path_finds_where_each_token_starts(self):
        import numpy as np
        from stt import token_starts

        heard = np.zeros((3, 12))  # three tokens heard at frames 2-3, 5-7 and 9-11
        heard[0, 2:4], heard[1, 5:8], heard[2, 9:] = 1, 1, 1
        starts = token_starts(heard)
        self.assertLess(starts[0], starts[1])
        self.assertEqual(list(starts[1:]), [5, 9])

    def test_language_follows_the_letters(self):
        from stt import script_language

        self.assertEqual([script_language(t) for t in ("Hello there.", "שלום hello", "مرحبا", "123")], ["en", "he", "ar", "he"])


class ParseTest(unittest.TestCase):
    def test_pause_marker(self):
        self.assertEqual(parse("שלום [הפסקה 2] עולם", "v"), [("speech", "v", "שלום"), ("pause", 2.0), ("speech", "v", "עולם")])

    def test_dialogue_keeps_one_speakers_lines_together(self):
        segments = parse("נועה: שלום\nנועה: מה נשמע?\nדני: טוב", "v", dialogue=True, speakers={"נועה": "a", "דני": "b"})
        self.assertEqual(segments, [("speech", "a", "שלום\nמה נשמע?"), ("speech", "b", "טוב")])


if __name__ == "__main__":
    unittest.main()
