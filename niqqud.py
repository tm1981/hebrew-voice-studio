"""Respect niqqud the user typed, per word, while auto-diacritizing the rest.

Dicta and Phonikud re-predict niqqud from the bare letters, so text pasted with niqqud
comes out with their guess instead ("הַסַּפָּר סָפַר", the barber counted, becomes
"הַסֵּפֶר סָפֵּר"). Instead: predict on the bare text, then keep every word the user pointed.
"""
import re

# Vowel points, dagesh/shin dots, meteg and cantillation; not maqaf, paseq or sof pasuq,
# which are punctuation.
NIQQUD = re.compile(r"[֑-ׇֽֿׁׂׅׄ]")
WORDS = re.compile(r"(\s+)")


def strip_niqqud(text: str) -> str:
    return NIQQUD.sub("", text)


def keep_typed_niqqud(typed: str, predicted: str) -> str:
    """Words in `typed` that carry niqqud win over the same word in `predicted`.
    If the predictor changed the word count (it shouldn't), trust the typed text."""
    typed_words, predicted_words = WORDS.split(typed), WORDS.split(predicted)
    if len(typed_words) != len(predicted_words):
        return typed
    return "".join(t if NIQQUD.search(t) else p for t, p in zip(typed_words, predicted_words))


def diacritize_respecting_typed(text: str, predict) -> str:
    """predict(bare_text) -> text with niqqud; typed niqqud is kept per word."""
    if not NIQQUD.search(text):
        return predict(text)
    return keep_typed_niqqud(text, predict(strip_niqqud(text)))


class RespectfulDicta:
    """Drop-in for the Dicta instance Chatterbox's tokenizer calls (add_diacritics)."""

    def __init__(self, dicta):
        self.dicta = dicta

    def add_diacritics(self, text: str) -> str:
        return diacritize_respecting_typed(text, self.dicta.add_diacritics)
