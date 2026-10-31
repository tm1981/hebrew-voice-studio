"""Text preparation applied before every engine: the user's pronunciation dictionary,
then Hebrew normalization (abbreviations, currency, dates, numbers).

Engines handle digits and abbreviations inconsistently ("15:30", "ד"ר", "250 ₪"), so
everything is turned into plain Hebrew words first. Numbers use hebrew-num2words (it
already agrees numbers with the noun's gender: "3 ילדים" -> "שלושה ילדים"); dates and
currency are rewritten here because it leaves them broken.

The dictionary lives in pronunciations.json: {"word": "how to read it"}. The replacement
can carry niqqud or be a phonetic spelling. Entries of 3+ letters also match behind the
Hebrew prefixes ו ה ב ל מ ש כ ("ותאמר" matches "תאמר"); shorter ones match the exact word.
"""
import json
import re
import threading
from pathlib import Path

import hebrew_num2words as num2words

from paths import DATA_DIR

DICTIONARY_FILE = DATA_DIR / "pronunciations.json"

HEBREW = "א-ת"
MARKS = "֑-ׇ"
LETTER = rf"[{HEBREW}{MARKS}A-Za-z0-9]"
PREFIXES = "והבלמשכ"
MIN_PREFIXED_LETTERS = 3

# Common abbreviations; a dictionary entry with the same key overrides these.
ABBREVIATIONS = {
    'ד"ר': "דוקטור",
    'עו"ד': "עורך דין",
    'רו"ח': "רואה חשבון",
    "פרופ'": "פרופסור",
    "גב'": "גברת",
    'ש"ח': "שקלים",
    'וכו\'': "וכולי",
    "מס'": "מספר",
    "רח'": "רחוב",
    'ת"א': "תל אביב",
    'בע"מ': "בעירבון מוגבל",
    'אח"כ': "אחר כך",
    'בי"ס': "בית ספר",
    'ביה"ס': "בית הספר",
    'צה"ל': "צהל",
    'ארה"ב': "ארצות הברית",
    'יו"ר': "יושב ראש",
    'מנכ"ל': "מנכל",
    'סמנכ"ל': "סמנכל",
    'כנ"ל': "כנזכר לעיל",
    'ז"ל': "זכרונו לברכה",
    'בס"ד': "בסיעתא דשמיא",
    'ע"י': "על ידי",
    'ע"פ': "על פי",
    'לפנה"ס': "לפני הספירה",
    'לספה"נ': "לספירת הנוצרים",
}

CURRENCY = {"₪": "שקלים", "$": "דולר", "€": "יורו", "£": "פאונד"}
_NUMBER = r"\d+(?:[.,]\d+)*"
_CURRENCY_AFTER = re.compile(rf"({_NUMBER})\s*([₪$€£])")
_CURRENCY_BEFORE = re.compile(rf"([₪$€£])\s*({_NUMBER})")
# An attached prefix ("ב-12/03") joins the expanded words without its hyphen ("בשנים עשר").
_DATE = re.compile(rf"(?:([{PREFIXES}]{{1,3}})-?)?(?<!\d)(\d{{1,2}})[/.](\d{{1,2}})[/.](\d{{4}}|\d{{2}})(?!\d)")
_GERSHAYIM = str.maketrans({"״": '"', "׳": "'", "“": '"', "”": '"', "’": "'"})


def _entry_pattern(word: str) -> re.Pattern:
    """Regex for a dictionary word: quote marks match both ASCII and Hebrew gershayim,
    and Hebrew prefixes are allowed for words long enough not to collide."""
    body = re.escape(word.translate(_GERSHAYIM)).replace('"', '["״]').replace("'", "['׳]")
    letters = len(re.findall(rf"[{HEBREW}]", word))
    prefix = f"(?:[{PREFIXES}]{{1,3}}-?)?" if letters >= MIN_PREFIXED_LETTERS else ""
    return re.compile(rf"(?<!{LETTER})({prefix}){body}(?!{LETTER})")


class Dictionary:
    """The user's pronunciation dictionary, saved to pronunciations.json."""

    def __init__(self, path: Path = DICTIONARY_FILE):
        self.path = path
        self.lock = threading.Lock()
        self.entries: dict[str, str] = {}
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8"))
        self._compile()

    def _compile(self):
        # Longest first, so "בית הספר" wins over "בית".
        self.patterns = [(_entry_pattern(w), r) for w, r in sorted(self.entries.items(), key=lambda e: -len(e[0]))]

    def _save(self):
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")
        self._compile()

    def set(self, word: str, reading: str):
        with self.lock:
            self.entries[word] = reading
            self._save()

    def remove(self, word: str):
        with self.lock:
            self.entries.pop(word, None)
            self._save()

    def apply(self, text: str) -> str:
        for pattern, reading in self.patterns:
            text = pattern.sub(lambda m, r=reading: m.group(1) + r, text)
        return text


_ABBREVIATION_PATTERNS = [(_entry_pattern(w), r) for w, r in sorted(ABBREVIATIONS.items(), key=lambda e: -len(e[0]))]


def _expand_date(m: re.Match) -> str:
    prefix, day, month, year = m.group(1) or "", int(m.group(2)), int(m.group(3)), int(m.group(4))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    if len(m.group(4)) == 2:
        year += 2000
    return prefix + num2words.expand_date(day, month, year)


def normalize(text: str) -> str:
    """Abbreviations, currency, dates and numbers into Hebrew words."""
    for pattern, reading in _ABBREVIATION_PATTERNS:
        text = pattern.sub(lambda m, r=reading: m.group(1) + r, text)
    text = _CURRENCY_BEFORE.sub(lambda m: f"{m.group(2)} {CURRENCY[m.group(1)]}", text)
    text = _CURRENCY_AFTER.sub(lambda m: f"{m.group(1)} {CURRENCY[m.group(2)]}", text)
    text = _DATE.sub(_expand_date, text)
    return num2words.normalize_numbers(text)


def prepare(text: str, dictionary: Dictionary, normalize_text: bool = True) -> str:
    """The dictionary first, so a user's entry beats the built-in abbreviations."""
    text = dictionary.apply(text)
    return normalize(text) if normalize_text else text
