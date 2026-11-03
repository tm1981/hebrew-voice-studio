"""Text preparation applied before every engine: the user's pronunciation dictionary,
then Hebrew normalization (abbreviations, currency, dates, numbers). Numbers in English
sentences become English words instead.

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
from num2words import num2words as num2words_en

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
# Numbers in English ("I am 10 years old") are read in English. The next word in the same
# sentence decides ("יש לי 3 apples" -> "three apples"), or the one before when the number
# ends the sentence ("born in 1990."); a number on its own stays Hebrew.
ENGLISH_CURRENCY = {"$": ("dollar", "cent"), "€": ("euro", "cent"), "£": ("pound", "penny"), "₪": ("shekel", "agora")}
ENGLISH_PLURAL = {"penny": "pence", "agora": "agorot"}
YEAR_BEFORE = {"in", "since", "from", "until", "till", "by", "of", "year", "circa", "early", "late", "mid"}
# Day/month without a year ("on 3/4") is a date only after one of these; otherwise a fraction.
DATE_BEFORE = {"on", "by", "until", "till", "from", "since", "before", "after", "dated"}
MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
# Same formats as _DATE (day first, "/" or ".", 2- or 4-digit year), plus "d/m" with no year.
_ENGLISH_DATE = re.compile(r"(?<![\w.,/:])(\d{1,2})([/.])(\d{1,2})(?:[/.](\d{4}|\d{2}))?(?![\w/]|[.,:]\d)")
_ENGLISH_NUMBER = re.compile(
    r"(?<![\w.,/:])(?:([$€£₪])\s*)?(\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.(\d+))?"
    r"(?::(\d{2}))?(?:(st|nd|rd|th)\b|\s*(%)|\s*([$€£₪]))?(?![\w/]|[.,:]\d)"
)
_WORD_BEFORE = re.compile(rf"([{HEBREW}A-Za-z][{HEBREW}{MARKS}A-Za-z'’]*)[^{HEBREW}{MARKS}A-Za-z.!?\n]*$")
_WORD_AFTER = re.compile(rf"^[^{HEBREW}A-Za-z.!?\n]*([{HEBREW}A-Za-z])")
_LATIN = re.compile(r"[A-Za-z]")
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

    def merge(self, entries: dict[str, str]):
        """Adds these words (a restored backup); a word already here takes the new reading."""
        with self.lock:
            self.entries.update(entries)
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


def _in_english(text: str, start: int, end: int) -> tuple[bool, str]:
    """Whether the number at text[start:end] is in English, and the word before it."""
    before = _WORD_BEFORE.search(text[max(0, start - 200) : start])
    after = _WORD_AFTER.search(text[end:])
    word = before.group(1).lower() if before else ""
    if after:
        return bool(_LATIN.match(after.group(1))), word
    return bool(before and _LATIN.match(word)), word


def _expand_english_date(m: re.Match) -> str:
    day, sep, month, year = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    english, before = _in_english(m.string, m.start(), m.end())
    if not (english and 1 <= day <= 31 and 1 <= month <= 12) or not (year or sep == "/" and before in DATE_BEFORE):
        return m.group(0)
    words = f"the {num2words_en(day, to='ordinal')} of {MONTHS[month - 1]}"
    if year:
        words += " " + num2words_en(int(year) + (2000 if len(year) == 2 else 0), to="year")
    return words


def _english_count(n: int, unit: str) -> str:
    plural = unit if n == 1 else ENGLISH_PLURAL.get(unit, unit + "s")
    return f"{num2words_en(n)} {plural}"


def _expand_english(m: re.Match) -> str:
    english, before = _in_english(m.string, m.start(), m.end())
    if not english:
        return m.group(0)
    cur_before, whole, fraction, minutes, ordinal, percent, cur_after = m.groups()
    n = int(whole.replace(",", ""))
    currency = cur_before or cur_after
    if currency and not (minutes or ordinal or percent):
        main, small = ENGLISH_CURRENCY[currency]
        words = _english_count(n, main)
        if fraction and int(fraction):
            words += " and " + _english_count(int(fraction.ljust(2, "0")[:2]), small)
        return words
    if minutes and not fraction:
        tail = "o'clock" if minutes == "00" else ("oh " if minutes[0] == "0" else "") + num2words_en(int(minutes))
        return f"{num2words_en(n)} {tail}"
    if ordinal and not fraction:
        return num2words_en(n, to="ordinal")
    if fraction:
        words = f"{num2words_en(n)} point " + " ".join(num2words_en(int(d)) for d in fraction)
    elif "," not in whole and 1100 <= n <= 2099 and before in YEAR_BEFORE:
        words = num2words_en(n, to="year")
    else:
        words = num2words_en(n)
    words = words.replace(",", "")  # "one thousand, two hundred": no pause in the middle
    return words + " percent" if percent else words


def normalize(text: str) -> str:
    """Abbreviations, currency, dates and numbers into words: English words for a number in
    English, Hebrew for the rest."""
    for pattern, reading in _ABBREVIATION_PATTERNS:
        text = pattern.sub(lambda m, r=reading: m.group(1) + r, text)
    text = _ENGLISH_DATE.sub(_expand_english_date, text)
    text = _ENGLISH_NUMBER.sub(_expand_english, text)
    text = _CURRENCY_BEFORE.sub(lambda m: f"{m.group(2)} {CURRENCY[m.group(1)]}", text)
    text = _CURRENCY_AFTER.sub(lambda m: f"{m.group(1)} {CURRENCY[m.group(2)]}", text)
    text = _DATE.sub(_expand_date, text)
    return num2words.normalize_numbers(text)


def prepare(text: str, dictionary: Dictionary, normalize_text: bool = True) -> str:
    """The dictionary first, so a user's entry beats the built-in abbreviations."""
    text = dictionary.apply(text)
    return normalize(text) if normalize_text else text
