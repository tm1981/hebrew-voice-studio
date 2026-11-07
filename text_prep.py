"""Text preparation applied before every engine: the user's pronunciation dictionary,
then Hebrew normalization (abbreviations, currency, dates, numbers). Numbers in English
and Arabic sentences become English and Arabic words instead.

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
# A Hebrew price is read here, not by hebrew-num2words: after "כרטיס" or "מספר" it reads the
# number digit by digit, and "49.90" came out "ארבעים ותשע תשעים". (one, many, cent one, cents)
HEBREW_MONEY = {
    "₪": ("שקל", "שקלים", "אגורה", "אגורות"),
    "$": ("דולר", "דולר", "סנט", "סנט"),
    "€": ("יורו", "יורו", "סנט", "סנט"),
    "£": ("פאונד", "פאונד", "פני", "פני"),
}
HEBREW_CURRENCY_WORDS = {"שקל": "₪", "שקלים": "₪", "דולר": "$", "דולרים": "$", "יורו": "€", "פאונד": "£"}
FEMININE_CENTS = {"אגורה"}
# Numbers in English ("I am 10 years old") are read in English. The next word in the same
# sentence decides ("יש לי 3 apples" -> "three apples"), or the one before when the number
# ends the sentence ("born in 1990."). So does a sentence with English and no Hebrew or Arabic
# ("No. 5", "16GB"), and for one with no letters ("2.") its line, then the whole text; a number
# with no letters anywhere stays Hebrew.
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
ARABIC = "ء-ي"
ARABIC_CURRENCY = {"₪": "شيكل", "$": "دولار", "€": "يورو", "£": "جنيه"}
ARABIC_MONTHS = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو",
                 "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")
_LETTERS = f"{HEBREW}A-Za-z{ARABIC}"
_LETTER_AFTER = re.compile(rf"^[^{_LETTERS}.!?؟\n]*([{_LETTERS}])")
_LETTER_BEFORE = re.compile(rf"([{_LETTERS}])[^{_LETTERS}.!?؟\n]*$")
_ARABIC_LETTER = re.compile(rf"[{ARABIC}]")
_ANY_LETTER = re.compile(rf"[{_LETTERS}]")
_NOT_ENGLISH_LETTER = re.compile(rf"[{HEBREW}{ARABIC}]")
_SENTENCE_END = re.compile(r"[.!?؟]\s")
# Digits and Latin letters in one word: "16GB", "3.5GHz", "M4", "1990s"
_GLUED_NUMBER = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?[A-Za-z]|[A-Za-z]+\d)[A-Za-z\d]*(?!\w)")
_GLUED_PARTS = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+")
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)")
_LIST_ITEM = re.compile(r"^[ \t]*(\d{1,3})\.[ \t]+(\S(?:.*\S)?)[ \t]*$", re.M)
_DECADE = re.compile(r"(\d{1,3}0)s")
ROUND_PLURAL = {"100": "hundreds", "1000": "thousands"}  # "1000s of people", not "one thousands"
# "24/7", "50/50", "3/4", "1 1/2", once _ENGLISH_DATE has taken the dates
CLOCK = {"am": "A M", "pm": "P M"}  # read as letters: "ten am" came out as "tenement"
_ENGLISH_PAIR = re.compile(r"(?<![\w.,/:])(?:(\d{1,2}) )?(\d+)/(\d+)(?![\w/]|[.,:]\d)")
FRACTION_NAMES = {2: "half", 4: "quarter"}
# Tenths are left out: "8/10" is usually a score, not a fraction.
FRACTION_DENOMINATORS = {2, 3, 4, 5, 6, 7, 8, 9, 12, 16}
# "2024-03-12", rewritten as "12/03/2024" so the date rule of each language reads it
_ISO_DATE = re.compile(r"(?<![\w.,/:])(?<!\d-)(\d{4})-(\d{2})-(\d{2})(?![\w/-]|[.,:]\d)")
_NUMBER = r"\d+(?:[.,]\d+)*"
_MONEY_AMOUNT = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?"
# "250 ₪", "₪250", "49.90 שקלים", with an attached prefix: "ב-250 ₪" -> "במאתיים וחמישים שקלים"
_HEBREW_MONEY = re.compile(
    rf"(?<![{HEBREW}\w.,/:])(?:([{PREFIXES}]{{1,3}})-)?"
    rf"(?:([₪$€£])\s*{_MONEY_AMOUNT}(?![\d\w]|[.,]\d)"
    rf"|{_MONEY_AMOUNT}(?:\s*([₪$€£])|\s+(שקלים|שקל|דולרים|דולר|יורו|פאונד)(?![{HEBREW}])))"
)
_SCALE_CURRENCY = re.compile(r"(מיליון|מיליארד|אלף)\s*([₪$€£])")  # "1.5 מיליון ₪"
_CURRENCY_AFTER = re.compile(rf"({_NUMBER})\s*([₪$€£])")
_CURRENCY_BEFORE = re.compile(rf"([₪$€£])\s*({_NUMBER})")
# An attached prefix ("ב-12/03") joins the expanded words without its hyphen ("בשנים עשר").
_DATE = re.compile(rf"(?:([{PREFIXES}]{{1,3}})-?)?(?<!\d)(\d{{1,2}})[/.](\d{{1,2}})[/.](\d{{4}}|\d{{2}})(?!\d)")
_ARABIC_DATE = re.compile(r"(?<![\d/.])(\d{1,2})[/.](\d{1,2})[/.](\d{4}|\d{2})(?!\d)")
_ARABIC_NUMBER = re.compile(rf"(?:([₪$€£])\s?)?(?<![\w.,/:])({_NUMBER})(?::(\d{{2}})(?!\d))?(?:\s?([%٪]))?(?:\s?([₪$€£]))?")
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


def _reorder_iso_date(m: re.Match) -> str:
    year, month, day = m.groups()
    return f"{day}/{month}/{year}" if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 else m.group(0)


def _hebrew_count(n: int, one: str, many: str, gender: str) -> str:
    if n == 1:
        return f"{one} {num2words.cardinal(1, gender)}"
    return f"{num2words.cardinal(n, gender, construct=n == 2)} {many}"


def _expand_hebrew_money(m: re.Match) -> str:
    """"49.90 ₪" -> "ארבעים ותשעה שקלים ותשעים אגורות", "1 ₪" -> "שקל אחד"."""
    prefix, symbol, whole, cents, whole2, cents2, symbol2, word = m.groups()
    symbol = symbol or symbol2 or HEBREW_CURRENCY_WORDS[word]
    whole, cents = int((whole or whole2).replace(",", "")), cents or cents2
    if whole >= 1_000_000:  # past what cardinal() reads; the general rules below still do
        return m.group(0)
    one, many, cent_one, cent_many = HEBREW_MONEY[symbol]
    if word and whole != 1:
        many = word  # "3 דולרים" stays "דולרים"
    parts = []
    if whole or not cents or not int(cents):
        parts.append(_hebrew_count(whole, one, many, num2words.MASC))
    if cents and int(cents):
        gender = num2words.FEM if cent_one in FEMININE_CENTS else num2words.MASC
        parts.append(_hebrew_count(int(cents.ljust(2, "0")), cent_one, cent_many, gender))
    return (prefix or "") + " ו".join(parts)


def _in_english(text: str, start: int, end: int) -> tuple[bool, str]:
    """Whether the number at text[start:end] is in English, and the word before it."""
    before = _WORD_BEFORE.search(text[max(0, start - 200) : start])
    after = _WORD_AFTER.search(text[end:])
    word = before.group(1).lower() if before else ""
    english = _LATIN.match(after.group(1)) if after else before and _LATIN.match(word)
    return bool(english) or _english_around(text, start), word


def _english_around(text: str, pos: int) -> bool:
    """Whether the sentence at pos is English: Latin letters and no Hebrew or Arabic. With no
    letters in it, its line decides, then the whole text."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    line = text[start : len(text) if end < 0 else end]
    pos -= start
    sentence_start = max((m.end() for m in _SENTENCE_END.finditer(line, 0, pos)), default=0)
    sentence_end = _SENTENCE_END.search(line, pos)
    for scope in (line[sentence_start : sentence_end.end() if sentence_end else len(line)], line, text):
        if _ANY_LETTER.search(scope):
            return bool(_LATIN.search(scope)) and not _NOT_ENGLISH_LETTER.search(scope)
    return False


def _in_arabic(text: str, start: int, end: int) -> bool:
    """Whether the number at text[start:end] is in Arabic, going by the nearest letter."""
    letter = _LETTER_AFTER.search(text[end:]) or _LETTER_BEFORE.search(text[max(0, start - 200) : start])
    return bool(letter and _ARABIC_LETTER.match(letter.group(1)))


def _arabic(n: int) -> str:
    return num2words_en(n, lang="ar")


def _expand_arabic_date(m: re.Match) -> str:
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (_in_arabic(m.string, m.start(), m.end()) and 1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    return f"{_arabic(day)} {ARABIC_MONTHS[month - 1]} {_arabic(year + (2000 if len(m.group(3)) == 2 else 0))}"


def _expand_arabic_number(m: re.Match) -> str:
    if not _in_arabic(m.string, m.start(2), m.end(2)):
        return m.group(0)
    cur_before, number, minutes, percent, cur_after = m.groups()
    whole, _, fraction = number.replace(",", "").partition(".")
    words = _arabic(int(whole))
    if minutes:
        words += " و " + _arabic(int(minutes)) if int(minutes) else ""
    if fraction:
        words += " فاصلة " + " ".join(_arabic(int(d)) for d in fraction)
    if percent:
        words += " بالمئة"
    currency = cur_before or cur_after
    return f"{words} {ARABIC_CURRENCY[currency]}" if currency else words


def _expand_english_date(m: re.Match) -> str:
    day, sep, month, year = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    english, before = _in_english(m.string, m.start(), m.end())
    if not (english and 1 <= day <= 31 and 1 <= month <= 12) or not (year or sep == "/" and before in DATE_BEFORE):
        return m.group(0)
    words = f"the {num2words_en(day, to='ordinal')} of {MONTHS[month - 1]}"
    if year:
        words += " " + num2words_en(int(year) + (2000 if len(year) == 2 else 0), to="year")
    return words


def _expand_list_item(m: re.Match) -> str:
    """"2. Open the app" -> "2: Open the app.": a number alone as a sentence often comes out
    garbled, and an item with no full stop runs into the next item's number. Of colon, comma,
    dash and semicolon, Whisper heard the number best after a colon: 12 of 12 with a clone."""
    number, item = m.groups()
    return f"{number}: {item}" + ("" if item[-1] in ".!?؟…:;," else ".")


def _split_glued_number(m: re.Match) -> str:
    """"16GB" -> "16 GB", so the number gets read; only in English, "1990s" -> "nineteen nineties"."""
    word = m.group(0)
    if _ORDINAL.fullmatch(word) or not _english_around(m.string, m.start()):
        return word
    decade = _DECADE.fullmatch(word)
    if decade:
        if decade.group(1) in ROUND_PLURAL:
            return ROUND_PLURAL[decade.group(1)]
        words = num2words_en(int(decade.group(1)), to="year")
        return words[:-1] + "ies" if words.endswith("y") else words + "s"
    return " ".join(CLOCK.get(part.lower(), part) for part in _GLUED_PARTS.findall(word))


def _english_fraction(a: int, b: int, mixed: bool) -> str | None:
    """"3/4" -> "three quarters", "1 1/2" -> "and a half"; None when it isn't a fraction."""
    if a >= b or b not in FRACTION_DENOMINATORS:
        return None
    name = FRACTION_NAMES.get(b) or num2words_en(b, to="ordinal")
    if a > 1:
        name = "halves" if name == "half" else name + "s"
    count = "a" if mixed and a == 1 else num2words_en(a)
    return f"{'and ' if mixed else ''}{count} {name}"


def _expand_english_pair(m: re.Match) -> str:
    if not _in_english(m.string, m.start(), m.end())[0]:
        return m.group(0)
    whole, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    words = _english_fraction(a, b, bool(whole)) or f"{num2words_en(a)}{' over ' if a < b else ' '}{num2words_en(b)}"
    return f"{num2words_en(int(whole))} {words}" if whole else words


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
    """Abbreviations, currency, dates and numbers into words: English or Arabic words for a
    number in those languages, Hebrew for the rest."""
    for pattern, reading in _ABBREVIATION_PATTERNS:
        text = pattern.sub(lambda m, r=reading: m.group(1) + r, text)
    text = _LIST_ITEM.sub(_expand_list_item, text)
    text = _ISO_DATE.sub(_reorder_iso_date, text)
    text = _ARABIC_DATE.sub(_expand_arabic_date, text)
    text = _ARABIC_NUMBER.sub(_expand_arabic_number, text)
    text = _GLUED_NUMBER.sub(_split_glued_number, text)
    text = _ENGLISH_DATE.sub(_expand_english_date, text)
    text = _ENGLISH_PAIR.sub(_expand_english_pair, text)
    text = _ENGLISH_NUMBER.sub(_expand_english, text)
    text = _HEBREW_MONEY.sub(_expand_hebrew_money, text)
    text = _SCALE_CURRENCY.sub(lambda m: f"{m.group(1)} {CURRENCY[m.group(2)]}", text)
    text = _CURRENCY_BEFORE.sub(lambda m: f"{m.group(2)} {CURRENCY[m.group(1)]}", text)
    text = _CURRENCY_AFTER.sub(lambda m: f"{m.group(1)} {CURRENCY[m.group(2)]}", text)
    text = _DATE.sub(_expand_date, text)
    return num2words.normalize_numbers(text)


def prepare(text: str, dictionary: Dictionary, normalize_text: bool = True) -> str:
    """The dictionary first, so a user's entry beats the built-in abbreviations."""
    text = dictionary.apply(text)
    return normalize(text) if normalize_text else text
