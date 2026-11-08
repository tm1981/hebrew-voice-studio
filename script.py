"""Turn text into speech and pause segments.

Pauses (always on):  [הפסקה] = 1 second, [הפסקה 2] / [pause 1.5] = that many seconds (max 10).
Dialogue (opt-in):   lines like "נועה: שלום, מה שלומך?" are spoken by the voice mapped to
                     that speaker; lines without a speaker continue the previous one.
"""
import re
from difflib import SequenceMatcher

PAUSE =re.compile(r"\[\s*(?:הפסקה|pause)\s*(\d+(?:\.\d+)?)?\s*\]", re.IGNORECASE)
SPEAKER_LINE = re.compile(r"^\s*([^:\n\[\]]{1,40}?)\s*:\s*(.+)$")
MAX_PAUSE = 10.0
# Added to the pause between sentences: after a new line (a heading, a list item, the next
# speaker) and after a blank line (a new paragraph), so a long text breathes like a person reading.
LINE_PAUSE = 0.25
PARAGRAPH_PAUSE = 0.5
_LINE_BREAK = re.compile(r"[ \t\r]*\n(?:[ \t\r]*\n)*[ \t\r]*")


def word_map(typed: str, spoken: list[str]) -> list[tuple[str, tuple[int, int] | None]]:
    """The typed text in pieces (words and the spaces between them), each word with the range of
    spoken words it became: "עולה 250 ₪" read as "עולה מאתיים וחמישים שקלים" gives עולה -> (0, 0),
    and 250 and ₪ -> (1, 3) together, so the page can mark what you typed while it's heard.
    Spaces, and words that weren't read (a [pause] marker), get None."""
    pieces = [p for p in re.split(r"(\s+)", typed) if p]
    words = [i for i, p in enumerate(pieces) if not p.isspace()]
    ranges: dict[int, tuple[int, int]] = {}
    lead = None  # spoken words before the first typed word that was read
    matcher = SequenceMatcher(None, [pieces[i] for i in words], spoken, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                ranges[words[i1 + k]] = (j1 + k, j1 + k)
        elif op == "replace":  # "250 ₪" -> three words: the typed words share the spoken ones
            for k in range(i1, i2):
                ranges[words[k]] = (j1, j2 - 1)
        elif op == "insert":  # spoken words with nothing typed for them join the word before
            before = [w for w in words[:i1] if w in ranges]
            if before:
                ranges[before[-1]] = (ranges[before[-1]][0], j2 - 1)
            else:
                lead = j1
    if ranges and lead is not None:  # the first word that was read takes them
        first = min(ranges)
        ranges[first] = (lead, ranges[first][1])
    return [(p, ranges.get(i)) for i, p in enumerate(pieces)]


def line_pieces(text: str) -> list[tuple[str, float]]:
    """The text cut at line breaks: [(piece, extra pause after it)], LINE_PAUSE after a new
    line, PARAGRAPH_PAUSE after one or more blank lines, nothing after the last piece."""
    pieces: list[tuple[str, float]] = []
    last = 0
    for m in _LINE_BREAK.finditer(text):
        if text[last : m.start()].strip():
            pieces.append((text[last : m.start()].strip(), PARAGRAPH_PAUSE if m.group(0).count("\n") > 1 else LINE_PAUSE))
        last = m.end()
    if text[last:].strip():
        pieces.append((text[last:].strip(), 0.0))
    if pieces:
        pieces[-1] = (pieces[-1][0], 0.0)
    return pieces


def detect_speakers(text: str) -> list[str]:
    """Speaker names in order of first appearance, for the UI to map to voices."""
    names = []
    for line in text.splitlines():
        m = SPEAKER_LINE.match(line)
        if m and m.group(1) not in names and not m.group(1).strip().isdigit():
            names.append(m.group(1))
    return names


def _split_pauses(text: str, voice: str) -> list[tuple]:
    segments, last = [], 0
    for m in PAUSE.finditer(text):
        if text[last:m.start()].strip():
            segments.append(("speech", voice, text[last:m.start()].strip()))
        segments.append(("pause", min(float(m.group(1) or 1.0), MAX_PAUSE)))
        last = m.end()
    if text[last:].strip():
        segments.append(("speech", voice, text[last:].strip()))
    return segments


def parse(text: str, default_voice: str, dialogue: bool = False, speakers: dict[str, str] | None = None) -> list[tuple]:
    """-> [("speech", voice_id, text) | ("pause", seconds)], adjacent speech of one voice merged."""
    speakers = speakers or {}
    raw: list[tuple] = []
    if not dialogue:
        raw = _split_pauses(text, default_voice)
    else:
        voice = default_voice
        for line in text.splitlines():
            m = SPEAKER_LINE.match(line)
            if m and not m.group(1).strip().isdigit():
                voice = speakers.get(m.group(1), default_voice)
                line = m.group(2)
            raw += _split_pauses(line, voice)
    merged: list[tuple] = []
    for seg in raw:
        if merged and seg[0] == "speech" and merged[-1][0] == "speech" and merged[-1][1] == seg[1]:
            merged[-1] = ("speech", seg[1], f"{merged[-1][2]}\n{seg[2]}")  # newline: a sentence break
        else:
            merged.append(seg)
    return merged
