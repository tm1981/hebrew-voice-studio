"""Turn text into speech and pause segments.

Pauses (always on):  [הפסקה] = 1 second, [הפסקה 2] / [pause 1.5] = that many seconds (max 10).
Dialogue (opt-in):   lines like "נועה: שלום, מה שלומך?" are spoken by the voice mapped to
                     that speaker; lines without a speaker continue the previous one.
"""
import re

PAUSE = re.compile(r"\[\s*(?:הפסקה|pause)\s*(\d+(?:\.\d+)?)?\s*\]", re.IGNORECASE)
SPEAKER_LINE = re.compile(r"^\s*([^:\n\[\]]{1,40}?)\s*:\s*(.+)$")
MAX_PAUSE = 10.0


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
