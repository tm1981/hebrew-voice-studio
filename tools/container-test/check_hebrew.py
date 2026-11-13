"""How correctly an engine reads Hebrew: the 15 test sentences with each of its voices (at most
three), listened to by the app's own speech-to-text.

    python tools/container-test/check_hebrew.py voxtream zonos3
    python tools/container-test/check_hebrew.py --rounds 4 voxtream     (60 sentences in a row per voice)

More rounds show whether an engine gets worse the longer it runs: each round is one line.

A sentence is "wrong" when words were added or dropped, or a run of words was replaced by a
different number of words; "one word off" is a same-length swap, often just the listener's
spelling. Whisper mishears a little too, so compare engines rather than read the numbers as exact.
"""
import difflib
import re
import sys
import time
from pathlib import Path

from common import OUT, free_gpu, post, read, voices

SENTENCES = [s for s in (Path(__file__).parent / "hebrew_sentences.txt").read_text(encoding="utf-8").split("\n") if s.strip()]


def words(text: str) -> list[str]:
    return re.sub(r"[^א-ת ]", "", text).split()


args = sys.argv[1:]
rounds = int(args.pop(args.index("--rounds") + 1)) if "--rounds" in args else 1
args = [a for a in args if a != "--rounds"]
ids = voices()
for engine in args:
    for voice in [i for i in ids if i.startswith(engine + "/")][:3]:
      for round_number in range(1, rounds + 1):
        wrong = off = failed = 0
        seconds = work = 0.0
        for n, text in enumerate(SENTENCES):
            started = time.time()
            r = read(text, voice)
            work += time.time() - started
            if not r.ok:
                failed += 1
                continue
            (OUT / f"hebrew_{engine}_{n}.wav").write_bytes(r.content)
            heard = post("/stt/transcribe", files={"file": ("a.wav", r.content, "audio/wav")}, data={"language": "he"}, timeout=600).json()
            seconds += heard.get("duration", 0)
            ops = difflib.SequenceMatcher(None, words(text), words(heard.get("text", "")), autojunk=False).get_opcodes()
            bad = any(op in ("insert", "delete") or (op == "replace" and i2 - i1 != j2 - j1) for op, i1, i2, j1, j2 in ops)
            wrong += bad
            off += (not bad) and any(op == "replace" for op, *_ in ops)
            if bad:
                print(f"    {voice} {n}: heard \"{heard.get('text', '').strip()}\"", flush=True)
        print(f"{voice} round {round_number}: wrong {wrong}/15, one word off {off}/15, failed {failed}, "
              f"{seconds:.0f}s of audio in {work:.0f}s", flush=True)
    free_gpu()
