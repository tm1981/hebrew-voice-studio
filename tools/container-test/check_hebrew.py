"""How correctly an engine reads Hebrew: test sentences with each of its voices (at most three),
listened to by the app's own speech-to-text.

    python tools/container-test/check_hebrew.py voxtream zonos3
    python tools/container-test/check_hebrew.py --rounds 4 voxtream     (60 sentences in a row per voice)
    python tools/container-test/check_hebrew.py --long qwen chatterbox  (8 long news-style sentences)

    python tools/container-test/check_hebrew.py --long --options "{\"temperature\": 0.3}" chatterbox

--options are the engine's Advanced settings for the run, as JSON.
More rounds show whether an engine gets worse the longer it runs: each round is one line.
--long reads hebrew_long.txt instead of the 15 short sentences: 100 to 150 letters each, with
numbers, names and quotes. What each was meant to say is the app's own preparation of it (numbers
as words), so a number the preparation itself gets wrong doesn't show here; tests/test_text_prep.py
covers that.

A sentence is "wrong" when words were added or dropped, or a run of words was replaced by a
different number of words; "one word off" is a same-length swap, often just the listener's
spelling. Whisper mishears a little too, so compare engines rather than read the numbers as exact.
"""
import difflib
import json
import re
import sys
import time
from pathlib import Path

from common import OUT, free_gpu, post, read, voices

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
rounds = int(args.pop(args.index("--rounds") + 1)) if "--rounds" in args else 1
long_ones = "--long" in args
settings = json.loads(args.pop(args.index("--options") + 1)) if "--options" in args else {}
args = [a for a in args if a not in ("--rounds", "--long", "--options")]

SENTENCES = [s for s in (HERE / ("hebrew_long.txt" if long_ones else "hebrew_sentences.txt")).read_text(encoding="utf-8").split("\n")
             if s.strip()]
EXPECTED = SENTENCES
if long_ones:  # compare with the words the engines are given
    sys.path.insert(0, str(HERE.parents[1]))
    from text_prep import Dictionary, prepare

    EXPECTED = [prepare(s, Dictionary(), True) for s in SENTENCES]
TOTAL = len(SENTENCES)


def words(text: str) -> list[str]:
    return re.sub(r"[^א-ת ]", "", text.replace("־", " ").replace("-", " ")).split()


ids = voices()
for engine in args:
    for voice in [i for i in ids if i.startswith(engine + "/")][:3]:
        for round_number in range(1, rounds + 1):
            wrong = off = failed = 0
            seconds = work = 0.0
            for n, text in enumerate(SENTENCES):
                started = time.time()
                r = read(text, voice, options={engine: settings})
                work += time.time() - started
                if not r.ok:
                    failed += 1
                    continue
                (OUT / f"hebrew_{engine}_{n}.wav").write_bytes(r.content)
                heard = post("/stt/transcribe", files={"file": ("a.wav", r.content, "audio/wav")}, data={"language": "he"},
                             timeout=600).json()
                seconds += heard.get("duration", 0)
                said = heard.get("text", "")
                if long_ones:  # the listener writes "22" and "18%": into words, the same way as the expected text
                    said = prepare(said, Dictionary(), True)
                ops = difflib.SequenceMatcher(None, words(EXPECTED[n]), words(said), autojunk=False).get_opcodes()
                bad = any(op in ("insert", "delete") or (op == "replace" and i2 - i1 != j2 - j1) for op, i1, i2, j1, j2 in ops)
                wrong += bad
                off += (not bad) and any(op == "replace" for op, *_ in ops)
                if bad:
                    print(f"    {voice} {n}: heard \"{heard.get('text', '').strip()}\"", flush=True)
                else:  # which words came out as other words
                    want, got = words(EXPECTED[n]), words(said)
                    swaps = [f"{' '.join(want[i1:i2])} > {' '.join(got[j1:j2])}" for op, i1, i2, j1, j2 in ops if op == "replace"]
                    if swaps:
                        print(f"    {voice} {n}: {'; '.join(swaps)}", flush=True)
            print(f"{voice} round {round_number}: wrong {wrong}/{TOTAL}, one word off {off}/{TOTAL}, failed {failed}, "
                  f"{seconds:.0f}s of audio in {work:.0f}s", flush=True)
    free_gpu()
