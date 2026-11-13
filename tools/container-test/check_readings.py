"""Twelve short readings that each broke something once: units, times, list numbers, a one-word
sentence, Hebrew and Arabic digits, a cloned voice. Prints "ok" or "FAILED" for each.

    python tools/container-test/check_readings.py
"""
import time

from common import OUT, read, voices

clone = next((v for v in voices() if v.startswith("chatterbox/") and v != "chatterbox/default"), "chatterbox/default")
READINGS = [
    ("a", "chatterbox/default", "The M4 chip has 16GB of memory and runs at 3.5GHz."),
    ("b", "chatterbox/default", "We open at 10am and close at 5pm, 24/7."),
    ("c", "chatterbox/default", "In the 1990s we had 2 cars."),
    ("d", "chatterbox/default", "No."),
    ("e", "chatterbox/default", "No. 5"),
    ("f", "chatterbox/default", "2. Open the app."),
    ("g", clone, "The new model has 4K video and 5G."),
    ("k", clone, "1. First item\n2. Second item\n3. Third item"),
    ("l", "chatterbox/default", "1. First item\n2. Second item\n3. Third item"),
    ("h", "chatterbox/default", "יש לי 5 ילדים."),
    ("i", "chatterbox/default", "اشترى 5 كتب."),
    ("j", "lightblue/noa", "The M4 chip has 16GB of memory."),
]
failed = 0
for tag, voice, text in READINGS:
    started = time.time()
    r = read(text, voice, timeout=300)
    if r.ok:
        (OUT / f"reading_{tag}.wav").write_bytes(r.content)
        print(tag, f"ok {time.time() - started:.1f}s", flush=True)
    else:
        failed += 1
        print(tag, "FAILED", r.status_code, r.text[:200], flush=True)
print(f"{len(READINGS) - failed} of {len(READINGS)} ok; audio in {OUT}")
raise SystemExit(1 if failed else 0)
