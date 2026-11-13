"""One sentence at speed 1.0, 1.1 and 0.9 through engines whose pace the server changes after
the audio is made (speed.py): the lengths should come out 10% shorter and longer.

    python tools/container-test/check_speed.py qwen drama
"""
import io
import sys

import soundfile as sf

from common import OUT, free_gpu, read, voices

TEXT = "היום נדבר על מזג האוויר בירושלים, ואחר כך נצא לטיול קצר בעיר העתיקה."
ids, failed = voices(), 0
for engine in sys.argv[1:] or ["qwen"]:
    voice = next((i for i in ids if i.startswith(engine + "/")), None)
    if not voice:
        print(engine, "has no voices here")
        continue
    normal = None
    for speed in (1.0, 1.1, 0.9):
        r = read(TEXT, voice, timeout=1200, speed=speed, options={engine: {"seed": 3}})  # the same take each time
        if not r.ok:
            failed += 1
            print(engine, speed, r.status_code, r.text[:200])
            continue
        (OUT / f"{engine}_speed_{speed}.wav").write_bytes(r.content)
        audio, rate = sf.read(io.BytesIO(r.content), dtype="float32")
        seconds = len(audio) / rate
        normal = normal or seconds
        ratio = normal / seconds
        failed += abs(ratio - speed) > 0.03
        print(f"{engine} at {speed}: {seconds:.2f}s, {ratio:.2f} times the normal pace", flush=True)
    free_gpu()
print("audio in", OUT)
raise SystemExit(1 if failed else 0)
