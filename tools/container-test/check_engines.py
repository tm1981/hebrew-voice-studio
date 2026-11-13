"""Three Hebrew sentences through each engine named, with the load time of the first, the GPU
memory it takes, and whether unloading gives it back.

    python tools/container-test/check_engines.py qwen magpie zonos3
    python tools/container-test/check_engines.py            (every GPU engine)
"""
import sys
import time

from common import OUT, free_gpu, get, gpu_used_mb, read, voices

SENTENCES = ["היום נדבר על מזג האוויר בירושלים.", "אני מאוד שמח לראות אתכם כאן היום.",
             "הרכבת יוצאת מהתחנה המרכזית בשעה שמונה."]
names = sys.argv[1:] or [e["name"] for e in get("/status").json()["engines"] if e["gpu"] and e["name"] != "stt"]
ids, failed = voices(), 0
for engine in names:
    mine = [i for i in ids if i.startswith(engine + "/")]
    if not mine:
        print(engine, "has no voices here (not in this image, or it only clones and no voice is recorded)", flush=True)
        continue
    free_gpu()
    time.sleep(4)
    before = gpu_used_mb()
    for n, text in enumerate(SENTENCES):
        started = time.time()
        r = read(text, mine[0])
        if r.ok:
            (OUT / f"{engine}_{n}.wav").write_bytes(r.content)
        else:
            failed += 1
        print(f"  {engine} {n}: {r.status_code} in {time.time() - started:.0f}s, GPU +{gpu_used_mb() - before} MB"
              + ("" if r.ok else f" {r.text[:300]}"), flush=True)
    free_gpu()
    time.sleep(5)
    print(f"  {engine}: after unload GPU +{gpu_used_mb() - before} MB", flush=True)
print("audio in", OUT)
raise SystemExit(1 if failed else 0)
