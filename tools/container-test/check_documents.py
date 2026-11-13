"""A long-document job from start to ZIP, then a pin and a download of one of its chapters.
Deletes the chapters it made.

    python tools/container-test/check_documents.py
"""
import time

import requests

from common import HEADERS, URL, get, post

text = "\n\n".join(f"Part {i}. " + " ".join(f"This is sentence {j} of the part." for j in range(40)) for i in range(3))
print("library before:", get("/settings/library").json())
r = post("/documents", files={"file": ("container test.txt", text.encode(), "text/plain")},
         data={"settings": '{"voice": "lightblue/noa"}'})
job = r.json()
print("job:", r.status_code, job.get("id"), "chapters:", job.get("total"))
while job.get("status") == "running":
    time.sleep(1)
    job = get(f"/jobs/{job['id']}").json()
print("finished:", job["status"], job["done"], "of", job["total"], job.get("error") or "")
ok = job["status"] == "done"
archive = get(f"/documents/{job['id']}/zip")
print("zip:", archive.status_code, len(archive.content), "bytes")
if job["items"]:
    item = job["items"][0]
    pinned = requests.put(f"{URL}/library/{item}/pin", headers=HEADERS, json={"pinned": True}).json()
    listed = [i["pinned"] for i in get("/library").json() if i["id"] == item]
    audio = get(f"/library/{item}/audio?format=mp3")
    print("pin:", pinned, "| in the list:", listed, "| mp3:", audio.status_code, len(audio.content), "bytes")
    ok = ok and archive.ok and listed == [True] and audio.ok
    for i in job["items"]:
        requests.delete(f"{URL}/library/{i}", headers=HEADERS)
print("library after:", get("/settings/library").json())
raise SystemExit(0 if ok else 1)
