"""Downloads the third-party code the app runs (BlueTTS) into vendor/, at a fixed version.

    python tools/fetch_vendor.py

It isn't kept in this repository because BlueTTS has its own (MIT) license.
The Docker image runs this while building.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"

BLUETTS = ("https://github.com/maxmelichov/BlueTTS/archive/712f7c9491ad7875fc985e24e166afe43ae649b0.zip",
           VENDOR / "BlueTTS")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main() -> int:
    url, target = BLUETTS
    if (target / "src" / "blue_onnx").exists():
        print(f"{target.relative_to(ROOT)} already there")
    else:
        print(f"Downloading BlueTTS into {target.relative_to(ROOT)}...")
        with zipfile.ZipFile(io.BytesIO(fetch(url))) as z:
            prefix = z.namelist()[0].split("/")[0] + "/"
            for name in z.namelist():
                rel = name[len(prefix):]
                if not rel or name.endswith("/"):
                    continue
                out = target / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
