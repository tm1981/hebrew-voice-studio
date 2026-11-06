"""Downloads the third-party code the app runs into vendor/, at fixed versions.

    python tools/fetch_vendor.py

It isn't kept in this repository because each project has its own license.
The Docker image runs this while building.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"

# name, zip of a pinned commit, folder, a file or folder that shows it's already there
PROJECTS = [
    ("BlueTTS", "https://github.com/maxmelichov/BlueTTS/archive/712f7c9491ad7875fc985e24e166afe43ae649b0.zip",
     VENDOR / "BlueTTS", "src/blue_onnx"),
    ("Breeze TTS", "https://github.com/breezeblue-ai/breeze-tts/archive/daa6cb2efe5471f09369229f322d20bb956cb3c3.zip",
     VENDOR / "breeze-tts", "models/breeze.py"),
    ("Ming-3.0-tts", "https://github.com/inclusionAI/Ming-3.0-tts/archive/a39b80660bdc5a4394d5e1d41af6dbfd9c443a54.zip",
     VENDOR / "Ming-3.0-tts", "ming_tts.py"),
    ("ComfyUI", "https://github.com/comfyanonymous/ComfyUI/archive/d4056d50c3c5b7d32b4e78e2bc2a724406053843.zip",
     VENDOR / "ComfyUI", "comfy/model_management.py"),
    ("ComfyUI-Drama3", "https://github.com/PedroMarinhoDev/ComfyUI-Drama3/archive/8cb3c64379d4988b9693cae9f3436b8fd54b22ad.zip",
     VENDOR / "ComfyUI-Drama3", "drama3/generate.py"),
]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main() -> int:
    for project, url, target, marker in PROJECTS:
        if (target / marker).exists():
            print(f"{target.relative_to(ROOT)} already there")
            continue
        print(f"Downloading {project} into {target.relative_to(ROOT)}...")
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
