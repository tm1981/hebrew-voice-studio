"""Model files that don't come through an engine's own Hugging Face download, fetched into
models/ the first time an engine needs them (Hugging Face models the engines load themselves
already land in models/hf). Lets a fresh install, or the Docker image with an empty models
volume, start without the manual download steps.

Only the standard library and huggingface_hub here: the GPU workers import it from their envs.
"""
import os
import tempfile
import urllib.request
from pathlib import Path

from paths import MODELS_DIR

# name -> (kind, source, files, target): ("hf", repo, [files] or None for the whole repo, folder)
# or ("url", url, None, file)
SOURCES = {
    "blue-onnx-v2": ("hf", "notmax123/blue-onnx-v2", None, "blue-onnx-v2"),
    "bluetts-2.5": ("hf", "notmax123/BlueTTS2.5-onnx", None, "bluetts-2.5"),
    "blue-codec": ("hf", "notmax123/blue-codec", ["model.safetensors", "README.md"], "blue-codec"),
    "renikud-plus": ("hf", "notmax123/RenikudPlus", ["model.onnx"], "renikud-plus"),
    "kokoro-hebrew": ("hf", "thewh1teagle/kokoro-hebrew-nc", None, "kokoro-hebrew"),
    "phonikud": ("hf", "thewh1teagle/phonikud-onnx", ["phonikud-1.0.int8.onnx"], "kokoro-hebrew"),
    "kokoro-english-voices": ("url", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                                     "model-files-v1.0/voices-v1.0.bin", None, "kokoro-hebrew/voices-english.bin"),
    "dicta": ("url", "https://github.com/thewh1teagle/dicta-onnx/releases/download/"
                     "model-files-v1.0/dicta-1.0.int8.onnx", None, "dicta-1.0.int8.onnx"),
}


def ensure(name: str, marker: str) -> Path:
    """Path of models/<marker>, downloading SOURCES[name] first if it isn't there."""
    path = MODELS_DIR / marker
    if path.exists():
        return path
    kind, source, files, target = SOURCES[name]
    print(f"Downloading {name} into {MODELS_DIR.name}/{Path(target).parts[0]} (first use)...", flush=True)
    if kind == "hf":
        from huggingface_hub import snapshot_download

        snapshot_download(source, allow_patterns=files, local_dir=str(MODELS_DIR / target))
    else:
        out = MODELS_DIR / target
        out.parent.mkdir(parents=True, exist_ok=True)
        # to a temp name first: two processes may want the same file, and a broken
        # download must not look finished
        fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".part")
        os.close(fd)
        try:
            urllib.request.urlretrieve(source, tmp)
            os.replace(tmp, out)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    if not path.exists():
        raise FileNotFoundError(f"{name}: downloaded, but {path} is still missing")
    return path
