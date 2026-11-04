"""Copies Hugging Face models the app needs from other caches on this PC into the project's
own cache (models/hf), so nothing is downloaded again and nothing outside the project is used.

Usage:  python tools/copy_hf_models.py [extra source cache dirs...]
Sources default to %USERPROFILE%/.cache/huggingface/hub. Symlinks are resolved, so the copy
has real files (Windows without Developer Mode can't create the links anyway). A model that
isn't found anywhere is simply downloaded into models/hf on the first start.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "hf" / "hub"
REPOS = [
    "ResembleAI/chatterbox",                          # Chatterbox Multilingual
    "k2-fsa/OmniVoice",                               # OmniVoice
    "openai/whisper-large-v3-turbo",                  # OmniVoice: transcribes cloned-voice recordings
    "multimodalart/higgs-audio-v3-tts-4b-transformers",  # Higgs Audio v3
    "bosonai/higgs-audio-v2-tokenizer",               # Higgs audio codec
    "dicta-il/dictabert-large-char-menaked",          # tokenizer for the Dicta and Phonikud niqqud models
    "vibevoice/VibeVoice-1.5B-hf",                    # VibeVoice 1.5B
    "vibevoice/VibeVoice-7B-hf",                      # VibeVoice 7B
]


def main():
    sources = [Path.home() / ".cache" / "huggingface" / "hub", *map(Path, sys.argv[1:])]
    TARGET.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        folder = "models--" + repo.replace("/", "--")
        if (TARGET / folder / "snapshots").exists():
            print(f"have     {repo}")
            continue
        source = next((s / folder for s in sources if (s / folder / "snapshots").exists()), None)
        if source is None:
            print(f"missing  {repo} (downloads on first start)")
            continue
        for part in ("refs", "snapshots"):
            if (source / part).exists():
                shutil.copytree(source / part, TARGET / folder / part, symlinks=False, dirs_exist_ok=True)
        size = sum(f.stat().st_size for f in (TARGET / folder).rglob("*") if f.is_file()) / 1e9
        print(f"copied   {repo} from {source.parent} ({size:.1f} GB)")


if __name__ == "__main__":
    main()
