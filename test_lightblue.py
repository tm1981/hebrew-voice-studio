"""LightBlue (BlueTTS) Hebrew smoke test: one sample per bundled voice.

Usage (voice env, Python 3.12):
    conda activate voice
    python test_lightblue.py
"""
import subprocess
import sys

# LightBlue opens its files without an encoding; on Windows that needs Python's UTF-8 mode.
if not sys.flags.utf8_mode:
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).parent
BLUE_REPO = ROOT / "vendor" / "BlueTTS"
sys.path.insert(0, str(BLUE_REPO))
from src.blue_onnx import limit_peak, load_text_to_speech, load_voice_style  # noqa: E402

ONNX_DIR = ROOT / "models" / "blue-onnx-v2"
# Local path: RenikudPlus's auto-download also fetches a datastore.json that is
# missing from its Hugging Face repo, and fails.
RENIKUD_MODEL = ROOT / "models" / "renikud-plus" / "model.onnx"
OUT_DIR = ROOT / "outputs" / "lightblue"
TEXT = "מזג האוויר בתל אביב נעים מאוד, והים שקט ויפה. הפגישה נקבעה ליום שלישי בשעה שלוש וחצי."

OUT_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
tts = load_text_to_speech(
    onnx_dir=str(ONNX_DIR), config_path=str(ONNX_DIR / "tts.json"), renikud_path=str(RENIKUD_MODEL)
)
print(f"Loaded in {time.perf_counter() - t0:.1f}s")

tts("שלום", lang="he", style=load_voice_style([str(BLUE_REPO / "voices" / "noa.json")]), total_step=5)  # warm-up
for voice_file in sorted((BLUE_REPO / "voices").glob("*.json")):
    style = load_voice_style([str(voice_file)])
    t0 = time.perf_counter()
    audio, _ = tts(TEXT, lang="he", style=style, total_step=5, cfg_scale=4.0)
    elapsed = time.perf_counter() - t0
    audio = limit_peak(audio[0] if audio.ndim == 2 else audio)
    seconds = len(audio) / tts.sample_rate
    out = OUT_DIR / f"{voice_file.stem}.wav"
    sf.write(out, audio, tts.sample_rate)
    print(f"{voice_file.stem}: {seconds:.1f}s audio in {elapsed:.2f}s (RTF {elapsed / seconds:.3f}) -> {out.name}")
