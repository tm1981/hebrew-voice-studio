"""Hebrew TTS smoke test for Chatterbox Multilingual.

Usage:
    python test_hebrew.py                      # default voice
    python test_hebrew.py --voice my_voice.wav # clone a voice (5-15s clean WAV)
    python test_hebrew.py --device cuda:1      # pick a GPU
"""
import argparse
import time
from pathlib import Path

import soundfile as sf
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox_fixes import DEFAULT_TEMPERATURE, guard_speech_tokens, load_dicta, optimize_t3

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs"

SENTENCES = [
    "שלום, מה שלומך היום?",
    "מזג האוויר בתל אביב נעים מאוד, והים שקט ויפה.",
    "אני מערכת המרת טקסט לדיבור שרצה מקומית על כרטיס המסך שלך.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", type=Path, help="reference WAV for voice cloning")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    load_dicta()

    print(f"Device: {args.device} ({torch.cuda.get_device_name(args.device) if 'cuda' in args.device else 'cpu'})")
    t0 = time.perf_counter()
    model = ChatterboxMultilingualTTS.from_pretrained(device=args.device)
    guard_speech_tokens(model)
    optimize_t3(model)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    OUT_DIR.mkdir(exist_ok=True)
    for i, text in enumerate(SENTENCES, 1):
        t0 = time.perf_counter()
        wav = model.generate(
            text,
            language_id="he",
            temperature=DEFAULT_TEMPERATURE,
            audio_prompt_path=str(args.voice) if args.voice else None,
        )
        elapsed = time.perf_counter() - t0
        audio = wav.squeeze(0).cpu().numpy()
        duration = len(audio) / model.sr
        out = OUT_DIR / f"hebrew_{i}.wav"
        sf.write(out, audio, model.sr)
        print(f"[{i}] {duration:.1f}s audio in {elapsed:.1f}s (RTF {elapsed / duration:.2f}) -> {out.name}")

    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
